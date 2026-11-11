"""RF5 / RF10 / T4.1 — os 6 modelos do artigo, com GPU automática.

Responsabilidades deste módulo:

  1. Ler `config/experiment.yml` — a **única** fonte de hiperparâmetros (RF5).
     Nenhum valor de modelo é escrito em Python aqui; qualquer `params` que
     apareça neste arquivo é bug.
  2. Instanciar os 6 estimadores de SDD §4 D4 (LogisticRegression,
     RandomForest, XGBoost, LightGBM, CatBoost, MLP).
  3. Detectar GPU automaticamente e cair para CPU **sem intervenção e sem
     falhar** (RF10, RH4).

⛔ Este módulo NÃO treina nada (SDD §0). Ele devolve estimadores *não
ajustados*; quem chama `.fit()` é o laço da matriz (T4.3) ou o smoke (T4.4),
sempre por `ajustar`, que só roteia os grupos do MLP agrupado (AD-057).

Por que a padronização (`escala: true`) mora no config
------------------------------------------------------
LogisticRegression e MLP são embrulhados num `Pipeline` com `StandardScaler`.
As 41 features variam de 0/1 (booleanas) a ~65.000 (`url_length`, uint16):
sem padronizar, o lbfgs e o adam não convergem em número razoável de
iterações e o "baseline" mediria a falha de convergência, não o modelo. Os 4
modelos de árvore são invariantes a transformação monotônica e ficam crus.

Isso está declarado em `experiment.yml` (`escala:`) em vez de embutido aqui
para que seja auditável junto com o resto do contrato — não é um ajuste
escondido no código.

Determinismo (RNF1)
-------------------
Todos os estimadores recebem a `seed` do config. Em CPU os seis são
determinísticos (o LightGBM só com `deterministic: true` +
`force_row_wise: true`, que o config já fixa). Em GPU, o CatBoost não garante
bit-a-bit entre execuções; a mitigação é que a matriz principal (Etapa 1 do
D5) tem seed fixa e é rodada uma vez, e a auditoria de determinismo do RNF1 é
feita em CPU, nesta máquina, pelo smoke.
"""

from __future__ import annotations

import copy
import logging
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import _multilayer_perceptron as _mlp_mod

import splits

logger = logging.getLogger(__name__)

# Raiz do repositório, a partir deste arquivo (src/models.py -> repo/).
# RNF3: nenhum caminho absoluto no código.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "experiment.yml"

# Ordem canônica dos 6 modelos (SDD §4 D4). Usada em relatório e nos laços,
# para que a saída não dependa da ordem de iteração de um dicionário.
MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
)


class ModelConfigError(RuntimeError):
    """Config de experimento inválida — falha alto e claro (RH1)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_experiment_config(path: str | Path | None = None) -> dict:
    """Carrega e valida `config/experiment.yml`.

    Valida o suficiente para que um erro de digitação no YAML falhe aqui, com
    mensagem acionável, e não 40 minutos adiante no meio da matriz (RH1).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ModelConfigError(
            f"config do experimento não encontrada: {path}. "
            f"Ela é o contrato de hiperparâmetros (RF5) — sem ela não há baseline."
        )
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ModelConfigError(f"{path} não contém um mapeamento YAML no topo")

    for key in ("seed", "modelos", "corpora", "regimes", "split", "avaliacao"):
        if key not in config:
            raise ModelConfigError(f"{path}: chave obrigatória ausente: '{key}'")

    modelos = config["modelos"]
    if not isinstance(modelos, dict):
        raise ModelConfigError(f"{path}: 'modelos' deve ser um mapeamento")

    missing = [name for name in MODEL_ORDER if name not in modelos]
    if missing:
        raise ModelConfigError(
            f"{path}: faltam modelos exigidos por SDD §4 D4: {missing}. "
            f"Os 6 são obrigatórios — a comparação é por cobertura de família."
        )
    extra = [name for name in modelos if name not in MODEL_ORDER]
    if extra:
        raise ModelConfigError(
            f"{path}: modelos não previstos em SDD §4 D4: {extra}. "
            f"Adicionar modelo é mudança de decisão travada — edite o SDD primeiro."
        )
    for name, spec in modelos.items():
        if not isinstance(spec, dict) or "params" not in spec:
            raise ModelConfigError(f"{path}: modelo '{name}' precisa de um bloco 'params'")

    return config


def apply_smoke_overrides(config: dict) -> dict:
    """Devolve uma cópia da config com os `smoke.overrides` aplicados.

    O smoke valida fiação, não desempenho (T4.4). Os overrides cortam o
    orçamento de cada modelo para caber no limite do RF9. A cópia é profunda
    de propósito: nenhuma execução real pode herdar acidentalmente um
    `n_estimators: 40`.

    O `smoke.bootstrap_n` também é aplicado aqui, sobre `avaliacao`: o IC por
    bootstrap custa ~7 min na execução real e não tem o que validar no portão
    além da fiação da chamada, que os testes já cobrem. Fica em 0 por default —
    e se a chave existir na `smoke`, ela manda.
    """
    config = copy.deepcopy(config)
    smoke = config.get("smoke") or {}
    overrides = smoke.get("overrides") or {}
    for name, params in overrides.items():
        if name not in config["modelos"]:
            raise ModelConfigError(
                f"smoke.overrides menciona modelo inexistente: '{name}'"
            )
        config["modelos"][name]["params"].update(params)
    if "bootstrap_n" in smoke:
        config["avaliacao"]["bootstrap_n"] = int(smoke["bootstrap_n"])
    config["_smoke"] = True
    return config


# ---------------------------------------------------------------------------
# Detecção de GPU (RF10, RH4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    """Resultado da detecção de GPU. `available=False` nunca é erro."""

    available: bool
    n_devices: int = 0
    names: tuple[str, ...] = field(default_factory=tuple)
    detected_by: str = "nenhum"
    reason: str = ""

    def describe(self) -> str:
        if not self.available:
            return f"CPU (sem GPU NVIDIA detectada: {self.reason})"
        return f"GPU x{self.n_devices} via {self.detected_by}: {', '.join(self.names)}"


def _detect_via_nvidia_smi() -> GpuInfo | None:
    """`nvidia-smi -L` lista as GPUs. Ausente => sem driver NVIDIA."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi indisponível: %s", exc)
        return None
    if proc.returncode != 0:
        logger.debug("nvidia-smi retornou %s: %s", proc.returncode, proc.stderr.strip())
        return None
    names = tuple(
        line.split(":", 1)[-1].split("(UUID")[0].strip()
        for line in proc.stdout.splitlines()
        if line.strip().startswith("GPU")
    )
    if not names:
        return None
    return GpuInfo(
        available=True, n_devices=len(names), names=names, detected_by="nvidia-smi"
    )


def _detect_via_cuda_driver() -> GpuInfo | None:
    """Plano B: fala com o driver CUDA por `ctypes`.

    Existe para o caso de o `nvidia-smi` não estar no PATH (acontece em
    contêiner e em instalação parcial) enquanto o driver está presente. Só
    conta dispositivos; não aloca nada, não inicializa contexto de GPU.
    """
    import ctypes

    candidates = ("nvcuda.dll", "libcuda.so.1", "libcuda.so")
    lib = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None
    try:
        if lib.cuInit(0) != 0:
            return None
        count = ctypes.c_int(0)
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return None
    except (AttributeError, OSError) as exc:
        logger.debug("driver CUDA presente mas não utilizável: %s", exc)
        return None
    if count.value <= 0:
        return None
    return GpuInfo(
        available=True,
        n_devices=count.value,
        names=tuple(f"cuda:{i}" for i in range(count.value)),
        detected_by="driver CUDA (ctypes)",
    )


@lru_cache(maxsize=1)
def detect_gpu() -> GpuInfo:
    """Detecta GPU NVIDIA sem treinar nada e sem perguntar nada (RH4, RH5).

    Ordem: variável de ambiente de override → `nvidia-smi` → driver CUDA.
    Resultado em cache: a detecção acontece uma vez por processo, não 48 vezes
    no laço da matriz.

    `PHISHING_FORCE_CPU=1` força CPU. Existe para tornar o smoke reprodutível
    entre esta máquina (sem GPU) e a do Lucas (com GPU) quando se quer
    comparar as duas — não é uma flag de operação normal.
    """
    if os.environ.get("PHISHING_FORCE_CPU", "").strip() not in ("", "0", "false"):
        return GpuInfo(
            available=False, reason="PHISHING_FORCE_CPU definido no ambiente"
        )

    for probe in (_detect_via_nvidia_smi, _detect_via_cuda_driver):
        info = probe()
        if info is not None:
            logger.info("GPU detectada: %s", info.describe())
            return info

    info = GpuInfo(
        available=False,
        reason="nvidia-smi ausente/sem dispositivo e driver CUDA não carregável",
    )
    logger.info("Nenhuma GPU detectada — executando em CPU. %s", info.reason)
    return info


def gpu_params(model_name: str, config: dict, gpu: GpuInfo | None = None) -> dict:
    """Parâmetros de dispositivo a injetar em um modelo, dado o hardware.

    Devolve sempre um dicionário explícito — inclusive no caso CPU. Deixar o
    default da biblioteca decidir seria abrir espaço para a mesma execução
    render números diferentes em máquinas diferentes sem que nada no artefato
    registre isso.
    """
    gpu = gpu if gpu is not None else detect_gpu()
    gpu_cfg = config.get("gpu") or {}
    auto = bool(gpu_cfg.get("auto", True))
    enabled_for = set(gpu_cfg.get("habilitado_para") or ())
    use_gpu = auto and gpu.available and model_name in enabled_for

    if model_name == "xgboost":
        return {"device": "cuda" if use_gpu else "cpu"}
    if model_name == "catboost":
        if use_gpu:
            return {"task_type": "GPU", "devices": "0"}
        return {"task_type": "CPU"}
    # LightGBM e os 3 do scikit-learn: sempre CPU (ver nota em experiment.yml).
    return {}


# ---------------------------------------------------------------------------
# Construção dos estimadores
# ---------------------------------------------------------------------------


def _seed_kwarg(model_name: str, seed: int) -> dict:
    """Nome do parâmetro de semente varia por biblioteca."""
    if model_name == "catboost":
        return {"random_seed": seed}
    return {"random_state": seed}


def _n_jobs_kwarg(model_name: str, config: dict) -> dict:
    """`n_jobs` só para os estimadores em que ele tem efeito.

    - `RandomForest` declara o seu diretamente em `experiment.yml`.
    - `MLPClassifier` não tem o parâmetro (o paralelismo vem do BLAS).
    - `CatBoost` usa `thread_count`, deixado no default para não conflitar com
      o modo GPU.
    - `LogisticRegression` **não** entra aqui: desde o scikit-learn 1.8 o
      parâmetro não tem efeito algum (só paralelizava o esquema OvR
      multiclasse, e o nosso problema é binário), emite `FutureWarning` e está
      marcado para remoção na 1.10. Passá-lo faria a execução do Lucas quebrar
      numa wheel mais nova por um parâmetro que nunca fez nada. Medido com
      scikit-learn 1.9.0 em 04/08/2026.
    """
    n_jobs = (config.get("cpu") or {}).get("n_jobs", -1)
    if model_name in ("xgboost", "lightgbm"):
        return {"n_jobs": n_jobs}
    return {}


class MLPValidacaoAgrupada(MLPClassifier):
    """MLP cujo *early stopping* valida num corte agrupado por eTLD+1 (AD-057).

    O `MLPClassifier` nativo separa a validação interna por linha, estratificada
    pelo rótulo. No Regime B isso deixa o domínio do próprio treino na
    validação e o critério de parada enxerga domínio visto (REEX-09). Aqui, com
    `groups`, o corte vem de `splits._grupo_estratificado_once`: nenhum grupo
    em comum entre as linhas de ajuste e a validação, que fica com
    `validation_fraction` dos grupos (k = 1/fração, estratificado pelo rótulo).

    Como funciona: o laço de épocas, o `tol`, o `n_iter_no_change` e a
    restauração dos melhores pesos são os do próprio sklearn. Durante o `fit`,
    o `train_test_split` do módulo `sklearn.neural_network._multilayer_perceptron`
    é trocado por uma função que devolve o corte já calculado. Depende de um
    nome interno do sklearn 1.9.0 (versão pinada); se a função substituta não
    for chamada exatamente uma vez, o `fit` levanta `RuntimeError`.

    Sem `groups` o `fit` levanta `ValueError` (com `early_stopping` ligado):
    cair na validação por linha manteria a proveniência `agrupada_etld1` sem que
    a validação fosse agrupada. O Regime A não usa esta classe, e sim o
    `MLPClassifier` nativo (`build_model(..., validacao_agrupada=False)`).
    """

    def fit(self, X, y, groups=None):
        if not self.early_stopping:
            return super().fit(X, y)
        if groups is None:
            raise ValueError(
                "MLPValidacaoAgrupada exige `groups` (eTLD+1 de cada linha de "
                "treino) para a validação interna agrupada; sem eles a validação "
                "cairia por linha. No Regime A use o MLPClassifier nativo."
            )

        grupos = np.asarray(groups)
        rotulos = np.asarray(y)
        n = len(rotulos)
        if len(grupos) != n:
            raise ValueError(
                f"groups tem {len(grupos)} elementos, mas y tem {n}."
            )
        fit_idx, val_idx = splits._grupo_estratificado_once(
            grupos, rotulos, self.validation_fraction, self.random_state
        )
        chamadas: list[int] = []

        def substituto(*arrays, **kwargs):
            chamadas.append(1)
            if len(arrays) != 2:
                raise RuntimeError(
                    "o train_test_split do sklearn mudou de uso interno "
                    f"({len(arrays)} arrays em vez de X e y): revise a versão do sklearn."
                )
            X_, y_ = arrays
            if len(X_) != n:
                raise RuntimeError(
                    f"o sklearn pediu o corte de {len(X_)} linhas, mas os grupos "
                    f"têm {n}: revise a versão do sklearn."
                )
            return X_[fit_idx], X_[val_idx], y_[fit_idx], y_[val_idx]

        with patch.object(_mlp_mod, "train_test_split", substituto):
            super().fit(X, y)
        if len(chamadas) != 1:
            raise RuntimeError(
                f"o train_test_split do sklearn foi chamado {len(chamadas)} vez(es) "
                f"no fit do MLP, e o esperado é 1: a versão do sklearn mudou."
            )
        self.fracao_validacao_obtida_ = len(val_idx) / n
        self.n_grupos_validacao_ = int(len(np.unique(grupos[val_idx])))
        return self


def _instantiate(model_name: str, params: dict, validacao_agrupada: bool = False):
    """Importa a biblioteca certa e instancia o estimador cru.

    Os imports são locais de propósito: importar xgboost/lightgbm/catboost no
    topo custa segundos e ninguém que só quer ler a config precisa pagar isso.
    """
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**params)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)
    if model_name == "mlp":
        params = dict(params)
        # YAML devolve lista; o sklearn exige tupla para hidden_layer_sizes.
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        if validacao_agrupada:
            return MLPValidacaoAgrupada(**params)
        return MLPClassifier(**params)
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params)
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params)
    raise ModelConfigError(
        f"modelo desconhecido: {model_name!r}. Válidos: {list(MODEL_ORDER)}"
    )


def build_model(
    model_name: str,
    config: dict,
    gpu: GpuInfo | None = None,
    validacao_agrupada: bool = False,
):
    """Instancia um dos 6 modelos, **não ajustado**, pronto para `.fit()`.

    Os hiperparâmetros vêm inteiramente de `config['modelos'][model_name]`
    (RF5). A este conjunto somam-se apenas: a semente do config (RNF1), o
    `n_jobs` do config e os parâmetros de dispositivo escolhidos pela detecção
    de GPU (RF10).

    `validacao_agrupada=True` só tem efeito no `mlp`: devolve o
    `MLPValidacaoAgrupada` (Regime B, AD-057) com os mesmos `params`. Sem a
    opção, o `mlp` é o `MLPClassifier` nativo.

    Devolve um `Pipeline(StandardScaler, estimador)` quando o config marca
    `escala: true` — ver nota no topo do módulo.
    """
    if model_name not in MODEL_ORDER:
        raise ModelConfigError(
            f"modelo desconhecido: {model_name!r}. Válidos: {list(MODEL_ORDER)}"
        )
    spec = config["modelos"][model_name]
    seed = int(config["seed"])

    params = dict(spec.get("params") or {})
    params.update(_seed_kwarg(model_name, seed))
    params.update(_n_jobs_kwarg(model_name, config))
    params.update(gpu_params(model_name, config, gpu=gpu))

    estimator = _instantiate(model_name, params, validacao_agrupada=validacao_agrupada)

    if spec.get("escala", False):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [("escala", StandardScaler()), ("clf", estimator)]
        )
    return estimator


def build_all_models(config: dict, gpu: GpuInfo | None = None) -> dict[str, Any]:
    """Os 6 modelos, na ordem canônica de `MODEL_ORDER`."""
    gpu = gpu if gpu is not None else detect_gpu()
    return {name: build_model(name, config, gpu=gpu) for name in MODEL_ORDER}


def model_provenance(
    model_name: str,
    config: dict,
    gpu: GpuInfo | None = None,
    validacao_agrupada: bool = False,
) -> dict:
    """Registro auditável de como um modelo foi construído (RNF4).

    Vai junto com as métricas para que qualquer número do artigo possa ser
    rastreado até os hiperparâmetros exatos que o produziram — inclusive o
    dispositivo, que muda o resultado em GBM.

    `validacao_interna` (MLP-01): `"agrupada_etld1"` quando o MLP usou a
    validação agrupada, `"linha"` quando usou a nativa. Os outros cinco modelos
    não têm validação interna e gravam `None`.
    """
    gpu = gpu if gpu is not None else detect_gpu()
    spec = config["modelos"][model_name]
    if model_name == "mlp":
        validacao_interna = "agrupada_etld1" if validacao_agrupada else "linha"
    else:
        validacao_interna = None
    return {
        "validacao_interna": validacao_interna,
        "model": model_name,
        "familia": spec.get("familia"),
        "escala": bool(spec.get("escala", False)),
        "params": dict(spec.get("params") or {}),
        "seed": int(config["seed"]),
        "device_params": gpu_params(model_name, config, gpu=gpu),
        "gpu_detected": gpu.available,
        "gpu_describe": gpu.describe(),
        "smoke": bool(config.get("_smoke", False)),
    }


def _final_estimator(model):
    """Estimador de fato, atravessando o `Pipeline` quando houver um."""
    return model.named_steps["clf"] if hasattr(model, "named_steps") else model


def ajustar(modelo, nome: str, X, y, grupos=None):
    """Único ponto de `fit` do projeto.

    Para o `mlp` com validação agrupada e `grupos` informados, passa os grupos
    ao `MLPValidacaoAgrupada` (pelo roteamento `passo__parâmetro` do `Pipeline`,
    válido com *metadata routing* desligado, o padrão). Para os demais modelos,
    e para o `mlp` nativo, `grupos` é ignorado. Com o `mlp` agrupado e sem
    `grupos`, o `fit` do `MLPValidacaoAgrupada` levanta `ValueError`. Devolve o
    modelo ajustado.
    """
    if (
        nome == "mlp"
        and grupos is not None
        and isinstance(_final_estimator(modelo), MLPValidacaoAgrupada)
    ):
        if hasattr(modelo, "named_steps"):
            modelo.fit(X, y, clf__groups=grupos)
        else:
            modelo.fit(X, y, groups=grupos)
    else:
        modelo.fit(X, y)
    return modelo


class _predicao_serial:
    """Força `n_jobs = 1` durante a predição, para valer o RNF1.

    Por que isto existe
    -------------------
    O `RandomForestClassifier.predict_proba` soma as probabilidades das árvores
    **acumulando em paralelo** num array compartilhado (uma thread por bloco de
    árvores). Soma em ponto flutuante não é associativa, então a ordem em que as
    threads terminam muda o último bit do resultado. Medido nesta máquina:
    diferença máxima de **2,22e-16** — exatamente 1 ULP — entre duas chamadas
    idênticas.

    Parece irrelevante, e seria, se as métricas não fossem baseadas em ordenação.
    AUROC e AUC-PR tratam scores empatados como um bloco; um empate desfeito por
    1 ULP muda o tratamento daquele bloco e o valor da métrica. Medido no smoke:
    **30 dos 192 registros** mudavam entre duas execuções, e num caso a mudança
    deslocou o limiar escolhido, alterando também Precision, F1, MCC, FPR e as
    contagens da matriz de confusão. Isso viola o RNF1 ("duas execuções produzem
    métricas idênticas") e, num artigo, significa que um número da tabela não é
    reproduzível.

    O ajuste (`fit`) não é afetado: as sementes das árvores são sorteadas em
    série antes do despacho paralelo, então a floresta é sempre a mesma. Só a
    redução da predição precisa ser serial.

    Custo medido (300 árvores, 160.000 linhas de teste — a escala do
    Grambeddings): 3,5 s em série contra 0,42 s em paralelo. Na matriz completa
    são 32 predições de RandomForest, ou seja ~2 minutos a mais numa execução
    orçada em 2 horas (RNF5). Determinismo exato por 2 minutos é troca óbvia.

    Aplica-se somente a estimadores que expõem `n_jobs`. XGBoost, LightGBM,
    CatBoost, MLP e LogisticRegression foram verificados como bit-a-bit
    determinísticos na predição e passam por aqui sem alteração.
    """

    def __init__(self, model):
        self._target = _final_estimator(model)
        self._original = None
        self._restore = False

    def __enter__(self):
        if hasattr(self._target, "n_jobs") and self._target.n_jobs != 1:
            self._original = self._target.n_jobs
            self._target.n_jobs = 1
            self._restore = True
        return self

    def __exit__(self, *exc):
        if self._restore:
            self._target.n_jobs = self._original
        return False


def positive_class_proba(model, X):
    """Probabilidade da classe positiva (phishing = 1), em ordem segura.

    Nunca assumir que a coluna 1 de `predict_proba` é a classe 1: se um split
    degenerado tiver só uma classe no treino, a matriz tem uma coluna só, e
    indexar `[:, 1]` estoura ou — pior — devolve a classe errada calada. Esta
    função resolve pelo `classes_` do estimador.

    Toda predição do projeto passa por aqui, e é por isso que a serialização
    do RNF1 (`_predicao_serial`) mora neste ponto: um caminho alternativo de
    predição reintroduziria a não-reprodutibilidade sem aviso.
    """
    with _predicao_serial(model):
        proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if proba.ndim == 1:
        return proba
    if 1 in classes:
        return proba[:, classes.index(1)]
    # Treino degenerado, sem nenhum exemplo positivo: a probabilidade de
    # phishing é 0 para tudo. Devolver isso explicitamente é melhor que
    # devolver a coluna da classe errada.
    import numpy as np

    return np.zeros(len(proba), dtype=float)


def feature_importance(model, model_name: str) -> dict | None:
    """Importância das 41 features, por ajuste. `None` quando não existe.

    Para que serve (§7.4)
    ---------------------
    A matriz mostra **que** a transferência entre corpora colapsa; ela não
    mostra **por quê**. Comparar em quais features cada corpus se apoia
    transforma "colapsa" em "colapsa porque a feature discriminativa do corpus
    A é ruído no B" — mecanismo, não sintoma. É o que separa uma auditoria de
    um artigo com explicação, e sai de graça: o ajuste já aconteceu.

    O que cada família devolve, e o que **não** é comparável
    -------------------------------------------------------
    - Árvores (`random_forest`, `xgboost`, `lightgbm`, `catboost`): o
      `feature_importances_` do próprio estimador, que é redução de impureza ou
      ganho, dependendo da biblioteca. Some 1 (ou é normalizada para tal).
    - `logistic_regression`: o coeficiente, **com sinal**. Vem de dentro do
      Pipeline, portanto na escala padronizada pelo `StandardScaler` — e é
      justamente isso que o torna comparável entre features de escalas
      diferentes (`url_length` vai a ~65.000, as booleanas a 1).
    - `mlp`: `None`. Uma rede de duas camadas não tem importância por feature
      sem método de atribuição (permutação, SHAP), e permutação custaria uma
      predição por feature por célula. Fora do escopo desta rodada.

    ⚠️ Valores de famílias diferentes **não** são comparáveis entre si: ganho de
    árvore e coeficiente linear não estão na mesma unidade nem na mesma escala.
    A leitura válida é dentro de uma família, entre corpora — que é exatamente
    a comparação que o artigo precisa fazer. Não monte um ranking único
    misturando as seis colunas.
    """
    import numpy as np

    import features

    estimador = getattr(model, "named_steps", {}).get("clf", model)

    bruto = getattr(estimador, "feature_importances_", None)
    if bruto is None:
        coef = getattr(estimador, "coef_", None)
        if coef is None:
            return None  # MLP, e qualquer modelo futuro sem atribuição
        bruto = np.asarray(coef).reshape(-1)

    bruto = np.asarray(bruto, dtype=float).reshape(-1)
    nomes = list(features.FEATURE_NAMES)
    if len(bruto) != len(nomes):
        # Nunca deve acontecer — mas devolver um dicionário desalinhado seria
        # pior que devolver nada, e é o tipo de erro silencioso que o B13
        # ensinou a não aceitar.
        logger.warning(
            "importancia com tamanho inesperado; descartada",
            model=model_name, obtido=len(bruto), esperado=len(nomes),
        )
        return None
    return {n: round(float(v), 8) for n, v in zip(nomes, bruto)}


__all__ = [
    "MODEL_ORDER",
    "DEFAULT_CONFIG_PATH",
    "ModelConfigError",
    "GpuInfo",
    "load_experiment_config",
    "apply_smoke_overrides",
    "detect_gpu",
    "gpu_params",
    "build_model",
    "build_all_models",
    "MLPValidacaoAgrupada",
    "ajustar",
    "model_provenance",
    "positive_class_proba",
    "feature_importance",
]
