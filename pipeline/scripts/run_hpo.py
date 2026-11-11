"""T4.5 — Etapa 2 do D5: ablação de ajuste de hiperparâmetros no Grambeddings.

    venv\\Scripts\\python.exe scripts\\run_hpo.py

RODE DEPOIS do `run_all.py`. Este script não produz o resultado principal do
artigo — produz a Tabela T4, que responde a uma objeção: *"os modelos transferem
mal porque estão mal configurados"*. A resposta é o delta seco→ajustado medido
**num corpus só**.

Por que num corpus só (SDD §4 D5)
---------------------------------
Ajustar hiperparâmetros por corpus destruiria a matriz: a fora-diagonal cairia
por overfit de tuning e não por diferença de distribuição, e a matriz deixaria
de medir o que o artigo afirma medir. Por isso o HPO acontece **exclusivamente
no Grambeddings** e **não** realimenta a matriz.

Este script NUNCA escreve em config/experiment.yml
--------------------------------------------------
O baseline seco continua sendo o resultado do artigo. Os parâmetros achados
aqui vivem em `results/hpo.json` e só alimentam a T4. Se algum dia alguém
copiar estes números para a `experiment.yml`, a matriz inteira deixa de ser
comparável e o D5 é violado — está travado por
`tests/test_run_hpo.py`, que confere que este script não escreve no YAML, que
cada espaço de busca contém o valor seco e que o objetivo é a validação.

ESTE SCRIPT TREINA MODELOS. Por SDD §0 ele não roda na máquina do Julio: sem
GPU detectada, exige `--permitir-cpu` explícito.

O que produz
------------
    results/hpo.json  ← por modelo: melhor conjunto, AUROC de validação seco e
                        ajustado, o delta medido no TESTE (que é a T4) e a
                        **trajetória** das tentativas
    results/hpo_fora_diagonal.json
                      ← por modelo: o modelo seco e o ajustado, ambos treinados no
                        Grambeddings, avaliados no teste dos quatro datasets com o
                        limiar da validação do Grambeddings (HPO-02)
    results/hpo/predicoes/ ← escores por (configuração, modelo), fora do git

Frente A (HPO-01, HPO-02, AD-057): o espaço de busca, as 30 tentativas, o
amostrador TPE e a semente 42 são os de antes. Mudam três coisas: a partição vem
de `splits.particionar_corpus` (com deduplicação pelo url_norm), o MLP usa a
validação interna agrupada por eTLD+1 no Regime B (a mesma da matriz) e o
script recusa iniciar com código modificado (RUN-03).

Por que a trajetória é gravada (05/08/2026)
-------------------------------------------
A objeção previsível à T4 é "vocês sub-ajustaram": 30 tentativas num espaço de
7 dimensões é pouco, e um delta pequeno poderia ser só busca curta. Guardar o
AUROC de validação de cada tentativa permite mostrar que a busca **estabilizou**
— o melhor-até-agora deixa de subir muito antes da trigésima. Sem isso a
resposta seria "confie em nós". Custo zero: o Optuna já tem os números.

O arquivo é reescrito **a cada modelo**, não só no fim. São 6 modelos e mais de
uma hora de máquina; cair no sexto e perder os cinco primeiros seria perder o
dia do Lucas por nada.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUT_DIR = REPO_ROOT / "results"
CORPUS_HPO = "grambeddings"

# Descricao curta do `--help`. NAO usar `__doc__` aqui: o console do
# Windows e cp1252 e a docstring tem setas/travessoes, entao `--help`
# morria com UnicodeEncodeError. A pendencia 10 da SDD 13.4 dizia que
# docstring era livre -- e era, ate `description=__doc__` transformar a
# docstring em saida impressa. Mantenha esta string em ASCII puro.
DESCRICAO_CLI = (
    "Ablacao de ajuste de hiperparametros (T4.5), exclusivamente no Grambeddings. Nao altera config/experiment.yml nem a matriz principal."
)

_T0 = time.perf_counter()


def log(msg: str = "") -> None:
    if msg:
        print(f"[{(time.perf_counter() - _T0) / 60:5.1f}min] {msg}", flush=True)
    else:
        print(flush=True)


def falhar(msg: str, *, dica: str = ""):
    print(f"\n{'=' * 74}", flush=True)
    print(f"HPO ABORTADO: {msg}", flush=True)
    if dica:
        print(f"\n{dica}", flush=True)
    print(f"{'=' * 74}", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Espaços de busca
# ---------------------------------------------------------------------------

# Centrados no baseline seco declarado em `experiment.yml`, não em faixas
# arbitrárias: a T4 mede "quanto se ganha ajustando **este** modelo", e uma
# faixa que não contenha o baseline mediria outra coisa. Cada espaço contém o
# valor seco — conferir isso é parte da revisão deste script.
#
# Conter o baseline **não** garante delta positivo: o TPE pode não amostrar o
# ponto seco em 30 tentativas, e o melhor da validação pode piorar no teste.
# Um delta negativo aqui é resultado legítimo, não defeito — e, aliás, reforça
# a tese: se nem ajustar melhora, o problema não é configuração.


def espaco(trial, nome_modelo: str) -> dict:
    if nome_modelo == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "max_iter": trial.suggest_categorical("max_iter", [1000, 2000, 4000]),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        }
    if nome_modelo == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if nome_modelo == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    if nome_modelo == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    if nome_modelo == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 800, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
    if nome_modelo == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", ["128,64", "256,128", "128", "256,128,64"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            "max_iter": trial.suggest_categorical("max_iter", [200, 400]),
        }
    raise ValueError(f"modelo sem espaço de busca definido: {nome_modelo}")


def normalizar(nome_modelo: str, params: dict) -> dict:
    """Converte o que o Optuna sabe sugerir no que o modelo espera receber.

    `hidden_layer_sizes` é o único caso: o Optuna não sugere tupla, então o
    espaço usa string e a conversão acontece aqui, num lugar só.
    """
    params = dict(params)
    if nome_modelo == "mlp" and isinstance(params.get("hidden_layer_sizes"), str):
        params["hidden_layer_sizes"] = tuple(
            int(x) for x in params["hidden_layer_sizes"].split(",")
        )
    return params


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------


def checar_arvore_limpa(raiz=None) -> None:
    """RUN-03: recusa iniciar com código modificado ou não rastreado.

    Usa a mesma conferência do `run_all.py` (`run_all.arvore_suja`): os dois
    scripts carimbam o que produzem com o commit do código.
    """
    import run_all

    sujas = run_all.arvore_suja(raiz)
    if sujas:
        falhar(
            "ha modificacao ou arquivo nao rastreado nos caminhos de codigo.",
            dica=(
                "O HPO e carimbado com o commit do codigo e so vale se o codigo que\n"
                "roda e o do commit. Faca o commit (ou descarte a mudanca) e rode de\n"
                f"novo. Caminhos conferidos:\n    {' '.join(run_all.CAMINHOS_CODIGO)}\n\n"
                "Arquivos:\n" + "\n".join(f"    {ln}" for ln in sujas)
            ),
        )


def carregar_quadros(config: dict, permitir_cpu: bool):
    """Carrega os quatro datasets (cache de características) e detecta a GPU."""
    from run_all import checar_corpora, checar_restricao_0, preparar_corpus

    import models

    gpu = models.detect_gpu()
    checar_restricao_0(permitir_cpu, gpu)

    nomes = list(config["corpora"]["treino_avaliacao"])
    if CORPUS_HPO not in nomes:
        raise ValueError(f"{CORPUS_HPO} nao esta nos corpora de treino/avaliacao: {nomes}")
    manifesto = checar_corpora(nomes)
    log(f"preparando {len(nomes)} datasets (canonizacao + 41 features, com cache)...")
    quadros = {n: preparar_corpus(n, manifesto, usar_cache=True) for n in nomes}
    return gpu, quadros


def particionar_quadros(config: dict, quadros: dict, regime: str):
    """Dedup e partição de cada dataset, na mesma fachada da matriz (HPO-02).

    Devolve `(indices, contagens)` por dataset. As posições indexam o quadro
    original; as contagens são as da deduplicação.
    """
    import splits

    val_size = float(
        config["split"]["val_size"] if regime == "etld1"
        else config["split"]["val_size_oficial"]
    )
    indices, contagens = {}, {}
    for nome, quadro in quadros.items():
        indices[nome], contagens[nome] = splits.particionar_corpus(
            quadro, regime,
            corpus=nome,
            seed=int(config["seed"]),
            test_size=float(config["split"]["test_size"]),
            val_size=val_size,
        )
    return indices, contagens


def auroc_de(modelo, X, y) -> float:
    import evaluate
    import models

    p = models.positive_class_proba(modelo, X)
    metrics, _ = evaluate.evaluate_predictions(y, p, threshold=0.5, ece_bins=15)
    auroc = metrics["auroc"]
    if auroc is None:
        raise RuntimeError("AUROC indefinido no Grambeddings — teste de classe única?")
    return float(auroc)


def ajustar_modelo(nome_modelo, config, gpu, treino, validacao, n_trials, seed,
                   grupos_tr=None, validacao_agrupada=False):
    """Roda o estudo Optuna de um modelo. Objetivo: AUROC na VALIDAÇÃO.

    Nunca no teste: escolher hiperparâmetro olhando o teste é o vazamento mais
    banal que existe, e seria especialmente constrangedor num artigo cuja tese
    é justamente sobre avaliação contaminada.

    `grupos_tr` (eTLD+1 de cada linha de treino) e `validacao_agrupada` valem
    para o MLP no Regime B: o ajuste passa por `models.ajustar`, o único ponto
    de `fit` do projeto, com a mesma validação interna agrupada da matriz.
    """
    import optuna

    import models

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_tr, y_tr = treino
    X_val, y_val = validacao

    def objetivo(trial):
        cfg = json.loads(json.dumps(config))  # cópia profunda barata
        cfg["modelos"][nome_modelo]["params"].update(
            normalizar(nome_modelo, espaco(trial, nome_modelo))
        )
        modelo = models.build_model(
            nome_modelo, cfg, gpu=gpu, validacao_agrupada=validacao_agrupada
        )
        models.ajustar(modelo, nome_modelo, X_tr, y_tr, grupos=grupos_tr)
        return auroc_de(modelo, X_val, y_val)

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"hpo-{nome_modelo}-{CORPUS_HPO}",
    )
    estudo.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    return estudo


def gravar(saida: dict) -> None:
    """Escreve `results/hpo.json`. Chamado a cada modelo, não só no fim."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hpo.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"gravado: results/hpo.json ({len(saida['modelos'])} de "
        f"{saida['n_modelos_pedidos']} modelos)")


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRICAO_CLI)
    parser.add_argument("--permitir-cpu", action="store_true",
                        help="autoriza treinar sem GPU (SDD secao 0)")
    parser.add_argument("--trials", type=int, default=30,
                        help="tentativas por modelo (escopo fixado na T4.5: ~30)")
    parser.add_argument("--modelos", type=str, default=None,
                        help="subconjunto separado por virgula; default = os 6")
    parser.add_argument("--regime", type=str, default="etld1", choices=["etld1", "oficial"],
                        help="regime do split. Default etld1: e o protocolo do "
                             "artigo (T2), entao o delta e comparavel a ele.")
    return parser


def gravar_fora_diagonal(saida: dict) -> None:
    """Escreve `results/hpo_fora_diagonal.json`. Chamado a cada modelo."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hpo_fora_diagonal.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"gravado: results/hpo_fora_diagonal.json ({len(saida['modelos'])} de "
        f"{saida['n_modelos_pedidos']} modelos)")


def avaliar_fora_da_diagonal(modelo, configuracao: str, nome_modelo: str,
                             matrizes: dict, indices: dict, config: dict,
                             regime: str) -> dict:
    """Avalia um modelo já treinado no Grambeddings no teste dos quatro datasets.

    O limiar é o da validação do Grambeddings (D6), transportado como está para
    os demais datasets. Grava as predições de `configuracao` ("seco" ou
    "ajustado") em `results/hpo/predicoes/` (HPO-02) e devolve
    `{"limiar", "avaliacoes"}`, com as métricas completas de cada dataset de
    teste e a referência (arquivo e sha256) das predições.
    """
    import evaluate
    import models
    import predicoes

    criterio = config["avaliacao"]["criterio_limiar"]
    ece_bins = int(config["avaliacao"]["ece_bins"])
    X_hpo, y_hpo, _ = matrizes[CORPUS_HPO]
    idx_val = indices[CORPUS_HPO].val
    p_val = models.positive_class_proba(modelo, X_hpo[idx_val])
    limiar = evaluate.select_threshold(y_hpo[idx_val], p_val, criterion=criterio)

    blocos, avaliacoes = [], []
    for nome, (X, y, linhas) in matrizes.items():
        idx = indices[nome].test
        p_te = models.positive_class_proba(modelo, X[idx])
        metricas, avisos = evaluate.evaluate_predictions(
            y[idx], p_te, threshold=limiar, ece_bins=ece_bins
        )
        blocos.append((nome, linhas[idx], y[idx], p_te))
        avaliacoes.append({
            "configuracao": configuracao,
            "train_corpus": CORPUS_HPO,
            "test_corpus": nome,
            "diagonal": nome == CORPUS_HPO,
            "n_test": int(len(idx)),
            "metrics": metricas,
            "warnings": list(avisos),
        })
    ref = predicoes.gravar_predicoes(
        OUT_DIR / "hpo", configuracao, regime, CORPUS_HPO, nome_modelo, blocos
    )
    for a in avaliacoes:
        a["predicoes"] = dict(ref)
    return {"limiar": float(limiar), "avaliacoes": avaliacoes}


def rodar(config: dict, gpu, quadros: dict, nomes: list[str], regime: str,
          n_trials: int, cabecalho: dict, indices: dict, contagens: dict) -> tuple[dict, dict]:
    """Ablação de cada modelo em `nomes` e avaliação fora da diagonal (HPO-01, HPO-02).

    Treina no Grambeddings (Regime `regime`, semente do config), com a partição
    deduplicada de `particionar_quadros`; busca os hiperparâmetros na validação;
    e avalia o modelo seco e o ajustado no teste dos quatro datasets. Grava
    `hpo.json` e `hpo_fora_diagonal.json` a cada modelo. Devolve os dois
    dicionários gravados.
    """
    import numpy as np

    import features
    import models

    agrupada = regime == "etld1"
    matrizes = {
        nome: (
            q[list(features.FEATURE_NAMES)].to_numpy(dtype=np.float32),
            q["label"].to_numpy(),
            q["linha"].to_numpy() if "linha" in q.columns else np.arange(len(q)),
        )
        for nome, q in quadros.items()
    }
    X_all, y_all, _ = matrizes[CORPUS_HPO]
    sp = indices[CORPUS_HPO]
    treino = (X_all[sp.train], y_all[sp.train])
    validacao = (X_all[sp.val], y_all[sp.val])
    teste = (X_all[sp.test], y_all[sp.test])
    grupos_tr = None
    if agrupada:
        import splits

        grupos_tr = splits.group_keys(quadros[CORPUS_HPO]).to_numpy()[sp.train]

    log(f"regime: {regime} | treino={len(treino[1])} val={len(validacao[1])} "
        f"teste={len(teste[1])} | dedup: "
        f"-{contagens[CORPUS_HPO]['n_duplicatas_removidas']} duplicatas, "
        f"-{contagens[CORPUS_HPO]['n_conflito_linhas']} em conflito")
    log(f"{len(nomes)} modelos x {n_trials} tentativas")
    log()

    saida = {
        **cabecalho,
        "corpus": CORPUS_HPO,
        "regime": regime,
        "n_trials": n_trials,
        "n_modelos_pedidos": len(nomes),
        "seed": int(config["seed"]),
        "dedup": contagens[CORPUS_HPO],
        "validacao_interna_mlp": "agrupada_etld1" if agrupada else "linha",
        "modelos": {},
    }
    saida_fd = {
        **cabecalho,
        "corpus_treino": CORPUS_HPO,
        "regime": regime,
        "seed": int(config["seed"]),
        "n_modelos_pedidos": len(nomes),
        "datasets_teste": list(quadros),
        "dedup": dict(contagens),
        "modelos": {},
    }

    for nome_modelo in nomes:
        log(f"=== {nome_modelo} " + "=" * (56 - len(nome_modelo)))

        # Baseline seco: o modelo exatamente como a matriz o usa.
        seco = models.build_model(nome_modelo, config, gpu=gpu, validacao_agrupada=agrupada)
        t0 = time.perf_counter()
        models.ajustar(seco, nome_modelo, *treino, grupos=grupos_tr)
        seg_seco = time.perf_counter() - t0
        val_seco = auroc_de(seco, *validacao)
        teste_seco = auroc_de(seco, *teste)

        try:
            estudo = ajustar_modelo(
                nome_modelo, config, gpu, treino, validacao,
                n_trials, int(config["seed"]),
                grupos_tr=grupos_tr, validacao_agrupada=agrupada,
            )
        except Exception as exc:  # noqa: BLE001
            falhar(f"{nome_modelo}: {type(exc).__name__}: {exc}",
                   dica=traceback.format_exc())

        # Reajusta com o melhor conjunto e mede no TESTE — é este o delta da T4.
        melhores = normalizar(nome_modelo, estudo.best_params)
        cfg = json.loads(json.dumps(config))
        cfg["modelos"][nome_modelo]["params"].update(melhores)
        ajustado = models.build_model(nome_modelo, cfg, gpu=gpu, validacao_agrupada=agrupada)
        t0 = time.perf_counter()
        models.ajustar(ajustado, nome_modelo, *treino, grupos=grupos_tr)
        seg_ajustado = time.perf_counter() - t0
        val_ajustado = auroc_de(ajustado, *validacao)
        teste_ajustado = auroc_de(ajustado, *teste)

        # HPO-02: os dois modelos, já treinados no Grambeddings, nos quatro testes.
        fd_seco = avaliar_fora_da_diagonal(
            seco, "seco", nome_modelo, matrizes, indices, config, regime
        )
        fd_ajustado = avaliar_fora_da_diagonal(
            ajustado, "ajustado", nome_modelo, matrizes, indices, config, regime
        )
        saida_fd["modelos"][nome_modelo] = {
            "params_secos": config["modelos"][nome_modelo]["params"],
            "params_ajustados": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in melhores.items()
            },
            "limiar_seco": fd_seco["limiar"],
            "limiar_ajustado": fd_ajustado["limiar"],
            "avaliacoes": fd_seco["avaliacoes"] + fd_ajustado["avaliacoes"],
        }

        # Trajetória: AUROC de validação de cada tentativa, na ordem em que o
        # TPE as propôs, mais o melhor-até-agora. É o que sustenta a resposta a
        # "vocês sub-ajustaram" — ver a nota no topo do arquivo.
        valores = [
            (None if t.value is None else round(float(t.value), 6))
            for t in estudo.trials
        ]
        melhor_ate_agora, corrente = [], None
        for v in valores:
            if v is not None and (corrente is None or v > corrente):
                corrente = v
            melhor_ate_agora.append(corrente)

        saida["modelos"][nome_modelo] = {
            "params_secos": config["modelos"][nome_modelo]["params"],
            "params_ajustados": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in melhores.items()
            },
            "auroc_val_seco": round(val_seco, 6),
            "auroc_val_ajustado": round(val_ajustado, 6),
            "auroc_teste_seco": round(teste_seco, 6),
            "auroc_teste_ajustado": round(teste_ajustado, 6),
            "delta_teste": round(teste_ajustado - teste_seco, 6),
            "fit_seconds_seco": round(seg_seco, 3),
            "fit_seconds_ajustado": round(seg_ajustado, 3),
            "n_trials_concluidos": len(estudo.trials),
            "n_trials_falhados": sum(1 for v in valores if v is None),
            "trajetoria_val": valores,
            "trajetoria_val_melhor_ate_agora": melhor_ate_agora,
        }
        log(f"  val   seco={val_seco:.4f} ajustado={val_ajustado:.4f}")
        log(f"  teste seco={teste_seco:.4f} ajustado={teste_ajustado:.4f} "
            f"delta={teste_ajustado - teste_seco:+.4f}")

        # Grava a cada modelo, nao so no fim: sao 6 modelos e mais de uma hora
        # de maquina, e cair no sexto nao pode custar os cinco primeiros.
        gravar(saida)
        gravar_fora_diagonal(saida_fd)
        log()

    gravar(saida)
    gravar_fora_diagonal(saida_fd)
    return saida, saida_fd


def main() -> int:
    import yaml

    args = criar_parser().parse_args()

    print("=" * 74)
    print("T4.5 - ABLACAO DE AJUSTE (Etapa 2 do D5) - so no Grambeddings")
    print("=" * 74)

    # RUN-03: antes de qualquer outra coisa e antes de carregar dados.
    checar_arvore_limpa()

    import evaluate
    import matrix
    import models
    import psl
    import run_all

    commit_codigo = run_all.commit_codigo()
    if commit_codigo is None:
        falhar("nao foi possivel obter o commit do codigo (`git log` falhou).",
               dica="O HPO e carimbado com o `commit_codigo`; sem ele nao ha carimbo.")

    config = yaml.safe_load(
        (REPO_ROOT / "config" / "experiment.yml").read_text(encoding="utf-8")
    )
    nomes = (
        [n.strip() for n in args.modelos.split(",") if n.strip()]
        if args.modelos else list(models.MODEL_ORDER)
    )
    invalidos = [n for n in nomes if n not in models.MODEL_ORDER]
    if invalidos:
        falhar(f"modelos invalidos: {invalidos}. Validos: {list(models.MODEL_ORDER)}")

    try:
        gpu, quadros = carregar_quadros(config, args.permitir_cpu)
        indices, contagens = particionar_quadros(config, quadros, args.regime)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        falhar(f"{type(exc).__name__}: {exc}", dica=traceback.format_exc())

    log(f"dispositivo: {gpu.describe()}")
    cabecalho = {
        "commit": evaluate.git_commit_hash(),
        "commit_codigo": commit_codigo,
        "run_id": evaluate.make_run_id(),
        "psl_sha256": psl.sha256(),
        "config_fingerprint": matrix.config_fingerprint(config),
        "gpu": gpu.describe(),
    }
    saida, _ = rodar(
        config, gpu, quadros, nomes, args.regime, args.trials, cabecalho,
        indices, contagens,
    )

    deltas = [m["delta_teste"] for m in saida["modelos"].values()]
    print()
    print("=" * 74)
    print(f"{len(nomes)} modelos ajustados - delta no teste: "
          f"min {min(deltas):+.4f} / max {max(deltas):+.4f} / "
          f"media {sum(deltas) / len(deltas):+.4f}")
    print("config/experiment.yml NAO foi tocado (D5). O baseline seco segue")
    print("sendo o resultado do artigo; isto aqui e so a Tabela T4.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
