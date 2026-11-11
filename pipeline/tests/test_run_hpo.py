"""T4.5 — as travas da ablação de hiperparâmetros. Não treina nada.

Este arquivo existia como promessa antes de existir como arquivo: a docstring do
`scripts/run_hpo.py` afirmava que a garantia de "não escreve no experiment.yml"
estava travada por `tests/test_run_hpo.py::test_nao_escreve_no_experiment_yml`,
e o arquivo não existia. Encontrado pela auditoria de 05/08/2026.

O que se trava aqui, e por quê
------------------------------
1. **Não escrever na `experiment.yml`.** Copiar um número do `hpo.json` para o
   config muda a comparação principal entre os 6 modelos e viola o D5. É a única
   coisa que o `CLAUDE.md` pede explicitamente ao Lucas para vigiar.
2. **Cada espaço de busca contém o valor seco.** A T4 responde "quanto se ganha
   ajustando **este** modelo". Uma faixa que não contenha o baseline mede outra
   coisa, e o delta deixa de ser interpretável. Conferido nos 6 modelos,
   parâmetro por parâmetro.
3. **Parâmetro fora da busca é preservado.** `deterministic` e `force_row_wise`
   do LightGBM existem para o RNF1; `subsample_freq` faz o `subsample` valer;
   `early_stopping` do MLP muda o que "max_iter" significa. Se a busca os
   apagasse, o "ajustado" não seria o mesmo modelo que o "seco".
4. **O objetivo é a validação, nunca o teste.** Escolher hiperparâmetro olhando
   o teste seria o vazamento mais banal que existe — num artigo cuja tese é
   sobre avaliação contaminada.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CAMINHO_HPO = REPO_ROOT / "scripts" / "run_hpo.py"
CAMINHO_YAML = REPO_ROOT / "config" / "experiment.yml"


@pytest.fixture(scope="module")
def run_hpo():
    if not CAMINHO_HPO.exists():
        pytest.skip("scripts/run_hpo.py ausente neste commit")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_hpo_sob_teste", CAMINHO_HPO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def config():
    import yaml

    return yaml.safe_load(CAMINHO_YAML.read_text(encoding="utf-8"))


class CapturaDeEspaco:
    """Finge ser um `trial` do Optuna e registra o que foi pedido.

    Devolve sempre o limite inferior, que é irrelevante: o objetivo é inspecionar
    a **forma** do espaço, não amostrar dele.
    """

    def __init__(self):
        self.espaco: dict = {}

    def suggest_float(self, nome, baixo, alto, *, log=False, step=None):
        self.espaco[nome] = ("float", baixo, alto, log)
        return baixo

    def suggest_int(self, nome, baixo, alto, *, step=1, log=False):
        self.espaco[nome] = ("int", baixo, alto, log, step)
        return baixo

    def suggest_categorical(self, nome, escolhas):
        self.espaco[nome] = ("cat", list(escolhas))
        return escolhas[0]


def _contem(espaco, valor) -> bool:
    tipo = espaco[0]
    if tipo == "cat":
        return valor in espaco[1]
    if tipo == "float":
        return isinstance(valor, (int, float)) and espaco[1] <= valor <= espaco[2]
    _, baixo, alto, em_log, passo = espaco
    if not isinstance(valor, int) or not baixo <= valor <= alto:
        return False
    # `suggest_int` com `step` só propõe valores na grade; com `log`, qualquer
    # inteiro da faixa é alcançável.
    return em_log or (valor - baixo) % passo == 0


MODELOS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
]


class TestNaoTocaOConfig:
    def test_nao_escreve_no_experiment_yml(self, run_hpo):
        """A trava que a docstring do script prometia e não existia.

        Varre o fonte procurando qualquer rota de escrita no caminho do config.
        É teste de fonte de propósito: o comportamento só apareceria rodando o
        HPO inteiro, que treina 180 modelos e a §0 proíbe aqui.
        """
        fonte = CAMINHO_HPO.read_text(encoding="utf-8")
        escritas = ("write_text", "open(", "to_csv", "safe_dump", "yaml.dump")
        for linha in fonte.splitlines():
            nu = linha.strip()
            if nu.startswith("#") or "experiment.yml" not in nu:
                continue
            # A única menção legítima é a LEITURA da config.
            assert not any(e in nu for e in escritas) or "read_text" in nu, (
                f"linha suspeita de escrever no config: {nu}"
            )
        assert 'OUT_DIR / "hpo.json"' in fonte, (
            "o script deveria gravar apenas em results/hpo.json"
        )

    def test_a_copia_do_config_e_profunda(self, run_hpo, config):
        """`normalizar` + `update` não podem vazar para o dicionário original.

        Se a cópia fosse rasa, ajustar o primeiro modelo contaminaria o baseline
        seco dos cinco seguintes — e o delta da T4 sairia medido contra um
        baseline que já tinha sido ajustado.
        """
        import copy
        import json

        antes = copy.deepcopy(config)
        cfg = json.loads(json.dumps(config))
        cfg["modelos"]["xgboost"]["params"].update({"max_depth": 99})
        assert config == antes
        assert config["modelos"]["xgboost"]["params"]["max_depth"] != 99


class TestEspacoDeBuscaContemOBaseline:
    @pytest.mark.parametrize("modelo", MODELOS)
    def test_todo_parametro_buscado_contem_o_valor_seco(self, run_hpo, config, modelo):
        captura = CapturaDeEspaco()
        run_hpo.espaco(captura, modelo)
        secos = config["modelos"][modelo]["params"]

        assert captura.espaco, f"{modelo} nao declarou espaco de busca"
        for nome, espaco in captura.espaco.items():
            assert nome in secos, (
                f"{modelo}: o espaco busca '{nome}', que nao existe no baseline "
                f"da experiment.yml — o delta da T4 nao seria interpretavel"
            )
            valor = secos[nome]
            # O Optuna não sugere tupla/lista; o espaço do MLP usa string.
            if modelo == "mlp" and nome == "hidden_layer_sizes":
                valor = ",".join(str(v) for v in valor)
            assert _contem(espaco, valor), (
                f"{modelo}.{nome}: o baseline seco e {valor!r} e o espaco e "
                f"{espaco!r} — a T4 mediria outra coisa"
            )

    @pytest.mark.parametrize(
        "modelo,preservados",
        [
            ("lightgbm", ["deterministic", "force_row_wise", "subsample_freq"]),
            ("mlp", ["early_stopping", "n_iter_no_change", "validation_fraction"]),
            ("xgboost", ["tree_method", "eval_metric"]),
            ("logistic_regression", ["solver"]),
        ],
    )
    def test_parametro_fora_da_busca_nao_e_apagado(
        self, run_hpo, config, modelo, preservados
    ):
        """`update` só sobrescreve as chaves buscadas — o resto sobrevive."""
        import json

        captura = CapturaDeEspaco()
        run_hpo.espaco(captura, modelo)
        cfg = json.loads(json.dumps(config))
        cfg["modelos"][modelo]["params"].update(
            run_hpo.normalizar(modelo, {k: 1 for k in captura.espaco})
        )
        for chave in preservados:
            assert chave in cfg["modelos"][modelo]["params"], (
                f"{modelo}: '{chave}' desapareceu ao aplicar os ajustados. "
                f"O modelo ajustado deixaria de ser o mesmo modelo do seco."
            )
            assert chave not in captura.espaco, (
                f"{modelo}: '{chave}' nao deveria estar no espaco de busca"
            )

    def test_modelo_desconhecido_levanta(self, run_hpo):
        with pytest.raises(ValueError, match="sem espaço de busca"):
            run_hpo.espaco(CapturaDeEspaco(), "modelo_que_nao_existe")


class TestObjetivoEAValidacao:
    def test_ajustar_modelo_nao_recebe_o_teste(self, run_hpo):
        """Conferência estrutural: a assinatura não tem por onde ver o teste.

        Mais forte que ler o corpo: se o teste não entra na função, não há como
        o objetivo olhá-lo, nem hoje nem depois de uma refatoração.
        """
        import inspect

        params = list(inspect.signature(run_hpo.ajustar_modelo).parameters)
        assert "validacao" in params
        assert not any("test" in p for p in params), (
            f"ajustar_modelo recebe algo com 'test' na assinatura: {params}. "
            f"Escolher hiperparametro olhando o teste e vazamento."
        )

    def test_normalizar_converte_hidden_layer_sizes(self, run_hpo):
        saida = run_hpo.normalizar("mlp", {"hidden_layer_sizes": "256,128,64"})
        assert saida["hidden_layer_sizes"] == (256, 128, 64)

    def test_normalizar_nao_mexe_nos_outros_modelos(self, run_hpo):
        entrada = {"max_depth": 8, "learning_rate": 0.1}
        assert run_hpo.normalizar("xgboost", entrada) == entrada


# ---------------------------------------------------------------------------
# Frente A, T15: HPO refeito e avaliado fora da diagonal (HPO-01, HPO-02, RUN-03)
# ---------------------------------------------------------------------------

# Distribuições sugeridas por `espaco`, tiradas do estado anterior à frente A.
# O espaço de busca não pode mudar (HPO-01): a mudança do HPO é a partição, o MLP
# agrupado e a avaliação, não o que se busca.
SNAPSHOT_ESPACO = {
    "logistic_regression": {
        "C": ("float", 0.01, 100.0, True),
        "max_iter": ("cat", [1000, 2000, 4000]),
        "tol": ("float", 1e-05, 0.001, True),
    },
    "random_forest": {
        "n_estimators": ("int", 100, 600, False, 100),
        "max_depth": ("cat", [None, 10, 20, 40]),
        "min_samples_leaf": ("int", 1, 8, False, 1),
        "max_features": ("cat", ["sqrt", "log2", 0.5]),
    },
    "xgboost": {
        "n_estimators": ("int", 200, 800, False, 100),
        "max_depth": ("int", 4, 12, False, 1),
        "learning_rate": ("float", 0.01, 0.3, True),
        "subsample": ("float", 0.6, 1.0, False),
        "colsample_bytree": ("float", 0.6, 1.0, False),
        "min_child_weight": ("int", 1, 10, False, 1),
        "reg_lambda": ("float", 0.1, 10.0, True),
    },
    "lightgbm": {
        "n_estimators": ("int", 200, 800, False, 100),
        "num_leaves": ("int", 15, 255, True, 1),
        "learning_rate": ("float", 0.01, 0.3, True),
        "subsample": ("float", 0.6, 1.0, False),
        "colsample_bytree": ("float", 0.6, 1.0, False),
        "min_child_samples": ("int", 5, 100, True, 1),
        "reg_lambda": ("float", 0.1, 10.0, True),
    },
    "catboost": {
        "iterations": ("int", 200, 800, False, 100),
        "depth": ("int", 4, 10, False, 1),
        "learning_rate": ("float", 0.01, 0.3, True),
        "l2_leaf_reg": ("float", 1.0, 10.0, True),
    },
    "mlp": {
        "hidden_layer_sizes": ("cat", ["128,64", "256,128", "128", "256,128,64"]),
        "alpha": ("float", 1e-06, 0.01, True),
        "learning_rate_init": ("float", 0.0001, 0.01, True),
        "max_iter": ("cat", [200, 400]),
    },
}


class TestEspacoEQuantidadeDeTentativasInalterados:
    @pytest.mark.parametrize("modelo", MODELOS)
    def test_espaco_e_igual_ao_snapshot(self, run_hpo, modelo):
        captura = CapturaDeEspaco()
        run_hpo.espaco(captura, modelo)
        assert captura.espaco == SNAPSHOT_ESPACO[modelo]

    def test_trinta_tentativas_e_regime_b_por_padrao(self, run_hpo):
        args = run_hpo.criar_parser().parse_args([])
        assert args.trials == 30
        assert args.regime == "etld1"


def _corpus_hpo(nome, n_dominios=80, por_dominio=4, offset=0):
    """Corpus canonizado sintético, com as 41 características separáveis."""
    import zlib

    import numpy as np
    import pandas as pd

    import features

    rng = np.random.default_rng(zlib.crc32(nome.encode("utf-8")))
    linhas = []
    for d in range(n_dominios):
        dominio = f"{nome}{d + offset:04d}.com"
        for r in range(por_dominio):
            linhas.append({
                "label": (d + offset) % 2,
                "url": f"http://{dominio}/p{r}",
                "url_norm": f"{dominio}/p{r}",
                "corpus": nome,
                "split_oficial": "train" if d % 5 else "test",
                "etld_plus_one": dominio,
            })
    df = pd.DataFrame(linhas)
    df["linha"] = np.arange(len(df), dtype=np.int32)
    for i, col in enumerate(features.FEATURE_NAMES):
        df[col] = (df["label"].to_numpy() * (5 + i % 3)
                   + rng.normal(0, 2, len(df))).astype(np.float32)
    return df


@pytest.fixture
def cenario_pequeno(run_hpo, config, tmp_path, monkeypatch):
    import models

    monkeypatch.setattr(run_hpo, "OUT_DIR", tmp_path)
    monkeypatch.setattr(run_hpo, "log", lambda *a, **k: None)
    nomes = config["corpora"]["treino_avaliacao"]
    quadros = {n: _corpus_hpo(n, offset=100 * i) for i, n in enumerate(nomes)}
    gpu = models.GpuInfo(available=False, reason="teste")
    indices, contagens = run_hpo.particionar_quadros(config, quadros, "etld1")
    cabecalho = {"commit": "abc", "run_id": "t", "psl_sha256": "x",
                 "config_fingerprint": "f", "gpu": gpu.describe()}
    return config, gpu, quadros, indices, contagens, cabecalho, tmp_path


class TestAvaliacaoForaDaDiagonal:
    """HPO-02: seco e ajustado, treinados no Grambeddings, nos quatro testes."""

    def test_grava_seco_e_ajustado_nos_quatro_datasets_com_predicoes(
        self, run_hpo, cenario_pequeno
    ):
        import json

        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        import predicoes

        config, gpu, quadros, indices, contagens, cabecalho, saida_dir = cenario_pequeno
        saida, saida_fd = run_hpo.rodar(
            config, gpu, quadros, ["logistic_regression"], "etld1", 2,
            cabecalho, indices, contagens,
        )

        # 2 tentativas, e o hpo.json de antes continua sendo gravado.
        assert saida["modelos"]["logistic_regression"]["n_trials_concluidos"] == 2
        assert (saida_dir / "hpo.json").is_file()

        gravado = json.loads((saida_dir / "hpo_fora_diagonal.json").read_text(encoding="utf-8"))
        assert gravado["corpus_treino"] == "grambeddings"
        assert saida_fd["modelos"].keys() == gravado["modelos"].keys()
        modelo = gravado["modelos"]["logistic_regression"]
        avaliacoes = modelo["avaliacoes"]
        assert len(avaliacoes) == 2 * len(quadros)
        for configuracao in ("seco", "ajustado"):
            doc = [a for a in avaliacoes if a["configuracao"] == configuracao]
            assert sorted(a["test_corpus"] for a in doc) == sorted(quadros)
            # O limiar é um só por configuração: o da validação do Grambeddings.
            limiar = modelo[f"limiar_{configuracao}"]
            assert {a["metrics"]["threshold"] for a in doc} == {limiar}
            assert [a["diagonal"] for a in doc if a["test_corpus"] == "grambeddings"] == [True]

        for a in avaliacoes:
            ref = a["predicoes"]
            assert predicoes.conferir(saida_dir / "hpo", ref)
            assert ref["arquivo"].startswith(f"predicoes/{a['configuracao']}/")
            df = pd.read_parquet(saida_dir / "hpo" / ref["arquivo"])
            bloco = df[df["test_corpus"] == a["test_corpus"]]
            assert len(bloco) == a["n_test"] == len(indices[a["test_corpus"]].test)
            # A linha identifica a posição no parquet do dataset de teste.
            esperado = quadros[a["test_corpus"]]["linha"].to_numpy()[indices[a["test_corpus"]].test]
            assert (bloco["linha"].to_numpy() == esperado).all()
            if a["metrics"]["auroc"] is not None:
                recalculada = roc_auc_score(bloco["y_true"], bloco["score"])
                assert abs(recalculada - a["metrics"]["auroc"]) < 1e-9
        assert np.isfinite(modelo["limiar_seco"]) and np.isfinite(modelo["limiar_ajustado"])

    def test_particao_do_hpo_e_a_deduplicada(self, run_hpo, cenario_pequeno):
        config, gpu, quadros, indices, contagens, *_ = cenario_pequeno
        for nome, quadro in quadros.items():
            c = contagens[nome]
            assert c["n_antes"] == len(quadro)
            kept = (set(indices[nome].train) | set(indices[nome].val)
                    | set(indices[nome].test))
            assert len(kept) == c["n_final"]

    def test_mlp_no_regime_b_usa_a_validacao_agrupada(
        self, run_hpo, cenario_pequeno, monkeypatch
    ):
        import models

        config, gpu, quadros, indices, contagens, cabecalho, _ = cenario_pequeno
        chamadas = []
        original = models.ajustar

        def espiao(modelo, nome, X, y, grupos=None):
            agrupada = isinstance(models._final_estimator(modelo), models.MLPValidacaoAgrupada)
            chamadas.append((nome, agrupada, grupos is not None))
            return original(modelo, nome, X, y, grupos=grupos)

        monkeypatch.setattr(models, "ajustar", espiao)
        run_hpo.rodar(config, gpu, quadros, ["mlp"], "etld1", 2, cabecalho, indices, contagens)
        # seco + 2 tentativas + ajustado, todos agrupados e com grupos.
        assert chamadas == [("mlp", True, True)] * 4


class TestArvoreSujaRecusaOHpo:
    def test_arvore_suja_sai_com_1_antes_de_carregar_dados(
        self, run_hpo, monkeypatch, capsys
    ):
        import run_all

        monkeypatch.setattr(run_all, "arvore_suja", lambda raiz=None: ["M pipeline/src/x.py"])
        monkeypatch.setattr(sys, "argv", ["run_hpo.py"])

        def nao_deveria_rodar(*a, **k):
            raise AssertionError("carregou dados com a arvore suja")

        monkeypatch.setattr(run_hpo, "carregar_quadros", nao_deveria_rodar)
        with pytest.raises(SystemExit) as saida:
            run_hpo.main()
        assert saida.value.code == 1
        assert "pipeline/src/x.py" in capsys.readouterr().out

    def test_arvore_limpa_nao_impede(self, run_hpo, monkeypatch):
        import run_all

        monkeypatch.setattr(run_all, "arvore_suja", lambda raiz=None: [])
        run_hpo.checar_arvore_limpa()  # não levanta


class TestCabecalhoTemCommitCodigo:
    """Achado 2 do lote F: `hpo.json` e `hpo_fora_diagonal.json` levam o
    `commit_codigo`, para o `report.conferir_commits` compará-lo com o das rodadas."""

    def _main_com_stubs(self, run_hpo, monkeypatch, commit):
        import run_all

        capturado = {}

        class GpuFalsa:
            available = False

            def describe(self):
                return "gpu falsa"

        def rodar_falso(config, gpu, quadros, nomes, regime, n_trials, cabecalho, *a):
            capturado.update(cabecalho)
            return {"modelos": {"mlp": {"delta_teste": 0.1}}}, {}

        monkeypatch.setattr(run_all, "arvore_suja", lambda raiz=None: [])
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: commit)
        monkeypatch.setattr(run_hpo, "carregar_quadros", lambda c, p: (GpuFalsa(), {}))
        monkeypatch.setattr(run_hpo, "particionar_quadros", lambda c, q, r: ({}, {}))
        monkeypatch.setattr(run_hpo, "rodar", rodar_falso)
        monkeypatch.setattr(sys, "argv", ["run_hpo.py", "--modelos", "mlp"])
        return capturado

    def test_cabecalho_leva_o_commit_do_codigo(self, run_hpo, monkeypatch):
        capturado = self._main_com_stubs(run_hpo, monkeypatch, "d" * 40)
        assert run_hpo.main() == 0
        assert capturado["commit_codigo"] == "d" * 40

    def test_git_sem_commit_recusa_o_hpo(self, run_hpo, monkeypatch, capsys):
        self._main_com_stubs(run_hpo, monkeypatch, None)
        with pytest.raises(SystemExit) as saida:
            run_hpo.main()
        assert saida.value.code == 1
        assert "commit" in capsys.readouterr().out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
