"""T4.3 — DoD: 48 combinações geradas e execução **retomável** (RH2).

O laço vive em `src/matrix.py` e é compartilhado por `run_smoke.py` (portão) e
`run_all.py` (execução real). Este arquivo cobre o que o smoke não exercita:
o livro-caixa, a impressão digital da config e a decisão de pular ajustes.

Sobre treinar nos testes (SDD §0)
--------------------------------
A classe `TestRetomadaDePontaAPonta` chama `.fit()`. É deliberado e proporcional:
250 linhas sintéticas, florestas de 2 árvores, MLP de 5 iterações — segundos no
total. "Retomável" é o DoD da T4.3, e não há como provar que a retomada funciona
sem executar o laço duas vezes. As demais classes não treinam nada.
"""

import json

import numpy as np
import pandas as pd
import pytest

import features
import matrix
import models
from matrix import MatrixError

FINGERPRINT = "abc123def456abcd"


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def _registro(regime="etld1", treino="a", teste="a", modelo="xgboost",
              fingerprint=FINGERPRINT):
    """Registro mínimo no esquema de SDD §7.3."""
    return {
        "run_id": "commit-20260805T000000Z",
        "regime": regime,
        "train_corpus": treino,
        "test_corpus": teste,
        "model": modelo,
        "metrics": {
            "auroc": 0.9, "auc_pr": 0.85,
            "precision": 0.8, "recall": 0.75, "f1": 0.77,
            "fpr": 0.02, "mcc": 0.7, "brier": 0.05, "ece": 0.03,
            "threshold": 0.5,
            "tp": 100, "fp": 10, "tn": 800, "fn": 30,
        },
        "fit_seconds": 1.0,
        "n_test": 940,
        "diagonal": treino == teste,
        "warnings": [],
        "config_fingerprint": fingerprint,
    }


def _corpus_sintetico(nome: str, n_dominios=60, por_dominio=4, offset=0):
    """Corpus já canonizado, com as 41 features preenchidas de forma separável."""
    import zlib

    import features as feats

    # Semente estável entre processos. `hash(str)` muda a cada execução
    # (PYTHONHASHSEED), e com ele os dados sintéticos: era a causa da falha
    # intermitente de `test_ic_presente_quando_bootstrap_ligado`.
    rng = np.random.default_rng(zlib.crc32(nome.encode("utf-8")))
    linhas = []
    for d in range(n_dominios):
        dominio = f"{nome}{d + offset:04d}.com"
        rotulo = (d + offset) % 2
        for r in range(por_dominio):
            linhas.append({
                "label": rotulo,
                "url": f"http://{dominio}/p{r}",
                "url_norm": f"{dominio}/p{r}",
                "corpus": nome,
                "split_oficial": "train" if d % 5 else "test",
                "etld_plus_one": dominio,
            })
    df = pd.DataFrame(linhas)
    df["linha"] = np.arange(len(df), dtype=np.int32)
    # Features sintéticas: correlacionadas com o rótulo, com ruído, para que os
    # modelos tenham o que aprender sem depender da extração real.
    for i, col in enumerate(feats.FEATURE_NAMES):
        base = df["label"].to_numpy() * (5 + i % 3)
        df[col] = (base + rng.normal(0, 2, len(df))).astype(np.float32)
    return df


def _config_minima():
    """Config real, com os 6 modelos cortados ao mínimo executável."""
    config = models.load_experiment_config()
    minimos = {
        "logistic_regression": {"max_iter": 20},
        "random_forest": {"n_estimators": 2},
        "xgboost": {"n_estimators": 2, "max_depth": 2},
        "lightgbm": {"n_estimators": 2, "num_leaves": 3},
        "catboost": {"iterations": 2, "depth": 2},
        "mlp": {"hidden_layer_sizes": [4], "max_iter": 5, "early_stopping": False},
    }
    import copy

    config = copy.deepcopy(config)
    for nome, params in minimos.items():
        config["modelos"][nome]["params"].update(params)
    config["split"]["test_size"] = 0.25
    config["split"]["val_size"] = 0.15
    config["split"]["val_size_oficial"] = 0.15
    return config


# ---------------------------------------------------------------------------
# Impressão digital da config
# ---------------------------------------------------------------------------
class TestConfigFingerprint:
    def test_estavel_para_a_mesma_config(self):
        config = models.load_experiment_config()
        assert matrix.config_fingerprint(config) == matrix.config_fingerprint(config)

    def test_muda_quando_hiperparametro_muda(self):
        import copy

        a = models.load_experiment_config()
        b = copy.deepcopy(a)
        b["modelos"]["xgboost"]["params"]["max_depth"] = 3
        assert matrix.config_fingerprint(a) != matrix.config_fingerprint(b)

    def test_muda_quando_a_seed_muda(self):
        import copy

        a = models.load_experiment_config()
        b = copy.deepcopy(a)
        b["seed"] = 7
        assert matrix.config_fingerprint(a) != matrix.config_fingerprint(b)

    def test_muda_entre_smoke_e_execucao_real(self):
        """Crítico: os overrides do smoke não podem contaminar `results/`.
        Impressões digitais diferentes fazem o livro-caixa real rejeitar
        registros de smoke."""
        real = models.load_experiment_config()
        smoke = models.apply_smoke_overrides(real)
        assert matrix.config_fingerprint(real) != matrix.config_fingerprint(smoke)

    def test_nao_muda_com_n_jobs(self):
        """Número de núcleos não altera resultado (o determinismo está travado
        em models._predicao_serial), então não deve invalidar uma retomada."""
        import copy

        a = models.load_experiment_config()
        b = copy.deepcopy(a)
        b["cpu"]["n_jobs"] = 4
        assert matrix.config_fingerprint(a) == matrix.config_fingerprint(b)


# ---------------------------------------------------------------------------
# Livro-caixa (RH2)
# ---------------------------------------------------------------------------
class TestLivroCaixa:
    def test_inexistente_devolve_vazio(self, tmp_path):
        regs, feitos, trunc = matrix.read_ledger(tmp_path / "nao_existe.jsonl",
                                                 FINGERPRINT, 2)
        assert regs == [] and feitos == set() and trunc == 0

    def test_ida_e_volta(self, tmp_path):
        ledger = tmp_path / "m.jsonl"
        novos = [_registro(teste="a"), _registro(teste="b")]
        matrix.append_ledger(ledger, novos)
        regs, feitos, _ = matrix.read_ledger(ledger, FINGERPRINT, 2)
        assert len(regs) == 2
        assert feitos == {("etld1", "a", "xgboost")}

    def test_ajuste_incompleto_e_refeito(self, tmp_path):
        """Um ajuste com 1 de 2 corpora de teste não dá linha de tabela; tem de
        ser refeito, e seus registros parciais descartados para não duplicar."""
        ledger = tmp_path / "m.jsonl"
        matrix.append_ledger(ledger, [_registro(teste="a")])
        regs, feitos, _ = matrix.read_ledger(ledger, FINGERPRINT, 2)
        assert feitos == set()
        assert regs == []

    def test_linha_truncada_e_descartada_sem_quebrar(self, tmp_path):
        """Queda no meio da escrita deixa JSON pela metade. Descartar é seguro:
        o ajuste é refeito."""
        ledger = tmp_path / "m.jsonl"
        matrix.append_ledger(ledger, [_registro(teste="a"), _registro(teste="b")])
        with ledger.open("a", encoding="utf-8") as f:
            f.write('{"regime": "etld1", "train_cor')  # truncada
        regs, feitos, trunc = matrix.read_ledger(ledger, FINGERPRINT, 2)
        assert trunc == 1
        assert len(regs) == 2
        assert feitos == {("etld1", "a", "xgboost")}

    def test_config_diferente_aborta(self, tmp_path):
        """A trava que impede misturar hiperparâmetros na mesma tabela (D5)."""
        ledger = tmp_path / "m.jsonl"
        matrix.append_ledger(ledger, [_registro(fingerprint="OUTRAconfig12345")])
        with pytest.raises(MatrixError, match="outra configuração"):
            matrix.read_ledger(ledger, FINGERPRINT, 2)

    def test_mensagem_de_erro_e_acionavel(self, tmp_path):
        ledger = tmp_path / "m.jsonl"
        matrix.append_ledger(ledger, [_registro(fingerprint="OUTRAconfig12345")])
        with pytest.raises(MatrixError) as exc:
            matrix.read_ledger(ledger, FINGERPRINT, 2)
        texto = str(exc.value)
        assert "experiment.yml" in texto
        assert "--recomecar" in texto

    def test_conta_ajustes_por_regime_e_modelo(self, tmp_path):
        ledger = tmp_path / "m.jsonl"
        novos = []
        for regime in ("oficial", "etld1"):
            for modelo in ("xgboost", "mlp"):
                for teste in ("a", "b"):
                    novos.append(_registro(regime=regime, modelo=modelo, teste=teste))
        matrix.append_ledger(ledger, novos)
        _, feitos, _ = matrix.read_ledger(ledger, FINGERPRINT, 2)
        assert len(feitos) == 4

    def test_fsync_grava_de_verdade(self, tmp_path):
        """O `fsync` do append é o que garante que uma queda de energia não
        perca ajustes já concluídos."""
        ledger = tmp_path / "m.jsonl"
        matrix.append_ledger(ledger, [_registro()])
        assert ledger.exists()
        assert json.loads(ledger.read_text(encoding="utf-8").strip())["model"] == "xgboost"


# ---------------------------------------------------------------------------
# Conferência e ordenação
# ---------------------------------------------------------------------------
class TestConferencia:
    def _completo(self, n=2):
        corpora = ["a", "b"][:n]
        return [
            _registro(regime=reg, treino=tr, teste=te, modelo=m)
            for reg in ("oficial", "etld1")
            for tr in corpora
            for te in corpora
            for m in models.MODEL_ORDER
        ]

    def test_matriz_completa_passa(self):
        config = models.load_experiment_config()
        matrix.check_records(self._completo(), config, 2)

    def test_registro_faltando_e_barrado(self):
        config = models.load_experiment_config()
        with pytest.raises(MatrixError, match="esperados"):
            matrix.check_records(self._completo()[:-1], config, 2)

    def test_duplicata_e_barrada(self):
        config = models.load_experiment_config()
        regs = self._completo()
        regs[-1] = regs[0]
        with pytest.raises(MatrixError, match="duplicado"):
            matrix.check_records(regs, config, 2)

    def test_metrica_invalida_e_barrada(self):
        """Fecha o ciclo do B1 no nível do laço."""
        config = models.load_experiment_config()
        regs = self._completo()
        regs[0]["metrics"]["mcc"] = 2.8497
        with pytest.raises(Exception, match="fora de"):
            matrix.check_records(regs, config, 2)

    def test_sem_fora_diagonal_e_barrado(self):
        """Uma matriz só com diagonal significa laço errado — e é metade do
        artigo faltando (a fora-diagonal É a contribuição C2).

        Os registros são marcados como diagonais mantendo chaves distintas: se
        eu duplicasse registros para bater a contagem, a checagem de duplicata
        dispararia primeiro e este teste passaria sem nunca exercitar o ramo
        que ele deveria cobrir.
        """
        config = models.load_experiment_config()
        regs = self._completo()
        for r in regs:
            r["diagonal"] = True
        with pytest.raises(MatrixError, match="fora da diagonal"):
            matrix.check_records(regs, config, 2)

    def test_ordem_canonica_e_independente_da_execucao(self):
        """Uma retomada executa em ordem diferente; sem ordenar, dois resultados
        idênticos apareceriam como mudança no `git diff`."""
        regs = self._completo()
        import random

        embaralhado = regs[:]
        random.Random(7).shuffle(embaralhado)
        assert matrix.sort_records(regs) == matrix.sort_records(embaralhado)

    def test_ordem_segue_model_order(self):
        regs = matrix.sort_records(self._completo())
        primeiros = [r["model"] for r in regs[:len(models.MODEL_ORDER)]]
        assert primeiros == list(models.MODEL_ORDER)


# ---------------------------------------------------------------------------
# Partições
# ---------------------------------------------------------------------------
class TestBuildSplits:
    def test_particiona_os_dois_regimes(self):
        config = _config_minima()
        quadros = {"a": _corpus_sintetico("a")}
        indices, resumos = matrix.build_splits(config, quadros)
        assert set(indices) == {("a", "oficial"), ("a", "etld1")}
        assert resumos["a/etld1"]["domain_overlap"]["train_test"] == 0

    def test_calcula_uma_unica_vez_e_devolve_os_dois(self):
        """Devolver índices E resumos juntos evita reparticionar 800 mil linhas
        só para gerar relatório — desperdício real da primeira versão."""
        config = _config_minima()
        quadros = {"a": _corpus_sintetico("a")}
        indices, resumos = matrix.build_splits(config, quadros)
        assert len(indices) == len(resumos) == 2


# ---------------------------------------------------------------------------
# Retomada de ponta a ponta (treina — ver nota no topo do arquivo)
# ---------------------------------------------------------------------------
class TestRetomadaDePontaAPonta:
    @pytest.fixture(scope="class")
    def cenario(self):
        return _config_minima(), {
            "a": _corpus_sintetico("a", n_dominios=50),
            "b": _corpus_sintetico("b", n_dominios=50, offset=500),
        }

    def test_primeira_execucao_gera_a_matriz_completa(self, cenario, tmp_path):
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        res = matrix.run_matrix(
            config, quadros, gpu, run_id="teste", ledger_path=tmp_path / "m.jsonl",
        )
        n_esperado = 2 * 2 * 6 * 2  # regimes x treino x modelos x teste
        assert len(res.records) == n_esperado
        assert res.n_fits_executados == 2 * 2 * 6
        assert res.n_fits_pulados == 0
        matrix.check_records(res.records, config, 2)

    def test_segunda_execucao_pula_tudo(self, cenario, tmp_path):
        """O DoD do RH2: retomar não refaz ajuste algum."""
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        ledger = tmp_path / "m.jsonl"

        primeira = matrix.run_matrix(config, quadros, gpu, run_id="r1",
                                     ledger_path=ledger)
        segunda = matrix.run_matrix(config, quadros, gpu, run_id="r2",
                                    ledger_path=ledger)

        assert segunda.n_fits_executados == 0, "retomada refez ajuste (RH2 quebrado)"
        assert segunda.n_fits_pulados == 24
        assert len(segunda.records) == len(primeira.records)

    def test_retomada_do_meio_completa_o_que_falta(self, cenario, tmp_path):
        """Simula queda: apaga metade dos ajustes do livro-caixa e exige que a
        retomada refaça exatamente os que faltam — 'se cair no 30º, retoma do 30º'."""
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        ledger = tmp_path / "m.jsonl"

        matrix.run_matrix(config, quadros, gpu, run_id="r1", ledger_path=ledger)

        linhas = ledger.read_text(encoding="utf-8").splitlines()
        # cada ajuste gravou 2 registros (2 corpora de teste); mantém 10 ajustes
        ledger.write_text("\n".join(linhas[: 10 * 2]) + "\n", encoding="utf-8")

        retomada = matrix.run_matrix(config, quadros, gpu, run_id="r2",
                                     ledger_path=ledger)
        assert retomada.n_fits_pulados == 10
        assert retomada.n_fits_executados == 14  # 24 - 10
        matrix.check_records(retomada.records, config, 2)

    def test_retomada_preserva_os_valores(self, cenario, tmp_path):
        """Os registros retomados do disco têm de ser idênticos aos originais —
        senão a tabela final mistura números de execuções diferentes."""
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        ledger = tmp_path / "m.jsonl"

        primeira = matrix.run_matrix(config, quadros, gpu, run_id="r1",
                                     ledger_path=ledger)
        linhas = ledger.read_text(encoding="utf-8").splitlines()
        ledger.write_text("\n".join(linhas[: 10 * 2]) + "\n", encoding="utf-8")
        retomada = matrix.run_matrix(config, quadros, gpu, run_id="r2",
                                     ledger_path=ledger)

        def chave(r):
            return (r["regime"], r["train_corpus"], r["test_corpus"], r["model"])

        antes = {chave(r): r["metrics"] for r in primeira.records}
        depois = {chave(r): r["metrics"] for r in retomada.records}
        assert set(antes) == set(depois)
        for k in antes:
            assert antes[k] == depois[k], f"métrica mudou na retomada: {k}"

    def test_sem_livro_caixa_nao_retoma(self, cenario, tmp_path):
        """É o modo do smoke: um portão sempre roda do zero."""
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        a = matrix.run_matrix(config, quadros, gpu, run_id="r1", ledger_path=None)
        b = matrix.run_matrix(config, quadros, gpu, run_id="r2", ledger_path=None)
        assert a.n_fits_executados == b.n_fits_executados == 24
        assert not (tmp_path / "m.jsonl").exists()

    def test_limiar_vem_da_validacao_do_corpus_de_treino(self, cenario, tmp_path):
        """SDD §4 D6: o limiar é escolhido na validação do TREINO e transportado
        para todos os corpora de teste — o mesmo valor na linha inteira."""
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        res = matrix.run_matrix(config, quadros, gpu, run_id="r1", ledger_path=None)

        por_ajuste: dict = {}
        for rec in res.records:
            chave = (rec["regime"], rec["train_corpus"], rec["model"])
            por_ajuste.setdefault(chave, set()).add(rec["metrics"]["threshold"])
        for chave, limiares in por_ajuste.items():
            assert len(limiares) == 1, (
                f"{chave} usou limiares diferentes por corpus de teste: {limiares}"
            )

    def test_diagonal_e_fora_diagonal_presentes(self, cenario):
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        res = matrix.run_matrix(config, quadros, gpu, run_id="r1", ledger_path=None)
        assert any(r["diagonal"] for r in res.records)
        assert any(not r["diagonal"] for r in res.records)

    def test_proveniencia_em_todo_registro(self, cenario):
        """RNF4: todo artefato carimbado."""
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        res = matrix.run_matrix(config, quadros, gpu, run_id="r1", ledger_path=None)
        for rec in res.records:
            assert rec["provenance"]["seed"] == config["seed"]
            assert rec["config_fingerprint"]
            assert rec["run_id"] == "r1"


# ---------------------------------------------------------------------------
# Anexos do registro: curvas, importâncias e IC (05/08/2026)
# ---------------------------------------------------------------------------
class TestAnexosDoRegistro:
    """Curvas e importâncias viajam no registro e saem em arquivo próprio.

    Viajam no registro porque o livro-caixa é o que sobrevive a uma queda: se a
    curva ficasse fora dele, uma execução retomada devolveria metade das curvas
    e o `curves.json` sairia incompleto sem ninguém notar. Saem em arquivo
    próprio porque o `matrix.json`/`matrix.csv` são o que o `report.py` lê.
    """

    @pytest.fixture(scope="class")
    def cenario(self):
        config = _config_minima()
        config["avaliacao"]["bootstrap_n"] = 40  # barato; o valor real é 500
        return config, {"a": _corpus_sintetico("a", n_dominios=40)}

    @pytest.fixture(scope="class")
    def registros(self, cenario):
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        return matrix.run_matrix(
            config, quadros, gpu, run_id="anexos", ledger_path=None
        ).records

    def test_curva_e_importancia_chegam_no_registro(self, registros):
        for rec in registros:
            assert "curves" in rec, "sem curva, a F1 nao tem fonte"
            assert "importances" in rec

    def test_ic_presente_quando_bootstrap_ligado(self, registros):
        for rec in registros:
            ic = rec["ci"]
            assert ic is not None
            assert ic["auroc_lo"] <= rec["metrics"]["auroc"] <= ic["auroc_hi"], (
                "o ponto tem de cair dentro do proprio intervalo"
            )
            assert ic["n_resamples"] == 40

    def test_ic_ausente_quando_bootstrap_desligado(self, cenario):
        """É o que o portão faz: RF9 tem orçamento de tempo."""
        config, quadros = cenario
        config = {**config, "avaliacao": {**config["avaliacao"], "bootstrap_n": 0}}
        gpu = models.GpuInfo(available=False, reason="teste")
        res = matrix.run_matrix(config, quadros, gpu, run_id="x", ledger_path=None)
        assert all("ci" not in r for r in res.records)

    def test_importancia_do_mlp_e_none(self, registros):
        """Rede não tem importância por feature sem método de atribuição."""
        por_modelo = {r["model"]: r["importances"] for r in registros}
        assert por_modelo["mlp"] is None
        for arvore in ("random_forest", "xgboost", "lightgbm", "catboost"):
            assert len(por_modelo[arvore]) == len(features.FEATURE_NAMES)
        assert len(por_modelo["logistic_regression"]) == len(features.FEATURE_NAMES)

    def test_ic_sobrevive_identico_a_retomada(self, cenario, tmp_path):
        """RNF1 pelo caminho que importa: o IC não pode depender da ordem.

        A retomada executa os ajustes em ordem diferente. Se a semente do
        bootstrap fosse derivada de um contador de execução, o intervalo mudaria
        entre a primeira e a segunda execução — e duas execuções deixariam de
        dar métricas idênticas.
        """
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        ledger = tmp_path / "m.jsonl"
        primeira = matrix.run_matrix(
            config, quadros, gpu, run_id="p", ledger_path=ledger
        ).records
        segunda = matrix.run_matrix(
            config, quadros, gpu, run_id="s", ledger_path=ledger
        ).records
        chave = lambda r: (r["regime"], r["train_corpus"], r["test_corpus"], r["model"])  # noqa: E731
        a = {chave(r): r["ci"] for r in primeira}
        b = {chave(r): r["ci"] for r in segunda}
        assert a == b

    def test_separar_anexos_limpa_o_registro_e_grava(self, registros, tmp_path):
        copia = [dict(r) for r in registros]
        escritos = matrix.separar_anexos(copia, tmp_path)
        assert "curves.json" in escritos and "importances.json" in escritos
        for rec in copia:
            assert "curves" not in rec and "importances" not in rec
            assert "ci" in rec, "o IC fica no registro: as tabelas leem dele"
        pontos = json.loads((tmp_path / "curves.json").read_text(encoding="utf-8"))
        assert len(pontos["grade"]) == 101
        assert len(pontos["celulas"]) == len(copia)
        imp = json.loads((tmp_path / "importances.json").read_text(encoding="utf-8"))
        # Importância é por AJUSTE, não por célula: 1 corpus x 2 regimes x 5
        # modelos com importância (o MLP não tem).
        assert len(imp["ajustes"]) == 2 * 5

    def test_csv_recusa_registro_com_anexo(self, registros):
        """A trava que impede a coluna `curves` de voltar ao matrix.csv."""
        import evaluate

        with pytest.raises(evaluate.MetricsError, match="anexo"):
            evaluate.records_to_dataframe([dict(registros[0])])


# ---------------------------------------------------------------------------
# Leave-one-corpus-out (D3-b)
# ---------------------------------------------------------------------------
class TestLeaveOneCorpusOut:
    """A T6 só é comparável com a fora-diagonal da T2 se três coisas valerem:
    o corpus retido não entra no treino, o teste é a MESMA partição da matriz,
    e nenhuma célula é diagonal.
    """

    @pytest.fixture(scope="class")
    def cenario(self):
        config = _config_minima()
        config["avaliacao"]["bootstrap_n"] = 0  # o que se testa aqui é o laço
        config["modelos"] = {
            k: v for k, v in config["modelos"].items()
            if k in ("logistic_regression", "random_forest")
        }
        return config, {
            "a": _corpus_sintetico("a", n_dominios=40),
            "b": _corpus_sintetico("b", n_dominios=40, offset=500),
            "c": _corpus_sintetico("c", n_dominios=40, offset=900),
        }

    @pytest.fixture(scope="class")
    def resultado(self, cenario):
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        original = list(models.MODEL_ORDER)
        try:
            models.MODEL_ORDER = tuple(config["modelos"])
            return matrix.run_loco(
                config, quadros, gpu, run_id="loco", ledger_path=None
            )
        finally:
            models.MODEL_ORDER = tuple(original)

    def test_um_ajuste_por_corpus_retido_por_regime_por_modelo(self, cenario, resultado):
        config, quadros = cenario
        esperado = len(config["regimes"]) * len(quadros) * len(config["modelos"])
        assert len(resultado.records) == esperado
        matrix.check_loco_records(resultado.records, config, len(quadros))

    def test_rotulo_de_treino_nao_e_um_corpus(self, cenario, resultado):
        _, quadros = cenario
        for rec in resultado.records:
            assert rec["train_corpus"].startswith(matrix.LOCO_PREFIXO)
            assert rec["train_corpus"] not in quadros
            assert rec["test_corpus"] in quadros

    def test_nenhuma_celula_e_diagonal(self, resultado):
        """Se uma for, o corpus retido entrou no treino."""
        assert not any(r["diagonal"] for r in resultado.records)

    def test_o_teste_e_a_mesma_particao_da_matriz(self, cenario, resultado):
        """Sem isto a T6 deixa de ser comparável com a fora-diagonal da T2 —
        que é o ponto inteiro da tabela."""
        config, quadros = cenario
        indices, _ = matrix.build_splits(config, quadros)
        for rec in resultado.records:
            esperado = len(indices[(rec["test_corpus"], rec["regime"])].test)
            assert rec["n_test"] == esperado

    def test_uniao_de_treino_exclui_o_corpus_retido(self, cenario, resultado):
        """Verificação por tamanho: o treino da união tem de ser exatamente a
        soma dos treinos dos OUTROS corpora, nem uma linha do retido."""
        config, quadros = cenario
        indices, _ = matrix.build_splits(config, quadros)
        for regime in config["regimes"]:
            for retido in quadros:
                outros = [n for n in quadros if n != retido]
                esperado = sum(len(indices[(n, regime)].train) for n in outros)
                com_retido = esperado + len(indices[(retido, regime)].train)
                assert esperado != com_retido, "cenario degenerado"
                # O tamanho do treino não vai no registro, então o que se
                # confere aqui é a aritmética que `run_loco` usa; o teste de
                # regressão de verdade é `test_nenhuma_celula_e_diagonal` mais a
                # ausência do retido em `outros`.
                assert retido not in outros

    def test_exige_ao_menos_dois_corpora(self, cenario):
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        with pytest.raises(MatrixError, match="ao menos 2 corpora"):
            matrix.run_loco(
                config, {"a": quadros["a"]}, gpu, run_id="x", ledger_path=None
            )

    def test_livro_caixa_proprio_retoma_o_loco(self, cenario, tmp_path):
        """O LOCO tem 1 avaliação por ajuste e a matriz tem N: no mesmo arquivo,
        `read_ledger` acharia todo ajuste do LOCO incompleto para sempre."""
        config, quadros = cenario
        gpu = models.GpuInfo(available=False, reason="teste")
        ledger = tmp_path / "loco.jsonl"
        original = list(models.MODEL_ORDER)
        try:
            models.MODEL_ORDER = tuple(config["modelos"])
            primeira = matrix.run_loco(
                config, quadros, gpu, run_id="p", ledger_path=ledger
            )
            segunda = matrix.run_loco(
                config, quadros, gpu, run_id="s", ledger_path=ledger
            )
        finally:
            models.MODEL_ORDER = tuple(original)
        assert primeira.n_fits_executados > 0
        assert segunda.n_fits_executados == 0
        assert segunda.n_fits_pulados == primeira.n_fits_executados


class TestCheckLocoRecords:
    def _regs(self):
        regs = []
        for c in ("a", "b"):
            for m in ("logistic_regression", "random_forest"):
                rec = _registro(
                    regime="etld1", treino=f"{matrix.LOCO_PREFIXO}{c}",
                    teste=c, modelo=m,
                )
                rec["diagonal"] = False  # nunca diagonal no LOCO
                regs.append(rec)
        return regs

    def _config(self):
        return {
            "regimes": ["etld1"],
            "modelos": {"logistic_regression": {}, "random_forest": {}},
        }

    def test_aceita_conjunto_completo(self):
        matrix.check_loco_records(self._regs(), self._config(), 2)

    def test_recusa_faltante(self):
        with pytest.raises(MatrixError, match="esperadas 4 avaliacoes"):
            matrix.check_loco_records(self._regs()[:-1], self._config(), 2)

    def test_recusa_duplicata(self):
        regs = self._regs()
        regs[1] = dict(regs[0])
        with pytest.raises(MatrixError, match="duplicado"):
            matrix.check_loco_records(regs, self._config(), 2)

    def test_recusa_celula_marcada_como_diagonal(self):
        """Uma diagonal no LOCO significa que a união vazou o corpus retido."""
        regs = self._regs()
        regs[0]["diagonal"] = True
        with pytest.raises(MatrixError, match="diagonal"):
            matrix.check_loco_records(regs, self._config(), 2)

    def test_recusa_rotulo_de_treino_sem_prefixo(self):
        regs = self._regs()
        regs[0]["train_corpus"] = "a"
        with pytest.raises(MatrixError, match="train_corpus inesperado"):
            matrix.check_loco_records(regs, self._config(), 2)


# ---------------------------------------------------------------------------
# Frente A, T12: predições, MLP agrupado, dedup e impressão digital
# (PRED-01 a PRED-03, MLP-01, DEDUP-05)
# ---------------------------------------------------------------------------
GPU_TESTE = models.GpuInfo(available=False, reason="teste")


def _cenario_dois_datasets():
    config = _config_minima()
    config["avaliacao"]["bootstrap_n"] = 0
    quadros = {
        "a": _corpus_sintetico("a", n_dominios=50),
        "b": _corpus_sintetico("b", n_dominios=50, offset=500),
    }
    return config, quadros


def _chave_celula(r):
    return (r["regime"], r["train_corpus"], r["test_corpus"], r["model"])


class TestPredicoesPorAjuste:
    """PRED-01 e PRED-02: cada ajuste grava um parquet e o livro-caixa o confere."""

    @pytest.fixture(scope="class")
    def rodada(self, tmp_path_factory):
        config, quadros = _cenario_dois_datasets()
        dir_rodada = tmp_path_factory.mktemp("rodada")
        res = matrix.run_matrix(
            config, quadros, GPU_TESTE, run_id="pred",
            ledger_path=dir_rodada / "matrix.jsonl", dir_rodada=dir_rodada,
        )
        return config, quadros, dir_rodada, res

    def test_cada_ajuste_grava_um_parquet(self, rodada):
        _, _, dir_rodada, res = rodada
        arquivos = sorted((dir_rodada / "predicoes" / "matriz").glob("*.parquet"))
        assert len(arquivos) == res.n_fits_executados == 2 * 2 * 6
        for rec in res.records:
            ref = rec["predicoes"]
            assert (dir_rodada / ref["arquivo"]).is_file()
            assert len(ref["sha256"]) == 64

    def test_auroc_recalculada_do_parquet_bate_com_a_do_registro(self, rodada):
        from sklearn.metrics import roc_auc_score

        _, _, dir_rodada, res = rodada
        conferidas = 0
        for rec in res.records:
            df = pd.read_parquet(dir_rodada / rec["predicoes"]["arquivo"])
            bloco = df[df["test_corpus"] == rec["test_corpus"]]
            assert len(bloco) == rec["n_test"]
            assert bloco["score"].dtype == np.float64
            auroc = rec["metrics"]["auroc"]
            if auroc is None:
                continue
            recalculada = roc_auc_score(bloco["y_true"], bloco["score"])
            assert abs(recalculada - auroc) < 1e-9
            conferidas += 1
        assert conferidas > 0

    def test_linha_aponta_para_o_parquet_do_dataset_de_teste(self, rodada):
        _, quadros, dir_rodada, res = rodada
        for rec in res.records[:12]:
            df = pd.read_parquet(dir_rodada / rec["predicoes"]["arquivo"])
            bloco = df[df["test_corpus"] == rec["test_corpus"]]
            quadro = quadros[rec["test_corpus"]]
            rotulos = quadro.set_index("linha").loc[bloco["linha"], "label"]
            assert (rotulos.to_numpy() == bloco["y_true"].to_numpy()).all()

    def test_proveniencia_registra_a_validacao_interna(self, rodada):
        _, _, _, res = rodada
        for rec in res.records:
            valor = rec["provenance"]["validacao_interna"]
            if rec["model"] != "mlp":
                assert valor is None
            elif rec["regime"] == "etld1":
                assert valor == "agrupada_etld1"
            else:
                assert valor == "linha"

    def test_retomada_refaz_so_o_ajuste_com_parquet_ausente_ou_alterado(
        self, tmp_path
    ):
        config, quadros = _cenario_dois_datasets()
        ledger = tmp_path / "matrix.jsonl"
        primeira = matrix.run_matrix(
            config, quadros, GPU_TESTE, run_id="r1",
            ledger_path=ledger, dir_rodada=tmp_path,
        )
        assert primeira.n_fits_executados == 24

        arquivos = sorted((tmp_path / "predicoes" / "matriz").glob("*.parquet"))
        ausente, alterado = arquivos[0], arquivos[5]
        ausente.unlink()
        alterado.write_bytes(b"adulterado")

        mensagens: list[str] = []
        segunda = matrix.run_matrix(
            config, quadros, GPU_TESTE, run_id="r2", ledger_path=ledger,
            dir_rodada=tmp_path, log=lambda m="": mensagens.append(m),
        )
        assert segunda.n_fits_executados == 2
        assert segunda.n_fits_pulados == 22
        assert ausente.is_file() and alterado.read_bytes() != b"adulterado"
        assert any("refazendo" in m for m in mensagens)
        # Sem registro duplicado: vale o último de cada célula.
        assert len(segunda.records) == 48
        matrix.check_records(segunda.records, config, 2)

        terceira = matrix.run_matrix(
            config, quadros, GPU_TESTE, run_id="r3", ledger_path=ledger,
            dir_rodada=tmp_path,
        )
        assert terceira.n_fits_executados == 0
        assert len(terceira.records) == 48

    def test_registro_sem_referencia_de_predicoes_e_refeito(self, tmp_path):
        """Livro-caixa sem o campo `predicoes` não vale quando a rodada persiste."""
        config, quadros = _cenario_dois_datasets()
        ledger = tmp_path / "matrix.jsonl"
        matrix.run_matrix(config, quadros, GPU_TESTE, run_id="r1", ledger_path=ledger)
        retomada = matrix.run_matrix(
            config, quadros, GPU_TESTE, run_id="r2", ledger_path=ledger,
            dir_rodada=tmp_path,
        )
        assert retomada.n_fits_executados == 24

    def test_exige_a_coluna_linha_quando_persiste(self, tmp_path):
        config, quadros = _cenario_dois_datasets()
        quadros["a"] = quadros["a"].drop(columns=["linha"])
        with pytest.raises(MatrixError, match="linha"):
            matrix.run_matrix(
                config, quadros, GPU_TESTE, run_id="x", dir_rodada=tmp_path,
            )


def _espiar_ajustar(monkeypatch):
    """Substitui `models.ajustar` por um espião que chama o original."""
    chamadas: list[dict] = []
    original = models.ajustar

    def espiao(modelo, nome, X, y, grupos=None):
        agrupada = isinstance(models._final_estimator(modelo), models.MLPValidacaoAgrupada)
        chamadas.append({"nome": nome, "agrupada": agrupada, "grupos": grupos, "n": len(y)})
        return original(modelo, nome, X, y, grupos=grupos)

    monkeypatch.setattr(models, "ajustar", espiao)
    return chamadas


class TestMlpAgrupadoNaMatrizENoLoco:
    """MLP-01 e MLP-02: o MLP recebe os grupos no Regime B e não recebe no A."""

    def test_matriz_passa_os_grupos_do_treino_no_regime_b(self, monkeypatch):
        config, quadros = _cenario_dois_datasets()
        chamadas = _espiar_ajustar(monkeypatch)
        indices, _ = matrix.build_splits(config, quadros)
        matrix.run_matrix(config, quadros, GPU_TESTE, run_id="g")

        mlp = [c for c in chamadas if c["nome"] == "mlp"]
        assert len(mlp) == 4  # 2 regimes x 2 treinos
        outros = [c for c in chamadas if c["nome"] != "mlp"]
        # Os demais modelos nunca são agrupados (`models.ajustar` ignora os grupos).
        assert all(not c["agrupada"] for c in outros)

        regime_b = [c for c in mlp if c["agrupada"]]
        regime_a = [c for c in mlp if not c["agrupada"]]
        assert len(regime_b) == len(regime_a) == 2
        assert all(c["grupos"] is None for c in regime_a)
        for chamada, treino in zip(regime_b, ("a", "b"), strict=True):
            esperado = (
                splits_group_keys(quadros[treino])[indices[(treino, "etld1")].train]
            )
            assert chamada["grupos"] is not None
            assert chamada["n"] == len(esperado)
            assert list(chamada["grupos"]) == list(esperado)

    def test_loco_concatena_os_grupos_das_fontes_na_ordem_de_x_tr(self, monkeypatch):
        config = _config_minima()
        config["avaliacao"]["bootstrap_n"] = 0
        config["modelos"] = {"mlp": config["modelos"]["mlp"]}
        quadros = {
            "a": _corpus_sintetico("a", n_dominios=40),
            "b": _corpus_sintetico("b", n_dominios=40, offset=500),
            "c": _corpus_sintetico("c", n_dominios=40, offset=900),
        }
        chamadas = _espiar_ajustar(monkeypatch)
        indices, _ = matrix.build_splits(config, quadros)
        original = list(models.MODEL_ORDER)
        try:
            models.MODEL_ORDER = ("mlp",)
            matrix.run_loco(config, quadros, GPU_TESTE, run_id="g")
        finally:
            models.MODEL_ORDER = tuple(original)

        regime_b = [c for c in chamadas if c["agrupada"]]
        regime_a = [c for c in chamadas if not c["agrupada"]]
        assert len(regime_b) == len(regime_a) == 3
        assert all(c["grupos"] is None for c in regime_a)
        for chamada, retido in zip(regime_b, quadros, strict=True):
            outros = [n for n in quadros if n != retido]
            esperado = np.concatenate([
                splits_group_keys(quadros[n])[indices[(n, "etld1")].train]
                for n in outros
            ])
            assert chamada["n"] == len(esperado)
            assert list(chamada["grupos"]) == list(esperado)


def splits_group_keys(quadro):
    import splits

    return splits.group_keys(quadro).to_numpy()


class TestImpressaoDigitalDaFrenteA:
    def test_muda_com_strip_scheme(self):
        config = models.load_experiment_config()
        com = {**config, "_strip_scheme": True}
        assert matrix.config_fingerprint(config) != matrix.config_fingerprint(com)

    def test_muda_com_o_sha256_da_psl(self, monkeypatch):
        import psl

        config = models.load_experiment_config()
        antes = matrix.config_fingerprint(config)
        monkeypatch.setattr(psl, "sha256", lambda: "0" * 64)
        assert matrix.config_fingerprint(config) != antes

    def test_muda_com_a_canonizacao(self, monkeypatch):
        # Achado 4 do lote F: a normalizacao gera o url_norm, chave da dedup.
        from data.url_canonicalization import URLCanonicalizer

        config = models.load_experiment_config()
        antes = matrix.config_fingerprint(config)
        monkeypatch.setattr(
            URLCanonicalizer, "_normalize_query", lambda self, query: query or None
        )
        assert matrix.config_fingerprint(config) != antes


class TestResumoDePaticoesComDedup:
    def test_resumo_traz_as_contagens_de_dedup(self):
        config = _config_minima()
        quadro = _corpus_sintetico("a", n_dominios=40)
        repetida = quadro.iloc[[3]].copy()  # mesma URL, mesmo rótulo
        quadro = pd.concat([quadro, repetida], ignore_index=True)
        quadro["linha"] = np.arange(len(quadro), dtype=np.int32)

        indices, resumos = matrix.build_splits(config, {"a": quadro})

        for regime in ("oficial", "etld1"):
            contagens = resumos[f"a/{regime}"]["dedup"]
            assert contagens["n_antes"] == len(quadro)
            assert contagens["n_duplicatas_removidas"] == 1
            assert contagens["n_final"] == len(quadro) - 1
            mantidas = set(indices[("a", regime)].train) | set(
                indices[("a", regime)].val) | set(indices[("a", regime)].test)
            assert len(quadro) - 1 not in mantidas  # a repetida (a última) saiu
        assert "n_repetidas_treino_teste" in resumos["a/oficial"]["dedup"]
        por_chave = matrix.contagens_dedup(resumos)
        assert set(por_chave) == {("a", "oficial"), ("a", "etld1")}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
