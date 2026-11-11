"""T4.4 — testes da amostragem do portão de entrega (`scripts/run_smoke.py`).

O DoD da T4.4 ("executa em ≤ 60 s nesta máquina") é verificado rodando o
script de verdade, não aqui — este arquivo cobre a única lógica não trivial
dele: a amostragem estratificada e o teto de 5.000 linhas.

⛔ Nenhum teste aqui treina modelo (SDD §0).
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def run_smoke():
    """Carrega `scripts/run_smoke.py` como módulo (`scripts/` não é pacote)."""
    caminho = REPO_ROOT / "scripts" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("run_smoke", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def corpus_real_disponivel():
    if not (REPO_ROOT / "corpora" / "iscx2016" / "data.parquet").exists():
        pytest.skip("corpus real ausente (parquet não é versionado)")
    return "iscx2016"


class TestAmostragem:
    def test_respeita_o_teto_do_dod(self, run_smoke, corpus_real_disponivel):
        """≤ 5.000 linhas é o teto do DoD e o limite operacional de SDD §0."""
        amostra = run_smoke.amostrar_corpus(corpus_real_disponivel, 3000, seed=42)
        assert len(amostra) <= 3000

    def test_mantem_as_quatro_colunas_do_contrato(self, run_smoke, corpus_real_disponivel):
        """Regressão: a primeira versão usava `groupby().apply()`, e no pandas 3
        o grupo chega sem as colunas de agrupamento — `label` e `split_oficial`
        desapareciam da amostra."""
        amostra = run_smoke.amostrar_corpus(corpus_real_disponivel, 2000, seed=42)
        assert list(amostra.columns) == ["label", "url", "corpus", "split_oficial"]

    def test_estratifica_split_e_rotulo(self, run_smoke, corpus_real_disponivel):
        """Os dois regimes precisam das duas classes nos dois lados do split
        oficial — senão o smoke reprova por causa da amostragem."""
        amostra = run_smoke.amostrar_corpus(corpus_real_disponivel, 2000, seed=42)
        presentes = set(
            zip(
                amostra["split_oficial"].astype(str),
                amostra["label"].astype(int),
                strict=True,
            )
        )
        assert presentes == {("train", 0), ("train", 1), ("test", 0), ("test", 1)}

    def test_proporcao_do_split_oficial_e_preservada(self, run_smoke, corpus_real_disponivel):
        completo = pd.read_parquet(
            REPO_ROOT / "corpora" / corpus_real_disponivel / "data.parquet",
            columns=["split_oficial"],
        )
        esperado = (completo["split_oficial"] == "train").mean()
        amostra = run_smoke.amostrar_corpus(corpus_real_disponivel, 3000, seed=42)
        obtido = (amostra["split_oficial"] == "train").mean()
        assert obtido == pytest.approx(esperado, abs=0.03)

    def test_determinismo(self, run_smoke, corpus_real_disponivel):
        """RNF1 começa na amostragem: seed igual, amostra igual."""
        a = run_smoke.amostrar_corpus(corpus_real_disponivel, 2000, seed=42)
        b = run_smoke.amostrar_corpus(corpus_real_disponivel, 2000, seed=42)
        assert a.equals(b)

    def test_seed_diferente_muda_a_amostra(self, run_smoke, corpus_real_disponivel):
        a = run_smoke.amostrar_corpus(corpus_real_disponivel, 2000, seed=42)
        b = run_smoke.amostrar_corpus(corpus_real_disponivel, 2000, seed=7)
        assert not a.equals(b)

    def test_corpus_menor_que_o_teto_vem_inteiro(self, run_smoke, corpus_real_disponivel):
        amostra = run_smoke.amostrar_corpus(corpus_real_disponivel, 10**9, seed=42)
        completo = pd.read_parquet(
            REPO_ROOT / "corpora" / corpus_real_disponivel / "data.parquet"
        )
        assert len(amostra) == len(completo)


class TestSaidaDoSmoke:
    def test_grava_em_diretorio_separado_de_results(self, run_smoke):
        """R5: nenhum número de smoke pode virar linha de tabela do artigo.
        A separação física de `results/` é a garantia."""
        assert run_smoke.OUT_DIR.name == "smoke"
        assert run_smoke.OUT_DIR.parent.name == "results"

    def test_leia_me_avisa_que_nao_e_resultado(self, run_smoke):
        aviso = run_smoke.OUT_DIR / "LEIA-ME.md"
        if not aviso.exists():
            pytest.skip("smoke ainda não foi executado nesta cópia do repo")
        texto = aviso.read_text(encoding="utf-8")
        assert "NÃO é resultado" in texto
        assert "R5" in texto

    def test_matriz_gravada_passa_na_validacao_de_faixa(self, run_smoke):
        """Fecha o ciclo do B1: o que está em disco tem de sobreviver a
        `validate_metrics`."""
        import json

        import evaluate

        matriz = run_smoke.OUT_DIR / "matrix.json"
        if not matriz.exists():
            pytest.skip("smoke ainda não foi executado nesta cópia do repo")
        registros = json.loads(matriz.read_text(encoding="utf-8"))
        assert len(registros) == 192  # 2 regimes x 4 treino x 6 modelos x 4 teste
        for rec in registros:
            evaluate.validate_metrics(rec["metrics"])
            assert rec["smoke"] is True


class TestSaidaCompativelComOConsoleWindows:
    """O console do Windows usa cp1252 por padrão, não UTF-8.

    Um caractere fora do cp1252 numa string impressa levanta
    `UnicodeEncodeError` e **derruba o script** — não é cosmético. Aconteceu de
    verdade em 04/08/2026 com um "⚠" (U+26A0) numa mensagem de aviso: o smoke
    completou os 48 ajustes e morreu ao imprimir o resumo final.

    Este teste varre as chamadas a `print`/`log`/`falhar` dos dois scripts e
    exige que todo literal impresso seja codificável em cp1252. Acentuação
    portuguesa (á, ç, ã, §, ·) passa; setas, sinais matemáticos (≤, →) e
    emoji não.

    Comentários e docstrings estão fora do escopo de propósito: eles não são
    impressos, e o SDD/CLAUDE.md usam esses símbolos à vontade.
    """

    ARQUIVOS = (
        "scripts/run_smoke.py",
        "scripts/run_all.py",
        "src/matrix.py",
        # `python -m src.diagnostics` (T2.3) também imprime tabelas no console.
        "src/diagnostics.py",
    )

    @staticmethod
    def _codificavel(texto: str) -> bool:
        try:
            texto.encode("cp1252")
            return True
        except UnicodeEncodeError:
            return False

    @pytest.mark.parametrize("arquivo", ARQUIVOS)
    def test_literais_impressos_sao_cp1252(self, arquivo):
        import ast

        caminho = REPO_ROOT / arquivo
        if not caminho.exists():
            pytest.skip(f"{arquivo} ainda não existe")

        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        problemas = []
        for no in ast.walk(arvore):
            if not isinstance(no, ast.Call):
                continue
            if getattr(no.func, "id", None) not in ("print", "log", "falhar"):
                continue
            for parte in ast.walk(no):
                if (
                    isinstance(parte, ast.Constant)
                    and isinstance(parte.value, str)
                    and not self._codificavel(parte.value)
                ):
                    ruins = sorted(
                        {f"U+{ord(c):04X}" for c in parte.value
                         if not self._codificavel(c)}
                    )
                    problemas.append(f"{arquivo}:{parte.lineno} {ruins}")

        assert not problemas, (
            "literais impressos com caractere fora do cp1252 — derrubam o script "
            "no console do Windows:\n  " + "\n  ".join(problemas)
        )


# ---------------------------------------------------------------------------
# Frente A, T14 (RUN-05): o portão cobre predições, dedup e MLP agrupado
# ---------------------------------------------------------------------------


def _metricas(y, p):
    import evaluate

    metricas, avisos = evaluate.evaluate_predictions(y, p, threshold=0.5, ece_bins=10)
    return metricas, avisos


def _ajuste_sintetico(dir_rodada, tipo="matriz", regime="etld1", treino="a",
                      modelo="xgboost", testes=("a", "b"), validacao="agrupada_etld1"):
    """Grava as predições de um ajuste sintético e devolve os registros dele."""
    import numpy as np

    import evaluate
    import predicoes

    rng = np.random.default_rng(0)
    blocos, dados = [], {}
    for nome in testes:
        n = 40
        y = np.tile([0, 1], n // 2)
        p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, n), 0, 1)
        blocos.append((nome, np.arange(n) + 100, y, p))
        dados[nome] = (y, p)
    ref = predicoes.gravar_predicoes(dir_rodada, tipo, regime, treino, modelo, blocos)
    registros = []
    for nome, (y, p) in dados.items():
        metricas, avisos = _metricas(y, p)
        registros.append(evaluate.make_result_record(
            run_id="t", regime=regime, train_corpus=treino, test_corpus=nome,
            model=modelo, metrics=metricas, fit_seconds=0.1, warnings=avisos,
            extra={"predicoes": dict(ref),
                   "provenance": {"validacao_interna": validacao}},
        ))
    return registros


class TestConferenciaDePredicoes:
    def test_consistente_passa(self, run_smoke, tmp_path):
        regs = _ajuste_sintetico(tmp_path)
        assert run_smoke.problemas_predicoes(tmp_path, regs, "matriz", 1) == []

    def test_parquet_ausente_e_apontado(self, run_smoke, tmp_path):
        regs = _ajuste_sintetico(tmp_path)
        (tmp_path / regs[0]["predicoes"]["arquivo"]).unlink()
        problemas = run_smoke.problemas_predicoes(tmp_path, regs, "matriz", 1)
        assert any("0 parquet" in p for p in problemas)
        assert any("sem predições válidas" in p for p in problemas)

    def test_parquet_alterado_e_apontado(self, run_smoke, tmp_path):
        regs = _ajuste_sintetico(tmp_path)
        (tmp_path / regs[0]["predicoes"]["arquivo"]).write_bytes(b"x")
        problemas = run_smoke.problemas_predicoes(tmp_path, regs, "matriz", 1)
        assert any("sem predições válidas" in p for p in problemas)

    def test_numero_de_parquets_diferente_do_de_ajustes(self, run_smoke, tmp_path):
        regs = _ajuste_sintetico(tmp_path)
        problemas = run_smoke.problemas_predicoes(tmp_path, regs, "matriz", 2)
        assert any("esperava 2" in p for p in problemas)

    def test_auroc_do_registro_diferente_da_recalculada(self, run_smoke, tmp_path):
        regs = _ajuste_sintetico(tmp_path)
        regs[0]["metrics"] = {**regs[0]["metrics"], "auroc": regs[0]["metrics"]["auroc"] - 1e-6}
        problemas = run_smoke.problemas_predicoes(tmp_path, regs, "matriz", 1)
        assert any("AUROC" in p for p in problemas)

    def test_tamanho_do_bloco_diferente_de_n_test(self, run_smoke, tmp_path):
        regs = _ajuste_sintetico(tmp_path)
        regs[0]["n_test"] += 1
        problemas = run_smoke.problemas_predicoes(tmp_path, regs, "matriz", 1)
        assert any("n_test" in p for p in problemas)

    def test_registro_sem_referencia_e_apontado(self, run_smoke, tmp_path):
        regs = _ajuste_sintetico(tmp_path)
        regs[1].pop("predicoes")
        problemas = run_smoke.problemas_predicoes(tmp_path, regs, "matriz", 1)
        assert len(problemas) == 1 and "sem predições válidas" in problemas[0]


def _resumo(n_antes=100, conflito=4, duplicatas=6, partes=(70, 10, 10), regime="etld1"):
    dedup = {"n_antes": n_antes, "n_conflito_linhas": conflito,
             "n_duplicatas_removidas": duplicatas,
             "n_final": n_antes - conflito - duplicatas}
    if regime == "oficial":
        dedup["n_repetidas_treino_teste"] = 3
    return {"dedup": dedup, "partitions": {
        nome: {"n_rows": n} for nome, n in zip(("train", "val", "test"), partes, strict=True)
    }}


class TestConferenciaDeDedup:
    REGIMES = ["oficial", "etld1"]

    def _resumos(self, **kw):
        return {"a/oficial": _resumo(regime="oficial", **kw), "a/etld1": _resumo(**kw)}

    def test_contagens_coerentes_passam(self, run_smoke):
        assert run_smoke.problemas_dedup(self._resumos(), {"a": 100}, self.REGIMES) == []

    def test_resumo_sem_contagens(self, run_smoke):
        resumos = self._resumos()
        del resumos["a/etld1"]["dedup"]
        problemas = run_smoke.problemas_dedup(resumos, {"a": 100}, self.REGIMES)
        assert problemas == ["a/etld1: resumo de partições sem contagens de dedup"]

    def test_n_antes_diferente_do_quadro(self, run_smoke):
        problemas = run_smoke.problemas_dedup(self._resumos(), {"a": 101}, self.REGIMES)
        assert any("n_antes" in p for p in problemas)

    def test_conta_que_nao_fecha(self, run_smoke):
        resumos = self._resumos()
        resumos["a/etld1"]["dedup"]["n_final"] += 1
        problemas = run_smoke.problemas_dedup(resumos, {"a": 100}, self.REGIMES)
        assert any("duplicatas" in p for p in problemas)

    def test_particoes_que_nao_somam_n_final(self, run_smoke):
        problemas = run_smoke.problemas_dedup(
            self._resumos(partes=(70, 10, 11)), {"a": 100}, self.REGIMES
        )
        assert any("partições somam" in p for p in problemas)

    def test_regime_a_sem_contagem_de_repetidas(self, run_smoke):
        resumos = self._resumos()
        del resumos["a/oficial"]["dedup"]["n_repetidas_treino_teste"]
        problemas = run_smoke.problemas_dedup(resumos, {"a": 100}, self.REGIMES)
        assert any("n_repetidas_treino_teste" in p for p in problemas)


class TestConferenciaDoMlpAgrupado:
    def _regs(self, b="agrupada_etld1", a="linha"):
        def rec(regime, treino, valor):
            return {"model": "mlp", "regime": regime, "train_corpus": treino,
                    "provenance": {"validacao_interna": valor}}

        return [rec("etld1", "a", b), rec("etld1", "uniao_sem_a", b),
                rec("oficial", "a", a), rec("oficial", "uniao_sem_a", a),
                {"model": "xgboost", "regime": "etld1", "train_corpus": "a",
                 "provenance": {"validacao_interna": None}}]

    def test_agrupado_no_b_e_por_linha_no_a_passa(self, run_smoke):
        assert run_smoke.problemas_mlp_agrupado(self._regs(), 2) == []

    def test_regime_b_sem_validacao_agrupada(self, run_smoke):
        problemas = run_smoke.problemas_mlp_agrupado(self._regs(b="linha"), 2)
        assert any("etld1" in p and "agrupada_etld1" in p for p in problemas)

    def test_regime_a_nao_pode_ser_agrupado(self, run_smoke):
        problemas = run_smoke.problemas_mlp_agrupado(self._regs(a="agrupada_etld1"), 2)
        assert any("oficial" in p for p in problemas)

    def test_contagem_do_espiao_diferente_do_esperado(self, run_smoke):
        problemas = run_smoke.problemas_mlp_agrupado(self._regs(), 1)
        assert any("esperava 2" in p for p in problemas)

    def test_espiao_conta_so_o_mlp_que_chegou_ao_fit_com_grupos(self, run_smoke):
        import copy

        import numpy as np

        import models

        config = copy.deepcopy(models.load_experiment_config())
        config["modelos"]["mlp"]["params"].update(
            {"hidden_layer_sizes": [4], "max_iter": 5}
        )
        gpu = models.GpuInfo(available=False, reason="teste")
        rng = np.random.default_rng(1)
        y = np.tile([0, 1], 200)
        X = rng.normal(0, 1, (400, 3)) + y[:, None]
        grupos = np.repeat(np.arange(100), 4).astype(object)
        original = models.ajustar
        with run_smoke.contar_mlp_agrupado() as contador:
            models.ajustar(models.build_model("mlp", config, gpu=gpu, validacao_agrupada=True),
                           "mlp", X, y, grupos=grupos)
            models.ajustar(models.build_model("mlp", config, gpu=gpu, validacao_agrupada=False),
                           "mlp", X, y, grupos=grupos)
            # Achado 3 do lote F: o MLP agrupado sem grupos falhava em silencio
            # (caia na validacao por linha); agora levanta e nao e contado.
            with pytest.raises(ValueError, match="groups"):
                models.ajustar(
                    models.build_model("mlp", config, gpu=gpu, validacao_agrupada=True),
                    "mlp", X, y, grupos=None,
                )
        assert contador["n"] == 1
        assert models.ajustar is original  # o espião é desfeito ao sair


class TestConferirFrenteA:
    def test_problema_reprova_o_portao(self, run_smoke, tmp_path, monkeypatch):
        monkeypatch.setattr(run_smoke, "log", lambda *a, **k: None)
        regs = _ajuste_sintetico(tmp_path)
        (tmp_path / regs[0]["predicoes"]["arquivo"]).unlink()
        config = {"regimes": ["etld1"], "modelos": {"xgboost": {}}}
        resumos = {"a/etld1": _resumo(partes=(70, 10, 10))}
        with pytest.raises(SystemExit) as saida:
            run_smoke.conferir_frente_a(
                tmp_path, regs, [], resumos, {"a": [0] * 100}, config, 0
            )
        assert saida.value.code == 1


class TestAmostraComLinha:
    def test_linha_e_a_posicao_no_parquet_do_corpus(self, run_smoke, corpus_real_disponivel):
        completo = pd.read_parquet(
            REPO_ROOT / "corpora" / corpus_real_disponivel / "data.parquet",
            columns=["url"],
        )
        amostra = run_smoke.amostrar_corpus(
            corpus_real_disponivel, 2000, seed=42, com_linha=True
        )
        assert amostra["linha"].dtype.name == "int32"
        assert list(amostra["linha"]) == sorted(amostra["linha"])
        assert (completo["url"].to_numpy()[amostra["linha"].to_numpy()]
                == amostra["url"].to_numpy()).all()

    def test_sem_a_opcao_o_contrato_de_quatro_colunas_continua(
        self, run_smoke, corpus_real_disponivel
    ):
        amostra = run_smoke.amostrar_corpus(corpus_real_disponivel, 2000, seed=42)
        assert "linha" not in amostra.columns

    def test_corpus_menor_que_o_teto_tambem_ganha_linha(
        self, run_smoke, corpus_real_disponivel
    ):
        amostra = run_smoke.amostrar_corpus(
            corpus_real_disponivel, 10**9, seed=42, com_linha=True
        )
        assert list(amostra["linha"]) == list(range(len(amostra)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
