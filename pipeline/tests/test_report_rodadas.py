"""T22 a T26: o `report.py` e as rodadas novas da frente A.

Resultados sintéticos em `tmp_path`, escritos por `_resultados`: nenhum teste
lê o `results/` real e nenhum treina modelo. Os valores esperados saem de contas
feitas à mão, comentadas em cada caso (spec RUN-01, RUN-03, SENS-02, REP-01,
HPO-02, PROV-02, DEDUP-04, PSL-04, DEDUP-05, FILT-01, BOOT-01).

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import analises
import report

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

CORPORA = ("grambeddings", "kaggle03", "phishstorm", "iscx2016")
MODELOS = ("logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost", "mlp")
REGIMES = ("oficial", "etld1")


# ---------------------------------------------------------------------------
# Montagem de resultados sintéticos
# ---------------------------------------------------------------------------


def _metricas(auroc=0.8, precision=0.7, recall=0.6, fpr=0.2, **extra):
    base = {
        "auroc": auroc, "auc_pr": None if auroc is None else auroc - 0.05,
        "precision": precision,
        "recall": recall, "f1": 0.65, "fpr": fpr, "mcc": 0.4, "brier": 0.15,
        "ece": 0.03, "threshold": 0.5, "tp": 6, "fp": 3, "tn": 8, "fn": 4,
    }
    base.update(extra)
    return base


def _registro(regime, treino, teste, modelo, **metricas):
    return {
        "regime": regime,
        "train_corpus": treino,
        "test_corpus": teste,
        "model": modelo,
        "diagonal": treino == teste,
        "n_test": 10,
        "metrics": _metricas(**metricas),
    }


def _matriz(valor=lambda regime, tr, te, m: {}, corpora=CORPORA, modelos=MODELOS):
    return [
        _registro(regime, tr, te, m, **valor(regime, tr, te, m))
        for regime in REGIMES
        for tr in corpora
        for te in corpora
        for m in modelos
    ]


def _loco(valor=lambda regime, te, m: {}, corpora=CORPORA, modelos=MODELOS):
    return [
        _registro(regime, f"uniao_sem_{te}", te, m, **valor(regime, te, m))
        for regime in REGIMES
        for te in corpora
        for m in modelos
    ]


def _gravar_json(caminho: Path, conteudo) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo), encoding="utf-8")


def _gravar_rodada(base: Path, matriz=None, loco=None, commit_codigo=None):
    """Grava `matrix.json`, `loco.json` e `run_meta.json` de uma rodada."""
    _gravar_json(base / "matrix.json", _matriz() if matriz is None else matriz)
    _gravar_json(base / "loco.json", _loco() if loco is None else loco)
    meta = {"commit": "head"}
    if commit_codigo is not None:
        meta["commit_codigo"] = commit_codigo
    _gravar_json(base / "run_meta.json", meta)
    _gravar_json(base / "importances.json", {"features": [], "ajustes": {}})


def _resultados(tmp_path, matriz=None, loco=None, commit_codigo=None) -> Path:
    """`tmp_path/pipeline/results` com a semente 42 e os arquivos de apoio."""
    results = tmp_path / "pipeline" / "results"
    _gravar_rodada(results, matriz, loco, commit_codigo)
    _gravar_json(results / "diagnostics.json", {})
    _gravar_json(results / "hpo.json", {})
    _gravar_json(results / "loco_importances.json", {})
    _gravar_json(results / "importances.json", {"features": [], "ajustes": {}})
    return results


# ---------------------------------------------------------------------------
# T22: carga das rodadas novas e conferência de commit (RUN-03)
# ---------------------------------------------------------------------------


def test_rodadas_adicionais_ausentes_nao_sao_erro(tmp_path):
    """Sem as rodadas novas, `carregar` devolve vazio e as leituras antigas seguem."""
    dados = report.carregar(_resultados(tmp_path))

    assert dados["sensibilidade_por_seed"] == {}
    assert dados["repeticao"] is None
    assert dados["hpo_fora_diagonal"] is None
    assert dados["offline"] is None
    assert dados["manifest"] is None
    assert list(dados["matriz_por_seed"]) == [42]


def test_carregar_le_as_rodadas_e_os_arquivos_novos(tmp_path):
    results = _resultados(tmp_path)
    sens = results / "sensibilidade_sem_esquema"
    _gravar_rodada(sens)
    _gravar_rodada(sens / "seed_07")
    _gravar_rodada(results / "repeticao_seed_42")
    _gravar_json(results / "hpo_fora_diagonal.json", {"modelos": {}})
    _gravar_json(results / "offline.json", {"filtragem": {}})
    _gravar_json(tmp_path / "pipeline" / "corpora" / "manifest.json", {"iscx2016": {}})

    dados = report.carregar(results)

    assert sorted(dados["sensibilidade_por_seed"]) == [7, 42]
    assert len(dados["sensibilidade_por_seed"][42]["matriz"]) == 2 * 16 * 6
    assert len(dados["sensibilidade_por_seed"][42]["loco"]) == 2 * 4 * 6
    assert dados["repeticao"]["run_meta"] == {"commit": "head"}
    assert dados["hpo_fora_diagonal"] == {"modelos": {}}
    assert dados["offline"] == {"filtragem": {}}
    assert dados["manifest"] == {"iscx2016": {}}


def test_rodada_sem_loco_nem_run_meta_carrega_com_vazios(tmp_path):
    results = _resultados(tmp_path)
    _gravar_json(results / "repeticao_seed_42" / "matrix.json", _matriz())

    dados = report.carregar(results)

    assert dados["repeticao"]["loco"] == []
    assert dados["repeticao"]["run_meta"] == {}


def test_dois_commit_codigo_diferentes_levantam_report_error_listando_os_dois(tmp_path):
    """RUN-03, critério 4: a mensagem lista os commits de cada rodada."""
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_rodada(results / "repeticao_seed_42", commit_codigo="bbbb222")

    with pytest.raises(report.ReportError) as erro:
        report.carregar(results)

    mensagem = str(erro.value)
    assert "aaaa111" in mensagem and "bbbb222" in mensagem
    assert "principal, semente 42" in mensagem
    assert "repetição da semente 42" in mensagem


def test_divergencia_entre_sementes_e_sensibilidade_tambem_reprova(tmp_path):
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_rodada(results / "seed_07", commit_codigo="aaaa111")
    _gravar_rodada(results / "sensibilidade_sem_esquema" / "seed_13", commit_codigo="cccc333")

    with pytest.raises(report.ReportError, match="cccc333"):
        report.carregar(results)


def test_commit_codigo_igual_ou_ausente_nas_rodadas_antigas_passa(tmp_path):
    """Iguais passam; rodada sem o campo (anterior à consolidada) fica fora da conta."""
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_rodada(results / "seed_07", commit_codigo="aaaa111")
    _gravar_rodada(results / "repeticao_seed_42")  # sem commit_codigo

    dados = report.carregar(results)

    assert sorted(dados["matriz_por_seed"]) == [7, 42]
    report.conferir_commits(dados)  # não levanta de novo


def test_rodada_nova_sem_commit_codigo_reprova(tmp_path):
    """Achado 1 do lote F: run_meta de esquema novo (AD-068) sem `commit_codigo`
    (campo ausente ou nulo, git falhou) nao e compativel; so a rodada antiga, sem
    nenhum campo novo, fica fora da conta."""
    for meta in (
        {"commit": "head", "psl_sha256": "x"},
        {"commit": "head", "psl_sha256": "x", "commit_codigo": None},
    ):
        results = _resultados(tmp_path, commit_codigo="aaaa111")
        _gravar_rodada(results / "seed_07", commit_codigo="aaaa111")
        _gravar_json(results / "seed_07" / "run_meta.json", meta)

        with pytest.raises(report.ReportError, match="semente 7"):
            report.carregar(results)


# Achado 2 do lote F: hpo.json, hpo_fora_diagonal.json e offline.json.


@pytest.mark.parametrize("arquivo", ["hpo.json", "hpo_fora_diagonal.json"])
def test_commit_codigo_do_hpo_diferente_reprova_nomeando_o_arquivo(tmp_path, arquivo):
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_json(results / arquivo, {"psl_sha256": "x", "commit_codigo": "bbbb222"})

    with pytest.raises(report.ReportError) as erro:
        report.carregar(results)

    assert arquivo in str(erro.value) and "bbbb222" in str(erro.value)


@pytest.mark.parametrize("arquivo", ["hpo.json", "hpo_fora_diagonal.json"])
def test_hpo_de_esquema_novo_sem_commit_codigo_reprova(tmp_path, arquivo):
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_json(results / arquivo, {"psl_sha256": "x"})

    with pytest.raises(report.ReportError, match=arquivo):
        report.carregar(results)


def test_hpo_com_o_mesmo_commit_codigo_passa(tmp_path):
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_json(results / "hpo.json", {"psl_sha256": "x", "commit_codigo": "aaaa111"})
    _gravar_json(results / "hpo_fora_diagonal.json",
                 {"psl_sha256": "x", "commit_codigo": "aaaa111"})

    report.carregar(results)  # não levanta


def test_commit_codigo_do_offline_diferente_reprova(tmp_path):
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_json(results / "offline.json", {"fontes": {}, "commit_codigo": "bbbb222"})

    with pytest.raises(report.ReportError, match=r"offline\.json"):
        report.carregar(results)


def _matriz_com_predicoes(sha):
    registros = _matriz()
    registros[0]["predicoes"] = {"arquivo": "predicoes/matriz/a.parquet", "sha256": sha}
    return registros


def test_fontes_do_offline_com_sha256_diferente_reprovam(tmp_path):
    """O `offline.json` guarda o sha256 das predições que leu; se a matriz carregada
    registra outro, o offline é de outra rodada."""
    results = _resultados(tmp_path, matriz=_matriz_com_predicoes("a" * 64),
                          commit_codigo="aaaa111")
    _gravar_json(results / "offline.json", {
        "commit_codigo": "aaaa111",
        "fontes": {"predicoes/matriz/a.parquet": "b" * 64},
    })

    with pytest.raises(report.ReportError) as erro:
        report.carregar(results)

    assert "offline.json" in str(erro.value) and "predicoes/matriz/a.parquet" in str(erro.value)


def test_fontes_do_offline_de_rodada_com_prefixo_sao_conferidas_na_rodada_certa(tmp_path):
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_rodada(results / "seed_07", matriz=_matriz_com_predicoes("a" * 64),
                   commit_codigo="aaaa111")
    _gravar_json(results / "offline.json", {
        "commit_codigo": "aaaa111",
        "fontes": {"seed_07/predicoes/matriz/a.parquet": "c" * 64},
    })

    with pytest.raises(report.ReportError, match=r"seed_07/predicoes/matriz/a\.parquet"):
        report.carregar(results)


def test_fontes_do_offline_iguais_as_da_matriz_passam(tmp_path):
    results = _resultados(tmp_path, matriz=_matriz_com_predicoes("a" * 64),
                          commit_codigo="aaaa111")
    _gravar_json(results / "offline.json", {
        "commit_codigo": "aaaa111",
        "fontes": {"predicoes/matriz/a.parquet": "a" * 64},
    })

    report.carregar(results)  # não levanta


def test_fonte_do_offline_que_a_matriz_nao_registra_reprova(tmp_path):
    """Predição lida pelo offline e ausente do `matrix.json` carregado: rodada trocada."""
    results = _resultados(tmp_path, commit_codigo="aaaa111")
    _gravar_json(results / "offline.json", {
        "commit_codigo": "aaaa111",
        "fontes": {"predicoes/matriz/fantasma.parquet": "d" * 64},
    })

    with pytest.raises(report.ReportError, match="fantasma"):
        report.carregar(results)


# ---------------------------------------------------------------------------
# Utilitários de leitura do que o relatório gera
# ---------------------------------------------------------------------------


def _fmt(valor: float, casas: int = 4) -> str:
    """Formato do artigo, escrito à parte do `report._num`: vírgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _macros_de(linhas: list[str]) -> dict:
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}", "\n".join(linhas))
    }


def _linha_da_tabela(tex: str, inicio: str) -> list[str]:
    """Células da linha que começa em `inicio` (sem a quebra `\\\\` final)."""
    for linha in tex.splitlines():
        if linha.startswith(inicio):
            return [c.strip() for c in linha.removesuffix("\\\\").split(" & ")]
    raise AssertionError(f"linha {inicio!r} ausente")


@pytest.fixture(scope="module")
def dados_reais():
    if not (REPO_ROOT / "results" / "matrix.json").exists():
        pytest.skip("results/matrix.json ausente")
    return report.carregar()


# ---------------------------------------------------------------------------
# T23: tabela de precisão, revocação e TFP (RUN-01)
# ---------------------------------------------------------------------------

# Valores sintéticos com forma fechada, para a conta sair à mão. `i` e `j` são
# os índices do treino e do teste em CORPORA e `k` o do modelo em MODELOS.
# Média dos seis modelos (k = 0..5, soma 15): precisão ganha 0,01 por k (0,075 /
# 6 = 0,025 de média), revocação perde 0,02 por k (0,05) e a TFP ganha 0,005
# por k (0,0125). O regime A (`oficial`) soma 0,1 de base.


def _deslocamento(regime: str) -> float:
    return 0.1 if regime == "oficial" else 0.0


def _valor_matriz(regime, tr, te, m):
    i, j, k = CORPORA.index(tr), CORPORA.index(te), MODELOS.index(m)
    off = 0.01 * (i * j + j) + _deslocamento(regime)
    return {
        "precision": 0.30 + off + 0.01 * k,
        "recall": 0.80 - off - 0.02 * k,
        "fpr": 0.10 + off / 2 + 0.005 * k,
    }


def _valor_loco(regime, te, m):
    j, k = CORPORA.index(te), MODELOS.index(m)
    off = 0.01 * j + _deslocamento(regime)
    return {
        "precision": 0.40 + off + 0.01 * k,
        "recall": 0.70 - off - 0.02 * k,
        "fpr": 0.20 + off / 2 + 0.005 * k,
    }


def _media_matriz(regime, tr, te):
    """Média dos seis modelos de uma célula da matriz, à mão."""
    i, j = CORPORA.index(tr), CORPORA.index(te)
    off = 0.01 * (i * j + j) + _deslocamento(regime)
    return {
        "precision": 0.30 + off + 0.025,
        "recall": 0.80 - off - 0.05,
        "fpr": 0.10 + off / 2 + 0.0125,
    }


def _media_loco(regime, te):
    j = CORPORA.index(te)
    off = 0.01 * j + _deslocamento(regime)
    return {
        "precision": 0.40 + off + 0.025,
        "recall": 0.70 - off - 0.05,
        "fpr": 0.20 + off / 2 + 0.0125,
    }


@pytest.fixture
def dados_limiar(tmp_path):
    matriz = _matriz(_valor_matriz)
    loco = _loco(_valor_loco)
    return report.carregar(_resultados(tmp_path, matriz=matriz, loco=loco))


def test_a_tabela_de_limiar_bate_com_os_registros_sinteticos(dados_limiar):
    tex = report.tabela10_limiar(dados_limiar)

    assert r"\label{tab:limiar}" in tex
    # Duas células da matriz e uma do LOCO, regime A depois regime B, na ordem
    # precisão, revocação e TFP.
    for rotulo, tr, te in (
        (r"Grambeddings $\to$ Kaggle-03", "grambeddings", "kaggle03"),
        (r"ISCX-URL2016 $\to$ PhishStorm", "iscx2016", "phishstorm"),
        (r"PhishStorm $\to$ PhishStorm", "phishstorm", "phishstorm"),
    ):
        celulas = _linha_da_tabela(tex, rotulo)[1:]
        esperado = []
        for regime in ("oficial", "etld1"):
            m = _media_matriz(regime, tr, te)
            esperado += [_fmt(m["precision"]), _fmt(m["recall"]), _fmt(m["fpr"])]
        assert celulas == esperado, rotulo

    celulas = _linha_da_tabela(tex, r"União $\to$ PhishStorm")[1:]
    esperado = []
    for regime in ("oficial", "etld1"):
        m = _media_loco(regime, "phishstorm")
        esperado += [_fmt(m["precision"]), _fmt(m["recall"]), _fmt(m["fpr"])]
    assert celulas == esperado
    # 16 células da matriz e 4 do LOCO: 20 linhas de dado mais 3 de média.
    assert tex.count(r"$\to$") == 20 + 1  # +1 no cabeçalho "Treino $\to$ teste"


def test_as_macros_de_limiar_batem_com_a_conta_a_mao_e_com_a_tabela(dados_limiar):
    macros = _macros_de(report.macros_limiar(dados_limiar))
    assert len(macros) == 18  # 3 métricas x 2 regimes x (diagonal, fora, LOCO)
    assert all(nome.isalpha() for nome in macros)

    for regime, sufixo in (("oficial", "A"), ("etld1", "B")):
        for chave, prefixo in (("precision", "precisao"), ("recall", "revocacao"), ("fpr", "tfp")):
            diagonal = [_media_matriz(regime, c, c)[chave] for c in CORPORA]
            fora = [
                _media_matriz(regime, tr, te)[chave]
                for tr in CORPORA for te in CORPORA if tr != te
            ]
            loco = [_media_loco(regime, te)[chave] for te in CORPORA]
            assert macros[f"{prefixo}DiagonalRegime{sufixo}"] == _fmt(sum(diagonal) / 4)
            assert macros[f"{prefixo}ForaDiagonalRegime{sufixo}"] == _fmt(sum(fora) / 12)
            assert macros[f"{prefixo}LocoRegime{sufixo}"] == _fmt(sum(loco) / 4)

    # As linhas de média da tabela são as mesmas das macros.
    tex = report.tabela10_limiar(dados_limiar)
    linha = _linha_da_tabela(tex, r"\textbf{Média da diagonal}")[1:]
    assert linha[0] == macros["precisaoDiagonalRegimeA"]
    assert linha[5] == macros["tfpDiagonalRegimeB"]


# A2, HPO-01: as três macros do menor delta do HPO trocaram de nome de propósito
# (sem alias). O fixture congelado guarda os nomes de antes; a expectativa passa a
# aceitar o nome novo no lugar de cada um deles.
MACROS_RENOMEADAS = {
    "hpoPiorModelo": "hpoMenorDeltaModelo",
    "hpoPiorPerda": "hpoMenorDeltaValor",
    "hpoPiorCusto": "hpoMenorDeltaCusto",
}


def _macros_de_antes(texto: str) -> list[str]:
    return [MACROS_RENOMEADAS.get(nome, nome) for nome in texto.split()]


def test_toda_macro_de_antes_continua_gerada(dados_reais):
    """RUN-04: nenhuma macro de `numeros.tex` some nem muda de nome, salvo as
    renomeadas de propósito (A2, HPO-01)."""
    antes = _macros_de_antes((FIXTURES / "macros_antes.txt").read_text(encoding="utf-8"))
    agora = set(_macros_de(report.numeros_prosa(dados_reais).splitlines()))
    assert not set(antes) - agora, sorted(set(antes) - agora)
    # As macros de limiar entram além das antigas.
    assert "precisaoDiagonalRegimeB" in agora


def test_as_macros_de_limiar_do_numeros_tex_vem_do_matrix_json(dados_reais):
    registros = json.loads((REPO_ROOT / "results" / "matrix.json").read_text(encoding="utf-8"))
    por_celula: dict = {}
    for r in registros:
        if r["regime"] == "etld1" and r["train_corpus"] != r["test_corpus"]:
            por_celula.setdefault((r["train_corpus"], r["test_corpus"]), []).append(
                r["metrics"]["recall"]
            )
    medias = [sum(v) / len(v) for v in por_celula.values()]
    esperado = _fmt(sum(medias) / len(medias))

    macros = _macros_de(report.numeros_prosa(dados_reais).splitlines())

    assert macros["revocacaoForaDiagonalRegimeB"] == esperado


# ---------------------------------------------------------------------------
# T24: macros de dedup, descartes, PSL e prevalência
# ---------------------------------------------------------------------------


def _dedup(antes, conflito_linhas, conflito_urls, duplicatas, sem_url_norm, **extra):
    return {
        "n_antes": antes,
        "n_conflito_linhas": conflito_linhas,
        "n_conflito_urls": conflito_urls,
        "n_duplicatas_removidas": duplicatas,
        "n_sem_url_norm": sem_url_norm,
        "n_final": antes - conflito_linhas - duplicatas,
        **extra,
    }


def _stats(ruim, vazia, invalida):
    return {
        "dropped_bad_label": ruim,
        "dropped_empty_url": vazia,
        "dropped_unparseable": invalida,
        "raw_lines": 10,
        "reconstructed_lines": 0,
        "records_out": 10,
    }


def _particao(n, prevalencia, desvio):
    return {"n": n, "prevalencia": prevalencia, "desvio": desvio}


@pytest.fixture
def dados_frente_a(tmp_path):
    results = _resultados(tmp_path)
    _gravar_json(results / "splits.json", {
        "grambeddings/etld1": {"dedup": _dedup(1000, 6, 3, 120, 2)},
        "grambeddings/oficial": {"dedup": _dedup(1000, 6, 3, 100, 2, n_repetidas_treino_teste=45)},
        "iscx2016/etld1": {"dedup": _dedup(19965, 10, 5, 1065, 0)},
        "kaggle03/etld1": {"n_train": 1},  # splits.json anterior à dedup: sem campo
    })
    _gravar_json(tmp_path / "pipeline" / "corpora" / "manifest.json", {
        "iscx2016": {"load_stats_by_split": {
            "train": _stats(2, 1, 4), "test": _stats(1, 0, 3)}},
        "grambeddings": {"load_stats_by_split": {
            "train": _stats(0, 0, 0), "test": _stats(0, 0, 0)}},
    })
    _gravar_json(results / "diagnostics.json", {
        "psl_privada_contra_icann": {"grambeddings": {
            "n_urls": 800000, "n_urls_mudam": 1234,
            "n_grupos_icann_afetados": 7, "n_grupos_resultantes": 55}},
        "dominios_duas_classes": {"kaggle03": {
            "bruto": {"n_dominios": 100, "n_dominios_duas_classes": 40},
            "deduplicado_etld1": {"n_dominios": 90, "n_dominios_duas_classes": 35}}},
        "desvio_prevalencia": {"etld1": {"iscx2016": {
            "n_dataset": 1000, "prevalencia_dataset": 0.5,
            "train": _particao(700, 0.52, 0.02),
            "val": _particao(100, 0.47, -0.03),
            "test": _particao(0, None, None)}}},
    })
    return report.carregar(results)


def test_macros_de_dedup_batem_com_o_splits_json(dados_frente_a):
    macros = _macros_de(report.macros_dedup(dados_frente_a))

    # Milhar em ponto, como o resto do artigo.
    assert macros["dedupAntesGrambeddingsRegimeB"] == "1.000"
    assert macros["dedupDuplicatasGrambeddingsRegimeB"] == "120"
    assert macros["dedupConflitoUrlsGrambeddingsRegimeB"] == "3"
    assert macros["dedupConflitoLinhasGrambeddingsRegimeB"] == "6"
    assert macros["dedupSemUrlNormGrambeddingsRegimeB"] == "2"
    assert macros["dedupFinalGrambeddingsRegimeB"] == "874"  # 1000 - 6 - 120
    assert macros["dedupDuplicatasGrambeddingsRegimeA"] == "100"
    assert macros["dedupRepetidasTreinoTesteGrambeddingsRegimeA"] == "45"
    assert macros["dedupAntesIscxRegimeB"] == "19.965"
    assert macros["dedupFinalIscxRegimeB"] == "18.890"  # 19965 - 10 - 1065
    # Só o Regime A conta URLs repetidas entre treino e teste.
    assert "dedupRepetidasTreinoTesteGrambeddingsRegimeB" not in macros
    # Entrada sem o campo `dedup` não gera macro.
    assert not [n for n in macros if "KaggleTres" in n]
    assert all(n.isalpha() for n in macros)


def test_macros_de_descarte_somam_as_particoes_por_motivo(dados_frente_a):
    macros = _macros_de(report.macros_descartes(dados_frente_a))

    # ISCX: rótulo 2 + 1, URL vazia 1 + 0, URL inválida 4 + 3; total 3 + 1 + 7.
    assert macros["descarteRotuloInvalidoIscx"] == "3"
    assert macros["descarteUrlVaziaIscx"] == "1"
    assert macros["descarteUrlInvalidaIscx"] == "7"
    assert macros["descarteTotalIscx"] == "11"
    assert macros["descarteTotalGrambeddings"] == "0"
    assert not [n for n in macros if "Phishstorm" in n]


def test_macros_da_psl_privada_batem_com_o_diagnostico(dados_frente_a):
    macros = _macros_de(report.macros_psl(dados_frente_a))

    assert macros == {
        "pslUrlsGrambeddings": "800.000",
        "pslUrlsMudamGrambeddings": "1.234",
        "pslGruposIcannAfetadosGrambeddings": "7",
        "pslGruposResultantesGrambeddings": "55",
    }


def test_macros_de_prevalencia_e_dominios_de_duas_classes(dados_frente_a):
    macros = _macros_de(report.macros_prevalencia(dados_frente_a))

    assert macros["dominiosDuasClassesBrutoKaggleTres"] == "40"
    assert macros["dominiosDuasClassesDedupKaggleTres"] == "35"
    # Pontos percentuais: 0,52 menos 0,50 vira 2,00; -0,03 vira -3,00.
    assert macros["prevalenciaDatasetIscxRegimeB"] == "50,00"
    assert macros["prevalenciaDesvioTreinoIscxRegimeB"] == "2,00"
    assert macros["prevalenciaDesvioValidacaoIscxRegimeB"] == "-3,00"
    # Partição vazia: desvio indefinido, no formato de valor ausente do módulo.
    assert macros["prevalenciaDesvioTesteIscxRegimeB"] == "--"
    assert not [n for n in macros if n.endswith("RegimeA")]


def test_os_blocos_da_frente_a_so_trazem_o_que_tem_fonte(tmp_path, dados_frente_a):
    titulos = [t for t, _ in report.blocos_frente_a(dados_frente_a)]
    # Os quatro blocos de dedup, descartes, PSL e prevalência (as fontes existem)
    # mais os da F04 que saem só da matriz e do LOCO (T30); nenhum bloco das
    # rodadas ausentes (sensibilidade, repetição, HPO, offline).
    for chave in ("DEDUP-04", "PROV-04", "PSL-04", "DEDUP-05"):
        assert sum(chave in t for t in titulos) == 1
    for chave in ("SENS-02", "REP-01", "HPO-02", "FILT-01", "BOOT-01"):
        assert not [t for t in titulos if chave in t]

    # Sem as fontes novas só sobram os blocos da F04 calculados da matriz e do
    # LOCO, que o `_resultados` sempre grava (T30: eles não dependem de rodada extra).
    vazio = report.carregar(_resultados(tmp_path / "vazio"))
    titulos_vazio = [t for t, _ in report.blocos_frente_a(vazio)]
    assert titulos_vazio
    assert all(any(c in t for c in ("ANL-02", "ANL-04", "ANL-06")) for t in titulos_vazio)


def test_descartes_do_numeros_tex_vem_do_manifest_real(dados_reais):
    manifest = json.loads(
        (REPO_ROOT / "corpora" / "manifest.json").read_text(encoding="utf-8")
    )
    esperado = sum(
        p["dropped_unparseable"] for p in manifest["sid321axn"]["load_stats_by_split"].values()
    )
    macros = _macros_de(report.numeros_prosa(dados_reais).splitlines())

    assert macros["descarteUrlInvalidaKaggleUm"] == str(esperado)


# ---------------------------------------------------------------------------
# T25: macros de sensibilidade, repetição e HPO fora da diagonal
# ---------------------------------------------------------------------------


def _matriz_auroc(diagonal_por_regime: dict, fora_por_regime: dict) -> list[dict]:
    """Matriz em que toda célula da diagonal (ou de fora) tem a mesma AUROC."""
    return _matriz(
        lambda regime, tr, te, m: {
            "auroc": (diagonal_por_regime if tr == te else fora_por_regime)[regime]
        }
    )


def _resultados_com_sensibilidade(tmp_path, so_semente_42=False) -> Path:
    # Principal. Semente 42: diagonal .90 (B) e .95 (A), fora .70 (B) e .90 (A).
    # Semente 7: diagonal .80 e .95, fora .60 e .90.
    results = _resultados(
        tmp_path,
        matriz=_matriz_auroc({"etld1": 0.90, "oficial": 0.95}, {"etld1": 0.70, "oficial": 0.90}),
    )
    sens = results / "sensibilidade_sem_esquema"
    # Sem esquema, semente 42: diagonal .85 e .94, fora .60 e .85.
    _gravar_rodada(sens, matriz=_matriz_auroc(
        {"etld1": 0.85, "oficial": 0.94}, {"etld1": 0.60, "oficial": 0.85}))
    if not so_semente_42:
        _gravar_rodada(results / "seed_07", matriz=_matriz_auroc(
            {"etld1": 0.80, "oficial": 0.95}, {"etld1": 0.60, "oficial": 0.90}))
        # Sem esquema, semente 7: diagonal .70 e .94, fora .50 e .85.
        _gravar_rodada(sens / "seed_07", matriz=_matriz_auroc(
            {"etld1": 0.70, "oficial": 0.94}, {"etld1": 0.50, "oficial": 0.85}))
    return results


def test_macros_de_sensibilidade_batem_com_a_conta_a_mao(tmp_path):
    dados = report.carregar(_resultados_com_sensibilidade(tmp_path))

    macros = _macros_de(report.macros_sensibilidade(dados))

    # Regime B, diagonal: principal (.90 + .80)/2 = .85; sem esquema
    # (.85 + .70)/2 = .775; diferença (-.05 - .10)/2 = -.075.
    assert macros["semEsquemaAurocPrincipalDiagonalRegimeB"] == "0,8500"
    assert macros["semEsquemaAurocDiagonalRegimeB"] == "0,7750"
    assert macros["semEsquemaDifDiagonalRegimeB"] == "-0,0750"
    assert macros["semEsquemaDifDiagonalRegimeBSementeQuarentaEDois"] == "-0,0500"
    assert macros["semEsquemaDifDiagonalRegimeBSementeSete"] == "-0,1000"
    # Regime B, fora: principal (.70 + .60)/2 = .65; sem esquema (.60 + .50)/2 = .55.
    assert macros["semEsquemaAurocPrincipalForaDiagonalRegimeB"] == "0,6500"
    assert macros["semEsquemaAurocForaDiagonalRegimeB"] == "0,5500"
    assert macros["semEsquemaDifForaDiagonalRegimeB"] == "-0,1000"
    # Regime A: .95 para .94 (-.01) na diagonal e .90 para .85 (-.05) fora, nas duas.
    assert macros["semEsquemaDifDiagonalRegimeA"] == "-0,0100"
    assert macros["semEsquemaDifForaDiagonalRegimeA"] == "-0,0500"
    assert all(n.isalpha() for n in macros)


def test_tabela_de_sensibilidade_imprime_as_medias_e_a_faixa(tmp_path):
    dados = report.carregar(_resultados_com_sensibilidade(tmp_path))

    tex = report.tabela11_sensibilidade(dados)

    assert r"\label{tab:sensibilidade}" in tex
    celulas = _linha_da_tabela(tex, "Regime B, diagonal")
    assert celulas[1:3] == ["0,8500", "0,7750"]
    # Diferença média e a faixa entre as duas sementes (-0,05 e -0,10).
    assert celulas[3] == "-0,0750 [-0,1000;~-0,0500]"
    assert "faixa entre as sementes" in tex


def test_com_uma_semente_so_a_tabela_de_sensibilidade_nao_promete_faixa(tmp_path):
    dados = report.carregar(_resultados_com_sensibilidade(tmp_path, so_semente_42=True))

    tex = report.tabela11_sensibilidade(dados)

    assert _linha_da_tabela(tex, "Regime B, diagonal")[3] == "-0,0500"
    assert "faixa entre" not in tex  # a legenda não promete o que não imprime


def test_sem_a_rodada_de_sensibilidade_nao_ha_macro_nem_tabela_nem_erro(tmp_path):
    dados = report.carregar(_resultados(tmp_path))

    assert report.macros_sensibilidade(dados) == []
    assert report.tabela11_sensibilidade(dados) is None


def _com_metricas(registros, chave, **novos):
    """Cópia dos registros com `novos` somados às métricas de um registro."""
    saida = []
    for r in registros:
        copia = {**r, "metrics": dict(r["metrics"])}
        if (r["regime"], r["train_corpus"], r["test_corpus"], r["model"]) == chave:
            copia["metrics"] = {**copia["metrics"], **novos}
        saida.append(copia)
    return saida


def test_macros_de_repeticao_batem_com_a_diferenca_plantada(tmp_path):
    results = _resultados(tmp_path)
    # Uma célula da matriz muda em AUROC e F1; uma do LOCO muda só em MCC.
    matriz = _com_metricas(
        _matriz(), ("etld1", "iscx2016", "grambeddings", "xgboost"),
        auroc=0.8 + 0.0123, f1=0.65 - 0.004,
    )
    loco = _com_metricas(
        _loco(), ("oficial", "uniao_sem_phishstorm", "phishstorm", "mlp"), mcc=0.45,
    )
    _gravar_rodada(results / "repeticao_seed_42", matriz=matriz, loco=loco)
    dados = report.carregar(results)

    macros = _macros_de(report.macros_repeticao(dados))

    assert macros["repeticaoCelulas"] == "240"  # 2 x 16 x 6 da matriz + 2 x 4 x 6 do LOCO
    assert macros["repeticaoMaxAuroc"] == "0,01230"
    assert macros["repeticaoModeloAuroc"] == "XGBoost"
    assert macros["repeticaoRegimeAuroc"] == "Regime B"
    assert macros["repeticaoTreinoAuroc"] == "ISCX-URL2016"
    assert macros["repeticaoTesteAuroc"] == "Grambeddings"
    assert macros["repeticaoMaxFUm"] == "0,00400"
    assert macros["repeticaoMaxMcc"] == "0,05000"
    assert macros["repeticaoModeloMcc"] == "MLP"
    assert macros["repeticaoRegimeMcc"] == "Regime A"
    assert macros["repeticaoTreinoMcc"] == "União"
    assert macros["repeticaoTesteMcc"] == "PhishStorm"
    # Métrica que não mudou: diferença zero, na primeira célula em ordem.
    assert macros["repeticaoMaxBrier"] == "0,00000"
    assert all(n.isalpha() for n in macros)

    tex = report.tabela12_repeticao(dados)
    assert _linha_da_tabela(tex, "AUROC")[1:3] == ["0,01230", "XGBoost"]
    assert _linha_da_tabela(tex, "MCC")[4] == r"União $\to$ PhishStorm"


def test_sem_a_repeticao_nao_ha_macro_nem_tabela_nem_erro(tmp_path):
    dados = report.carregar(_resultados(tmp_path))

    assert report.macros_repeticao(dados) == []
    assert report.tabela12_repeticao(dados) is None


def test_repeticao_com_celulas_diferentes_levanta_report_error(tmp_path):
    results = _resultados(tmp_path)
    _gravar_rodada(results / "repeticao_seed_42", matriz=_matriz()[:-1])
    dados = report.carregar(results)

    with pytest.raises(report.ReportError, match="repetição da semente 42"):
        report.macros_repeticao(dados)


def _avaliacoes(seco: dict, ajustado: dict) -> list[dict]:
    return [
        {"configuracao": configuracao, "test_corpus": corpus,
         "metrics": _metricas(auroc=auroc)}
        for configuracao, por_corpus in (("seco", seco), ("ajustado", ajustado))
        for corpus, auroc in por_corpus.items()
    ]


def _resultados_com_hpo(tmp_path) -> Path:
    results = _resultados(tmp_path)
    seco = {"grambeddings": 0.90, "kaggle03": 0.70, "phishstorm": 0.60, "iscx2016": 0.50}
    _gravar_json(results / "hpo_fora_diagonal.json", {"modelos": {
        # Ganhos de +0,01, +0,02, -0,03 e 0,00.
        "logistic_regression": {"avaliacoes": _avaliacoes(seco, {
            "grambeddings": 0.91, "kaggle03": 0.72, "phishstorm": 0.57, "iscx2016": 0.50})},
        # Ganhos de +0,05, -0,01, 0,00 e indefinido (AUROC ajustada nula).
        "mlp": {"avaliacoes": _avaliacoes(seco, {
            "grambeddings": 0.95, "kaggle03": 0.69, "phishstorm": 0.60, "iscx2016": None})},
    }})
    return results


def test_macros_do_hpo_fora_da_diagonal_batem_com_as_entradas(tmp_path):
    dados = report.carregar(_resultados_com_hpo(tmp_path))

    macros = _macros_de(report.macros_hpo_fora(dados))

    assert macros["hpoForaDeltaLogisticaGrambeddings"] == "0,01000"
    assert macros["hpoForaDeltaLogisticaKaggleTres"] == "0,02000"
    assert macros["hpoForaDeltaLogisticaPhishstorm"] == "-0,03000"
    assert macros["hpoForaDeltaLogisticaIscx"] == "0,00000"
    assert macros["hpoForaDeltaMlpGrambeddings"] == "0,05000"
    assert macros["hpoForaDeltaMlpKaggleTres"] == "-0,01000"
    assert macros["hpoForaDeltaMlpIscx"] == "n/d"
    # Extremos entre as células definidas.
    assert macros["hpoForaDeltaMax"] == "0,05000"
    assert macros["hpoForaDeltaMin"] == "-0,03000"
    assert all(n.isalpha() for n in macros)

    tex = report.tabela13_hpo_fora(dados)
    assert _linha_da_tabela(tex, "Reg. logística")[1:] == [
        "0,0100", "0,0200", "-0,0300", "0,0000"]
    assert _linha_da_tabela(tex, "MLP")[1:] == ["0,0500", "-0,0100", "0,0000", "n/d"]


def test_sem_hpo_fora_da_diagonal_nao_ha_macro_nem_tabela_nem_erro(tmp_path):
    dados = report.carregar(_resultados(tmp_path))

    assert report.macros_hpo_fora(dados) == []
    assert report.tabela13_hpo_fora(dados) is None


def test_gerar_escreve_as_tabelas_das_rodadas_presentes_e_pula_as_ausentes(
    tmp_path, monkeypatch
):
    """Rodada ausente não é erro: a tabela dela simplesmente não sai."""
    # As tabelas antigas dependem de arquivos reais; só as novas importam aqui.
    monkeypatch.setattr(report, "TABELAS", {
        "tabela11-sensibilidade.tex": report.tabela11_sensibilidade,
        "tabela12-repeticao.tex": report.tabela12_repeticao,
        "tabela13-hpo-fora.tex": report.tabela13_hpo_fora,
    })
    results = _resultados_com_sensibilidade(tmp_path)
    destino = tmp_path / "tabelas"

    escritos = report.gerar(results, destino)

    assert [p.name for p in escritos] == ["tabela11-sensibilidade.tex"]
    assert not (destino / "tabela12-repeticao.tex").exists()


# ---------------------------------------------------------------------------
# T26: tabelas e macros da filtragem e do bootstrap pareado
# ---------------------------------------------------------------------------


def _filtrado(regime, treino, teste, modelo, fracao, auroc, com_metricas=True):
    return {
        "regime": regime,
        "train_corpus": treino,
        "test_corpus": teste,
        "model": modelo,
        "n_teste": 100,
        "n_removidas": round(fracao * 100),
        "fracao_removida": fracao,
        "n_restantes": 100 - round(fracao * 100),
        "metrics": _metricas(auroc=auroc) if com_metricas else None,
        "auroc_indefinida": auroc is None,
    }


def _pareado(treino, teste, modelo, n, delta, ic):
    """`ic` é `[baixo, alto]` ou `None` (IC indefinido)."""
    return {
        "regime_par": "B-A",
        "train_corpus": treino,
        "test_corpus": teste,
        "model": modelo,
        "n_intersecao": n,
        "n_por_classe": {"positivos": n // 2, "negativos": n - n // 2},
        "delta": {"auroc": delta, "auc_pr": delta, "f1": delta},
        "ic": None if ic is None else {"auroc": ic, "auc_pr": ic, "f1": ic},
        "ic_indefinido": ic is None,
    }


def _offline_sintetico() -> dict:
    return {
        "parametros": {"seed": 42, "bootstrap_n": 500, "bootstrap_nivel": 0.95},
        "filtragem": {"principal": {"42": [
            # Regime B: AUROC filtrada .80 e .60 (média .70); removidas 25%.
            _filtrado("etld1", "grambeddings", "kaggle03", "m1", 0.25, 0.80),
            _filtrado("etld1", "grambeddings", "kaggle03", "m2", 0.25, 0.60),
            # Célula em que a filtragem deixa uma classe só (AUROC indefinida) e
            # outra sem nenhuma linha (métricas nulas): sai n/d, removida 100%.
            _filtrado("etld1", "iscx2016", "grambeddings", "m1", 1.0, None, com_metricas=False),
            _filtrado("etld1", "iscx2016", "grambeddings", "m2", 1.0, None),
            # LOCO: removidas 50% e 30% (média 40%), AUROC .90 e .70 (média .80).
            _filtrado("etld1", "uniao_sem_phishstorm", "phishstorm", "m1", 0.5, 0.90),
            _filtrado("etld1", "uniao_sem_phishstorm", "phishstorm", "m2", 0.3, 0.70),
            # Regime A: uma célula, 10% removidas, AUROC .95.
            _filtrado("oficial", "grambeddings", "kaggle03", "m1", 0.10, 0.95),
        ]}},
        "bootstrap_pareado": [
            _pareado("grambeddings", "kaggle03", "m1", 1000, -0.10, [-0.12, -0.08]),
            _pareado("grambeddings", "kaggle03", "m2", 1000, -0.20, [-0.30, 0.05]),
            _pareado("iscx2016", "iscx2016", "m1", 20, 0.05, None),
            _pareado("uniao_sem_phishstorm", "phishstorm", "m1", 500, 0.30, [0.20, 0.40]),
            _pareado("uniao_sem_phishstorm", "phishstorm", "m2", 500, 0.10, [0.05, 0.15]),
        ],
    }


@pytest.fixture
def dados_offline(tmp_path):
    # AUROC original: 0,9 na diagonal e 0,7 fora, nos dois regimes; LOCO 0,75.
    results = _resultados(
        tmp_path,
        matriz=_matriz_auroc(
            {"etld1": 0.9, "oficial": 0.9}, {"etld1": 0.7, "oficial": 0.7}),
        loco=_loco(lambda regime, te, m: {"auroc": 0.75}),
    )
    _gravar_json(results / "offline.json", _offline_sintetico())
    return report.carregar(results)


def test_tabela_da_filtragem_imprime_n_d_para_auroc_indefinida(dados_offline):
    tex = report.tabela14_filtragem(dados_offline)

    assert r"\label{tab:filtragem}" in tex
    # Regime A (removidas, original, filtrada) e depois Regime B.
    assert _linha_da_tabela(tex, r"Grambeddings $\to$ Kaggle-03")[1:] == [
        "10,00", "0,7000", "0,9500", "25,00", "0,7000", "0,7000"]
    # AUROC indefinida (uma classe só e nenhuma linha) vira n/d; sem registro, tudo n/d
    # menos a AUROC original, que vem da matriz.
    assert _linha_da_tabela(tex, r"ISCX-URL2016 $\to$ Grambeddings")[1:] == [
        "n/d", "0,7000", "n/d", "100,00", "0,7000", "n/d"]
    assert _linha_da_tabela(tex, r"União $\to$ PhishStorm")[1:] == [
        "n/d", "0,7500", "n/d", "40,00", "0,7500", "0,8000"]
    # A diagonal não é filtrada e não tem linha.
    assert r"Grambeddings $\to$ Grambeddings" not in tex


def test_tabela_da_filtragem_mantem_sete_colunas_e_dezesseis_linhas(dados_offline):
    """TAB-05: a correção do Overfull é de largura; a forma é a de `5009f2c`
    (12 células fora da diagonal e 4 do LOCO, 7 colunas)."""
    tex = report.tabela14_filtragem(dados_offline)

    assert r"\begin{tabular}{lrrrrrr}" in tex
    dados_da_tabela = [
        ln for ln in tex.splitlines()
        if r"$\to$" in ln and not ln.startswith("Treino")
    ]
    assert len(dados_da_tabela) == 16
    assert len([ln for ln in dados_da_tabela if ln.startswith("União")]) == 4
    assert all(ln.count(" & ") == 6 for ln in dados_da_tabela)


def test_macros_da_filtragem_batem_com_a_conta_a_mao(dados_offline):
    macros = _macros_de(report.macros_filtragem(dados_offline))

    # Regime B, fora da diagonal: removidas (25 + 100)/2 = 62,5 pontos; AUROC só
    # da célula definida (média .8 e .6 = .7). LOCO: 40 pontos e .8.
    assert macros["filtragemRemovidaForaDiagonalRegimeB"] == "62,50"
    assert macros["filtragemAurocForaDiagonalRegimeB"] == "0,7000"
    assert macros["filtragemRemovidaLocoRegimeB"] == "40,00"
    assert macros["filtragemAurocLocoRegimeB"] == "0,8000"
    assert macros["filtragemCelulasIndefinidasRegimeB"] == "1"
    # Regime A: uma célula e nenhum LOCO no offline.json.
    assert macros["filtragemRemovidaForaDiagonalRegimeA"] == "10,00"
    assert macros["filtragemAurocForaDiagonalRegimeA"] == "0,9500"
    assert "filtragemRemovidaLocoRegimeA" not in macros
    assert macros["filtragemCelulasIndefinidasRegimeA"] == "0"
    assert all(n.isalpha() for n in macros)


def test_filtragem_toda_indefinida_vira_n_d_na_macro(tmp_path):
    results = _resultados(tmp_path)
    _gravar_json(results / "offline.json", {"filtragem": {"principal": {"42": [
        _filtrado("etld1", "grambeddings", "kaggle03", "m1", 1.0, None),
    ]}}})

    macros = _macros_de(report.macros_filtragem(report.carregar(results)))

    assert macros["filtragemAurocForaDiagonalRegimeB"] == "n/d"
    assert macros["filtragemRemovidaForaDiagonalRegimeB"] == "100,00"
    assert macros["filtragemCelulasIndefinidasRegimeB"] == "1"


def test_tabela_do_bootstrap_pareado_marca_ic_indefinido_como_n_d(dados_offline):
    tex = report.tabela15_bootstrap_pareado(dados_offline)

    assert r"\label{tab:bootstrap-pareado}" in tex
    # Interseção, Delta médio, mínimo a máximo e modelos cujo IC exclui zero.
    # Célula 1: deltas -0,10 e -0,20, média -0,15; só o IC do primeiro exclui zero.
    assert _linha_da_tabela(tex, r"Grambeddings $\to$ Kaggle-03")[1:] == [
        "1.000", "-0,1500", "-0,2000 a -0,1000", "1/2"]
    # Diagonal com IC indefinido (menos de 30 linhas de uma classe): n/d.
    assert _linha_da_tabela(tex, r"ISCX-URL2016 $\to$ ISCX-URL2016")[1:] == [
        "20", "0,0500", "0,0500", "n/d"]
    # LOCO: média (0,30 + 0,10)/2 = 0,20 e os dois ICs excluem zero.
    assert _linha_da_tabela(tex, r"União $\to$ PhishStorm")[1:] == [
        "500", "0,2000", "0,1000 a 0,3000", "2/2"]
    # Célula ausente do offline.json: tudo n/d.
    assert _linha_da_tabela(tex, r"Kaggle-03 $\to$ Grambeddings")[1:] == [
        "n/d", "n/d", "n/d", "n/d"]
    assert r"95\%" in tex
    legenda = tex.split(r"\caption{")[1].split(r"\label")[0]
    assert len(legenda.split()) <= 40


def test_macros_do_bootstrap_pareado_batem_com_a_conta_a_mao(dados_offline):
    macros = _macros_de(report.macros_bootstrap_pareado(dados_offline))

    assert macros["bootstrapAjustes"] == "5"
    assert macros["bootstrapInterseccaoMin"] == "20"
    assert macros["bootstrapInterseccaoMax"] == "1.000"
    # Quatro ICs definidos e um indefinido; exclui zero: o 1º da célula 1 e os dois do LOCO.
    assert macros["bootstrapIcDefinidos"] == "4"
    assert macros["bootstrapIcIndefinidos"] == "1"
    assert macros["bootstrapIcExcluiZero"] == "3"
    assert macros["bootstrapReamostras"] == "500"
    # Média das células: diagonal 0,05; fora -0,15; LOCO 0,20.
    assert macros["bootstrapDeltaAurocMedioDiagonal"] == "0,0500"
    assert macros["bootstrapDeltaAurocMedioForaDiagonal"] == "-0,1500"
    assert macros["bootstrapDeltaAurocMedioLoco"] == "0,2000"
    assert all(n.isalpha() for n in macros)


# ---------------------------------------------------------------------------
# A2 (FIL-01, FIL-02, FIL-04): filtragem nas sementes 7 e 13
# ---------------------------------------------------------------------------

SUFIXO_SEMENTE = {7: "SementeSete", 13: "SementeTreze"}
SUBDIR_SEMENTE = {7: "seed_07", 13: "seed_13"}

# Por semente: [(fracao removida, AUROC filtrada)] de cada um dos dois modelos, por
# regime e célula; e a AUROC original do LOCO por corpus de teste, por regime.
# Valores escolhidos à mão, diferentes entre as sementes e entre os regimes.
FILTRAGEM_SINTETICA = {
    7: {
        "etld1": {
            ("grambeddings", "kaggle03"): [(0.10, 0.80), (0.10, 0.60)],
            ("kaggle03", "phishstorm"): [(0.30, 0.40), (0.50, 0.60)],
            ("uniao_sem_iscx2016", "iscx2016"): [(0.50, 0.90), (0.30, 0.70)],
            ("uniao_sem_kaggle03", "kaggle03"): [(0.20, 0.60), (0.20, 0.50)],
        },
        "oficial": {("grambeddings", "kaggle03"): [(0.10, 0.96), (0.10, 0.94)]},
        "original_loco": {
            "etld1": {"iscx2016": 0.60, "kaggle03": 0.50, "grambeddings": 0.40, "phishstorm": 0.30},
            "oficial": {"iscx2016": 0.90, "kaggle03": 0.90, "grambeddings": 0.90, "phishstorm": 0.90},
        },
    },
    13: {
        "etld1": {
            ("grambeddings", "kaggle03"): [(0.40, 0.55), (0.20, 0.35)],
            ("kaggle03", "phishstorm"): [(0.60, 0.20), (0.60, 0.30)],
            ("uniao_sem_iscx2016", "iscx2016"): [(0.10, 0.85), (0.30, 0.65)],
            ("uniao_sem_kaggle03", "kaggle03"): [(0.40, 0.45), (0.40, 0.25)],
        },
        "oficial": {
            ("grambeddings", "kaggle03"): [(0.20, 0.70), (0.00, 0.50)],
            ("uniao_sem_phishstorm", "phishstorm"): [(0.10, 0.88), (0.10, 0.82)],
        },
        "original_loco": {
            "etld1": {"iscx2016": 0.70, "kaggle03": 0.55, "grambeddings": 0.35, "phishstorm": 0.20},
            "oficial": {"iscx2016": 0.80, "kaggle03": 0.80, "grambeddings": 0.80, "phishstorm": 0.60},
        },
    },
}


def _registros_filtrados(seed: int) -> list[dict]:
    registros = []
    for regime in ("etld1", "oficial"):
        for (treino, teste), pares in FILTRAGEM_SINTETICA[seed][regime].items():
            for modelo, (fracao, auroc) in zip(("m1", "m2"), pares, strict=True):
                registros.append(_filtrado(regime, treino, teste, modelo, fracao, auroc))
    return registros


def _resultados_de_tres_sementes(tmp_path, offline: dict, sementes_carregadas=(7, 13)) -> dict:
    """Rodadas 42, 7 e 13 em `tmp_path`, com o `offline.json` dado; devolve `dados`."""
    results = _resultados(
        tmp_path,
        matriz=_matriz_auroc({"etld1": 0.9, "oficial": 0.9}, {"etld1": 0.7, "oficial": 0.7}),
        loco=_loco(lambda regime, te, m: {"auroc": 0.75}),
    )
    for seed in sementes_carregadas:
        original = FILTRAGEM_SINTETICA[seed]["original_loco"]
        _gravar_rodada(
            results / SUBDIR_SEMENTE[seed],
            matriz=_matriz_auroc({"etld1": 0.9, "oficial": 0.9}, {"etld1": 0.7, "oficial": 0.7}),
            loco=_loco(lambda regime, te, m, original=original: {"auroc": original[regime][te]}),
        )
    _gravar_json(results / "offline.json", offline)
    return report.carregar(results)


def _offline_de_tres_sementes() -> dict:
    offline = _offline_sintetico()
    for seed in (7, 13):
        offline["filtragem"]["principal"][str(seed)] = _registros_filtrados(seed)
    return offline


def _esperado_da_filtragem(seed: int, regime: str) -> dict:
    """Média dos modelos por célula e média das células do grupo, pela regra da spec."""
    celulas = {
        chave: (sum(f for f, _a in pares) / len(pares), sum(a for _f, a in pares) / len(pares))
        for chave, pares in FILTRAGEM_SINTETICA[seed][regime].items()
    }

    def grupo(e_loco: bool) -> tuple[float, float] | None:
        doc = [v for (tr, _te), v in celulas.items() if tr.startswith("uniao_sem_") is e_loco]
        if not doc:
            return None
        return (sum(f for f, _a in doc) / len(doc), sum(a for _f, a in doc) / len(doc))

    original = FILTRAGEM_SINTETICA[seed]["original_loco"][regime]
    return {
        "fora": grupo(False),
        "loco": grupo(True),
        "original_loco": sum(original.values()) / len(original),
    }


@pytest.mark.parametrize("seed", [7, 13])
def test_macros_da_filtragem_das_sementes_extras_batem_com_a_regra_da_spec(tmp_path, seed):
    """FIL-01 e FIL-02: fração removida e AUROC filtrada fora da diagonal e no LOCO, e
    AUROC original do LOCO, nos dois regimes, calculadas aqui das entradas escolhidas."""
    dados = _resultados_de_tres_sementes(tmp_path, _offline_de_tres_sementes())
    macros = _macros_de(report.macros_filtragem(dados))
    s = SUFIXO_SEMENTE[seed]

    for regime, r in (("etld1", "B"), ("oficial", "A")):
        esperado = _esperado_da_filtragem(seed, regime)
        for chave, nome in (("fora", "ForaDiagonal"), ("loco", "Loco")):
            if esperado[chave] is None:
                assert f"filtragemRemovida{nome}Regime{r}{s}" not in macros
                assert f"filtragemAuroc{nome}Regime{r}{s}" not in macros
                continue
            fracao, auroc = esperado[chave]
            assert macros[f"filtragemRemovida{nome}Regime{r}{s}"] == _fmt_virgula(fracao * 100, 2)
            assert macros[f"filtragemAuroc{nome}Regime{r}{s}"] == _fmt_virgula(auroc, 4)
        assert macros[f"locoAurocOriginalRegime{r}{s}"] == _fmt_virgula(
            esperado["original_loco"], 4)


def test_macros_da_filtragem_das_sementes_extras_tem_os_valores_a_mao_da_semente_7(tmp_path):
    """Ancora de duas contas feitas à mão: B fora da diagonal e A sem LOCO."""
    dados = _resultados_de_tres_sementes(tmp_path, _offline_de_tres_sementes())
    macros = _macros_de(report.macros_filtragem(dados))

    # Células (.10, .70) e (.40, .50): removidas 25 pontos, AUROC .60. LOCO: células
    # (.40, .80) e (.20, .55), logo removidas 30 pontos e AUROC .675.
    assert macros["filtragemRemovidaForaDiagonalRegimeBSementeSete"] == "25,00"
    assert macros["filtragemAurocForaDiagonalRegimeBSementeSete"] == "0,6000"
    assert macros["filtragemRemovidaLocoRegimeBSementeSete"] == "30,00"
    assert macros["filtragemAurocLocoRegimeBSementeSete"] == "0,6750"
    # Original do LOCO no Regime B: (.60 + .50 + .40 + .30) / 4 = .45. No A: .90.
    assert macros["locoAurocOriginalRegimeBSementeSete"] == "0,4500"
    assert macros["locoAurocOriginalRegimeASementeSete"] == "0,9000"
    # Regime A não tem célula LOCO filtrada na semente 7.
    assert "filtragemRemovidaLocoRegimeASementeSete" not in macros


def _fmt_virgula(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def test_macros_da_semente_42_da_filtragem_mantem_nomes_e_valores(tmp_path):
    """IDT-03: carregar as sementes 7 e 13 não muda nenhuma macro da 42."""
    so_42 = _macros_de(report.macros_filtragem(_resultados_de_tres_sementes(
        tmp_path / "a", _offline_sintetico(), sementes_carregadas=())))
    com_tres = _macros_de(report.macros_filtragem(
        _resultados_de_tres_sementes(tmp_path / "b", _offline_de_tres_sementes())))

    # As contagens de FIL-03 são macros novas e dependem de quantas sementes estão
    # carregadas; só as macros que já existiam entram na comparação.
    so_42 = {n: v for n, v in so_42.items() if not n.startswith("filtragemSementesEleva")}
    assert so_42 and all(not n.endswith(("SementeSete", "SementeTreze")) for n in so_42)
    assert {n: com_tres[n] for n in so_42} == so_42
    # Valores de antes, os mesmos do teste da semente 42 sem sufixo.
    assert so_42["filtragemRemovidaForaDiagonalRegimeB"] == "62,50"
    assert so_42["filtragemAurocLocoRegimeB"] == "0,8000"


def test_rodada_da_semente_7_carregada_sem_o_bloco_no_offline_reprova(tmp_path):
    """FIL-04: a mensagem nomeia a semente e manda rodar `make offline`."""
    dados = _resultados_de_tres_sementes(
        tmp_path, _offline_sintetico(), sementes_carregadas=(7,))

    with pytest.raises(report.ReportError) as erro:
        report.macros_filtragem(dados)
    assert "7" in str(erro.value) and "make offline" in str(erro.value)


def test_so_a_semente_13_sem_bloco_reprova_citando_a_13(tmp_path):
    offline = _offline_sintetico()
    offline["filtragem"]["principal"]["7"] = _registros_filtrados(7)
    dados = _resultados_de_tres_sementes(tmp_path, offline)

    with pytest.raises(report.ReportError) as erro:
        report.macros_filtragem(dados)
    assert "13" in str(erro.value) and "make offline" in str(erro.value)


def test_sem_a_rodada_da_semente_7_carregada_nao_ha_macro_dela_nem_erro(tmp_path):
    """A rodada da 7 ausente com o bloco da 7 no offline: nada da 7 sai e nada falha."""
    offline = _offline_de_tres_sementes()
    dados = _resultados_de_tres_sementes(tmp_path, offline, sementes_carregadas=(13,))

    macros = _macros_de(report.macros_filtragem(dados))
    assert not [n for n in macros if n.endswith("SementeSete")]
    assert [n for n in macros if n.endswith("SementeTreze")]


# ---------------------------------------------------------------------------
# A2 (BOO-01, BOO-02): modelos com IC pareado que exclui zero na diagonal
# ---------------------------------------------------------------------------

# Por dataset da diagonal: o IC do Delta AUROC de cada um de três modelos (`None` =
# indefinido) e quantos excluem zero, contados à mão.
IC_DA_DIAGONAL = {
    "grambeddings": ([[0.01, 0.20], [-0.05, 0.10], [-0.30, -0.02]], 2),  # o do meio cruza zero
    "kaggle03": ([[0.10, 0.20], [0.05, 0.06], [-0.20, -0.10]], 3),
    "phishstorm": ([None, [-0.10, 0.10], [-0.01, 0.30]], 0),             # indefinido e dois que cruzam
    "iscx2016": ([[0.01, 0.20], [-0.20, -0.05], None], 2),
}


def _dados_com_bootstrap_da_diagonal(tmp_path) -> dict:
    registros = [
        _pareado(corpus, corpus, f"m{i}", 1000, 0.05, ic)
        for corpus, (ics, _k) in IC_DA_DIAGONAL.items()
        for i, ic in enumerate(ics)
    ]
    # Células de fora da diagonal não entram na contagem, mesmo com IC que exclui zero.
    registros.append(_pareado("grambeddings", "kaggle03", "m0", 1000, 0.5, [0.4, 0.6]))
    results = _resultados(tmp_path)
    _gravar_json(results / "offline.json", {
        "parametros": {"bootstrap_n": 500, "bootstrap_nivel": 0.95},
        "bootstrap_pareado": registros,
    })
    return report.carregar(results)


def test_macros_do_ic_pareado_que_exclui_zero_na_diagonal_contam_por_dataset(tmp_path):
    """BOO-01: a contagem por dataset sai dos ICs escolhidos, sem contar indefinido."""
    dados = _dados_com_bootstrap_da_diagonal(tmp_path)
    macros = _macros_de(report.macros_bootstrap_pareado(dados))

    sufixos = {"grambeddings": "Grambeddings", "kaggle03": "KaggleTres",
               "phishstorm": "Phishstorm", "iscx2016": "Iscx"}
    for corpus, (_ics, esperado) in IC_DA_DIAGONAL.items():
        assert macros[f"bootstrapModelosIcExcluiZeroDiagonal{sufixos[corpus]}"] == str(esperado)


def test_a_soma_das_macros_da_diagonal_e_a_dos_numeradores_da_tabela(tmp_path):
    """BOO-02: soma das quatro macros = soma dos "k" do "k/n" das linhas da diagonal."""
    dados = _dados_com_bootstrap_da_diagonal(tmp_path)
    macros = _macros_de(report.macros_bootstrap_pareado(dados))
    tex = report.tabela15_bootstrap_pareado(dados)

    soma_macros = sum(
        int(v) for n, v in macros.items() if n.startswith("bootstrapModelosIcExcluiZeroDiagonal"))
    numeradores = 0
    for corpus in IC_DA_DIAGONAL:
        rotulo = report._rotulo(corpus, report.ROTULO_CORPUS)
        k, _n = _linha_da_tabela(tex, rf"{rotulo} $\to$ {rotulo}")[-1].split("/")
        numeradores += int(k)
    assert soma_macros == numeradores == 7


# ---------------------------------------------------------------------------
# A2 (FIL-03): sementes em que a filtragem eleva a AUROC no ISCX-URL2016
# ---------------------------------------------------------------------------


def _dados_com_filtradas_do_iscx(tmp_path, filtradas: dict) -> tuple[dict, list[str]]:
    """Três sementes com AUROC original 0,70 fora da diagonal e 0,75 no LOCO.

    `filtradas[(regime, treino)]` é `{semente: AUROC filtrada ou None}` da célula
    com teste no ISCX-URL2016 (um modelo só: a média da célula é o próprio valor).
    Devolve os dados e as linhas de `macros_filtragem`.
    """
    principal: dict = {"42": [], "7": [], "13": []}
    for (regime, treino), por_semente in filtradas.items():
        for semente, auroc in por_semente.items():
            principal[str(semente)].append(
                _filtrado(regime, treino, "iscx2016", "m1", 0.2, auroc))
    results = _resultados(
        tmp_path,
        matriz=_matriz_auroc({"etld1": 0.9, "oficial": 0.9}, {"etld1": 0.7, "oficial": 0.7}),
        loco=_loco(lambda regime, te, m: {"auroc": 0.75}),
    )
    for seed in (7, 13):
        _gravar_rodada(
            results / SUBDIR_SEMENTE[seed],
            matriz=_matriz_auroc({"etld1": 0.9, "oficial": 0.9}, {"etld1": 0.7, "oficial": 0.7}),
            loco=_loco(lambda regime, te, m: {"auroc": 0.75}),
        )
    _gravar_json(results / "offline.json", {"filtragem": {"principal": principal}})
    dados = report.carregar(results)
    return dados, report.macros_filtragem(dados)


def test_sementes_em_que_a_filtragem_eleva_a_auroc_no_iscx_contam_3_1_e_0(tmp_path):
    """FIL-03: filtrada acima da original em 3, 1 e 0 sementes dão 3, 1 e 0.

    Original fora da diagonal 0,70 e no LOCO 0,75.
    """
    _dados, linhas = _dados_com_filtradas_do_iscx(tmp_path, {
        ("etld1", "grambeddings"): {42: 0.75, 7: 0.80, 13: 0.71},       # 3 acima de 0,70
        ("etld1", "kaggle03"): {42: 0.65, 7: 0.72, 13: 0.60},           # só a 7
        ("etld1", "phishstorm"): {42: 0.69, 7: 0.60, 13: 0.50},         # nenhuma acima: 0
        ("etld1", "uniao_sem_iscx2016"): {42: 0.76, 7: 0.80, 13: 0.70},  # 0,76 e 0,80 > 0,75
        ("oficial", "grambeddings"): {42: 0.90, 7: 0.60, 13: 0.50},     # só a 42
        ("oficial", "kaggle03"): {42: 0.60, 7: 0.60, 13: 0.60},         # 0
        ("oficial", "phishstorm"): {42: 0.71, 7: 0.72, 13: 0.73},       # 3
        ("oficial", "uniao_sem_iscx2016"): {42: 0.74, 7: 0.75, 13: 0.76},  # só a 13
    })
    macros = _macros_de(linhas)

    assert macros["filtragemSementesElevaGrambeddingsIscxRegimeB"] == "3"
    assert macros["filtragemSementesElevaKaggleTresIscxRegimeB"] == "1"
    assert macros["filtragemSementesElevaPhishstormIscxRegimeB"] == "0"
    assert macros["filtragemSementesElevaUniaoIscxRegimeB"] == "2"
    assert macros["filtragemSementesElevaGrambeddingsIscxRegimeA"] == "1"
    assert macros["filtragemSementesElevaKaggleTresIscxRegimeA"] == "0"
    assert macros["filtragemSementesElevaPhishstormIscxRegimeA"] == "3"
    assert macros["filtragemSementesElevaUniaoIscxRegimeA"] == "1"


def test_semente_com_auroc_filtrada_indefinida_nao_conta_e_vai_no_comentario(tmp_path):
    """FIL-03 e Edge Cases: indefinida não é "acima"; o comentário da macro registra."""
    _dados, linhas = _dados_com_filtradas_do_iscx(tmp_path, {
        ("etld1", "grambeddings"): {42: None, 7: 0.80, 13: 0.71},
        ("etld1", "kaggle03"): {42: 0.80, 7: 0.80, 13: 0.80},
    })
    macros = _macros_de(linhas)
    por_nome = {
        re.match(r"\\newcommand\{\\([a-zA-Z]+)\}", ln).group(1): ln
        for ln in linhas if ln.startswith(r"\newcommand")
    }

    # A semente 42 é indefinida; 7 e 13 estão acima de 0,70: contam 2, e não 3.
    assert macros["filtragemSementesElevaGrambeddingsIscxRegimeB"] == "2"
    assert "1 sementes com AUROC indefinida" in por_nome[
        "filtragemSementesElevaGrambeddingsIscxRegimeB"]
    # Sem indefinida, o comentário não a cita.
    assert macros["filtragemSementesElevaKaggleTresIscxRegimeB"] == "3"
    assert "indefinida" not in por_nome["filtragemSementesElevaKaggleTresIscxRegimeB"]


def test_sem_offline_json_nao_ha_tabela_nem_macro_nem_erro(tmp_path):
    dados = report.carregar(_resultados(tmp_path))

    assert report.tabela14_filtragem(dados) is None
    assert report.tabela15_bootstrap_pareado(dados) is None
    assert report.macros_filtragem(dados) == []
    assert report.macros_bootstrap_pareado(dados) == []
    # T30: os blocos da F04 saem da matriz e do LOCO e não de `offline.json`; só
    # o de ANL-06 usa o offline, e sem ele traz o erro e a razão, sem o Delta F1.
    titulos = [t for t, _ in report.blocos_frente_a(dados)]
    assert not [t for t in titulos if "FILT-01" in t or "BOOT-01" in t]
    assert not [n for n in _macros_de(report.macros_efeito(dados)) if "DeltaFUm" in n and "Media" not in n]


def test_gerar_sobre_os_resultados_atuais_continua_passando(dados_reais, tmp_path):
    """Com as rodadas novas presentes, o relatório gera o que gerava e as tabelas delas.

    Equivale ao `make tabelas` sobre `results/`, só que num diretório temporário
    (o `latex/tabelas/` não é tocado).

    Antes da rodada consolidada (AD-055, que substitui o AD-010) `results/` não
    trazia as rodadas adicionais e este teste exigia a AUSÊNCIA das tabelas 11 a
    15. Com a rodada consolidada commitada elas existem e são geradas. O caso
    das rodadas ausentes continua coberto, com dados sintéticos, pelos testes
    `test_rodadas_adicionais_ausentes_nao_sao_erro`,
    `test_sem_a_rodada_de_sensibilidade_nao_ha_macro_nem_tabela_nem_erro` e
    `test_gerar_escreve_as_tabelas_das_rodadas_presentes_e_pula_as_ausentes`.
    """
    escritos = report.gerar(REPO_ROOT / "results", tmp_path)
    nomes = {p.name for p in escritos}

    assert "tabela10-limiar.tex" in nomes
    novas_condicionais = {
        "tabela11-sensibilidade.tex", "tabela12-repeticao.tex", "tabela13-hpo-fora.tex",
        "tabela14-filtragem.tex", "tabela15-bootstrap-pareado.tex",
    }
    assert novas_condicionais <= nomes
    assert "numeros.tex" in nomes


# ---------------------------------------------------------------------------
# T30: tabelas e macros das análises da F04 (ANL-01, ANL-02, ANL-04, ANL-06)
# ---------------------------------------------------------------------------


def _resultados_tres_sementes(tmp_path, valor42, valor7, valor13, loco=None):
    """Semente 42 em `results/` e as sementes 7 e 13 em `seed_NN/`, só com a AUROC."""
    results = _resultados(tmp_path, matriz=_matriz(valor42), loco=loco)
    _gravar_rodada(results / "seed_07", matriz=_matriz(valor7), loco=loco)
    _gravar_rodada(results / "seed_13", matriz=_matriz(valor13), loco=loco)
    return results


def _so_modelo(regime, tr, te, m):
    # AUROC só do modelo: 0,5 + 0,01 por índice do modelo.
    return {"auroc": 0.5 + 0.01 * MODELOS.index(m)}


def _por_treino(regime, tr, te, m):
    return {"auroc": 0.9 - 0.1 * CORPORA.index(tr) + 0.001 * MODELOS.index(m)}


def _por_teste_e_modelo(regime, tr, te, m):
    base = 0.6 + 0.05 * CORPORA.index(te) + 0.01 * MODELOS.index(m)
    return {"auroc": base + (0.02 if regime == "oficial" else 0.0)}


def test_macros_da_decomposicao_batem_com_as_funcoes_de_analises(tmp_path):
    """ANL-01: macros por regime, por semente e na média das três sementes."""
    results = _resultados_tres_sementes(tmp_path, _so_modelo, _por_treino, _por_teste_e_modelo)
    dados = report.carregar(results)

    macros = _macros_de(report.macros_decomposicao(dados))

    # Semente 42, à mão: a AUROC só depende do modelo, então ele explica 100%.
    for r in ("A", "B"):
        assert macros[f"decompModeloRegime{r}SementeQuarentaEDois"] == "100,00"
        assert macros[f"decompTreinoRegime{r}SementeQuarentaEDois"] == "0,00"
        assert macros[f"decompTesteRegime{r}SementeQuarentaEDois"] == "0,00"
        assert macros[f"decompRestoRegime{r}SementeQuarentaEDois"] == "0,00"
        assert macros[f"decompValoresRegime{r}"] == "72"
        assert macros[f"decompSementesRegime{r}"] == "3"
    # As demais sementes e a média vêm de `analises` sobre os mesmos registros.
    sufixos = {42: "QuarentaEDois", 7: "Sete", 13: "Treze"}
    for regime, r in (("oficial", "A"), ("etld1", "B")):
        fracoes = {
            s: analises.decompor_variancia(dados["matriz_por_seed"][s], regime) for s in sufixos
        }
        for campo, nome in (("treino", "Treino"), ("teste", "Teste"),
                            ("modelo", "Modelo"), ("resto", "Resto")):
            for s, sufixo in sufixos.items():
                esperado = _fmt(100 * fracoes[s][campo], 2)
                if esperado == "-0,00":
                    esperado = "0,00"
                assert macros[f"decomp{nome}Regime{r}Semente{sufixo}"] == esperado
            media = sum(fracoes[s][campo] for s in sufixos) / 3
            esperado = _fmt(100 * media, 2)
            assert macros[f"decomp{nome}Regime{r}Media"] == esperado
    # A semente 7 depende do treino e a 13 do teste e do modelo: os números diferem.
    assert macros["decompTreinoRegimeBSementeSete"] != macros["decompTreinoRegimeBSementeTreze"]
    assert all(n.isalpha() for n in macros)


def test_tabela_da_decomposicao_tem_uma_linha_por_semente_e_a_media(tmp_path):
    results = _resultados_tres_sementes(tmp_path, _so_modelo, _por_treino, _por_teste_e_modelo)
    dados = report.carregar(results)

    tex = report.tabela16_decomposicao(dados)

    assert r"\label{tab:variacao}" in tex
    assert _linha_da_tabela(tex, "Semente 42")[1:] == ["0,00", "0,00", "100,00", "0,00", "72"]
    assert tex.count("Semente 7 &") == 2 and tex.count("Média das sementes &") == 2
    assert "por semente e na média" in tex


def test_decomposicao_com_so_a_semente_42_nao_promete_media(tmp_path):
    dados = report.carregar(_resultados(tmp_path, matriz=_matriz(_so_modelo)))

    tex = report.tabela16_decomposicao(dados)

    assert "Média das sementes" not in tex and "por semente e na média" not in tex
    macros = _macros_de(report.macros_decomposicao(dados))
    # Uma semente só: a média é a própria semente e o número de sementes diz isso.
    assert macros["decompSementesRegimeB"] == "1"
    assert macros["decompModeloRegimeBMedia"] == "100,00"


def test_decomposicao_sem_variancia_nao_gera_macro_nem_tabela(tmp_path):
    dados = report.carregar(_resultados(tmp_path))  # AUROC igual em todo ajuste

    assert report.macros_decomposicao(dados) == []
    assert report.tabela16_decomposicao(dados) is None


def _auroc_por_par(regime, tr, te, m):
    # AUROC só do treino (0,5 + 0,05 por índice mais um) fora da diagonal; 0,95 na diagonal.
    if tr == te:
        return {"auroc": 0.95}
    return {"auroc": 0.5 + 0.05 * (CORPORA.index(tr) + 1)}


def _dados_uniao(tmp_path, com_semente_7=True):
    """LOCO 0,62 na semente 42 e 0,72 na 7.

    Fontes por teste (AUROC 0,55, 0,60, 0,65 e 0,70 para treinar em Grambeddings,
    Kaggle-03, PhishStorm e ISCX). Para o Grambeddings: fontes 0,60, 0,65 e 0,70.
    """
    loco42 = _loco(lambda regime, te, m: {"auroc": 0.62})
    results = _resultados(tmp_path, matriz=_matriz(_auroc_por_par), loco=loco42)
    if com_semente_7:
        _gravar_rodada(
            results / "seed_07", matriz=_matriz(_auroc_por_par),
            loco=_loco(lambda regime, te, m: {"auroc": 0.72}),
        )
    return report.carregar(results)


def test_macros_da_uniao_batem_com_a_conta_a_mao(tmp_path):
    """ANL-02: melhor, pior, média e média sem ISCX por dataset de teste e regime."""
    macros = _macros_de(report.macros_uniao(_dados_uniao(tmp_path)))

    # Teste Grambeddings: fontes 0,60 (Kaggle-03), 0,65 (PhishStorm), 0,70 (ISCX).
    assert macros["uniaoLocoGrambeddingsRegimeB"] == "0,6200"
    assert macros["uniaoMelhorGrambeddingsRegimeB"] == "0,7000"
    assert macros["uniaoPiorGrambeddingsRegimeB"] == "0,6000"
    assert macros["uniaoMediaGrambeddingsRegimeB"] == "0,6500"
    assert macros["uniaoMediaSemIscxGrambeddingsRegimeB"] == "0,6250"
    assert macros["uniaoMelhorFonteGrambeddingsRegimeB"] == "ISCX-URL2016"
    assert macros["uniaoPiorFonteGrambeddingsRegimeB"] == "Kaggle-03"
    # Teste ISCX: fontes 0,55, 0,60 e 0,65; sem ISCX é igual à média.
    assert macros["uniaoMediaIscxRegimeB"] == "0,6000"
    assert macros["uniaoMediaSemIscxIscxRegimeB"] == "0,6000"
    # Em quantos dos quatro testes o LOCO (0,62) fica abaixo de cada referência:
    # melhor 4; pior 0; média 2 (Grambeddings 0,65 e Kaggle-03 0,6333); sem ISCX 1.
    assert macros["uniaoDatasetsRegimeB"] == "4"
    assert macros["uniaoAbaixoMelhorRegimeB"] == "4"
    assert macros["uniaoAbaixoPiorRegimeB"] == "0"
    assert macros["uniaoAbaixoMediaRegimeB"] == "2"
    assert macros["uniaoAbaixoMediaSemIscxRegimeB"] == "1"
    # Média das sementes: LOCO (0,62 + 0,72) / 2 = 0,67; as fontes não mudam.
    assert macros["uniaoLocoGrambeddingsRegimeBMediaSementes"] == "0,6700"
    assert macros["uniaoMelhorGrambeddingsRegimeBMediaSementes"] == "0,7000"
    assert all(n.isalpha() for n in macros)


def test_macros_da_uniao_na_media_das_sementes_vem_de_analises(tmp_path):
    dados = _dados_uniao(tmp_path)
    macros = _macros_de(report.macros_uniao(dados))

    por_semente = {
        s: analises.uniao_contra_fontes(
            dados["matriz_por_seed"][s], dados["loco_por_seed"][s], "etld1")
        for s in (42, 7)
    }
    for teste, sufixo in (("grambeddings", "Grambeddings"), ("iscx2016", "Iscx")):
        loco = (por_semente[42][teste]["loco"] + por_semente[7][teste]["loco"]) / 2
        media = (por_semente[42][teste]["media"] + por_semente[7][teste]["media"]) / 2
        assert macros[f"uniaoLoco{sufixo}RegimeBMediaSementes"] == _fmt(loco)
        assert macros[f"uniaoMedia{sufixo}RegimeBMediaSementes"] == _fmt(media)
    # LOCO médio 0,67 contra a média das fontes (0,65, 0,6333, 0,6167 e 0,60): nenhum
    # abaixo. Contra a melhor (0,70, 0,70, 0,70 e 0,65): abaixo em três, não no ISCX.
    assert macros["uniaoAbaixoMediaRegimeBMediaSementes"] == "0"
    assert macros["uniaoAbaixoMelhorRegimeBMediaSementes"] == "3"


def test_tabela_da_uniao_imprime_os_cinco_valores_por_dataset(tmp_path):
    tex = report.tabela17_uniao(_dados_uniao(tmp_path, com_semente_7=False))

    assert r"\label{tab:uniao-fontes}" in tex
    assert _linha_da_tabela(tex, "Grambeddings")[1:] == [
        "0,6200", "0,7000", "0,6000", "0,6500", "0,6250"]
    assert _linha_da_tabela(tex, "ISCX-URL2016")[1:] == [
        "0,6200", "0,6500", "0,5500", "0,6000", "0,6000"]


def _splits_para_composicao():
    def resumo(n):
        return {"n_train": n, "n_val": 1, "n_test": 1}

    return {
        f"{c}/etld1": resumo(n)
        for c, n in zip(CORPORA, (100, 300, 600, 1000))
    }


def test_composicao_da_uniao_em_macros_e_tabela(tmp_path):
    results = _resultados(tmp_path)
    _gravar_json(results / "splits.json", _splits_para_composicao())
    dados = report.carregar(results)

    macros = _macros_de(report.macros_composicao(dados))

    # Retido o ISCX: união de 100 + 300 + 600 = 1000, frações 10, 30 e 60%.
    assert macros["composicaoTotalRetidoIscxRegimeB"] == "1.000"
    assert macros["composicaoRetidoIscxFonteGrambeddingsRegimeB"] == "10,00"
    assert macros["composicaoRetidoIscxFonteKaggleTresRegimeB"] == "30,00"
    assert macros["composicaoRetidoIscxFontePhishstormRegimeB"] == "60,00"
    # Retido o Grambeddings: 300 + 600 + 1000 = 1900; Kaggle-03 é 300/1900.
    assert macros["composicaoTotalRetidoGrambeddingsRegimeB"] == "1.900"
    assert macros["composicaoRetidoGrambeddingsFonteKaggleTresRegimeB"] == "15,79"
    assert all(n.isalpha() for n in macros)
    tex = report.tabela18_composicao(dados)
    assert _linha_da_tabela(tex, "União sem ISCX-URL2016")[1:] == [
        "1.000", "10,00", "30,00", "60,00", "retido"]


def test_composicao_sem_splits_nao_gera_macro_nem_tabela(tmp_path):
    dados = report.carregar(_resultados(tmp_path))

    assert report.macros_composicao(dados) == []
    assert report.tabela18_composicao(dados) is None


def _com_taxa(base, ajustes_tp_fp):
    """Cópia da matriz com `tp` e `fp` trocados nos ajustes de `ajustes_tp_fp`.

    `n_test` é 10, logo a taxa de predição positiva é `(tp + fp) / 10`.
    """
    saida = []
    for r in base:
        chave = (r["regime"], r["train_corpus"], r["test_corpus"], r["model"])
        copia = {**r, "metrics": dict(r["metrics"])}
        if chave in ajustes_tp_fp:
            copia["metrics"]["tp"], copia["metrics"]["fp"] = ajustes_tp_fp[chave]
        saida.append(copia)
    return saida


def _dados_degeneradas(tmp_path):
    # Padrão (tp 6, fp 3): taxa 0,9, não degenerada. Regime B: dois modelos de
    # Grambeddings -> Kaggle-03 sem nenhum positivo (taxa 0) e o MLP de
    # Kaggle-03 -> ISCX só com positivos (taxa 1,0).
    ajustes = {
        ("etld1", "grambeddings", "kaggle03", "logistic_regression"): (0, 0),
        ("etld1", "grambeddings", "kaggle03", "random_forest"): (0, 0),
        ("etld1", "kaggle03", "iscx2016", "mlp"): (5, 5),
        # Taxa 0,5: não degenerada. O limite exato (0,05 e 0,95) é testado em
        # `test_analises.py`, onde `n_test` pode ser 100.
        ("etld1", "phishstorm", "kaggle03", "xgboost"): (5, 0),
    }
    return report.carregar(_resultados(tmp_path, matriz=_com_taxa(_matriz(), ajustes)))


def test_macros_das_degeneradas_contam_ajustes_e_celulas(tmp_path):
    macros = _macros_de(report.macros_degeneradas(_dados_degeneradas(tmp_path)))

    assert macros["degeneradosForaDiagonalRegimeB"] == "3"
    assert macros["degeneradosAjustesForaDiagonalRegimeB"] == "72"
    assert macros["degeneradasCelulasForaDiagonalRegimeB"] == "2"
    assert macros["degeneradosForaDiagonalRegimeA"] == "0"
    assert macros["degeneradasCelulasForaDiagonalRegimeA"] == "0"
    assert macros["degeneradosLocoRegimeB"] == "0"
    assert macros["degeneradosAjustesLocoRegimeB"] == "24"
    assert all(n.isalpha() for n in macros)


def test_tabela_das_degeneradas_traz_a_contagem_por_celula(tmp_path):
    tex = report.tabela19_degeneradas(_dados_degeneradas(tmp_path))

    assert r"\label{tab:degeneradas}" in tex
    celula = _linha_da_tabela(tex, r"Grambeddings $\to$ Kaggle-03")
    assert celula[4] == "0/6" and celula[8] == "2/6"      # Regime A, Regime B
    assert _linha_da_tabela(tex, r"Kaggle-03 $\to$ ISCX-URL2016")[8] == "1/6"
    # Taxa média da célula Grambeddings -> Kaggle-03 no Regime B: quatro modelos
    # com 0,9 e dois com 0, média 0,6.
    assert _linha_da_tabela(tex, r"Grambeddings $\to$ Kaggle-03")[7] == "0,6000"


def _dados_efeito(tmp_path):
    # Diagonal: Regime A 0,95 e B 0,85 (erros 0,05 e 0,15, razão 3). LOCO: A 0,90 e B 0,80.
    matriz = _matriz(
        lambda regime, tr, te, m: {"auroc": (0.95 if regime == "oficial" else 0.85) if tr == te else 0.7}
    )
    loco = _loco(lambda regime, te, m: {"auroc": 0.90 if regime == "oficial" else 0.80})
    results = _resultados(tmp_path, matriz=matriz, loco=loco)
    _gravar_json(results / "offline.json", _offline_sintetico())
    return report.carregar(results)


def test_macros_do_efeito_batem_com_a_conta_a_mao(tmp_path):
    macros = _macros_de(report.macros_efeito(_dados_efeito(tmp_path)))

    # Diagonal: 1 - 0,95 = 0,05; 1 - 0,85 = 0,15; razão 3,00.
    assert macros["efeitoErroADiagonalGrambeddings"] == "0,0500"
    assert macros["efeitoErroBDiagonalGrambeddings"] == "0,1500"
    assert macros["efeitoRazaoDiagonalGrambeddings"] == "3,00"
    # O bootstrap só traz o ISCX m1 na diagonal: Delta 0,05 e IC indefinido.
    assert macros["efeitoDeltaFUmDiagonalIscx"] == "0,0500"
    assert macros["efeitoModelosIcDiagonalIscx"] == "0"
    assert "efeitoDeltaFUmDiagonalGrambeddings" not in macros
    # LOCO PhishStorm: erros 0,10 e 0,20, razão 2,00, Delta (0,30 + 0,10) / 2 = 0,20,
    # dois ICs que excluem zero.
    assert macros["efeitoErroALocoPhishstorm"] == "0,1000"
    assert macros["efeitoErroBLocoPhishstorm"] == "0,2000"
    assert macros["efeitoRazaoLocoPhishstorm"] == "2,00"
    assert macros["efeitoDeltaFUmLocoPhishstorm"] == "0,2000"
    assert macros["efeitoModelosIcLocoPhishstorm"] == "2"
    assert macros["efeitoModelosIcExcluiZeroLocoPhishstorm"] == "2"
    # Médias dos grupos.
    assert macros["efeitoErroADiagonalMedia"] == "0,0500"
    assert macros["efeitoRazaoDasMediasDiagonal"] == "3,00"
    assert macros["efeitoRazaoLocoMedia"] == "2,00"
    assert all(n.isalpha() for n in macros)


def test_tabela_do_efeito_traz_erro_razao_e_delta_f1(tmp_path):
    tex = report.tabela20_efeito(_dados_efeito(tmp_path))

    assert r"\label{tab:efeito}" in tex
    assert _linha_da_tabela(tex, "ISCX-URL2016")[1:] == [
        "0,0500", "0,1500", "3,00", "0,0500", "n/d"]
    assert _linha_da_tabela(tex, r"União $\to$ PhishStorm")[1:] == [
        "0,1000", "0,2000", "2,00", "0,2000", "2/2"]
    assert r"95\%" in tex


def test_efeito_sem_loco_nao_gera_macro_nem_tabela(tmp_path):
    dados = report.carregar(_resultados(tmp_path))
    dados["loco_por_seed"] = {}

    assert report.macros_efeito(dados) == []
    assert report.tabela20_efeito(dados) is None


def test_tabelas_e_macros_da_f04_sobre_os_resultados_reais(dados_reais, tmp_path):
    """Os cinco arquivos novos saem dos resultados atuais e as macros não colidem."""
    escritos = {p.name: p for p in report.gerar(REPO_ROOT / "results", tmp_path)}
    for nome in ("tabela16-decomposicao.tex", "tabela17-uniao.tex", "tabela18-composicao.tex",
                 "tabela19-degeneradas.tex", "tabela20-efeito.tex"):
        assert nome in escritos

    nomes = re.findall(
        r"\\newcommand\{\\([a-zA-Z]+)\}", escritos["numeros.tex"].read_text(encoding="utf-8"))
    assert len(nomes) == len(set(nomes)), "macro definida duas vezes"
    antes = _macros_de_antes((FIXTURES / "macros_antes.txt").read_text(encoding="utf-8"))
    assert set(antes) <= set(nomes)
    # A decomposição da semente 42 vem de `analises` sobre os registros reais.
    d = analises.decompor_variancia(dados_reais["matriz_por_seed"][42], "etld1")
    macros = _macros_de(escritos["numeros.tex"].read_text(encoding="utf-8").splitlines())
    assert macros["decompTreinoRegimeBSementeQuarentaEDois"] == _fmt(100 * d["treino"], 2)


# ---------------------------------------------------------------------------
# T31, ANL-03: média e faixa das três sementes ao lado da semente 42
# ---------------------------------------------------------------------------


def test_celula_de_media_traz_a_media_e_a_faixa_embaixo():
    # Três sementes: 0,90, 0,80 e 0,85. Média 0,85, faixa de 0,80 a 0,90.
    celula = report._celula_media([0.90, 0.80, 0.85])
    assert celula == r"\shortstack[r]{0,8500\\{\tiny [0,8000--0,9000]}}"
    # Com valor negativo na faixa, o separador é o do chamador.
    assert report._celula_media([-0.01, 0.01], 4, r";\,") == (
        r"\shortstack[r]{0,0000\\{\tiny [-0,0100;\,0,0100]}}"
    )
    # Valor ausente sai da média; um valor só, ou nenhum, não tem faixa.
    assert report._celula_media([0.9, None, 0.7]) == r"\shortstack[r]{0,8000\\{\tiny [0,7000--0,9000]}}"
    assert report._celula_media([0.9, None]) == "0,9000"
    assert report._celula_media([None, None]) == "--"


def _com_precisao_deslocada(valor, deslocamento):
    """A precisão de cada ajuste ganha `deslocamento`; revocação e TFP ficam."""
    def novo(regime, tr, te, m):
        v = dict(valor(regime, tr, te, m))
        v["precision"] += deslocamento
        return v
    return novo


def _resultados_limiar_tres_sementes(tmp_path):
    """Precisão da semente 7 com +0,02 e da 13 com -0,02: a média das três é a da 42."""
    results = _resultados(tmp_path, matriz=_matriz(_valor_matriz), loco=_loco(_valor_loco))
    for sub, desl in (("seed_07", 0.02), ("seed_13", -0.02)):
        _gravar_rodada(
            results / sub,
            matriz=_matriz(_com_precisao_deslocada(_valor_matriz, desl)),
            loco=_loco(
                lambda regime, te, m, d=desl: {
                    **_valor_loco(regime, te, m),
                    "precision": _valor_loco(regime, te, m)["precision"] + d,
                }
            ),
        )
    return report.carregar(results)


def test_a_tabela_de_limiar_traz_media_e_faixa_das_tres_sementes(tmp_path):
    dados = _resultados_limiar_tres_sementes(tmp_path)

    tex = report.tabela10_limiar(dados)

    celulas = _linha_da_tabela(tex, r"Grambeddings $\to$ Kaggle-03")[1:]
    assert len(celulas) == 12  # seis da semente 42 e seis da média
    m = _media_matriz("oficial", "grambeddings", "kaggle03")
    # A semente 42 fica como era.
    assert celulas[0] == _fmt(m["precision"])
    # Precisão no Regime A: valores p - 0,02, p e p + 0,02: média p, faixa de p - 0,02 a p + 0,02.
    p = m["precision"]
    assert celulas[6] == (
        rf"\shortstack[r]{{{_fmt(p)}\\{{\tiny [{_fmt(p - 0.02)}--{_fmt(p + 0.02)}]}}}}"
    )
    # A revocação não muda entre as sementes: a média é ela e a faixa é um ponto.
    r = m["recall"]
    assert celulas[7] == rf"\shortstack[r]{{{_fmt(r)}\\{{\tiny [{_fmt(r)}--{_fmt(r)}]}}}}"
    # Linha de média do grupo e LOCO também trazem as colunas.
    assert len(_linha_da_tabela(tex, r"\textbf{Média da diagonal}")[1:]) == 12
    assert len(_linha_da_tabela(tex, r"União $\to$ PhishStorm")[1:]) == 12
    assert "3 sementes com a faixa" in tex
    assert r"\resizebox{\textwidth}" in tex


def test_com_so_a_semente_42_a_tabela_de_limiar_nao_tem_as_colunas_nem_as_promete(dados_limiar):
    tex = report.tabela10_limiar(dados_limiar)

    assert len(_linha_da_tabela(tex, r"Grambeddings $\to$ Kaggle-03")[1:]) == 6
    assert "shortstack" not in tex and "resizebox" not in tex
    legenda = tex.split(r"\caption{")[1].split(r"\label")[0]
    assert "semente" in legenda and "sementes" not in legenda and "faixa" not in legenda


def _so_a_semente_42(dados):
    return dict(
        dados,
        matriz_por_seed={42: dados["matriz_por_seed"][42]},
        loco_por_seed={42: dados["loco_por_seed"][42]},
    )


def _entre_sementes(dados_reais, regime, corpus, modelo, metrica):
    """AUROC (ou outra) de uma célula da diagonal em cada semente, direto do JSON carregado."""
    return [
        next(
            r["metrics"][metrica] for r in dados_reais["matriz_por_seed"][s]
            if r["regime"] == regime and r["diagonal"] and r["train_corpus"] == corpus
            and r["model"] == modelo
        )
        for s in (42, 7, 13)
    ]


def test_a_tabela_da_diagonal_traz_a_media_e_a_faixa_da_auroc_dos_dois_regimes(dados_reais):
    tex = report.tabela3_diagonal(dados_reais)

    linha = next(
        ln for ln in tex.splitlines() if ln.startswith("Reg. logística &")
    )
    celulas = [c.strip() for c in linha.removesuffix("\\\\").split(" & ")]
    assert len(celulas) == 8  # A2 (TAB-02): sem a coluna do critério antigo
    a = _entre_sementes(dados_reais, "oficial", "grambeddings", "logistic_regression", "auroc")
    b = _entre_sementes(dados_reais, "etld1", "grambeddings", "logistic_regression", "auroc")
    for celula, valores in ((celulas[6], a), (celulas[7], b)):
        media = sum(valores) / 3
        assert celula == (
            rf"\shortstack[r]{{{_fmt(media)}\\{{\tiny [{_fmt(min(valores))}--{_fmt(max(valores))}]}}}}"
        )
    assert "Faixa das 3 sementes" in tex


def test_a_tabela_da_diagonal_com_so_a_semente_42_volta_as_sete_colunas(dados_reais):
    tex = report.tabela3_diagonal(_so_a_semente_42(dados_reais))

    linha = next(ln for ln in tex.splitlines() if ln.startswith("Reg. logística &"))
    assert linha.count(" & ") == 5 and "shortstack" not in tex  # A2 (TAB-02): seis colunas
    assert not [ln for ln in tex.splitlines() if "resizebox" in ln and not ln.startswith("%")]
    assert r"\multicolumn{6}{l}{\textbf{Grambeddings}" in tex
    assert "sementes" not in tex.split(r"\caption{")[1].split(r"\label")[0]


def _colunas_da_linha(linha: str) -> int:
    """Colunas de uma linha do `tabular`: `&` mais 1, contando o vão de cada `\\multicolumn`."""
    celulas = linha.rstrip().removesuffix("\\\\").split("&")
    total = 0
    for c in celulas:
        m = re.match(r"\\multicolumn\{(\d+)\}", c.strip())
        total += int(m.group(1)) if m else 1
    return total


@pytest.mark.parametrize("com_media, colunas", [(True, 8), (False, 6)])
def test_a_tabela_da_diagonal_sem_o_criterio_antigo_e_com_colunas_coerentes(
    dados_reais, com_media, colunas
):
    """TAB-02 e TAB-03: sem `$\\le$ IC` nem `largura do IC`; cabeçalho, subtítulos e linhas
    de dados com o mesmo número de colunas do formato do `tabular`."""
    dados = dados_reais if com_media else _so_a_semente_42(dados_reais)
    tex = report.tabela3_diagonal(dados)
    assert r"\le$ IC" not in tex and "largura do IC" not in tex

    formato = re.search(r"\\begin\{tabular\}\{(\w+)\}", tex).group(1)
    assert len(formato) == colunas

    linhas = tex.splitlines()
    cabecalho = [
        ln for ln in linhas
        if ln.startswith((" & \\multicolumn{2}{c}{AUROC, regime}", "Modelo &"))
    ]
    modelos = [ln for ln in linhas if ln.startswith(tuple(v + " &" for v in report.ROTULO_MODELO.values()))]
    subtitulos = [ln for ln in linhas if ln.startswith(r"\multicolumn{")]
    assert len(cabecalho) == 2 and len(modelos) == 24 and len(subtitulos) == 4
    for ln in (*cabecalho, *modelos, *subtitulos):
        assert _colunas_da_linha(ln) == colunas, ln
    for ln in subtitulos:
        assert "AUROC saturada no Regime A em " in ln, ln


def test_a_tabela_do_loco_traz_a_media_e_a_faixa_das_tres_sementes(dados_reais):
    tex = report.tabela6_loco(dados_reais)

    linha = _linha_da_tabela(tex, "PhishStorm")
    assert len(linha) == 11  # rótulo, quatro da semente 42, quatro da média, duas de sobreposição
    uniao = []
    for s in (42, 7, 13):
        registros = [
            r["metrics"]["auroc"] for r in dados_reais["loco_por_seed"][s]
            if r["regime"] == "etld1" and r["test_corpus"] == "phishstorm"
        ]
        uniao.append(sum(registros) / len(registros))
    media = sum(uniao) / 3
    # Segunda coluna da média das sementes: a AUROC da união.
    assert linha[6] == (
        rf"\shortstack[r]{{{_fmt(media)}\\{{\tiny [{_fmt(min(uniao))}--{_fmt(max(uniao))}]}}}}"
    )
    # A semente 42 fica como era: a união é a terceira coluna.
    assert linha[2] == _fmt(uniao[0])
    assert "faixa à direita" in tex


def test_a_tabela_do_loco_com_so_a_semente_42_volta_as_seis_colunas(dados_reais):
    tex = report.tabela6_loco(_so_a_semente_42(dados_reais))

    assert len(_linha_da_tabela(tex, "PhishStorm")) == 7
    assert "shortstack" not in tex
    assert "sementes" not in tex.split(r"\caption{")[1].split(r"\label")[0]


# ---------------------------------------------------------------------------
# T32, ANL-05: auditoria de viés ampliada
# ---------------------------------------------------------------------------

EXTRAS = ("has_https", "has_www", "www_count")
CARACTERISTICAS_SINTETICAS = ["a", "b", "c", "d", "e", "f", "g", "h", *EXTRAS]
# Topo de cada dataset (postos 1 a 5) na importância sintética.
TOPO_SINTETICO = {
    "grambeddings": ["a", "b", "c", "d", "e"],
    "kaggle03": ["a", "b", "c", "f", "g"],
    "phishstorm": ["a", "b", "c", "d", "h"],
    "iscx2016": ["has_https", "a", "b", "c", "d"],
}


def _importancias_sinteticas():
    ajustes = {}
    for corpus, topo in TOPO_SINTETICO.items():
        valores = {f: 0.0 for f in CARACTERISTICAS_SINTETICAS}
        for posto, nome in enumerate(topo):
            valores[nome] = 5.0 - posto  # 5, 4, 3, 2, 1: sem empate no topo
        ajustes[f"etld1|{corpus}|random_forest"] = valores
    return {"features": CARACTERISTICAS_SINTETICAS, "ajustes": ajustes}


def _media_sintetica(corpus, classe, nome):
    i = CARACTERISTICAS_SINTETICAS.index(nome)
    return 0.5 * i + 0.1 * list(TOPO_SINTETICO).index(corpus) + (0.25 if classe == "phishing" else 0.0)


def _diagnostico_da_auditoria(com_medias=True):
    esquema = {
        corpus: {
            "benigno": {"n": 100, "explicito": 0.90, "http": 0.50, "https": 0.40, "nenhum": 0.10},
            "phishing": {"n": 100, "explicito": 0.60, "http": 0.55, "https": 0.05, "nenhum": 0.40},
        }
        for corpus in TOPO_SINTETICO
    }
    medias = {
        corpus: {
            classe: {
                "n": 100,
                "medias": {n: _media_sintetica(corpus, classe, n) for n in CARACTERISTICAS_SINTETICAS},
            }
            for classe in ("benigno", "phishing")
        }
        for corpus in TOPO_SINTETICO
    } if com_medias else {}
    return {"esquema_por_classe": esquema, "medias_por_classe": medias}


def _dados_da_auditoria(tmp_path, com_medias=True):
    results = _resultados(tmp_path)
    _gravar_json(results / "diagnostics.json", _diagnostico_da_auditoria(com_medias))
    _gravar_json(results / "importances.json", _importancias_sinteticas())
    return report.carregar(results)


def test_selecao_das_caracteristicas_da_auditoria_e_o_topo_mais_as_tres_extras():
    selecao = report.caracteristicas_da_auditoria(_importancias_sinteticas())

    # Topo por dataset na ordem dos datasets, sem repetir; has_https já está no
    # topo do ISCX, então só has_www e www_count entram como extras.
    assert selecao == ["a", "b", "c", "d", "e", "f", "g", "h", "has_https", "has_www", "www_count"]


def test_a_tabela_da_auditoria_bate_com_o_diagnostico_sintetico(tmp_path):
    tex = report.tabela21_auditoria(_dados_da_auditoria(tmp_path))

    assert r"\label{tab:auditoria-vies}" in tex
    # Esquema: por dataset, benigno e phishing, em pontos percentuais com uma casa.
    por_dataset = ["90,0", "60,0"] * 4
    assert _linha_da_tabela(tex, "Com esquema")[1:] == por_dataset
    assert _linha_da_tabela(tex, r"\emph{http}")[1:] == ["50,0", "55,0"] * 4
    assert _linha_da_tabela(tex, r"\emph{https}")[1:] == ["40,0", "5,0"] * 4
    # Característica do topo: média por dataset e classe, duas casas.
    esperado = [
        _fmt(_media_sintetica(corpus, classe, "c"), 2)
        for corpus in TOPO_SINTETICO for classe in ("benigno", "phishing")
    ]
    assert _linha_da_tabela(tex, r"\texttt{c}")[1:] == esperado
    # As três extras e o underscore escapado.
    assert _linha_da_tabela(tex, r"\texttt{www\_count}")[1:] == [
        _fmt(_media_sintetica(corpus, classe, "www_count"), 2)
        for corpus in TOPO_SINTETICO for classe in ("benigno", "phishing")
    ]
    assert tex.count(r"\texttt{has\_www}") >= 2  # legenda e linha
    linhas_de_caracteristica = [ln for ln in tex.splitlines() if ln.startswith(r"\texttt{")]
    assert len(linhas_de_caracteristica) == 11


def test_a_tabela_da_auditoria_mantem_nove_colunas_e_dezoito_linhas(dados_reais):
    """TAB-05: a correção do Overfull é de largura; a forma é a de `5009f2c`
    (9 colunas; 3 linhas de esquema e 15 de características)."""
    tex = report.tabela21_auditoria(dados_reais)

    assert r"\begin{tabular}{lrrrrrrrr}" in tex
    linhas = [
        ln for ln in tex.splitlines()
        if ln.count(" & ") == 8 and not ln.startswith(" & ")
    ]
    assert len(linhas) == 18
    assert len([ln for ln in linhas if ln.startswith(r"\texttt{")]) == 15
    assert len([ln for ln in linhas if ln.startswith(("Com esquema", r"\emph{http"))]) == 3


def test_as_macros_da_auditoria_batem_com_o_diagnostico_sintetico(tmp_path):
    macros = _macros_de(report.macros_auditoria(_dados_da_auditoria(tmp_path)))

    assert macros["esquemaExplicitoGrambeddingsBenigno"] == "90,00"
    assert macros["esquemaHttpsKaggleTresPhishing"] == "5,00"
    assert macros["esquemaHttpPhishstormBenigno"] == "50,00"
    assert macros["auditoriaMediaHasHttpsIscxPhishing"] == _fmt(
        _media_sintetica("iscx2016", "phishing", "has_https"), 2)
    assert macros["auditoriaMediaWwwCountGrambeddingsBenigno"] == _fmt(
        _media_sintetica("grambeddings", "benigno", "www_count"), 2)
    assert all(n.isalpha() for n in macros)


def test_auditoria_sem_o_diagnostico_novo_nao_gera_tabela_nem_macro(tmp_path):
    dados = report.carregar(_resultados(tmp_path))  # diagnostics.json sem as chaves do ANL-05

    assert report.tabela21_auditoria(dados) is None
    assert report.macros_auditoria(dados) == []


def test_auditoria_sem_medias_traz_so_o_esquema_e_a_legenda_nao_promete_o_resto(tmp_path):
    tex = report.tabela21_auditoria(_dados_da_auditoria(tmp_path, com_medias=False))

    assert _linha_da_tabela(tex, "Com esquema")[1:] == ["90,0", "60,0"] * 4
    assert not [ln for ln in tex.splitlines() if ln.startswith(r"\texttt{")]
    legenda = tex.split(r"\caption{")[1].split(r"\label")[0]
    assert "características" not in legenda and "importances.json" not in tex
