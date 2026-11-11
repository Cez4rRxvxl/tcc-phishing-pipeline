"""Defesas do `scripts/run_all.py` que nenhum outro arquivo cobre.

O foco é o **cache de features**, que a auditoria de 05/08/2026 encontrou aberto:
`contrato_features_fingerprint()` hasheava só nomes e dtypes, então corrigir a
T3.4 (que mudou *valores* em 10% das linhas) deixava o hash idêntico e o cache
da rodada 1 era aceito como válido na rodada 2.

O efeito era pior que o defeito que as canárias fecharam. O livro-caixa era
recusado corretamente — e então o pipeline recarregava as features velhas do
cache e estampava nelas a impressão digital da rodada nova. Livro-caixa dizendo
rodada 2, dado da rodada 1, sem rastro. E o `run_smoke.py` não usa cache, então
o portão passava com as features certas: não havia como o portão pegar.

Evidência medida antes da correção, no `cache/features/iscx2016.parquet` de
04/08 (contrato `b79d8ff8214c265f`, o mesmo que o `results/run_meta.json` da
rodada 1 registrou):

    subdomain_count   2169 de 19965 linhas divergentes (10,86%)
    subdomain_ratio   2169 (10,86%)
    has_www            298 ( 1,49%)
    domain_length        7 ( 0,04%)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# O contrato que a rodada 1 gravou. Um cache com este valor **não** pode mais
# ser aceito: ele foi produzido antes da T3.4.
CONTRATO_RODADA_1 = "b79d8ff8214c265f"


@pytest.fixture(scope="module")
def run_all():
    """Importa `scripts/run_all.py` sem depender de `scripts/` no pythonpath."""
    caminho = REPO_ROOT / "scripts" / "run_all.py"
    if not caminho.exists():
        pytest.skip("scripts/run_all.py ausente neste commit")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_all_sob_teste", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class TestContratoDeFeaturesFechaOCache:
    def test_e_deterministico(self, run_all):
        assert (
            run_all.contrato_features_fingerprint()
            == run_all.contrato_features_fingerprint()
        )

    def test_nao_e_mais_o_contrato_da_rodada_1(self, run_all):
        """Regressão literal: se voltar a bater, o cache velho volta a passar."""
        assert run_all.contrato_features_fingerprint() != CONTRATO_RODADA_1

    def test_muda_quando_a_logica_de_extracao_muda(self, run_all, monkeypatch):
        """O ponto todo: valor de feature diferente ⇒ cache invalidado.

        Sem isto o cache é validado por um hash que descreve o *esquema* e não
        o *conteúdo* — foi exatamente o defeito de 05/08/2026.
        """
        from features.lexical_features import LexicalFeatureExtractor

        antes = run_all.contrato_features_fingerprint()
        original = LexicalFeatureExtractor._extract_count_features

        def com_defeito(self, url, parsed):
            saida = original(self, url, parsed)
            # Reintroduz a contagem ingênua de pontos, que era o bug T3.4-c.
            if parsed.netloc:
                saida["subdomain_count"] = max(0, len(parsed.netloc.split(".")) - 2)
            return saida

        monkeypatch.setattr(
            LexicalFeatureExtractor, "_extract_count_features", com_defeito
        )
        assert run_all.contrato_features_fingerprint() != antes

    def test_nao_muda_quando_so_a_docstring_muda(self, run_all, monkeypatch):
        """Era o argumento contra hashear o fonte: comentário não é resultado."""
        from features.lexical_features import LexicalFeatureExtractor

        antes = run_all.contrato_features_fingerprint()
        monkeypatch.setattr(
            LexicalFeatureExtractor._extract_count_features,
            "__doc__",
            "outra explicacao, mesmo comportamento",
        )
        assert run_all.contrato_features_fingerprint() == antes

    def test_cache_com_contrato_velho_e_recomputado(self, run_all, monkeypatch, tmp_path):
        """Prova de comportamento, não de hash: o cache velho é ignorado.

        Monta um cache falso cujo `contrato` é o da rodada 1 e confere que
        `preparar_corpus` **não** o devolve — recomputa a partir do parquet.
        """
        import pandas as pd

        corpus = "corpus_de_teste"
        bruto = pd.DataFrame(
            {
                "label": [0, 1, 0, 1],
                "url": [
                    # sufixo composto: `co.uk` é o eTLD, então só `www` conta
                    "https://www.bbc.co.uk/a",
                    # `com.br` é o eTLD, então só `phish` conta
                    "http://phish.site.com.br/login.php",
                    # porta não é subdomínio nem entra no domain_length (T3.4-b)
                    "https://site.com:8080/x",
                    "http://a.b.c.exemplo.com/y",
                ],
                "corpus": [corpus] * 4,
                "split_oficial": ["train", "train", "test", "test"],
            }
        )
        corpora_dir = tmp_path / "corpora" / corpus
        corpora_dir.mkdir(parents=True)
        bruto.to_parquet(corpora_dir / "data.parquet", index=False)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Cache mentiroso: uma linha só, e `subdomain_count` absurdo. Se o
        # `preparar_corpus` devolver isto, o cache foi aceito.
        veneno = pd.DataFrame(
            {c: [999] for c in run_all.COLUNAS_CACHE},
        )
        veneno.to_parquet(cache_dir / f"{corpus}.parquet", index=False)
        (cache_dir / f"{corpus}.meta.json").write_text(
            json.dumps(
                {
                    "sha256_origem": "sha-de-mentira",
                    "contrato": CONTRATO_RODADA_1,
                    "n_rows": 1,
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(run_all, "CORPORA_DIR", tmp_path / "corpora")
        monkeypatch.setattr(run_all, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(run_all, "log", lambda *a, **k: None)

        manifesto = {corpus: {"sha256": "sha-de-mentira"}}
        quadro = run_all.preparar_corpus(corpus, manifesto, usar_cache=True)

        assert len(quadro) == 4, "o cache envenenado foi aceito"
        # E o valor recomputado é o da T3.4 corrigida, não a contagem de pontos.
        # A contagem ingênua daria [2, 2, 0, 4]; a definição por PSL dá:
        assert list(quadro["subdomain_count"]) == [1, 1, 0, 3]
        assert quadro.loc[2, "domain_length"] == len("site.com")


class TestCustoRegistradoIncluiOLoco:
    """O `run_meta.json` subnotificava o custo pelo tempo do LOCO inteiro.

    A `gravar()` roda **antes** do bloco do LOCO — de propósito, para que a
    matriz esteja em disco se o LOCO cair — e gravava um único campo
    `minutos_desta_execucao` com o relógio daquele instante. Medido na entrega
    da rodada 2: o arquivo dizia 15,6 min numa execução cujos ajustes somam
    46,6 min. Nenhum número científico dependia disso; a tabela de orçamento da
    §13.7 e a conferência do RNF5 dependiam.

    O portão (`run_smoke.py`) não cobre este caminho: `atualizar_custo_com_loco`
    só existe no `run_all.py`. Daí estes testes.
    """

    def test_minutos_de_ajuste_nao_conta_o_mesmo_treino_4_vezes(self, run_all):
        """Cada ajuste da matriz aparece em 4 registros com o mesmo fit_seconds."""
        registros = [
            {"regime": "etld1", "train_corpus": "a", "model": "m",
             "test_corpus": t, "fit_seconds": 60.0}
            for t in ["a", "b", "c", "d"]
        ]
        # 4 registros, 1 ajuste de 60 s = 1,0 min -- nao 4,0.
        assert run_all.minutos_de_ajuste(registros) == 1.0

    def test_ajustes_distintos_somam(self, run_all):
        registros = [
            {"regime": "etld1", "train_corpus": "a", "model": "m1",
             "test_corpus": "a", "fit_seconds": 60.0},
            {"regime": "etld1", "train_corpus": "a", "model": "m2",
             "test_corpus": "a", "fit_seconds": 120.0},
            # mesmo modelo e corpus, regime diferente = outro ajuste
            {"regime": "oficial", "train_corpus": "a", "model": "m1",
             "test_corpus": "a", "fit_seconds": 60.0},
        ]
        assert run_all.minutos_de_ajuste(registros) == 4.0

    def test_atualizar_custo_preenche_as_fases_e_o_total(self, run_all,
                                                        tmp_path, monkeypatch):
        monkeypatch.setattr(run_all, "OUT_DIR", tmp_path)
        monkeypatch.setattr(run_all, "log", lambda *a, **k: None)
        (tmp_path / "run_meta.json").write_text(
            json.dumps({
                "minutos_desta_execucao": 0.0,
                "minutos_fase_matriz": 0.0,
                "minutos_fase_loco": None,
                "minutos_ajuste_matriz": 8.7,
                "minutos_ajuste_loco": None,
            }),
            encoding="utf-8",
        )
        loco = [
            {"regime": "etld1", "train_corpus": "uniao_sem_a", "model": "m",
             "test_corpus": "a", "fit_seconds": 600.0},
        ]

        # `minutos_fase_matriz=0.0` porque o relógio deste teste é o do import
        # do módulo, não o de uma execução real. O que se prova aqui é a
        # aritmética dos campos, não a duração.
        run_all.atualizar_custo_com_loco(loco, minutos_fase_matriz=0.0)

        meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["minutos_fase_matriz"] == 0.0
        assert meta["minutos_fase_loco"] is not None
        # O total deixa de ser o da fase da matriz: passa a vir do relógio.
        assert meta["minutos_desta_execucao"] >= meta["minutos_fase_matriz"]
        # Estes dois são o que se recupera de um `results/` antigo.
        assert meta["minutos_ajuste_loco"] == 10.0
        assert meta["minutos_ajuste_total"] == 18.7

    def test_fase_do_loco_nunca_e_negativa(self, run_all, tmp_path, monkeypatch):
        """Duração negativa parece dado e não parece defeito — por isso o clamp.

        Acontece se alguém chamar a função com um `minutos_fase_matriz` lido do
        `run_meta.json` de **outra** execução, cujo relógio não é este.
        """
        monkeypatch.setattr(run_all, "OUT_DIR", tmp_path)
        monkeypatch.setattr(run_all, "log", lambda *a, **k: None)
        (tmp_path / "run_meta.json").write_text(
            json.dumps({"minutos_fase_matriz": 999.0, "minutos_ajuste_matriz": 0.0}),
            encoding="utf-8",
        )
        run_all.atualizar_custo_com_loco([], minutos_fase_matriz=999.0)
        meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["minutos_fase_loco"] == 0.0


# ---------------------------------------------------------------------------
# Frente A, T13: árvore limpa, diretórios, sessões, run_meta e cache (RUN-02,
# RUN-03, SENS-02, REP-01)
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    import subprocess

    return subprocess.run(
        ["git", "-c", "user.name=teste", "-c", "user.email=t@t", *args],
        cwd=str(cwd), capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo_git(tmp_path):
    """Repositório git mínimo com a mesma forma do monorepo."""
    raiz = tmp_path / "repo"
    (raiz / "pipeline" / "src").mkdir(parents=True)
    (raiz / "pipeline" / "scripts").mkdir()
    (raiz / "pipeline" / "results").mkdir()
    (raiz / "latex").mkdir()
    (raiz / "pipeline" / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (raiz / "pipeline" / "results" / "r.json").write_text("{}\n", encoding="utf-8")
    (raiz / "latex" / "t.tex").write_text("a\n", encoding="utf-8")
    _git(raiz, "init", "-q")
    _git(raiz, "add", "-A")
    _git(raiz, "commit", "-q", "-m", "inicial")
    return raiz


class TestArvoreLimpa:
    """RUN-03: código sujo reprova; `results/` e o texto não."""

    def test_arvore_limpa_nao_tem_linhas(self, run_all, repo_git):
        assert run_all.arvore_suja(repo_git) == []

    def test_arquivo_rastreado_modificado_em_src_reprova(self, run_all, repo_git):
        (repo_git / "pipeline" / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
        sujas = run_all.arvore_suja(repo_git)
        assert len(sujas) == 1 and "pipeline/src/a.py" in sujas[0]

    def test_arquivo_nao_rastreado_em_src_reprova(self, run_all, repo_git):
        (repo_git / "pipeline" / "src" / "novo.py").write_text("y = 1\n", encoding="utf-8")
        sujas = run_all.arvore_suja(repo_git)
        assert len(sujas) == 1 and "pipeline/src/novo.py" in sujas[0]

    def test_arquivo_nao_rastreado_em_scripts_reprova(self, run_all, repo_git):
        (repo_git / "pipeline" / "scripts" / "x.py").write_text("y\n", encoding="utf-8")
        assert run_all.arvore_suja(repo_git)

    def test_modificacao_so_em_results_nao_impede(self, run_all, repo_git):
        (repo_git / "pipeline" / "results" / "r.json").write_text("{\"a\": 1}\n",
                                                                  encoding="utf-8")
        (repo_git / "pipeline" / "results" / "novo.json").write_text("{}\n",
                                                                     encoding="utf-8")
        assert run_all.arvore_suja(repo_git) == []

    def test_modificacao_so_no_texto_nao_impede(self, run_all, repo_git):
        (repo_git / "latex" / "t.tex").write_text("b\n", encoding="utf-8")
        assert run_all.arvore_suja(repo_git) == []

    def test_main_sai_com_1_antes_de_carregar_dados(self, run_all, monkeypatch, capsys):
        monkeypatch.setattr(run_all, "arvore_suja", lambda raiz=None: ["M pipeline/src/x.py"])
        monkeypatch.setattr(sys, "argv", ["run_all.py"])

        def nao_deveria_rodar(*a, **k):
            raise AssertionError("carregou dados com a arvore suja")

        for nome in ("checar_dependencias", "checar_corpora", "preparar_corpus"):
            monkeypatch.setattr(run_all, nome, nao_deveria_rodar)
        with pytest.raises(SystemExit) as saida:
            run_all.main()
        assert saida.value.code == 1
        assert "pipeline/src/x.py" in capsys.readouterr().out

    def test_commit_codigo_ignora_commits_so_de_texto(self, run_all, repo_git):
        antes = run_all.commit_codigo(repo_git)
        assert antes == _git(repo_git, "rev-parse", "HEAD")
        (repo_git / "latex" / "t.tex").write_text("c\n", encoding="utf-8")
        _git(repo_git, "commit", "-q", "-am", "so texto")
        assert run_all.commit_codigo(repo_git) == antes
        assert _git(repo_git, "rev-parse", "HEAD") != antes
        (repo_git / "pipeline" / "src" / "a.py").write_text("x = 3\n", encoding="utf-8")
        _git(repo_git, "commit", "-q", "-am", "codigo")
        assert run_all.commit_codigo(repo_git) == _git(repo_git, "rev-parse", "HEAD")


class TestDiretoriosDaRodada:
    """Os diretórios de saída e de cache do design (SENS-02, REP-01)."""

    def _r(self, run_all, **kw):
        return run_all.resolver_saidas(kw.pop("seed", None), 42, **kw)

    def test_rodada_principal(self, run_all):
        s = self._r(run_all)
        assert s.out_dir == run_all.REPO_ROOT / "results"
        assert s.cache_dir == run_all.REPO_ROOT / "cache" / "features"
        assert s.ledger == s.out_dir / "matrix.jsonl"
        assert s.ledger_loco == s.out_dir / "loco.jsonl"

    def test_semente_canonica_explicita_e_a_principal(self, run_all):
        assert self._r(run_all, seed=42).out_dir == run_all.REPO_ROOT / "results"

    def test_outra_semente(self, run_all):
        s = self._r(run_all, seed=7)
        assert s.out_dir == run_all.REPO_ROOT / "results" / "seed_07"
        assert s.cache_dir == run_all.REPO_ROOT / "cache" / "features"

    def test_sem_esquema(self, run_all):
        s = self._r(run_all, sem_esquema=True)
        assert s.out_dir == run_all.REPO_ROOT / "results" / "sensibilidade_sem_esquema"
        assert s.cache_dir == run_all.REPO_ROOT / "cache" / "features_sem_esquema"

    def test_sem_esquema_com_outra_semente(self, run_all):
        s = self._r(run_all, sem_esquema=True, seed=13)
        assert s.out_dir == (
            run_all.REPO_ROOT / "results" / "sensibilidade_sem_esquema" / "seed_13"
        )
        assert s.cache_dir == run_all.REPO_ROOT / "cache" / "features_sem_esquema"

    def test_repeticao(self, run_all):
        s = self._r(run_all, repeticao=True)
        assert s.out_dir == run_all.REPO_ROOT / "results" / "repeticao_seed_42"
        assert s.cache_dir == run_all.REPO_ROOT / "cache" / "features"
        assert s.ledger.parent == s.out_dir

    def test_repeticao_com_semente_canonica_explicita(self, run_all):
        assert self._r(run_all, repeticao=True, seed=42).out_dir.name == "repeticao_seed_42"

    def test_repeticao_com_semente_nao_canonica_e_recusada(self, run_all):
        with pytest.raises(run_all.ArgumentoInvalido, match="canonica"):
            self._r(run_all, repeticao=True, seed=7)

    def test_repeticao_com_sem_esquema_e_recusada(self, run_all):
        with pytest.raises(run_all.ArgumentoInvalido):
            self._r(run_all, repeticao=True, sem_esquema=True)

    def test_main_recusa_repeticao_com_semente_nao_canonica(self, run_all, monkeypatch):
        monkeypatch.setattr(run_all, "arvore_suja", lambda raiz=None: [])
        monkeypatch.setattr(run_all, "checar_dependencias", lambda: None)
        monkeypatch.setattr(
            run_all, "checar_corpora",
            lambda nomes: pytest.fail("carregou dados com argumentos invalidos"),
        )
        monkeypatch.setattr(sys, "argv", ["run_all.py", "--repeticao", "--seed", "7"])
        with pytest.raises(SystemExit) as saida:
            run_all.main()
        assert saida.value.code == 1


class TestSessoes:
    """RUN-02: duas sessões somam no `tempo_parede_s`."""

    def test_duas_sessoes_simuladas(self, run_all, tmp_path):
        from datetime import UTC, datetime, timedelta

        t0 = datetime(2026, 11, 9, 8, 0, 0, tzinfo=UTC)
        (tmp_path / "run_meta.json").write_text(json.dumps({"commit": "abc"}),
                                                encoding="utf-8")
        run_all.registrar_sessao(t0, t0 + timedelta(seconds=90), tmp_path)
        run_all.registrar_sessao(t0 + timedelta(hours=1),
                                 t0 + timedelta(hours=1, seconds=30.5), tmp_path)

        meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
        assert len(meta["sessoes"]) == 2
        assert [s["segundos"] for s in meta["sessoes"]] == [90.0, 30.5]
        assert meta["tempo_parede_s"] == 120.5
        assert meta["commit"] == "abc"  # o resto do meta fica
        assert set(meta["sessoes"][0]) == {"inicio", "fim", "segundos"}

    def test_sessoes_sobrevivem_sem_run_meta(self, run_all, tmp_path):
        """Uma sessão que caiu antes do `run_meta.json` ainda conta na próxima."""
        from datetime import UTC, datetime, timedelta

        t0 = datetime(2026, 11, 9, 8, 0, 0, tzinfo=UTC)
        run_all.registrar_sessao(t0, t0 + timedelta(seconds=10), tmp_path)
        assert not (tmp_path / "run_meta.json").exists()
        assert len(run_all.ler_sessoes(tmp_path)) == 1
        assert run_all.tempo_parede_s(run_all.ler_sessoes(tmp_path)) == 10.0

    def test_sem_arquivo_de_sessoes_nao_ha_sessao(self, run_all, tmp_path):
        assert run_all.ler_sessoes(tmp_path) == []
        assert run_all.tempo_parede_s([]) == 0.0


class TestCommitPorSessao:
    """RUN-03, achado 1 da revisao do lote F: o commit do codigo e registrado em
    cada sessao e uma retomada com codigo diferente e recusada."""

    @pytest.fixture(autouse=True)
    def _restaurar_commit_sessao(self, run_all, monkeypatch):
        # `checar_commit_da_rodada` fixa um global do modulo (compartilhado entre
        # testes); o monkeypatch o devolve ao valor original ao fim de cada teste.
        monkeypatch.setattr(run_all, "COMMIT_SESSAO", None)

    def _sessao(self, run_all, pasta, commit):
        from datetime import UTC, datetime, timedelta

        t0 = datetime(2026, 11, 9, 8, 0, 0, tzinfo=UTC)
        run_all.registrar_sessao(t0, t0 + timedelta(seconds=5), pasta, commit=commit)

    def test_a_sessao_grava_o_commit_do_codigo(self, run_all, tmp_path):
        self._sessao(run_all, tmp_path, "a" * 40)
        assert run_all.ler_sessoes(tmp_path)[0]["commit_codigo"] == "a" * 40

    def test_retomada_com_o_mesmo_commit_passa(self, run_all, tmp_path, monkeypatch):
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: "a" * 40)
        self._sessao(run_all, tmp_path, "a" * 40)
        assert run_all.checar_commit_da_rodada(tmp_path) == "a" * 40

    def test_retomada_com_codigo_novo_e_recusada(self, run_all, tmp_path, monkeypatch, capsys):
        self._sessao(run_all, tmp_path, "a" * 40)
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: "b" * 40)
        with pytest.raises(SystemExit) as saida:
            run_all.checar_commit_da_rodada(tmp_path)
        assert saida.value.code == 1
        texto = capsys.readouterr().out
        assert "--recomecar" in texto and "a" * 40 in texto and "b" * 40 in texto

    def test_recomecar_aceita_commit_diferente(self, run_all, tmp_path, monkeypatch):
        self._sessao(run_all, tmp_path, "a" * 40)
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: "b" * 40)
        assert run_all.checar_commit_da_rodada(tmp_path, recomecar=True) == "b" * 40

    def test_rodada_nova_sem_sessoes_passa(self, run_all, tmp_path, monkeypatch):
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: "c" * 40)
        assert run_all.checar_commit_da_rodada(tmp_path) == "c" * 40

    def test_git_sem_commit_recusa_a_rodada(self, run_all, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: None)
        with pytest.raises(SystemExit) as saida:
            run_all.checar_commit_da_rodada(tmp_path)
        assert saida.value.code == 1
        assert "commit" in capsys.readouterr().out

    def test_run_meta_leva_o_commit_da_sessao_e_nao_o_do_fim(
        self, run_all, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: "z" * 40)
        monkeypatch.setattr(run_all, "COMMIT_SESSAO", "a" * 40)
        assert run_all.montar_meta_proveniencia({}, None, False, False)[
            "commit_codigo"
        ] == "a" * 40


class TestRunMetaDaFrenteA:
    """RUN-02: o `run_meta.json` carrega os campos novos do design."""

    CAMPOS_NOVOS = {
        "commit_codigo", "psl_sha256", "strip_scheme", "repeticao",
        "particoes", "dedup", "sessoes", "tempo_parede_s",
    }

    def _gravar(self, run_all, tmp_path, monkeypatch, **kw):
        import evaluate
        import models

        monkeypatch.setattr(run_all, "OUT_DIR", tmp_path)
        monkeypatch.setattr(run_all, "log", lambda *a, **k: None)
        monkeypatch.setattr(run_all, "commit_codigo", lambda raiz=None: "c0d160" * 6 + "abcd")
        metricas, avisos = evaluate.evaluate_predictions(
            [0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6], threshold=0.5, ece_bins=10
        )
        registro = evaluate.make_result_record(
            run_id="t", regime="etld1", train_corpus="a", test_corpus="a",
            model="xgboost", metrics=metricas, fit_seconds=1.0, warnings=avisos,
        )
        config = {"seed": 42, "split": {}, "avaliacao": {}, "modelos": {},
                  "regimes": ["etld1"]}
        resumo = {"a/etld1": {"dedup": {"n_antes": 10, "n_final": 9}}}
        gpu = models.GpuInfo(available=False, reason="teste")
        run_all.gravar([registro], resumo, config, "fp", gpu, ["a"], **kw)
        return json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))

    def test_tem_todos_os_campos_novos(self, run_all, tmp_path, monkeypatch):
        import psl

        particoes = {"arquivo": "predicoes/particoes.parquet", "sha256": "ab" * 32}
        meta = self._gravar(run_all, tmp_path, monkeypatch, particoes=particoes,
                            strip_scheme=True, repeticao=False)
        assert self.CAMPOS_NOVOS <= set(meta)
        assert meta["commit_codigo"].endswith("abcd")
        assert meta["psl_sha256"] == psl.sha256()
        assert meta["strip_scheme"] is True and meta["repeticao"] is False
        assert meta["particoes"] == particoes
        assert meta["dedup"] == {"a/etld1": {"n_antes": 10, "n_final": 9}}
        assert meta["sessoes"] == [] and meta["tempo_parede_s"] == 0.0
        assert meta["commit"]  # o campo antigo continua

    def test_sessao_anterior_entra_na_regravacao(self, run_all, tmp_path, monkeypatch):
        from datetime import UTC, datetime, timedelta

        t0 = datetime(2026, 11, 9, tzinfo=UTC)
        run_all.registrar_sessao(t0, t0 + timedelta(seconds=42), tmp_path)
        meta = self._gravar(run_all, tmp_path, monkeypatch)
        assert meta["tempo_parede_s"] == 42.0 and len(meta["sessoes"]) == 1
        assert meta["repeticao"] is False and meta["strip_scheme"] is False


class TestCacheDeOutraPslOuOpcao:
    """O cache de outra PSL ou de outra opção é recusado e recalculado."""

    @pytest.fixture
    def cenario(self, run_all, tmp_path, monkeypatch):
        import pandas as pd

        corpus = "corpus_cache"
        bruto = pd.DataFrame(
            {
                "label": [0, 1, 0, 1],
                "url": ["https://www.bbc.co.uk/a", "http://phish.site.com.br/login.php",
                        "site.com/x", "http://a.b.c.exemplo.com/y"],
                "corpus": [corpus] * 4,
                "split_oficial": ["train", "train", "test", "test"],
            }
        )
        pasta = tmp_path / "corpora" / corpus
        pasta.mkdir(parents=True)
        bruto.to_parquet(pasta / "data.parquet", index=False)
        mensagens: list[str] = []
        monkeypatch.setattr(run_all, "CORPORA_DIR", tmp_path / "corpora")
        monkeypatch.setattr(run_all, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(run_all, "log", lambda msg="": mensagens.append(msg))
        return corpus, {corpus: {"sha256": "sha-origem"}}, mensagens

    def test_cache_proprio_e_reaproveitado(self, run_all, cenario):
        corpus, manifesto, mensagens = cenario
        run_all.preparar_corpus(corpus, manifesto, usar_cache=True)
        mensagens.clear()
        run_all.preparar_corpus(corpus, manifesto, usar_cache=True)
        assert any("(cache)" in m for m in mensagens)

    def test_quadro_tem_linha_e_url_norm(self, run_all, cenario):
        import numpy as np

        corpus, manifesto, _ = cenario
        quadro = run_all.preparar_corpus(corpus, manifesto, usar_cache=True)
        assert list(quadro["linha"]) == [0, 1, 2, 3]
        assert quadro["linha"].dtype == np.int32
        assert quadro["url_norm"].notna().all()
        # E o quadro lido do cache mantém as duas colunas.
        de_cache = run_all.preparar_corpus(corpus, manifesto, usar_cache=True)
        assert list(de_cache["linha"]) == [0, 1, 2, 3]
        assert "url_norm" in de_cache.columns

    def test_cache_de_outra_psl_e_recusado(self, run_all, cenario, monkeypatch):
        import psl

        corpus, manifesto, mensagens = cenario
        run_all.preparar_corpus(corpus, manifesto, usar_cache=True)
        mensagens.clear()
        monkeypatch.setattr(psl, "sha256", lambda: "f" * 64)
        run_all.preparar_corpus(corpus, manifesto, usar_cache=True)
        assert any("invalidado" in m and "PSL" in m for m in mensagens)
        assert not any("(cache)" in m for m in mensagens)

    def test_cache_de_outra_opcao_e_recusado(self, run_all, cenario):
        corpus, manifesto, mensagens = cenario
        com_esquema = run_all.preparar_corpus(corpus, manifesto, usar_cache=True)
        mensagens.clear()
        sem_esquema = run_all.preparar_corpus(
            corpus, manifesto, usar_cache=True, strip_scheme=True
        )
        assert any("invalidado" in m and "strip_scheme" in m for m in mensagens)
        # Recalculou de fato: `url_length` (texto inteiro) perde o prefixo.
        assert sem_esquema["url_length"].iloc[0] < com_esquema["url_length"].iloc[0]
        # E o cache passou a ser da nova opção.
        mensagens.clear()
        run_all.preparar_corpus(corpus, manifesto, usar_cache=True, strip_scheme=True)
        assert any("(cache)" in m for m in mensagens)

    def test_contrato_do_cache_muda_com_a_canonizacao(self, run_all, monkeypatch):
        # Achado 4 do lote F: o cache guarda `url_norm` e `etld_plus_one`, entao
        # uma mudanca na normalizacao tem de invalidar o cache.
        from data.url_canonicalization import URLCanonicalizer

        base = run_all.contrato_features_fingerprint()
        monkeypatch.setattr(
            URLCanonicalizer, "_normalize_query", lambda self, query: query or None
        )
        assert run_all.contrato_features_fingerprint() != base

    def test_contrato_muda_com_a_opcao_e_com_a_psl(self, run_all, monkeypatch):
        import psl

        base = run_all.contrato_features_fingerprint()
        assert run_all.contrato_features_fingerprint(strip_scheme=True) != base
        monkeypatch.setattr(psl, "sha256", lambda: "e" * 64)
        assert run_all.contrato_features_fingerprint() != base


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
