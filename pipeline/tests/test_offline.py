"""T17 e T18: filtragem offline da sobreposição de eTLD+1 (spec FILT-01, critérios 1 a 4)
e *bootstrap* pareado do Delta entre regimes (BOOT-01, critérios 5 e 6).

Predições e partições sintéticas em que as linhas sobrepostas são conhecidas e
os valores esperados saem de contas feitas à mão (comentadas em cada caso).
Nenhum teste treina modelo.

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

import json

import numpy as np
import pandas as pd
import pytest

import offline
import predicoes
import splits
from offline import OfflineError

# ---------------------------------------------------------------------------
# Montagem de uma rodada sintética
# ---------------------------------------------------------------------------


def _montar_particoes(dir_rodada, part):
    """Grava `particoes.parquet` pelo caminho real (`predicoes.gravar_particoes`).

    `part` mapeia regime -> corpus -> lista de (partição, grupo), uma por linha.
    A coluna `linha` é `100 + posição`, para que posição e identidade difiram.
    """
    indices, quadros, contagens = {}, {}, {}
    for regime, corpora in part.items():
        for corpus, linhas in corpora.items():
            n = len(linhas)
            # `gravar_particoes` recebe um quadro por corpus. O quadro de um
            # corpus pode ter mais linhas num regime que noutro (o `a` do regime
            # `oficial` tem uma a mais): guarda-se o maior, que contém o menor
            # como prefixo, e cada SplitIndices só aponta posições existentes.
            if corpus not in quadros or n > len(quadros[corpus]):
                quadros[corpus] = pd.DataFrame(
                    {
                        "linha": np.arange(n, dtype=np.int32) + 100,
                        "etld_plus_one": [g for _, g in linhas],
                        "url": [f"http://x/{i}" for i in range(n)],
                    }
                )
            pos = {
                nome: np.array([i for i, (p, _) in enumerate(linhas) if p == nome], dtype=int)
                for nome in ("train", "val", "test")
            }
            indices[(corpus, regime)] = splits.SplitIndices(
                corpus, regime, pos["train"], pos["val"], pos["test"]
            )
            contagens[(corpus, regime)] = {"n_final": n}
    return predicoes.gravar_particoes(dir_rodada, indices, quadros, contagens)


def _montar(dir_rodada, part, ajustes, nome_json="matrix.json", meta=True):
    """Grava partições, predições por ajuste e o JSON de registros."""
    ref_part = _montar_particoes(dir_rodada, part)
    registros = []
    for aj in ajustes:
        blocos = [
            (corpus, np.asarray(lin) + 100, np.asarray(y), np.asarray(s, dtype=np.float64))
            for corpus, (lin, y, s) in aj["blocos"].items()
        ]
        ref = predicoes.gravar_predicoes(
            dir_rodada, aj["tipo"], aj["regime"], aj["treino"], aj["modelo"], blocos
        )
        for corpus, (lin, *_) in aj["blocos"].items():
            registros.append(
                {
                    "regime": aj["regime"],
                    "train_corpus": aj["treino"],
                    "test_corpus": corpus,
                    "model": aj["modelo"],
                    "diagonal": corpus == aj["treino"],
                    "n_test": len(lin),
                    "metrics": {"threshold": aj["limiar"]},
                    "predicoes": ref,
                }
            )
    (dir_rodada / nome_json).write_text(json.dumps(registros), encoding="utf-8")
    if meta:
        (dir_rodada / "run_meta.json").write_text(
            json.dumps({"particoes": ref_part}), encoding="utf-8"
        )
    return ref_part


# Partições: regime -> corpus -> (partição, grupo) por linha (posição 0, 1, ...).
_A_ETLD1 = [
    ("train", "g1"), ("train", "g1"), ("train", "g2"), ("val", "g3"),
    ("test", "g4"), ("test", "g4"),
]
_B = [
    ("train", "h1"), ("val", "h2"),
    ("test", "g1"), ("test", "g3"), ("test", "g4"), ("test", "x1"),
    ("test", "x2"), ("test", "x3"), ("test", "x4"), ("test", "x5"),
]
_C = [("train", "k1"), ("test", "g1"), ("test", "g2"), ("test", "g9"), ("test", "g10")]
_E = [("test", "g1"), ("test", "g2")]

PART = {
    "etld1": {"a": _A_ETLD1, "b": _B, "c": _C, "e": _E},
    # No regime A o treino de `a` também viu o grupo x1.
    "oficial": {"a": _A_ETLD1 + [("train", "x1")], "b": _B, "c": _C, "e": _E},
}

# Testes (posições 2..9 de `b`): grupos g1, g3, g4, x1, x2, x3, x4, x5.
_Y_B = [1, 0, 1, 0, 1, 0, 1, 0]
_S_B = [0.1, 0.95, 0.9, 0.8, 0.7, 0.3, 0.6, 0.2]


def _ajuste(regime, limiar=0.65):
    return {
        "tipo": "matriz",
        "regime": regime,
        "treino": "a",
        "modelo": "m",
        "limiar": limiar,
        "blocos": {
            "a": ([4, 5], [1, 0], [0.9, 0.1]),                         # diagonal
            "b": (list(range(2, 10)), _Y_B, _S_B),
            "c": ([1, 2, 3, 4], [0, 0, 1, 1], [0.2, 0.3, 0.8, 0.9]),
            "e": ([0, 1], [0, 1], [0.4, 0.6]),
        },
    }


def _por_chave(resultado):
    return {(r["regime"], r["train_corpus"], r["test_corpus"], r["model"]): r for r in resultado}


@pytest.fixture
def rodada(tmp_path):
    _montar(tmp_path, PART, [_ajuste("etld1"), _ajuste("oficial")])
    return tmp_path


# ---------------------------------------------------------------------------
# FILT critério 1: remove as linhas de teste cujo eTLD+1 está no treino ou na validação
# ---------------------------------------------------------------------------


class TestRemocao:
    def test_remove_so_as_linhas_com_grupo_do_treino_ou_da_validacao(self, rodada):
        r = _por_chave(offline.filtrar_sobreposicao(rodada))
        # Regime B: `a` viu g1, g2 (treino) e g3 (validação). Em `b`, saem as
        # linhas g1 e g3 (2 de 8); g4 só está no TESTE de `a` e fica.
        celula = r[("etld1", "a", "b", "m")]
        assert celula["n_teste"] == 8
        assert celula["n_removidas"] == 2
        assert celula["fracao_removida"] == pytest.approx(0.25)
        assert celula["n_restantes"] == 6

    def test_regime_a_usa_as_particoes_do_proprio_regime(self, rodada):
        r = _por_chave(offline.filtrar_sobreposicao(rodada))
        # No regime A o treino de `a` também viu x1: saem g1, g3 e x1 (3 de 8).
        celula = r[("oficial", "a", "b", "m")]
        assert celula["n_removidas"] == 3
        assert celula["fracao_removida"] == pytest.approx(3 / 8)

    def test_grupo_do_teste_do_proprio_treino_nao_conta_como_visto(self, rodada):
        # `g4` está só no teste de `a`: não é visto por `a`, então as linhas g4
        # de `b` permanecem (e entram nas 6 restantes do teste acima).
        r = _por_chave(offline.filtrar_sobreposicao(rodada))
        assert r[("etld1", "a", "b", "m")]["n_restantes"] == 6


# ---------------------------------------------------------------------------
# FILT critério 2: n e fração removidas, métricas recalculadas com o limiar original
# ---------------------------------------------------------------------------


class TestMetricasRecalculadas:
    def test_auroc_recalculada_bate_com_a_conta_a_mao(self, rodada):
        r = _por_chave(offline.filtrar_sobreposicao(rodada))
        # Regime B, restantes: positivos g4 .9, x2 .7, x4 .6; negativos x1 .8,
        # x3 .3, x5 .2. Pares (pos, neg) ordenados corretamente: .9 vence os 3,
        # .7 vence 2, .6 vence 2: 7 de 9.
        assert r[("etld1", "a", "b", "m")]["metrics"]["auroc"] == pytest.approx(7 / 9)
        # Regime A, restantes (sem x1): .9, .7, .6 contra .3, .2: 6 de 6.
        assert r[("oficial", "a", "b", "m")]["metrics"]["auroc"] == pytest.approx(1.0)

    def test_usa_o_limiar_do_registro_e_nao_outro(self, tmp_path):
        _montar(tmp_path, PART, [_ajuste("etld1", 0.65), _ajuste("oficial", 0.65)])
        r = _por_chave(offline.filtrar_sobreposicao(tmp_path))
        m = r[("etld1", "a", "b", "m")]["metrics"]
        # Limiar 0,65: g4 .9 TP, x1 .8 FP, x2 .7 TP, x3 TN, x4 .6 FN, x5 TN.
        assert m["threshold"] == 0.65
        assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)

        outro = tmp_path / "outro"
        outro.mkdir()
        _montar(outro, PART, [_ajuste("etld1", 0.5), _ajuste("oficial", 0.5)])
        m2 = _por_chave(offline.filtrar_sobreposicao(outro))[("etld1", "a", "b", "m")]["metrics"]
        # Limiar 0,5: x4 .6 passa a TP.
        assert m2["threshold"] == 0.5
        assert (m2["tp"], m2["fp"], m2["tn"], m2["fn"]) == (3, 1, 2, 0)

    def test_saida_ordenada_por_regime_treino_teste_modelo(self, rodada):
        saida = offline.filtrar_sobreposicao(rodada)
        chaves = [(r["regime"], r["train_corpus"], r["test_corpus"], r["model"]) for r in saida]
        assert chaves == sorted(chaves)


# ---------------------------------------------------------------------------
# FILT critério 3: uma classe só (ou nenhuma linha) marca a AUROC como indefinida
# ---------------------------------------------------------------------------


class TestIndefinida:
    def test_celula_sem_uma_classe_marca_auroc_indefinida_e_nao_interrompe(self, rodada):
        r = _por_chave(offline.filtrar_sobreposicao(rodada))
        # `c`: g1 e g2 saem; sobram g9 e g10, ambas positivas.
        celula = r[("etld1", "a", "c", "m")]
        assert celula["n_removidas"] == 2
        assert celula["auroc_indefinida"] is True
        assert celula["metrics"]["auroc"] is None
        # As demais células seguem calculadas.
        assert r[("etld1", "a", "b", "m")]["auroc_indefinida"] is False

    def test_celula_sem_nenhuma_linha_restante(self, rodada):
        r = _por_chave(offline.filtrar_sobreposicao(rodada))
        # `e`: g1 e g2 estão no treino de `a`; nada sobra.
        celula = r[("etld1", "a", "e", "m")]
        assert celula["n_removidas"] == celula["n_teste"] == 2
        assert celula["fracao_removida"] == 1.0
        assert celula["metrics"] is None
        assert celula["auroc_indefinida"] is True


# ---------------------------------------------------------------------------
# FILT critério 4: a diagonal fica fora
# ---------------------------------------------------------------------------


def test_diagonal_nao_entra_na_filtragem(rodada):
    saida = offline.filtrar_sobreposicao(rodada)
    assert saida
    assert all(r["train_corpus"] != r["test_corpus"] for r in saida)
    assert len(saida) == 6  # 3 testes fora da diagonal x 2 regimes


# ---------------------------------------------------------------------------
# LOCO: grupos vistos são a união das três fontes; limiar do registro
# ---------------------------------------------------------------------------


class TestLoco:
    @pytest.fixture
    def loco(self, tmp_path):
        regime = "etld1"
        part = {
            regime: {
                "a": [("train", "ga"), ("test", "za")],
                "b": [("val", "gb"), ("test", "zb")],
                "c": [("train", "gc"), ("test", "zc")],
                # O treino do retido NÃO conta: gd só está no treino de `d`.
                "d": [("train", "gd"), ("test", "ga"), ("test", "gb"), ("test", "gc"),
                      ("test", "gd"), ("test", "gz"), ("test", "zb")],
            }
        }
        ajuste = {
            "tipo": "loco",
            "regime": regime,
            "treino": offline.LOCO_PREFIXO + "d",
            "modelo": "m",
            "limiar": 0.3,
            # linhas 1..6 de `d`: ga, gb, gc, gd, gz, zb
            "blocos": {"d": ([1, 2, 3, 4, 5, 6], [1, 0, 1, 0, 1, 0],
                             [0.9, 0.2, 0.8, 0.4, 0.7, 0.1])},
        }
        _montar(tmp_path, part, [ajuste], nome_json="loco.json")
        return tmp_path

    def test_grupos_vistos_sao_a_uniao_das_tres_fontes(self, loco):
        (celula,) = offline.filtrar_sobreposicao(loco)
        # Saem ga (treino de a), gb (validação de b) e gc (treino de c);
        # ficam gd (treino do próprio retido), gz e zb.
        assert celula["n_teste"] == 6
        assert celula["n_removidas"] == 3
        assert celula["n_restantes"] == 3

    def test_limiar_usado_e_o_do_registro(self, loco):
        (celula,) = offline.filtrar_sobreposicao(loco)
        m = celula["metrics"]
        # Restantes: gd y0 .4, gz y1 .7, zb y0 .1. Limiar 0,3: gd FP, gz TP, zb TN.
        assert m["threshold"] == 0.3
        assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 0)
        assert m["auroc"] == pytest.approx(1.0)

    def test_prefixo_igual_ao_do_matrix(self):
        import matrix

        assert offline.LOCO_PREFIXO == matrix.LOCO_PREFIXO


# ---------------------------------------------------------------------------
# Erros: nunca silenciar
# ---------------------------------------------------------------------------


class TestErros:
    def test_particoes_ausentes(self, tmp_path):
        with pytest.raises(OfflineError, match="particoes.parquet"):
            offline.filtrar_sobreposicao(tmp_path)

    def test_predicoes_adulteradas_levantam_erro(self, rodada):
        arquivo = next((rodada / "predicoes" / "matriz").glob("*.parquet"))
        arquivo.write_bytes(arquivo.read_bytes() + b"x")
        with pytest.raises(predicoes.PredicaoError):
            offline.filtrar_sobreposicao(rodada)

    def test_particoes_adulteradas_levantam_erro(self, rodada):
        arquivo = rodada / predicoes.ARQUIVO_PARTICOES
        arquivo.write_bytes(arquivo.read_bytes() + b"x")
        with pytest.raises(OfflineError, match="adulteradas"):
            offline.filtrar_sobreposicao(rodada)

    def test_registro_sem_referencia_de_predicoes(self, rodada):
        caminho = rodada / "matrix.json"
        registros = json.loads(caminho.read_text(encoding="utf-8"))
        for r in registros:
            r.pop("predicoes")
        caminho.write_text(json.dumps(registros), encoding="utf-8")
        with pytest.raises(OfflineError, match="sem referência"):
            offline.filtrar_sobreposicao(rodada)

    def test_n_test_divergente_das_predicoes(self, rodada):
        caminho = rodada / "matrix.json"
        registros = json.loads(caminho.read_text(encoding="utf-8"))
        for r in registros:
            r["n_test"] += 1
        caminho.write_text(json.dumps(registros), encoding="utf-8")
        with pytest.raises(OfflineError, match="n_test"):
            offline.filtrar_sobreposicao(rodada)


# ---------------------------------------------------------------------------
# T18: bootstrap pareado (spec BOOT-01, critérios 5 e 6)
# ---------------------------------------------------------------------------

CONFIG = {"seed": 42, "avaliacao": {"bootstrap_n": 200, "bootstrap_nivel": 0.95}}


def _montar_pares(dir_rodada, celulas, nome_json="matrix.json"):
    """Predições e registros de células com os dois regimes, sem partições.

    Cada item de `celulas` tem regime, treino, teste, modelo, linhas, y, score e
    limiar. Um parquet por registro (o modelo é único por célula nos testes).
    """
    registros = []
    for c in celulas:
        ref = predicoes.gravar_predicoes(
            dir_rodada, "matriz", c["regime"], c["treino"], c["modelo"],
            [(c["teste"], np.asarray(c["linhas"]), np.asarray(c["y"]),
              np.asarray(c["score"], dtype=np.float64))],
        )
        registros.append(
            {
                "regime": c["regime"], "train_corpus": c["treino"],
                "test_corpus": c["teste"], "model": c["modelo"],
                "diagonal": c["treino"] == c["teste"], "n_test": len(c["linhas"]),
                "metrics": {"threshold": c["limiar"]}, "predicoes": ref,
            }
        )
    (dir_rodada / nome_json).write_text(json.dumps(registros), encoding="utf-8")


def _par(modelo, linhas_a, linhas_b, limiar_a=0.4, limiar_b=0.35, semente=0,
         y_de=lambda linha: int(linha % 2 == 0), treino="a", teste="b"):
    """Células dos regimes A e B com escores contínuos (sem empates)."""
    rng = np.random.default_rng(semente)
    celulas = []
    for regime, linhas, limiar in (("oficial", linhas_a, limiar_a), ("etld1", linhas_b, limiar_b)):
        y = np.array([y_de(i) for i in linhas])
        celulas.append(
            {"regime": regime, "treino": treino, "teste": teste, "modelo": modelo,
             "linhas": list(linhas), "y": y, "limiar": limiar,
             "score": 0.25 * y + 0.5 * rng.random(len(y))}
        )
    return celulas


def _por_celula(resultado):
    return {(r["train_corpus"], r["test_corpus"], r["model"]): r for r in resultado}


class TestBootstrapPareado:
    def test_predicoes_identicas_dao_delta_zero_e_ic_zero(self, tmp_path):
        # 60 linhas, 30 de cada classe (o mínimo que define o IC).
        celulas = _par("m1", range(60), range(60))
        celulas[1]["score"] = celulas[0]["score"].copy()
        celulas[1]["limiar"] = celulas[0]["limiar"]
        _montar_pares(tmp_path, celulas)
        (r,) = offline.bootstrap_pareado(tmp_path, CONFIG)
        assert r["regime_par"] == "B-A"
        assert r["n_intersecao"] == 60
        assert r["delta"] == {"auroc": 0.0, "auc_pr": 0.0, "f1": 0.0}
        assert r["ic"] == {m: [0.0, 0.0] for m in ("auroc", "auc_pr", "f1")}
        assert r["ic_indefinido"] is False

    def test_ic_bate_com_o_calculo_de_referencia(self, tmp_path):
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        # A tem as linhas 0..99 e B as 20..119: a interseção é 20..99 (80 linhas,
        # 40 de cada classe, pois y é a paridade da linha).
        celulas = _par("m1", range(100), range(20, 120))
        _montar_pares(tmp_path, celulas)
        (r,) = offline.bootstrap_pareado(tmp_path, CONFIG)
        assert r["n_intersecao"] == 80
        assert r["n_por_classe"] == {"positivos": 40, "negativos": 40}

        # Referência escrita no teste: sklearn sobre as linhas comuns e o mesmo
        # protocolo de sorteio (estratificado, positivos e depois negativos).
        comuns = np.arange(20, 100)
        a, b = celulas[0], celulas[1]
        s_a = a["score"][comuns - 0]
        s_b = b["score"][comuns - 20]
        y = a["y"][comuns]

        def metricas(idx, score, limiar):
            yi, si = y[idx], score[idx]
            return np.array([roc_auc_score(yi, si), average_precision_score(yi, si),
                             f1_score(yi, (si >= limiar).astype(int))])

        todas = np.arange(len(y))
        delta = metricas(todas, s_b, 0.35) - metricas(todas, s_a, 0.4)
        assert [r["delta"][m] for m in ("auroc", "auc_pr", "f1")] == pytest.approx(delta)

        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rng = np.random.default_rng(offline.semente_da_celula(("B-A", "a", "b", "m1"), 42))
        deltas = []
        for _ in range(CONFIG["avaliacao"]["bootstrap_n"]):
            sp = pos[rng.integers(0, len(pos), size=len(pos))]
            sn = neg[rng.integers(0, len(neg), size=len(neg))]
            idx = np.concatenate([sp, sn])
            deltas.append(metricas(idx, s_b, 0.35) - metricas(idx, s_a, 0.4))
        deltas = np.array(deltas)
        for j, m in enumerate(("auroc", "auc_pr", "f1")):
            esperado = np.percentile(deltas[:, j], [2.5, 97.5])
            assert r["ic"][m] == pytest.approx(esperado, abs=1e-5)

    def test_duas_execucoes_e_ordens_diferentes_dao_o_mesmo_json(self, tmp_path):
        celulas = _par("m1", range(100), range(20, 120)) + _par(
            "m2", range(80), range(80), semente=5, teste="c"
        )
        _montar_pares(tmp_path, celulas)
        primeira = offline.bootstrap_pareado(tmp_path, CONFIG)
        segunda = offline.bootstrap_pareado(tmp_path, CONFIG)

        caminho = tmp_path / "matrix.json"
        registros = json.loads(caminho.read_text(encoding="utf-8"))
        caminho.write_text(json.dumps(registros[::-1]), encoding="utf-8")
        invertida = offline.bootstrap_pareado(tmp_path, CONFIG)

        assert json.dumps(primeira) == json.dumps(segunda) == json.dumps(invertida)
        assert [(r["train_corpus"], r["test_corpus"], r["model"]) for r in primeira] == sorted(
            (r["train_corpus"], r["test_corpus"], r["model"]) for r in primeira
        )

    def test_semente_da_celula_depende_da_chave_e_nao_da_ordem(self):
        a = offline.semente_da_celula(("B-A", "a", "b", "m"), 42)
        assert a == offline.semente_da_celula(("B-A", "a", "b", "m"), 42)
        assert a != offline.semente_da_celula(("B-A", "a", "c", "m"), 42)
        assert a != offline.semente_da_celula(("B-A", "a", "b", "m"), 7)

    def test_menos_de_30_linhas_de_uma_classe_marca_ic_indefinido(self, tmp_path):
        # 60 linhas com 10 positivas (linha % 6 == 0): a classe rara tem < 30.
        celulas = _par("m1", range(60), range(60), y_de=lambda i: int(i % 6 == 0))
        _montar_pares(tmp_path, celulas)
        (r,) = offline.bootstrap_pareado(tmp_path, CONFIG)
        assert r["n_por_classe"] == {"positivos": 10, "negativos": 50}
        assert r["ic_indefinido"] is True
        assert r["ic"] is None
        # O Delta pontual continua calculado.
        assert r["delta"]["auroc"] is not None

    def test_interseccao_vazia(self, tmp_path):
        celulas = _par("m1", range(0, 40), range(100, 140))
        _montar_pares(tmp_path, celulas)
        (r,) = offline.bootstrap_pareado(tmp_path, CONFIG)
        assert r["n_intersecao"] == 0
        assert r["ic_indefinido"] is True
        assert r["delta"] == {"auroc": None, "auc_pr": None, "f1": None}

    def test_celula_sem_um_dos_regimes_levanta_erro(self, tmp_path):
        celulas = _par("m1", range(60), range(60))[:1]
        _montar_pares(tmp_path, celulas)
        with pytest.raises(OfflineError, match="etld1"):
            offline.bootstrap_pareado(tmp_path, CONFIG)

    def test_inclui_a_diagonal_e_o_loco(self, tmp_path):
        diagonal = _par("m1", range(60), range(60), treino="a", teste="a")
        loco = _par("m2", range(60), range(60), semente=3,
                    treino=offline.LOCO_PREFIXO + "d", teste="d")
        _montar_pares(tmp_path, diagonal)
        _montar_pares(tmp_path, loco, nome_json="loco.json")
        r = _por_celula(offline.bootstrap_pareado(tmp_path, CONFIG))
        assert ("a", "a", "m1") in r
        assert (offline.LOCO_PREFIXO + "d", "d", "m2") in r


# ---------------------------------------------------------------------------
# T18: CLI e offline.json
# ---------------------------------------------------------------------------


class TestCli:
    @pytest.fixture
    def config_yml(self, tmp_path_factory):
        caminho = tmp_path_factory.mktemp("cfg") / "experiment.yml"
        caminho.write_text(
            "seed: 42\navaliacao:\n  bootstrap_n: 50\n  bootstrap_nivel: 0.95\n",
            encoding="utf-8",
        )
        return caminho

    def test_grava_offline_json_com_filtragem_bootstrap_e_fontes(self, rodada, config_yml):
        _montar(rodada / "seed_07", PART, [_ajuste("etld1"), _ajuste("oficial")])
        assert offline.main(["--results-dir", str(rodada), "--config", str(config_yml)]) == 0
        saida = json.loads((rodada / "offline.json").read_text(encoding="utf-8"))

        # `commit_codigo` entrou no achado 2 do lote F (a expectativa antiga nao o previa).
        assert set(saida) == {"gerado_em", "parametros", "commit_codigo", "fontes",
                              "filtragem", "bootstrap_pareado"}
        assert set(saida["filtragem"]["principal"]) == {"42", "7"}
        assert len(saida["filtragem"]["principal"]["42"]) == 6
        # O bootstrap pareado é só da semente canônica da rodada principal.
        assert {r["train_corpus"] for r in saida["bootstrap_pareado"]} == {"a"}
        assert len(saida["bootstrap_pareado"]) == 4  # a -> a, b, c, e
        # sha256 de cada arquivo lido, com o prefixo do diretório da rodada.
        assert "predicoes/particoes.parquet" in saida["fontes"]
        assert "seed_07/predicoes/particoes.parquet" in saida["fontes"]
        assert any(k.startswith("seed_07/predicoes/matriz/") for k in saida["fontes"])
        assert all(len(v) == 64 for v in saida["fontes"].values())

    def test_dois_gerados_em_instantes_diferentes_so_diferem_no_carimbo(self, rodada, config_yml):
        import yaml

        config = yaml.safe_load(config_yml.read_text(encoding="utf-8"))
        a = offline.executar(rodada, config, gerado_em="t1")
        b = offline.executar(rodada, config, gerado_em="t2")
        a.pop("gerado_em"), b.pop("gerado_em")
        assert json.dumps(a) == json.dumps(b)

    @staticmethod
    def _com_commit(diretorio, commit):
        caminho = diretorio / "run_meta.json"
        meta = json.loads(caminho.read_text(encoding="utf-8"))
        meta["commit_codigo"] = commit
        caminho.write_text(json.dumps(meta), encoding="utf-8")

    def test_offline_json_leva_o_commit_codigo_das_rodadas_lidas(self, rodada, config_yml):
        import yaml

        _montar(rodada / "seed_07", PART, [_ajuste("etld1"), _ajuste("oficial")])
        self._com_commit(rodada, "a" * 40)
        self._com_commit(rodada / "seed_07", "a" * 40)
        config = yaml.safe_load(config_yml.read_text(encoding="utf-8"))
        assert offline.executar(rodada, config)["commit_codigo"] == "a" * 40

    def test_rodadas_de_commits_diferentes_recusam_o_offline(self, rodada, config_yml):
        import yaml

        _montar(rodada / "seed_07", PART, [_ajuste("etld1"), _ajuste("oficial")])
        self._com_commit(rodada, "a" * 40)
        self._com_commit(rodada / "seed_07", "b" * 40)
        config = yaml.safe_load(config_yml.read_text(encoding="utf-8"))
        with pytest.raises(OfflineError, match="commit_codigo"):
            offline.executar(rodada, config)

    def test_sem_rodada_com_predicoes_sai_com_1_e_nao_grava(self, tmp_path, config_yml, capsys):
        assert offline.main(["--results-dir", str(tmp_path), "--config", str(config_yml)]) == 1
        assert not (tmp_path / "offline.json").exists()
        assert "nenhuma rodada" in capsys.readouterr().err

    def test_rodadas_explicitas(self, rodada, config_yml):
        _montar(rodada / "sensibilidade_sem_esquema", PART, [_ajuste("etld1"), _ajuste("oficial")])
        code = offline.main(["--results-dir", str(rodada), "--config", str(config_yml),
                             "--rodadas", "sensibilidade_sem_esquema"])
        assert code == 0
        saida = json.loads((rodada / "offline.json").read_text(encoding="utf-8"))
        assert list(saida["filtragem"]) == ["sensibilidade_sem_esquema"]
        assert saida["bootstrap_pareado"] == []
