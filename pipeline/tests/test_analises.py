"""T21: diferenças entre rodadas (spec REP-01 e SENS-02).

Registros sintéticos no formato do livro-caixa; os valores esperados saem de
contas feitas à mão, comentadas em cada caso. Nenhum teste lê `results/`.

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

import json
from pathlib import Path

import pytest

import analises
from analises import AnalisesError


def _rec(regime, treino, teste, modelo, **metricas):
    return {
        "regime": regime,
        "train_corpus": treino,
        "test_corpus": teste,
        "model": modelo,
        "metrics": metricas,
    }


def _rodada(regime="etld1", corpora=("c1", "c2"), modelos=("m1", "m2"), **base):
    return [
        _rec(regime, tr, te, m, **base)
        for tr in corpora
        for te in corpora
        for m in modelos
    ]


def _com(registros, chave, **novos):
    """Cópia da rodada com `novos` sobrescrevendo métricas de um registro."""
    saida = []
    for r in registros:
        copia = {**r, "metrics": dict(r["metrics"])}
        if (r["regime"], r["train_corpus"], r["test_corpus"], r["model"]) == chave:
            copia["metrics"].update(novos)
        saida.append(copia)
    return saida


# ---------------------------------------------------------------------------
# REP-01: diferenças_repeticao
# ---------------------------------------------------------------------------


def test_rodadas_iguais_exceto_numa_celula_devolvem_celula_modelo_e_diferenca():
    """REP-01: a célula alterada, o modelo e a diferença como máximo da métrica."""
    base = _rodada(auroc=0.90, auc_pr=0.80, f1=0.70, mcc=0.50)
    alterada = ("etld1", "c1", "c2", "m2")
    b = _com(base, alterada, auroc=0.90 + 0.013, f1=0.70 - 0.02)

    saida = analises.diferencas_repeticao(base, b)

    assert saida["n_celulas"] == 8
    auroc = saida["metricas"]["auroc"]
    assert auroc["diferenca_max"] == pytest.approx(0.013)
    assert (auroc["regime"], auroc["train_corpus"], auroc["test_corpus"], auroc["model"]) == alterada
    # A diferença de F1 é em módulo: queda de 0,02.
    f1 = saida["metricas"]["f1"]
    assert f1["diferenca_max"] == pytest.approx(0.02)
    assert f1["model"] == "m2" and f1["test_corpus"] == "c2"


def test_a_maior_diferenca_vence_e_cada_metrica_tem_a_sua_celula():
    base = _rodada(auroc=0.90, mcc=0.50)
    b = _com(base, ("etld1", "c1", "c1", "m1"), auroc=0.904)  # diferença 0,004
    b = _com(b, ("etld1", "c2", "c1", "m2"), auroc=0.893)  # diferença 0,007: máxima
    b = _com(b, ("etld1", "c1", "c2", "m1"), mcc=0.45)  # só o MCC muda aqui

    saida = analises.diferencas_repeticao(base, b)["metricas"]

    assert saida["auroc"]["diferenca_max"] == pytest.approx(0.007)
    assert (saida["auroc"]["train_corpus"], saida["auroc"]["test_corpus"]) == ("c2", "c1")
    assert saida["mcc"]["diferenca_max"] == pytest.approx(0.05)
    assert (saida["mcc"]["train_corpus"], saida["mcc"]["test_corpus"]) == ("c1", "c2")


def test_empate_fica_com_a_primeira_celula_em_ordem_e_nao_depende_da_ordem_de_entrada():
    base = _rodada(auroc=0.90)
    b = _com(base, ("etld1", "c2", "c2", "m1"), auroc=0.91)
    b = _com(b, ("etld1", "c1", "c1", "m2"), auroc=0.91)  # empata em 0,01

    direta = analises.diferencas_repeticao(base, b)["metricas"]["auroc"]
    invertida = analises.diferencas_repeticao(base[::-1], b[::-1])["metricas"]["auroc"]

    assert direta == invertida
    assert (direta["train_corpus"], direta["model"]) == ("c1", "m2")


def test_metrica_none_em_qualquer_execucao_sai_da_comparacao():
    base = _rodada(auroc=0.90)
    a = _com(base, ("etld1", "c1", "c1", "m1"), auroc=None)
    b = _com(base, ("etld1", "c2", "c2", "m1"), auroc=None)
    b = _com(b, ("etld1", "c1", "c2", "m2"), auroc=0.95)

    saida = analises.diferencas_repeticao(a, b)["metricas"]

    assert saida["auroc"]["diferenca_max"] == pytest.approx(0.05)
    # Métrica ausente em todos os registros: nenhuma célula comparável.
    assert saida["mcc"] is None


def test_celulas_diferentes_entre_as_execucoes_levantam_erro():
    base = _rodada(auroc=0.9)
    with pytest.raises(AnalisesError, match="mesmas células"):
        analises.diferencas_repeticao(base, base[:-1])


def test_registro_repetido_levanta_erro_nomeando_a_celula():
    base = _rodada(auroc=0.9)
    with pytest.raises(AnalisesError, match="etld1/c1/c1/m1"):
        analises.diferencas_repeticao(base + [base[0]], base)


# ---------------------------------------------------------------------------
# SENS-02: diferencas_sensibilidade
# ---------------------------------------------------------------------------


def _celulas(regime, valores):
    """`valores`: {(treino, teste): [auroc de m1, auroc de m2, ...]}."""
    return [
        _rec(regime, tr, te, f"m{i + 1}", auroc=v)
        for (tr, te), vs in valores.items()
        for i, v in enumerate(vs)
    ]


def _semente_42():
    # Regime B, dois modelos por célula.
    # Principal: diagonal (c1,c1)=média(.9,.7)=.8 e (c2,c2)=.6, média .7;
    # fora (c1,c2)=média(.5,.7)=.6 e (c2,c1)=.4, média .5.
    principal = _celulas("etld1", {
        ("c1", "c1"): [0.9, 0.7], ("c2", "c2"): [0.6, 0.6],
        ("c1", "c2"): [0.5, 0.7], ("c2", "c1"): [0.4, 0.4],
    })
    # Sem esquema: diagonal .7 e .5 (média .6); fora .5 e .3 (média .4).
    sem = _celulas("etld1", {
        ("c1", "c1"): [0.8, 0.6], ("c2", "c2"): [0.5, 0.5],
        ("c1", "c2"): [0.4, 0.6], ("c2", "c1"): [0.3, 0.3],
    })
    return principal, sem


def _semente_7():
    principal = _celulas("etld1", {
        ("c1", "c1"): [0.9], ("c2", "c2"): [0.9],
        ("c1", "c2"): [0.8], ("c2", "c1"): [0.8],
    })
    sem = _celulas("etld1", {
        ("c1", "c1"): [0.85], ("c2", "c2"): [0.85],
        ("c1", "c2"): [0.6], ("c2", "c1"): [0.6],
    })
    return principal, sem


def _regime_a():
    principal = _celulas("oficial", {("c1", "c1"): [0.95], ("c1", "c2"): [0.90]})
    sem = _celulas("oficial", {("c1", "c1"): [0.94], ("c1", "c2"): [0.85]})
    return principal, sem


def test_diferenca_de_sensibilidade_por_regime_semente_e_media_bate_com_a_conta_a_mao():
    p42, s42 = _semente_42()
    p7, s7 = _semente_7()
    pa, sa = _regime_a()
    principal = {42: p42 + pa, 7: p7 + pa, 13: p7}  # a 13 não existe na outra rodada
    sem_esquema = {42: s42 + sa, 7: s7 + sa}

    saida = analises.diferencas_sensibilidade(principal, sem_esquema)

    assert set(saida) == {"etld1", "oficial"}
    b = saida["etld1"]
    assert sorted(b["por_semente"]) == [7, 42]  # só as sementes comuns
    # Semente 42: diagonal .7 -> .6 (-0,1); fora .5 -> .4 (-0,1).
    assert b["por_semente"][42]["diagonal"] == pytest.approx(
        {"principal": 0.7, "sem_esquema": 0.6, "diferenca": -0.1}
    )
    assert b["por_semente"][42]["fora_diagonal"] == pytest.approx(
        {"principal": 0.5, "sem_esquema": 0.4, "diferenca": -0.1}
    )
    # Semente 7: diagonal .9 -> .85 (-0,05); fora .8 -> .6 (-0,2).
    assert b["por_semente"][7]["diagonal"]["diferenca"] == pytest.approx(-0.05)
    assert b["por_semente"][7]["fora_diagonal"]["diferenca"] == pytest.approx(-0.2)
    # Média das duas sementes. Diagonal: principal (.7+.9)/2 = .8, sem
    # (.6+.85)/2 = .725, diferença (-.1-.05)/2 = -.075. Fora: principal
    # (.5+.8)/2 = .65, sem (.4+.6)/2 = .5, diferença (-.1-.2)/2 = -.15.
    assert b["media"]["diagonal"] == pytest.approx(
        {"principal": 0.8, "sem_esquema": 0.725, "diferenca": -0.075}
    )
    assert b["media"]["fora_diagonal"] == pytest.approx(
        {"principal": 0.65, "sem_esquema": 0.5, "diferenca": -0.15}
    )
    # O outro regime não se mistura: .95 -> .94 e .90 -> .85 nas duas sementes.
    a = saida["oficial"]
    assert a["media"]["diagonal"]["diferenca"] == pytest.approx(-0.01)
    assert a["media"]["fora_diagonal"]["diferenca"] == pytest.approx(-0.05)


def test_grupo_sem_celula_fica_none_e_sai_da_media():
    # Só a diagonal existe na semente 7 do regime B: o grupo "fora" é None ali.
    p42, s42 = _semente_42()
    diag7 = [_rec("etld1", "c1", "c1", "m1", auroc=0.9)]
    sem7 = [_rec("etld1", "c1", "c1", "m1", auroc=0.8)]

    saida = analises.diferencas_sensibilidade({42: p42, 7: diag7}, {42: s42, 7: sem7})

    fora7 = saida["etld1"]["por_semente"][7]["fora_diagonal"]
    assert fora7 == {"principal": None, "sem_esquema": None, "diferenca": None}
    # A média de "fora" é a da semente 42 sozinha.
    assert saida["etld1"]["media"]["fora_diagonal"]["diferenca"] == pytest.approx(-0.1)


def test_sem_semente_em_comum_levanta_erro():
    p42, s42 = _semente_42()
    with pytest.raises(AnalisesError, match="semente em comum"):
        analises.diferencas_sensibilidade({42: p42}, {7: s42})


# ---------------------------------------------------------------------------
# T27, ANL-01: decomposição da variação da AUROC fora da diagonal
# ---------------------------------------------------------------------------

FIXTURE_ANTIGA = Path(__file__).parent / "fixtures" / "auroc_fora_diagonal_seed42_antiga.json"


def _registros_do_fixture():
    """As 144 AUROC congeladas da semente 42 antiga, no formato dos registros."""
    brutos = json.loads(FIXTURE_ANTIGA.read_text(encoding="utf-8"))
    return [
        _rec(b["regime"], b["train_corpus"], b["test_corpus"], b["model"], auroc=b["auroc"])
        for b in brutos
    ]


def test_decomposicao_reproduz_os_valores_do_orientador_na_semente_42_antiga():
    """ANL-01: Regime B 68,68%, 3,24% e 0,55%; Regime A 71,43%, 2,48% e 0,45%."""
    registros = _registros_do_fixture()

    b = analises.decompor_variancia(registros, "etld1")
    a = analises.decompor_variancia(registros, "oficial")

    assert b["n"] == 72 and a["n"] == 72
    assert 100 * b["treino"] == pytest.approx(68.68, abs=0.01)
    assert 100 * b["teste"] == pytest.approx(3.24, abs=0.01)
    assert 100 * b["modelo"] == pytest.approx(0.55, abs=0.01)
    assert 100 * a["treino"] == pytest.approx(71.43, abs=0.01)
    assert 100 * a["teste"] == pytest.approx(2.48, abs=0.01)
    assert 100 * a["modelo"] == pytest.approx(0.45, abs=0.01)
    # O resto é 1 menos a soma das três frações.
    assert b["resto"] == pytest.approx(1 - b["treino"] - b["teste"] - b["modelo"])


def test_decomposicao_so_modelo_explica_quando_o_valor_depende_so_do_modelo():
    # Três fontes, seis pares fora da diagonal, dois modelos; AUROC = 0,5 para
    # m1 e 0,6 para m2 em toda célula. Total = 12 * 0,05^2 = 0,03; entre os
    # modelos = 12 * 0,05^2 = 0,03, logo 100%; treino e teste, 0%.
    registros = [
        _rec("etld1", tr, te, m, auroc=0.5 if m == "m1" else 0.6)
        for tr in "abc"
        for te in "abc"
        if tr != te
        for m in ("m1", "m2")
    ]

    saida = analises.decompor_variancia(registros, "etld1")

    assert saida["modelo"] == pytest.approx(1.0)
    assert saida["treino"] == pytest.approx(0.0)
    assert saida["teste"] == pytest.approx(0.0)
    assert saida["resto"] == pytest.approx(0.0, abs=1e-12)


def test_decomposicao_so_treino_calculada_a_mao():
    # Um modelo. AUROC depende só do treino: a = 0,9, b = 0,7, c = 0,5.
    # Pares: a->b, a->c (0,9); b->a, b->c (0,7); c->a, c->b (0,5). Média 0,7;
    # soma de quadrados total = 2*0,04 + 2*0 + 2*0,04 = 0,16.
    # Treino: três níveis de 2 valores, médias 0,9, 0,7, 0,5: entre = 0,16, ou 100%.
    # Teste: a recebe (0,7; 0,5), média 0,6; b recebe (0,9; 0,5), média 0,7;
    # c recebe (0,9; 0,7), média 0,8; entre = 2*0,01 + 0 + 2*0,01 = 0,04, ou 25%.
    # As frações marginais se sobrepõem (treino e teste não são ortogonais fora
    # da diagonal), então o resto, 1 menos a soma, fica em -25%.
    valor = {"a": 0.9, "b": 0.7, "c": 0.5}
    registros = [
        _rec("etld1", tr, te, "m1", auroc=valor[tr])
        for tr in "abc"
        for te in "abc"
        if tr != te
    ]

    saida = analises.decompor_variancia(registros, "etld1")

    assert saida["n"] == 6
    assert saida["treino"] == pytest.approx(1.0)
    assert saida["teste"] == pytest.approx(0.25)
    assert saida["modelo"] == pytest.approx(0.0)
    assert saida["resto"] == pytest.approx(-0.25)


def test_decomposicao_ignora_diagonal_outro_regime_e_auroc_ausente():
    base = [
        _rec("etld1", "a", "b", "m1", auroc=0.9),
        _rec("etld1", "b", "a", "m1", auroc=0.7),
        _rec("etld1", "a", "b", "m2", auroc=0.9),
        _rec("etld1", "b", "a", "m2", auroc=0.7),
    ]
    ruido = [
        _rec("etld1", "a", "a", "m1", auroc=0.1),  # diagonal
        _rec("oficial", "a", "b", "m1", auroc=0.2),  # outro regime
        _rec("etld1", "b", "a", "m3", auroc=None),  # AUROC indefinida
    ]

    com_ruido = analises.decompor_variancia(base + ruido, "etld1")
    sem_ruido = analises.decompor_variancia(base, "etld1")

    assert com_ruido == sem_ruido
    assert com_ruido["n"] == 4


def test_decomposicao_sem_valores_ou_sem_variancia_levanta_erro():
    with pytest.raises(AnalisesError, match="nenhuma AUROC"):
        analises.decompor_variancia([_rec("etld1", "a", "a", "m1", auroc=0.9)], "etld1")
    constantes = [
        _rec("etld1", "a", "b", "m1", auroc=0.8),
        _rec("etld1", "b", "a", "m1", auroc=0.8),
    ]
    with pytest.raises(AnalisesError, match="variância nula"):
        analises.decompor_variancia(constantes, "etld1")


# ---------------------------------------------------------------------------
# T28, ANL-02: união do LOCO contra as fontes isoladas
# ---------------------------------------------------------------------------


def _matriz_uniao():
    """Teste `t` com três fontes e dois modelos; o regime oficial como ruído.

    AUROC por modelo (m1, m2) e a média por fonte, feita à mão:
      s1: 0,8 e 0,6 -> 0,7
      s2: 0,5 e 0,5 -> 0,5
      iscx2016: 0,9 e 0,9 -> 0,9
    Para o teste `iscx2016` as fontes são s1 e s2 (0,4 e 0,2 -> 0,3; 0,2 e 0,2 -> 0,2).
    """
    valores = {
        ("s1", "t"): (0.8, 0.6),
        ("s2", "t"): (0.5, 0.5),
        ("iscx2016", "t"): (0.9, 0.9),
        ("s1", "iscx2016"): (0.4, 0.2),
        ("s2", "iscx2016"): (0.2, 0.2),
    }
    registros = []
    for (tr, te), (v1, v2) in valores.items():
        registros.append(_rec("etld1", tr, te, "m1", auroc=v1))
        registros.append(_rec("etld1", tr, te, "m2", auroc=v2))
    # Diagonal e outro regime não entram.
    registros.append(_rec("etld1", "t", "t", "m1", auroc=0.99))
    registros.append(_rec("oficial", "s1", "t", "m1", auroc=0.01))
    return registros


def test_uniao_contra_fontes_melhor_pior_media_e_media_sem_iscx():
    loco = [
        _rec("etld1", "uniao_sem_t", "t", "m1", auroc=0.85),
        _rec("etld1", "uniao_sem_t", "t", "m2", auroc=0.75),
        _rec("etld1", "uniao_sem_iscx2016", "iscx2016", "m1", auroc=0.4),
        _rec("etld1", "uniao_sem_iscx2016", "iscx2016", "m2", auroc=0.2),
    ]

    saida = analises.uniao_contra_fontes(_matriz_uniao(), loco, "etld1")

    t = saida["t"]
    assert t["loco"] == pytest.approx(0.80)  # (0,85 + 0,75) / 2
    assert t["fontes"] == pytest.approx({"s1": 0.7, "s2": 0.5, "iscx2016": 0.9})
    assert t["melhor"] == {"treino": "iscx2016", "auroc": pytest.approx(0.9)}
    assert t["pior"] == {"treino": "s2", "auroc": pytest.approx(0.5)}
    assert t["media"] == pytest.approx(0.7)  # (0,7 + 0,5 + 0,9) / 3
    assert t["media_sem_iscx"] == pytest.approx(0.6)  # (0,7 + 0,5) / 2
    assert t["loco_menos_melhor"] == pytest.approx(-0.10)
    assert t["loco_menos_pior"] == pytest.approx(0.30)
    assert t["loco_menos_media"] == pytest.approx(0.10)
    assert t["loco_menos_media_sem_iscx"] == pytest.approx(0.20)
    # Teste ISCX: as fontes são s1 (0,3) e s2 (0,2); sem ISCX é igual à média.
    iscx = saida["iscx2016"]
    assert iscx["loco"] == pytest.approx(0.3)
    assert iscx["melhor"]["treino"] == "s1" and iscx["pior"]["treino"] == "s2"
    assert iscx["media"] == pytest.approx(0.25)
    assert iscx["media_sem_iscx"] == pytest.approx(0.25)


def test_uniao_contra_fontes_nao_depende_da_ordem_nem_do_regime_alheio():
    loco = [_rec("etld1", "uniao_sem_t", "t", "m1", auroc=0.8)]
    matriz = _matriz_uniao()

    a = analises.uniao_contra_fontes(matriz, loco, "etld1")
    b = analises.uniao_contra_fontes(list(reversed(matriz)), loco, "etld1")

    assert a == b
    # Empate de AUROC entre fontes: o desempate é pelo nome, nas duas ordens.
    empate = [_rec("etld1", tr, "t", "m1", auroc=0.6) for tr in ("zz", "aa")]
    assert analises.uniao_contra_fontes(empate, loco, "etld1")["t"]["melhor"]["treino"] == "aa"
    assert analises.uniao_contra_fontes(empate[::-1], loco, "etld1")["t"]["melhor"]["treino"] == "aa"


def test_uniao_sem_loco_deixa_as_diferencas_em_none():
    saida = analises.uniao_contra_fontes(_matriz_uniao(), [], "etld1")

    assert saida["t"]["loco"] is None
    assert saida["t"]["loco_menos_melhor"] is None
    assert saida["t"]["media"] == pytest.approx(0.7)


def _resumo_split(n_train):
    return {"n_train": n_train, "n_val": 1, "n_test": 1}


def test_composicao_da_uniao_por_dataset_de_origem():
    # Três datasets com 100, 300 e 600 linhas de treino depois da dedup.
    # Retido `a`: a união é b (300) + c (600) = 900, frações 1/3 e 2/3.
    # Retido `c`: a (100) + b (300) = 400, frações 0,25 e 0,75.
    splits = {
        "a/etld1": _resumo_split(100),
        "b/etld1": _resumo_split(300),
        "c/etld1": _resumo_split(600),
        "a/oficial": _resumo_split(10),
        "b/oficial": _resumo_split(20),
    }

    saida = analises.composicao_uniao(splits)

    a = saida["etld1"]["a"]
    assert a["total"] == 900
    assert a["fontes"]["b"] == {"n_train": 300, "fracao": pytest.approx(1 / 3)}
    assert a["fontes"]["c"] == {"n_train": 600, "fracao": pytest.approx(2 / 3)}
    assert "a" not in a["fontes"]
    c = saida["etld1"]["c"]
    assert c["total"] == 400
    assert c["fontes"]["a"]["fracao"] == pytest.approx(0.25)
    assert c["fontes"]["b"]["fracao"] == pytest.approx(0.75)
    # O outro regime tem o seu próprio resumo.
    assert saida["oficial"]["a"] == {
        "total": 20, "fontes": {"b": {"n_train": 20, "fracao": 1.0}},
    }


# ---------------------------------------------------------------------------
# T29, ANL-04: células degeneradas
# ---------------------------------------------------------------------------


def _ajuste(treino, teste, modelo, positivos_previstos, n_test=100, regime="etld1", **metricas):
    """Registro com `positivos_previstos` = tp + fp entre `n_test` linhas."""
    tp = positivos_previstos // 2
    registro = _rec(
        regime, treino, teste, modelo,
        tp=tp, fp=positivos_previstos - tp, tn=0, fn=0, **metricas,
    )
    registro["n_test"] = n_test
    return registro


def test_taxa_positiva_004_e_096_sao_degeneradas_005_e_095_nao():
    registros = [
        _ajuste("a", "b", "m1", 4),    # 0,04
        _ajuste("a", "b", "m2", 96),   # 0,96
        _ajuste("a", "b", "m3", 5),    # 0,05: no limite, não degenerada
        _ajuste("a", "b", "m4", 95),   # 0,95: no limite, não degenerada
        _ajuste("a", "b", "m5", 50),
    ]

    saida = analises.celulas_degeneradas(registros)

    assert [(c["model"], c["taxa_positiva"]) for c in saida] == [("m1", 0.04), ("m2", 0.96)]


def test_degenerada_traz_f1_mcc_e_ignora_a_diagonal():
    registros = [
        _ajuste("a", "b", "m1", 100, f1=0.667, mcc=0.0),   # tudo positivo
        _ajuste("a", "a", "m1", 100, f1=0.667, mcc=0.0),   # diagonal: fora
        _ajuste("uniao_sem_b", "b", "m1", 1, f1=0.02, mcc=0.01),  # LOCO conta como fora
    ]

    saida = analises.celulas_degeneradas(registros)

    assert [(c["train_corpus"], c["test_corpus"]) for c in saida] == [
        ("a", "b"), ("uniao_sem_b", "b"),
    ]
    assert saida[0]["taxa_positiva"] == pytest.approx(1.0)
    assert saida[0]["f1"] == 0.667 and saida[0]["mcc"] == 0.0 and saida[0]["n_test"] == 100


def test_limite_configuravel_e_registro_sem_contagens_fica_de_fora():
    registros = [
        _ajuste("a", "b", "m1", 8),  # 0,08
        _rec("etld1", "a", "c", "m1", f1=0.5),  # sem tp, fp e n_test
    ]
    registros[1]["n_test"] = 100

    assert analises.celulas_degeneradas(registros) == []
    assert len(analises.celulas_degeneradas(registros, limite=0.10)) == 1


# ---------------------------------------------------------------------------
# T29, ANL-06: métricas do critério de efeito
# ---------------------------------------------------------------------------


def _par(treino, teste, modelo, delta_f1, ic_f1):
    return {
        "regime_par": "B-A",
        "train_corpus": treino,
        "test_corpus": teste,
        "model": modelo,
        "n_intersecao": 1000,
        "delta": {"auroc": 0.0, "auc_pr": 0.0, "f1": delta_f1},
        "ic": None if ic_f1 is None else {"auroc": [0.0, 0.0], "auc_pr": [0.0, 0.0], "f1": ic_f1},
        "ic_indefinido": ic_f1 is None,
    }


def test_metricas_de_efeito_erro_razao_e_delta_f1_com_ic():
    matriz = [
        _rec("oficial", "c", "c", "m1", auroc=0.95),
        _rec("oficial", "c", "c", "m2", auroc=0.97),   # média A = 0,96, erro 0,04
        _rec("etld1", "c", "c", "m1", auroc=0.85),
        _rec("etld1", "c", "c", "m2", auroc=0.87),     # média B = 0,86, erro 0,14
        _rec("oficial", "c", "d", "m1", auroc=0.5),    # fora da diagonal: não entra
    ]
    loco = [
        _rec("oficial", "uniao_sem_c", "c", "m1", auroc=0.90),   # erro A 0,10
        _rec("etld1", "uniao_sem_c", "c", "m1", auroc=0.80),     # erro B 0,20
    ]
    offline = {
        "bootstrap_pareado": [
            _par("c", "c", "m1", -0.02, [-0.03, -0.01]),
            _par("c", "c", "m2", -0.04, [-0.05, 0.01]),
            _par("uniao_sem_c", "c", "m1", -0.10, None),
            _par("c", "d", "m1", -0.50, [-0.6, -0.4]),   # fora da diagonal: não entra
        ]
    }

    saida = analises.metricas_efeito(offline, matriz, loco)

    diag = saida["diagonal"]["c"]
    assert diag["erro_a"] == pytest.approx(0.04)
    assert diag["erro_b"] == pytest.approx(0.14)
    assert diag["razao"] == pytest.approx(3.5)            # 0,14 / 0,04
    assert diag["delta_f1"] == pytest.approx(-0.03)       # (-0,02 + -0,04) / 2
    assert diag["n_modelos_ic"] == 2
    assert diag["n_modelos_ic_exclui_zero"] == 1          # só o de m1; o de m2 cruza zero
    assert diag["ic_f1_envelope"] == pytest.approx([-0.05, 0.01])
    assert diag["por_modelo"]["m1"] == {
        "delta_f1": -0.02, "ic": [-0.03, -0.01], "ic_indefinido": False,
    }
    uniao = saida["loco"]["c"]
    assert uniao["erro_a"] == pytest.approx(0.10)
    assert uniao["erro_b"] == pytest.approx(0.20)
    assert uniao["razao"] == pytest.approx(2.0)
    assert uniao["delta_f1"] == pytest.approx(-0.10)
    # IC indefinido: sem modelo com IC, sem envelope, e o modelo marcado.
    assert uniao["n_modelos_ic"] == 0 and uniao["ic_f1_envelope"] is None
    assert uniao["por_modelo"]["m1"]["ic_indefinido"] is True
    assert uniao["por_modelo"]["m1"]["ic"] is None
    # A célula fora da diagonal não gera entrada.
    assert set(saida["diagonal"]) == {"c"}


def test_razao_fica_none_quando_o_erro_do_regime_a_e_nulo():
    matriz = [
        _rec("oficial", "c", "c", "m1", auroc=1.0),
        _rec("etld1", "c", "c", "m1", auroc=0.9),
    ]

    saida = analises.metricas_efeito({}, matriz, [])

    assert saida["diagonal"]["c"]["erro_a"] == 0.0
    assert saida["diagonal"]["c"]["razao"] is None
    assert saida["diagonal"]["c"]["delta_f1"] is None
