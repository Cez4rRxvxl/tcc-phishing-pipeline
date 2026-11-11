"""Análises sem treino sobre os registros das rodadas (frente A, fase 1 e F04).

Funções puras: recebem as listas de registros de `matrix.json` e `loco.json`
(o formato do livro-caixa, um registro por regime, treino, teste e modelo) e
devolvem dicionários. Nada aqui lê arquivo, treina modelo ou formata texto: o
`report.py` carrega os JSON, chama estas funções e só formata o resultado. Assim
o `make tabelas` continua rodando sem predições e as contas ficam testáveis.

Esta primeira parte traz as duas comparações entre rodadas:

- `diferencas_repeticao` (REP-01, REEX-12): a maior diferença absoluta por
  métrica entre duas execuções da mesma semente, e a célula onde ocorre;
- `diferencas_sensibilidade` (SENS-02, REEX-05): a AUROC média por regime da
  rodada principal e da rodada sem esquema, e a diferença, por semente e na
  média das sementes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# Métricas contínuas comparadas entre as duas execuções. Os limiares e as
# contagens da matriz de confusão ficam de fora: o limiar é escolhido na
# validação e `tp`, `fp`, `tn`, `fn` são inteiros que seguem dele.
METRICAS_COMPARADAS = (
    "auroc", "auc_pr", "precision", "recall", "f1", "fpr", "mcc", "brier", "ece",
)

# Grupos de células da matriz: a diagonal (treino e teste no mesmo dataset) e
# o resto.
GRUPOS = ("diagonal", "fora_diagonal")


class AnalisesError(ValueError):
    """Entrada incoerente: nunca devolve resultado pela metade."""


def _chave(registro: Mapping) -> tuple[str, str, str, str]:
    return (
        registro["regime"],
        registro["train_corpus"],
        registro["test_corpus"],
        registro["model"],
    )


def _indexar(registros: Sequence[Mapping], nome: str) -> dict:
    indice: dict = {}
    for r in registros:
        chave = _chave(r)
        if chave in indice:
            raise AnalisesError(f"{nome}: registro repetido para {'/'.join(chave)}")
        indice[chave] = r
    return indice


def _media(valores) -> float | None:
    limpos = [v for v in valores if v is not None]
    return sum(limpos) / len(limpos) if limpos else None


# ---------------------------------------------------------------------------
# Repetição da semente 42 (REP-01)
# ---------------------------------------------------------------------------


def diferencas_repeticao(a: Sequence[Mapping], b: Sequence[Mapping]) -> dict:
    """Maior diferença absoluta por métrica entre duas execuções da mesma semente.

    `a` e `b` são as listas de registros das duas rodadas (a da matriz e a do
    LOCO juntas: a chave inclui o dataset de treino, que no LOCO é a união).
    Devolve::

        {"n_celulas": int,
         "metricas": {metrica: {"diferenca_max", "regime", "train_corpus",
                                "test_corpus", "model"} | None}}

    A célula de uma métrica é a primeira, em ordem de (regime, treino, teste,
    modelo), onde a diferença é máxima, então o empate não depende da ordem dos
    registros. Métrica `None` em qualquer das duas execuções tira a célula
    daquela métrica; uma métrica sem nenhuma célula comparável vem `None`.
    As duas rodadas têm de cobrir exatamente as mesmas células.
    """
    ia = _indexar(a, "primeira execução")
    ib = _indexar(b, "segunda execução")
    if ia.keys() != ib.keys():
        apenas_a = sorted(ia.keys() - ib.keys())
        apenas_b = sorted(ib.keys() - ia.keys())
        raise AnalisesError(
            "as duas execuções não cobrem as mesmas células: "
            f"{len(apenas_a)} só na primeira (ex.: {apenas_a[:1]}), "
            f"{len(apenas_b)} só na segunda (ex.: {apenas_b[:1]})"
        )

    metricas: dict = {}
    for metrica in METRICAS_COMPARADAS:
        melhor = None
        for chave in sorted(ia):
            va = ia[chave]["metrics"].get(metrica)
            vb = ib[chave]["metrics"].get(metrica)
            if va is None or vb is None:
                continue
            dif = abs(va - vb)
            if melhor is None or dif > melhor[0]:
                melhor = (dif, chave)
        if melhor is None:
            metricas[metrica] = None
            continue
        dif, (regime, treino, teste, modelo) = melhor
        metricas[metrica] = {
            "diferenca_max": dif,
            "regime": regime,
            "train_corpus": treino,
            "test_corpus": teste,
            "model": modelo,
        }
    return {"n_celulas": len(ia), "metricas": metricas}


# ---------------------------------------------------------------------------
# Sensibilidade sem esquema (SENS-02)
# ---------------------------------------------------------------------------


def _auroc_por_grupo(registros: Sequence[Mapping], regime: str) -> dict:
    """AUROC média de um regime: média dos modelos por célula, depois das células.

    É a mesma convenção do restante do relatório (a célula é a média dos
    modelos). Devolve `{"diagonal": x | None, "fora_diagonal": y | None}`.
    """
    por_celula: dict = {}
    for r in registros:
        if r["regime"] != regime:
            continue
        valor = r["metrics"].get("auroc")
        if valor is None:
            continue
        por_celula.setdefault((r["train_corpus"], r["test_corpus"]), []).append(valor)
    medias = {celula: sum(v) / len(v) for celula, v in por_celula.items()}
    return {
        "diagonal": _media([v for (tr, te), v in medias.items() if tr == te]),
        "fora_diagonal": _media([v for (tr, te), v in medias.items() if tr != te]),
    }


def _par(principal: float | None, sem_esquema: float | None) -> dict:
    diferenca = None if principal is None or sem_esquema is None else sem_esquema - principal
    return {"principal": principal, "sem_esquema": sem_esquema, "diferenca": diferenca}


def diferencas_sensibilidade(
    principal: Mapping[int, Sequence[Mapping]],
    sem_esquema: Mapping[int, Sequence[Mapping]],
) -> dict:
    """AUROC média por regime nas duas rodadas e a diferença (sem esquema menos principal).

    `principal` e `sem_esquema` mapeiam a semente aos registros da matriz. Só as
    sementes presentes nas duas entram, em ordem crescente. Devolve, por regime::

        {"por_semente": {semente: {grupo: {"principal", "sem_esquema", "diferenca"}}},
         "media": {grupo: {"principal", "sem_esquema", "diferenca"}}}

    com `grupo` em `diagonal` e `fora_diagonal`. A média é a das sementes
    comuns; a diferença média é a média das diferenças por semente. Valor
    indefinido (grupo sem célula com AUROC) fica `None` e sai da média.
    """
    sementes = sorted(set(principal) & set(sem_esquema))
    if not sementes:
        raise AnalisesError(
            "a rodada principal e a sem esquema não têm semente em comum "
            f"({sorted(principal)} contra {sorted(sem_esquema)})"
        )
    regimes = sorted(
        {r["regime"] for s in sementes for r in principal[s]}
        | {r["regime"] for s in sementes for r in sem_esquema[s]}
    )
    saida: dict = {}
    for regime in regimes:
        por_semente = {}
        for s in sementes:
            base = _auroc_por_grupo(principal[s], regime)
            nova = _auroc_por_grupo(sem_esquema[s], regime)
            por_semente[s] = {g: _par(base[g], nova[g]) for g in GRUPOS}
        media = {}
        for g in GRUPOS:
            pares = [por_semente[s][g] for s in sementes]
            media[g] = {
                "principal": _media([p["principal"] for p in pares]),
                "sem_esquema": _media([p["sem_esquema"] for p in pares]),
                "diferenca": _media([p["diferenca"] for p in pares]),
            }
        saida[regime] = {"por_semente": por_semente, "media": media}
    return saida


# ---------------------------------------------------------------------------
# Decomposição da variação da AUROC fora da diagonal (ANL-01)
# ---------------------------------------------------------------------------

# Fator da decomposição e a chave do registro que o define.
FATORES_VARIANCIA = (
    ("treino", "train_corpus"),
    ("teste", "test_corpus"),
    ("modelo", "model"),
)


def decompor_variancia(registros: Sequence[Mapping], regime: str) -> dict:
    """Fração da soma de quadrados da AUROC fora da diagonal explicada por fator.

    Usa as células fora da diagonal (treino diferente de teste) de um regime,
    um valor por (treino, teste, modelo); registro sem AUROC fica de fora. Para
    cada fator (fonte de treino, fonte de teste, modelo) a fração é a soma de
    quadrados entre os níveis do fator, `sum(n_nivel * (media_nivel - media)^2)`,
    dividida pela soma de quadrados total, `sum((x - media)^2)`. A fração é
    marginal: a matriz fora da diagonal não é um cruzamento completo (o teste
    nunca é o treino), então uma decomposição sequencial dependeria da ordem dos
    fatores, e a marginal não depende. O resto é `1 - soma` e junta interação e
    resíduo. Devolve frações entre 0 e 1::

        {"n": int, "treino": f, "teste": f, "modelo": f, "resto": f}

    Sem valores, ou com variância total nula, levanta `AnalisesError`.
    """
    valores = [
        r
        for r in registros
        if r["regime"] == regime
        and r["train_corpus"] != r["test_corpus"]
        and r["metrics"].get("auroc") is not None
    ]
    if not valores:
        raise AnalisesError(f"decomposição: nenhuma AUROC fora da diagonal no regime {regime}")
    xs = [r["metrics"]["auroc"] for r in valores]
    media = sum(xs) / len(xs)
    total = sum((x - media) ** 2 for x in xs)
    if total == 0:
        raise AnalisesError(f"decomposição: variância nula no regime {regime}")

    saida: dict = {"n": len(xs)}
    for nome, campo in FATORES_VARIANCIA:
        por_nivel: dict = {}
        for r, x in zip(valores, xs):
            por_nivel.setdefault(r[campo], []).append(x)
        entre = sum(len(v) * (sum(v) / len(v) - media) ** 2 for v in por_nivel.values())
        saida[nome] = entre / total
    saida["resto"] = 1 - sum(saida[nome] for nome, _ in FATORES_VARIANCIA)
    return saida


# ---------------------------------------------------------------------------
# União do LOCO contra as fontes isoladas (ANL-02)
# ---------------------------------------------------------------------------

# O dataset que o orientador pede para tirar da média das fontes (ANL-02).
CORPUS_ISCX = "iscx2016"


def _auroc_por_celula(registros: Sequence[Mapping], regime: str) -> dict:
    """`{(treino, teste): média dos modelos}` da AUROC de um regime."""
    acumulado: dict = {}
    for r in registros:
        if r["regime"] != regime:
            continue
        valor = r["metrics"].get("auroc")
        if valor is None:
            continue
        acumulado.setdefault((r["train_corpus"], r["test_corpus"]), []).append(valor)
    return {k: sum(v) / len(v) for k, v in acumulado.items()}


def uniao_contra_fontes(
    matriz: Sequence[Mapping], loco: Sequence[Mapping], regime: str
) -> dict:
    """AUROC do LOCO contra a das fontes isoladas, por dataset de teste (ANL-02).

    A AUROC de uma fonte é a média dos modelos na célula (treino na fonte,
    teste no dataset); a do LOCO é a média dos modelos treinados na união das
    outras três fontes. As fontes de um teste são os datasets de treino
    diferentes dele na matriz. Devolve, por dataset de teste::

        {"loco": x | None,
         "fontes": {treino: auroc},
         "melhor": {"treino", "auroc"}, "pior": {"treino", "auroc"},
         "media": m, "media_sem_iscx": m_sem,
         "loco_menos_melhor": d, "loco_menos_pior": d,
         "loco_menos_media": d, "loco_menos_media_sem_iscx": d}

    `media_sem_iscx` tira o ISCX-URL2016 das fontes; num teste que já é o ISCX
    ela é igual à `media`. O desempate entre fontes de mesma AUROC é pelo nome
    do dataset, para que a escolha não dependa da ordem dos registros. Teste
    sem nenhuma fonte com AUROC fica de fora; diferença sem LOCO fica `None`.
    """
    celulas = _auroc_por_celula(matriz, regime)
    por_loco: dict = {}
    for (_uniao, teste), valor in _auroc_por_celula(loco, regime).items():
        por_loco.setdefault(teste, []).append(valor)
    loco_por_teste = {t: sum(v) / len(v) for t, v in por_loco.items()}

    saida: dict = {}
    for teste in sorted({te for _tr, te in celulas}):
        fontes = {tr: v for (tr, te), v in sorted(celulas.items()) if te == teste and tr != te}
        if not fontes:
            continue
        ordem = sorted(fontes.items(), key=lambda kv: (kv[1], kv[0]))
        pior_tr, pior = ordem[0]
        melhor_tr, melhor = max(sorted(fontes.items()), key=lambda kv: kv[1])
        media = sum(fontes.values()) / len(fontes)
        sem_iscx = [v for tr, v in fontes.items() if tr != CORPUS_ISCX]
        media_sem = sum(sem_iscx) / len(sem_iscx) if sem_iscx else None
        uniao = loco_por_teste.get(teste)

        def dif(ref, _u=uniao):
            return None if _u is None or ref is None else _u - ref

        saida[teste] = {
            "loco": uniao,
            "fontes": fontes,
            "melhor": {"treino": melhor_tr, "auroc": melhor},
            "pior": {"treino": pior_tr, "auroc": pior},
            "media": media,
            "media_sem_iscx": media_sem,
            "loco_menos_melhor": dif(melhor),
            "loco_menos_pior": dif(pior),
            "loco_menos_media": dif(media),
            "loco_menos_media_sem_iscx": dif(media_sem),
        }
    return saida


def composicao_uniao(splits_json: Mapping) -> dict:
    """Linhas de treino por dataset de origem na união de cada LOCO (ANL-02).

    `splits_json` é o `splits.json` da rodada (chaves `dataset/regime`, com
    `n_train` já depois da deduplicação). A união que treina para reter o
    dataset `r` é a das partições de treino dos outros. Devolve, por regime e
    por dataset retido::

        {regime: {retido: {"total": n, "fontes": {origem: {"n_train", "fracao"}}}}}
    """
    por_regime: dict = {}
    for chave, resumo in splits_json.items():
        corpus, _, regime = chave.rpartition("/")
        por_regime.setdefault(regime, {})[corpus] = resumo["n_train"]
    saida: dict = {}
    for regime, n_train in sorted(por_regime.items()):
        retidos: dict = {}
        for retido in sorted(n_train):
            outros = {c: n for c, n in sorted(n_train.items()) if c != retido}
            total = sum(outros.values())
            retidos[retido] = {
                "total": total,
                "fontes": {
                    c: {"n_train": n, "fracao": n / total if total else None}
                    for c, n in outros.items()
                },
            }
        saida[regime] = retidos
    return saida


# ---------------------------------------------------------------------------
# Células degeneradas (ANL-04)
# ---------------------------------------------------------------------------

LIMITE_DEGENERADA = 0.05
_TOLERANCIA = 1e-12


def taxa_predicao_positiva(registro: Mapping) -> float | None:
    """`(tp + fp) / n_test`; `None` sem `n_test` ou sem contagens."""
    n_test = registro.get("n_test")
    m = registro["metrics"]
    if not n_test or m.get("tp") is None or m.get("fp") is None:
        return None
    return (m["tp"] + m["fp"]) / n_test


def celulas_degeneradas(registros: Sequence[Mapping], limite: float = LIMITE_DEGENERADA) -> list:
    """Ajustes fora da diagonal que predizem quase tudo numa classe só (ANL-04).

    Um ajuste é degenerado quando a taxa de predição positiva é menor que
    `limite` ou maior que `1 - limite`; os valores exatamente iguais ao limite
    não são degenerados. É o caso do F1 próximo de 0,667 com MCC perto de zero:
    o classificador prediz tudo como phishing numa amostra de prevalência 0,5.
    Considera só os registros fora da diagonal (`train_corpus` diferente de
    `test_corpus`; a união do LOCO conta como fora). Devolve a lista, ordenada
    por (regime, treino, teste, modelo), de::

        {"regime", "train_corpus", "test_corpus", "model",
         "taxa_positiva", "f1", "mcc", "n_test"}
    """
    saida = []
    for r in sorted(registros, key=_chave):
        if r["train_corpus"] == r["test_corpus"]:
            continue
        taxa = taxa_predicao_positiva(r)
        if taxa is None:
            continue
        if taxa < limite - _TOLERANCIA or taxa > 1 - limite + _TOLERANCIA:
            saida.append(
                {
                    "regime": r["regime"],
                    "train_corpus": r["train_corpus"],
                    "test_corpus": r["test_corpus"],
                    "model": r["model"],
                    "taxa_positiva": taxa,
                    "f1": r["metrics"].get("f1"),
                    "mcc": r["metrics"].get("mcc"),
                    "n_test": r["n_test"],
                }
            )
    return saida


# ---------------------------------------------------------------------------
# Métricas do critério de efeito (ANL-06)
# ---------------------------------------------------------------------------

REGIME_A = "oficial"
REGIME_B = "etld1"
PREFIXO_UNIAO = "uniao_sem_"


def _ic_exclui_zero(ic) -> bool:
    return ic[0] > 0 or ic[1] < 0


def metricas_efeito(offline: Mapping, matriz: Sequence[Mapping], loco: Sequence[Mapping]) -> dict:
    """Erro de ordenação por regime, razão e Delta F1 com IC, por célula (ANL-06).

    Para cada célula da diagonal e do LOCO (um valor por regime é a média dos
    modelos): `erro_a` e `erro_b` são `1 - AUROC` nos Regimes A e B, `razao` é
    `erro_b / erro_a` (maior que 1 quando o Regime B ordena pior) e `delta_f1`
    é a média dos modelos do Delta de F1 (B menos A) do *bootstrap* pareado em
    `offline["bootstrap_pareado"]`. A média de intervalos não é o intervalo da
    média, então o IC fica por modelo (`por_modelo`), junto com a contagem dos
    que excluem zero e o envelope (menor limite inferior, maior limite
    superior), que só delimita onde estão os ICs. Devolve::

        {"diagonal": {corpus: celula}, "loco": {retido: celula}}
        celula = {"auroc_a", "auroc_b", "erro_a", "erro_b", "razao",
                  "delta_f1", "n_modelos_ic", "n_modelos_ic_exclui_zero",
                  "ic_f1_envelope", "por_modelo": {modelo: {"delta_f1", "ic", "ic_indefinido"}}}

    Valor sem dado (célula fora do *bootstrap*, erro de ordenação nulo) fica
    `None`.
    """
    aurocs = {
        "diagonal": {
            (r, te): v
            for r in (REGIME_A, REGIME_B)
            for (tr, te), v in _auroc_por_celula(matriz, r).items()
            if tr == te
        },
        "loco": {
            (r, te): v
            for r in (REGIME_A, REGIME_B)
            for (tr, te), v in _auroc_por_celula(loco, r).items()
        },
    }
    por_celula: dict = {}
    for e in offline.get("bootstrap_pareado") or []:
        tr, te = e["train_corpus"], e["test_corpus"]
        if tr.startswith(PREFIXO_UNIAO):
            grupo = "loco"
        elif tr == te:
            grupo = "diagonal"
        else:
            continue
        por_celula.setdefault((grupo, te), []).append(e)

    saida: dict = {"diagonal": {}, "loco": {}}
    for grupo in ("diagonal", "loco"):
        testes = {te for (_r, te) in aurocs[grupo]} | {te for (g, te) in por_celula if g == grupo}
        for te in sorted(testes):
            auroc_a = aurocs[grupo].get((REGIME_A, te))
            auroc_b = aurocs[grupo].get((REGIME_B, te))
            erro_a = None if auroc_a is None else 1 - auroc_a
            erro_b = None if auroc_b is None else 1 - auroc_b
            razao = None if not erro_a or erro_b is None else erro_b / erro_a
            lista = sorted(por_celula.get((grupo, te), []), key=lambda e: e["model"])
            deltas = [e["delta"].get("f1") for e in lista]
            definidos = [e for e in lista if not e["ic_indefinido"] and e.get("ic")]
            limites = [e["ic"]["f1"] for e in definidos if e["ic"].get("f1")]
            saida[grupo][te] = {
                "auroc_a": auroc_a,
                "auroc_b": auroc_b,
                "erro_a": erro_a,
                "erro_b": erro_b,
                "razao": razao,
                "delta_f1": _media(deltas),
                "n_modelos_ic": len(limites),
                "n_modelos_ic_exclui_zero": sum(_ic_exclui_zero(ic) for ic in limites),
                "ic_f1_envelope": (
                    [min(ic[0] for ic in limites), max(ic[1] for ic in limites)]
                    if limites else None
                ),
                "por_modelo": {
                    e["model"]: {
                        "delta_f1": e["delta"].get("f1"),
                        "ic": None if e["ic_indefinido"] else (e.get("ic") or {}).get("f1"),
                        "ic_indefinido": bool(e["ic_indefinido"]),
                    }
                    for e in lista
                },
            }
    return saida


__all__ = [
    "AnalisesError",
    "LIMITE_DEGENERADA",
    "celulas_degeneradas",
    "metricas_efeito",
    "taxa_predicao_positiva",
    "CORPUS_ISCX",
    "FATORES_VARIANCIA",
    "GRUPOS",
    "METRICAS_COMPARADAS",
    "composicao_uniao",
    "uniao_contra_fontes",
    "decompor_variancia",
    "diferencas_repeticao",
    "diferencas_sensibilidade",
]
