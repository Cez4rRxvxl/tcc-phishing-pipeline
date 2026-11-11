"""T2.3 — viés de construção e vazamento de domínio (SDD §3.2, §3.3, §13.4-7).

Este módulo produz as **duas tabelas do artigo que não dependem do Lucas**:

    Tabela 1 (T1, prioridade P1) → vazamento de domínio nos splits oficiais
    Tabela 5 (T5, prioridade P2) → gap de viés de construção nos 6 corpora

e mais um número que a §13.4-7 do SDD registra como pendência fechada por
tarefa: a **matriz 4×4 de sobreposição de domínio entre corpora distintos**,
que a seção de Ameaças à Validade (T5.6) precisa citar. Antes desta tarefa,
`splits.cross_corpus_domain_overlap()` existia e não tinha chamador nenhum —
o número não chegava a `results/` e a R5 (todo número do artigo sai do
código) seria violada na hora de escrever a seção.

Uso:

    venv/Scripts/python.exe -m src.diagnostics          # grava results/diagnostics.json

Roda nesta máquina sem violar SDD §0: é canonização de URL e contagem de
conjuntos, não treino. Nenhum `.fit()` aparece aqui. Custo medido: ~6 min
para os 6 corpora (2,55 M de URLs), dominado pela canonização.

⚠️ Sobre os números da §3.2/§3.3 do SDD
---------------------------------------
As tabelas originais foram medidas em 04/08/2026 por um script de auditoria
*ad hoc* que nunca entrou no repositório. Ao implementar esta tarefa, as
definições foram reconstruídas e conferidas contra aquelas tabelas. Duas
divergências apareceram e estão registradas em SDD §11:

1. O "eTLD+1" da auditoria original era, de fato, o heurístico ingênuo dos
   **dois últimos rótulos do host** (`.'.join(host.split('.')[-2:])`), que
   trata `co.uk` como se fosse um domínio registrável e funde
   `a.co.uk` com `b.co.uk`. É exatamente o defeito que a T3.4 manda corrigir
   com `tldextract`. Este módulo usa o eTLD+1 **de verdade**, o mesmo que
   `splits.py` usa para particionar no Regime B — se a Tabela 1 contasse
   "domínio" de um jeito e o split de outro, o artigo estaria afirmando que
   o Regime B corrige um vazamento medido sob outra definição.
2. O gap de `path` do PDRCNN foi medido antes da reconstrução das linhas
   coladas da T2.1 (~2,9% das linhas daquele corpus), com um critério de
   path mais frouxo (qualquer `/` depois do host, inclusive barra final).

Os valores reproduzíveis são os que este módulo grava. Quem escrever o
artigo lê `results/diagnostics.json`, nunca a tabela do SDD (R5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# `src/` no path, mesma convenção de `scripts/run_smoke.py` e
# `src/features/__init__.py`: funciona tanto em `python -m src.diagnostics`
# quanto em `import diagnostics` (pytest usa `pythonpath = ["src"]`).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import dedup  # noqa: E402
import evaluate  # noqa: E402
import models  # noqa: E402
import psl  # noqa: E402
import splits  # noqa: E402
from data.url_canonicalization import canonicalize_url  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SRC_DIR.parent
DEFAULT_CORPORA_DIR = REPO_ROOT / "corpora"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / "diagnostics.json"

# Ordem canônica dos 6 corpora na saída: os 4 de treino/avaliação (D2)
# primeiro, depois os 2 de demonstração de viés. A ordem dentro de cada
# tabela é reordenada por magnitude (ver `executar`).
TODOS_OS_CORPORA = ("grambeddings", "kaggle03", "phishstorm", "iscx2016", "pdrcnn", "sid321axn")

# Limiar que separa "limpo" de "contaminado" no veredito da Tabela 5.
#
# Não é um número escolhido para produzir o resultado desejado: a
# distribuição medida do gap máximo é bimodal com um vazio enorme no meio
# (os quatro corpora limpos ficam abaixo de 7 p; os dois contaminados acima
# de 59 p). Qualquer limiar entre ~7 e ~59 produz a mesma partição — 20 p é
# apenas um valor redondo dentro desse vazio. Ele está aqui, e é gravado no
# JSON de saída, para que a regra seja auditável em vez de implícita.
LIMIAR_VIES_P = 20.0

# Schemes que contam como "URL distribuída com scheme explícito".
SCHEMES = ("http://", "https://", "ftp://")


class DiagnosticsError(RuntimeError):
    """Falha estrutural no diagnóstico — nunca deve ser silenciada."""


# ---------------------------------------------------------------------------
# Atributos estruturais do viés de construção (SDD §3.2)
# ---------------------------------------------------------------------------


def tem_scheme(url: str) -> bool:
    """A URL foi distribuída com scheme explícito?

    Teste de **prefixo**, não de substring: `httpsecure.com/x` não tem scheme
    algum, e tratá-la como tendo é o bug B14 do extrator (SDD §3.1).

    A presença de scheme é o atributo de viés mais forte que medimos: no
    `dataset.csv` legado o gap chegava a ~98 p, porque os benignos eram uma
    lista top-1M de domínios nus e os maliciosos vinham de feed com URL
    completa. A regra trivial "contém `://`?" resolvia o corpus.
    """
    if not isinstance(url, str):
        return False
    return url.lower().startswith(SCHEMES)


def tem_path(url: str) -> bool:
    """A URL tem caminho de verdade depois do host?

    Critério: existe uma `/` depois do host **e algo depois dela**. Uma barra
    final solta (`http://exemplo.com/`) não conta como path — ela é sintaxe,
    não conteúdo, e contá-la infla o atributo justamente nos corpora cujo
    lado phishing vem de feed (que costuma normalizar a barra final).

    Implementação textual, deliberadamente sem `urlparse`: no Python 3.14
    `urlparse` **levanta ValueError** em host malformado com bytes binários
    (`_check_bracketed_host`), e o corpus `sid321axn` tem 18 linhas assim
    (SDD §3.2). Um diagnóstico não pode morrer por causa do lixo que ele
    existe para descrever.
    """
    if not isinstance(url, str):
        return False
    depois_do_scheme = url.split("://", 1)[-1]
    i = depois_do_scheme.find("/")
    return i != -1 and len(depois_do_scheme) > i + 1


def _pct(mask: np.ndarray) -> float:
    """Percentual de `True` em `mask`, arredondado a 2 casas."""
    if len(mask) == 0:
        return 0.0
    return round(100.0 * float(np.mean(mask)), 2)


def _gap(urls_benignas: pd.Series, urls_phishing: pd.Series, atributo) -> dict:
    """Gap de um atributo estrutural entre as duas classes, em pontos percentuais.

    `gap` = |%(phishing) − %(benigno)|. Gap alto significa que o atributo
    identifica a **origem do arquivo**, não a classe: o modelo aprende de
    qual metade do corpus a linha veio.
    """
    pct_benigno = _pct(np.fromiter((atributo(u) for u in urls_benignas), dtype=bool,
                                   count=len(urls_benignas)))
    pct_phishing = _pct(np.fromiter((atributo(u) for u in urls_phishing), dtype=bool,
                                    count=len(urls_phishing)))
    return {
        "benigno_pct": pct_benigno,
        "phishing_pct": pct_phishing,
        "gap_p": round(abs(pct_phishing - pct_benigno), 2),
    }


def medir_vies(frame: pd.DataFrame) -> dict:
    """Tabela 5 (SDD §3.2) para um corpus: balanço, gaps e veredito.

    O gap é medido sobre o **split de treino** da fonte, como a §3.2 declara —
    é o conjunto de que o modelo aprende, e portanto onde o viés faz efeito.
    `gap_corpus_inteiro_p` acompanha cada atributo para deixar explícito que
    o viés não é um artefato do recorte de treino.
    """
    for col in ("label", "url", "split_oficial"):
        if col not in frame.columns:
            raise DiagnosticsError(f"coluna '{col}' ausente — ver contrato de SDD §7.3")

    treino = frame[frame["split_oficial"] == "train"]
    teste = frame[frame["split_oficial"] == "test"]
    if len(treino) == 0:
        raise DiagnosticsError("corpus sem split de treino oficial")

    n_phishing = int((frame["label"] == 1).sum())
    n_benigno = int(len(frame) - n_phishing)

    out: dict = {
        "n_treino": int(len(treino)),
        "n_teste": int(len(teste)),
        "n_total": int(len(frame)),
        "balanco": {
            "benigno": n_benigno,
            "phishing": n_phishing,
            "benigno_pct": _pct((frame["label"] == 0).to_numpy()),
            "phishing_pct": _pct((frame["label"] == 1).to_numpy()),
        },
    }

    for nome_atributo, atributo in (("scheme", tem_scheme), ("path", tem_path)):
        medida = _gap(
            treino.loc[treino["label"] == 0, "url"],
            treino.loc[treino["label"] == 1, "url"],
            atributo,
        )
        medida["gap_corpus_inteiro_p"] = _gap(
            frame.loc[frame["label"] == 0, "url"],
            frame.loc[frame["label"] == 1, "url"],
            atributo,
        )["gap_p"]
        out[nome_atributo] = medida

    out["gap_max_p"] = round(max(out["scheme"]["gap_p"], out["path"]["gap_p"]), 2)
    out["veredito"] = "contaminado" if out["gap_max_p"] >= LIMIAR_VIES_P else "limpo"
    return out


# ---------------------------------------------------------------------------
# Vazamento de domínio no split oficial (SDD §3.3) — o achado central
# ---------------------------------------------------------------------------


def medir_vazamento_split_oficial(frame: pd.DataFrame, etld_col: str = "etld_plus_one") -> dict:
    """Tabela 1 (SDD §3.3): quanto do teste oficial já apareceu no treino oficial.

    `contaminacao_dominios_pct` é a métrica que a §3.3 reporta: dos domínios
    eTLD+1 distintos do teste, quantos % também estão no treino.

    `contaminacao_linhas_pct` é a mesma pergunta pesada por linha — dos
    exemplos de teste, quantos % pertencem a um domínio que o modelo já viu.
    Ela é o número que de fato descreve o quanto a métrica publicada está
    inflada, e por isso vale mais que a contagem de domínios na Discussão.

    Linhas sem eTLD+1 extraível (URL vazia, host malformado, lixo binário)
    ficam **fora** das contagens de domínio — não são domínios — e são
    reportadas em `linhas_sem_etld1` para que a exclusão seja visível.
    """
    if etld_col not in frame.columns:
        raise DiagnosticsError(
            f"coluna '{etld_col}' ausente — canonize antes (ver `carregar_corpus_canonizado`)"
        )

    # Normaliza ausência (None/pd.NA) para "" antes de comparar: com dtype
    # "string" do pandas, `etld != ""` propagaria pd.NA e o `&` com um array
    # booleano quebraria.
    etld_col_valores = frame[etld_col]
    etld = (
        etld_col_valores.astype("object")
        .where(etld_col_valores.notna(), "")
        .to_numpy(dtype=object)
    )
    valido = etld != ""
    e_treino = (frame["split_oficial"] == "train").to_numpy()
    e_teste = (frame["split_oficial"] == "test").to_numpy()

    dominios_treino = set(etld[valido & e_treino])
    dominios_teste_por_linha = etld[valido & e_teste]
    dominios_teste = set(dominios_teste_por_linha)
    sobrepostos = dominios_treino & dominios_teste

    n_linhas_teste = len(dominios_teste_por_linha)
    linhas_contaminadas = (
        int(np.isin(dominios_teste_por_linha, list(sobrepostos)).sum()) if sobrepostos else 0
    )

    # URLs literalmente idênticas (AD-076, LIT-01 e LIT-02): valores distintos da
    # coluna `url` bruta presentes no treino e no teste, em string exata (sem
    # normalizar caixa, esquema nem barra final), antes de qualquer dedup.
    # URL nula não entra na contagem.
    url_valores = frame["url"]
    url_valida = url_valores.notna().to_numpy()
    url_objeto = url_valores.astype("object").to_numpy(dtype=object)
    urls_identicas = set(url_objeto[url_valida & e_treino]) & set(
        url_objeto[url_valida & e_teste]
    )

    return {
        "dominios_treino": len(dominios_treino),
        "dominios_teste": len(dominios_teste),
        "dominios_sobrepostos": len(sobrepostos),
        "urls_identicas_treino_teste": len(urls_identicas),
        "contaminacao_dominios_pct": (
            round(100.0 * len(sobrepostos) / len(dominios_teste), 2) if dominios_teste else 0.0
        ),
        "linhas_teste_com_dominio_valido": n_linhas_teste,
        "linhas_teste_contaminadas": linhas_contaminadas,
        "contaminacao_linhas_pct": (
            round(100.0 * linhas_contaminadas / n_linhas_teste, 2) if n_linhas_teste else 0.0
        ),
        "linhas_sem_etld1": {
            "treino": int((~valido & e_treino).sum()),
            "teste": int((~valido & e_teste).sum()),
        },
        "exemplo": escolher_exemplo_vazamento(frame, etld, valido, e_treino, e_teste),
    }


def escolher_exemplo_vazamento(
    frame: pd.DataFrame,
    etld,
    valido,
    e_treino,
    e_teste,
) -> dict | None:
    """Um par treino/teste concreto do domínio mais vazado, para citar no artigo.

    Existe porque o artigo não mostrava **uma única URL** (achado do
    roteiro de revisão): um trabalho sobre características léxicas de URL
    em que o leitor nunca vê um endereço obriga-o a imaginar o objeto de estudo
    o tempo inteiro. O par entra na prosa por macro do `numeros.tex`, como
    qualquer valor medido (PC3), e **não** digitado à mão.

    🚨 As duas URLs são exigidas DISTINTAS, e essa condição não é decorativa.
    O roteiro de revisão propunha um par escolhido a mão em que a URL do lado
    "teste" (`.../ba1n3qguq.html`) está **nas duas partições** do ISCX: usá-la
    ilustraria duplicação exata de URL, que é um defeito diferente e mais forte
    do que o vazamento de **domínio** que o C1 mede. Um revisor que conferisse
    encontraria o exemplo principal do artigo demonstrando outra coisa.
    Derivar o par com esta restrição torna esse erro impossível de reintroduzir.

    Determinístico: domínio sobreposto com mais linhas de teste (desempate pelo
    nome), e de cada lado a URL mais curta (desempate lexicográfico). Sem isso
    o exemplo mudaria de execução para execução e o `.tex` deixaria de
    reproduzir byte a byte.
    """
    sobrepostos = set(etld[valido & e_treino]) & set(etld[valido & e_teste])
    if not sobrepostos:
        return None

    etld_teste = etld[valido & e_teste]
    contagem = Counter(d for d in etld_teste if d in sobrepostos)
    # `-n` primeiro e o nome depois: `max` sobre a tupla desempata pelo nome em
    # ordem crescente, o que é estável entre execuções e entre versões do
    # Counter (cuja ordem de iteração acompanha a de inserção).
    dominio = min(contagem.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    urls = frame["url"].to_numpy(dtype=object)
    labels = frame["label"].to_numpy()
    do_dominio = etld == dominio
    lado = {}
    for nome, mascara in (("treino", e_treino), ("teste", e_teste)):
        sel = do_dominio & mascara & valido
        # `strict=True` não é ritual: as duas fatias vêm da MESMA máscara
        # booleana, então divergir de comprimento significaria que `urls` e
        # `labels` deixaram de vir do mesmo frame — melhor estourar que emparelhar
        # URL com o rótulo da linha errada em silêncio.
        lado[nome] = sorted(
            {
                (str(u), int(rot))
                for u, rot in zip(urls[sel], labels[sel], strict=True)
            },
            key=lambda ul: (len(ul[0]), ul[0]),
        )
    if not lado["treino"] or not lado["teste"]:
        return None

    urls_teste = {u for u, _ in lado["teste"]}
    urls_treino = {u for u, _ in lado["treino"]}
    treino = next((ul for ul in lado["treino"] if ul[0] not in urls_teste), None)
    teste = next((ul for ul in lado["teste"] if ul[0] not in urls_treino), None)
    if treino is None or teste is None:
        # Todo o domínio é duplicata exata: não há par que ilustre vazamento de
        # DOMÍNIO sem ilustrar duplicação de URL. Devolver None é honesto; o
        # artigo simplesmente não cita exemplo deste corpus.
        return None

    return {
        "dominio": dominio,
        "url_treino": treino[0],
        "url_teste": teste[0],
        "label_treino": treino[1],
        "label_teste": teste[1],
        "linhas_treino": int((do_dominio & e_treino & valido).sum()),
        "linhas_teste": int((do_dominio & e_teste & valido).sum()),
    }


# ---------------------------------------------------------------------------
# Sobreposição de domínio ENTRE corpora (SDD §13.4-7, §11) — DoD da T2.3
# ---------------------------------------------------------------------------


def _particionar_para_diagnostico(
    frames: dict[str, pd.DataFrame],
    regime: str,
    seed: int,
    test_size: float,
    val_size: float,
    val_size_oficial: float,
) -> dict:
    """Partições do diagnóstico — as **mesmas** que a matriz do artigo usa.

    Existe como função só porque duas medições dependem de as partições serem
    idênticas: a matriz par-a-par (T2.3) e a contaminação do LOCO (T4.10). Se
    cada uma sorteasse a sua, os dois números do `diagnostics.json` seriam de
    experimentos diferentes e a comparação entre eles — que é exatamente o que
    a §13.4-17 pede — não significaria nada.

    Desde a frente A (DEDUP-05, design componente 12) o particionamento passa
    por `splits.particionar_corpus`, a fachada que deduplica pelo `url_norm` e
    que a matriz também usa; os quadros precisam ter a coluna `url_norm`.
    """
    if regime not in splits.REGIMES:
        raise DiagnosticsError(f"regime desconhecido: {regime!r}. Válidos: {splits.REGIMES}")
    return {
        nome: splits.particionar_corpus(
            frame,
            regime,
            corpus=nome,
            seed=seed,
            test_size=test_size,
            val_size=val_size if regime == "etld1" else val_size_oficial,
        )[0]
        for nome, frame in frames.items()
    }


def sobreposicao_loco(
    frames: dict[str, pd.DataFrame],
    regime: str,
    seed: int,
    test_size: float,
    val_size: float,
    val_size_oficial: float,
    indices: dict | None = None,
) -> dict:
    """Contaminação de domínio do leave-one-corpus-out (T4.10, §13.4-17).

    Para cada corpus retido, mede quantos domínios do teste dele reaparecem na
    **união dos treinos dos outros três** — que é exatamente o que
    `matrix.run_loco` concatena para treinar, e testando na mesma partição
    (`indices[retido].test`) que a matriz usa, como o D3-b exige.

    Por que a matriz par-a-par não respondia isto
    ---------------------------------------------
    `matriz_sobreposicao_entre_corpora` só faz par a par, e a união de três
    conjuntos de domínios não se recupera de três medições separadas: somar
    superestima (os três se intersectam), pegar o maior subestima. O SDD só
    tinha limites derivados — PhishStorm retido em [20,84%; 28,02%], ISCX em
    [10,03%; 18,19%], kaggle03 em [4,94%; 9,24%], Grambeddings em
    [1,41%; 1,99%]. A T6 apareceria ao lado da T2 (média 4,79%) carregando até
    ~6× mais contaminação sem declarar quanto.

    Vale a doutrina do D3, igual à da matriz: **medir e declarar, nunca
    filtrar**. Filtrar a união mudaria o que a T6 mede.

    Diferente da matriz, aqui **não há diagonal e não há valor esperado de
    zero**: o corpus retido nunca entra no próprio treino (isso é conferido em
    `matrix.check_loco_records`), mas os outros três podem compartilhar
    domínios com ele à vontade — as feeds públicas de phishing são as mesmas.
    """
    if len(frames) < 2:
        raise DiagnosticsError(
            f"sobreposição do LOCO exige ao menos 2 corpora; recebi {list(frames)}"
        )

    indices = indices or _particionar_para_diagnostico(
        frames, regime, seed, test_size, val_size, val_size_oficial
    )

    por_retido: dict = {}
    for retido in frames:
        outros = [n for n in frames if n != retido]
        celula = splits.union_domain_overlap(
            [(frames[n], indices[n].train) for n in outros],
            frames[retido],
            indices[retido].test,
        )
        celula["train_corpora"] = outros
        por_retido[retido] = celula
    return por_retido


def _resumir_loco(por_retido: dict) -> dict:
    """Os números que a T5.6 e a legenda da T6 vão citar, já prontos."""
    pct = [c["shared_domain_pct"] for c in por_retido.values()]
    linhas = [c["rows_with_shared_domain_pct"] for c in por_retido.values()]
    pior = max(por_retido, key=lambda n: por_retido[n]["shared_domain_pct"])
    return {
        "min_pct": round(min(pct), 2) if pct else 0.0,
        "max_pct": round(max(pct), 2) if pct else 0.0,
        "media_pct": round(float(np.mean(pct)), 2) if pct else 0.0,
        "linhas_max_pct": round(max(linhas), 2) if linhas else 0.0,
        "linhas_media_pct": round(float(np.mean(linhas)), 2) if linhas else 0.0,
        "corpus_retido_mais_contaminado": pior,
    }


def matriz_sobreposicao_entre_corpora(
    frames: dict[str, pd.DataFrame],
    regime: str,
    seed: int,
    test_size: float,
    val_size: float,
    val_size_oficial: float,
    indices: dict | None = None,
) -> dict:
    """Matriz 4×4: domínios do treino de A que reaparecem no teste de B.

    Cada célula chama `splits.cross_corpus_domain_overlap()` sobre exatamente
    as partições que a matriz cruzada do artigo (T2/T3) vai usar — mesmo
    regime, mesma seed, mesmas frações. Só assim o resíduo reportado em
    Ameaças à Validade corresponde ao experimento de fato executado.

    A **diagonal do regime `etld1` tem de ser 0,0%** por construção: ali o
    treino e o teste saem do mesmo particionador agrupado por domínio.
    Isso é verificado, não assumido — é o teste mais barato de que a fiação
    está certa. Fora da diagonal não há garantia nenhuma, e é justamente esse
    resíduo que o artigo declara em vez de filtrar (SDD §4 D3, §11).

    Atenção à diagonal do regime `oficial`: ela **não** repete a Tabela 1.
    Aqui o "treino" já teve a validação recortada fora (10% do treino
    oficial), então a contagem de domínios é ligeiramente menor.
    """
    indices = indices or _particionar_para_diagnostico(
        frames, regime, seed, test_size, val_size, val_size_oficial
    )

    matriz: dict = {}
    for treino, frame_tr in frames.items():
        matriz[treino] = {}
        for teste, frame_te in frames.items():
            celula = splits.cross_corpus_domain_overlap(
                frame_tr,
                indices[treino].train,
                frame_te,
                indices[teste].test,
            )
            celula["diagonal"] = treino == teste
            matriz[treino][teste] = celula

    if regime == "etld1":
        for nome in frames:
            pct = matriz[nome][nome]["shared_domain_pct"]
            if pct != 0.0:
                raise DiagnosticsError(
                    f"diagonal do regime 'etld1' em '{nome}' deu {pct}% de sobreposição, "
                    f"e tem de ser 0,0% por construção (o treino e o teste saem do mesmo "
                    f"particionador agrupado por domínio). Isso indica fiação quebrada "
                    f"em splits.py, não um fato do corpus."
                )
    return matriz


def _resumir_matriz(matriz: dict) -> dict:
    """Os poucos números que a T5.6 precisa citar, já prontos."""
    diagonal, fora = [], []
    for treino, linha in matriz.items():
        for teste, celula in linha.items():
            (diagonal if treino == teste else fora).append(celula["shared_domain_pct"])
    fora_linhas = [
        celula["rows_with_shared_domain_pct"]
        for treino, linha in matriz.items()
        for teste, celula in linha.items()
        if treino != teste
    ]
    return {
        "diagonal_max_pct": round(max(diagonal), 2) if diagonal else 0.0,
        "fora_diagonal_min_pct": round(min(fora), 2) if fora else 0.0,
        "fora_diagonal_max_pct": round(max(fora), 2) if fora else 0.0,
        "fora_diagonal_media_pct": round(float(np.mean(fora)), 2) if fora else 0.0,
        "fora_diagonal_linhas_max_pct": round(max(fora_linhas), 2) if fora_linhas else 0.0,
        "fora_diagonal_linhas_media_pct": (
            round(float(np.mean(fora_linhas)), 2) if fora_linhas else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Diagnósticos da frente A (sem treino): PSL privada, domínios com as duas
# classes, desvio de prevalência, esquema e médias por classe
# ---------------------------------------------------------------------------
#
# Nenhuma função desta seção usa `models` nem ajusta modelo: são contagens e
# médias sobre os quadros e as partições (um teste confere os nomes usados).


def psl_privada_contra_icann(hosts) -> dict:
    """Quanto a seção privada da PSL muda o agrupamento (REEX-13, PSL-04).

    Compara o eTLD+1 de cada host com a seção de domínios privados e só com a
    seção ICANN (`psl.etld_mais_um(host, privada=...)`). Devolve:

    - `n_urls`: URLs recebidas;
    - `n_urls_mudam`: URLs cujo eTLD+1 muda (ex.: `a.github.io` passa de
      `github.io` para `a.github.io`);
    - `n_grupos_icann_afetados`: eTLD+1 da lista ICANN que se desfazem em mais
      de um grupo para ao menos uma URL (`github.io`);
    - `n_grupos_resultantes`: eTLD+1 distintos que essas URLs passam a ter.

    Host vazio ou sem eTLD+1 em alguma das duas listas não conta como mudança.
    """
    serie = pd.Series(list(hosts), dtype="object")
    validos = serie[serie.notna() & (serie != "")]
    mudam = 0
    icann: set[str] = set()
    resultantes: set[str] = set()
    for host, n in validos.value_counts().items():
        com_privada = psl.etld_mais_um(host, privada=True)
        so_icann = psl.etld_mais_um(host, privada=False)
        if com_privada and so_icann and com_privada != so_icann:
            mudam += int(n)
            icann.add(so_icann)
            resultantes.add(com_privada)
    return {
        "n_urls": int(len(serie)),
        "n_urls_mudam": mudam,
        "n_grupos_icann_afetados": len(icann),
        "n_grupos_resultantes": len(resultantes),
    }


def dominios_duas_classes(frame: pd.DataFrame, etld_col: str = "etld_plus_one") -> dict:
    """eTLD+1 com URLs das duas classes (REEX-08, DEDUP-05).

    Linhas sem eTLD+1 ficam de fora, como em `medir_vazamento_split_oficial`.
    Devolve `n_dominios`, `n_dominios_duas_classes` e `n_linhas_duas_classes`
    (linhas que pertencem a esses domínios).
    """
    if etld_col not in frame.columns:
        raise DiagnosticsError(f"coluna '{etld_col}' ausente, canonize antes")
    etld = frame[etld_col].astype("object")
    valido = etld.notna() & (etld != "")
    por_dominio = frame.loc[valido, "label"].groupby(etld[valido]).nunique()
    duas = por_dominio.index[por_dominio.to_numpy() > 1]
    return {
        "n_dominios": int(len(por_dominio)),
        "n_dominios_duas_classes": int(len(duas)),
        "n_linhas_duas_classes": int(etld[valido].isin(duas).sum()),
    }


def desvio_prevalencia(frame: pd.DataFrame, split: splits.SplitIndices) -> dict:
    """Prevalência de phishing por partição contra a do *dataset* deduplicado.

    A referência é a prevalência das linhas elegíveis (treino, validação e
    teste do `split`, já deduplicadas). `desvio` é a prevalência da partição
    menos a do *dataset*, em fração (0,01 é um ponto percentual). Partição
    vazia traz `prevalencia` e `desvio` nulos.
    """
    y = frame["label"].to_numpy()
    elegiveis = np.concatenate([split.train, split.val, split.test])
    if len(elegiveis) == 0:
        raise DiagnosticsError(f"partição de '{split.corpus}' ({split.regime}) sem linhas")
    prevalencia_dataset = float(np.mean(y[elegiveis] == 1))
    saida: dict = {"n_dataset": int(len(elegiveis)), "prevalencia_dataset": prevalencia_dataset}
    for nome, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
        if len(idx) == 0:
            saida[nome] = {"n": 0, "prevalencia": None, "desvio": None}
            continue
        prevalencia = float(np.mean(y[idx] == 1))
        saida[nome] = {
            "n": int(len(idx)),
            "prevalencia": prevalencia,
            "desvio": prevalencia - prevalencia_dataset,
        }
    return saida


def esquema_por_classe(frame: pd.DataFrame) -> dict:
    """Presença de esquema por classe: `http`, `https` e nenhum (ANL-05).

    Por classe (`benigno` = 0, `phishing` = 1): `n`, a fração com esquema
    explícito (`tem_scheme`, que inclui `ftp`), com `http://`, com `https://` e
    sem esquema. Teste de prefixo, como em `tem_scheme`.
    """
    for col in ("label", "url"):
        if col not in frame.columns:
            raise DiagnosticsError(f"coluna '{col}' ausente")
    saida = {}
    for nome, rotulo in (("benigno", 0), ("phishing", 1)):
        urls = frame.loc[frame["label"] == rotulo, "url"].tolist()
        n = len(urls)
        explicito = sum(tem_scheme(u) for u in urls)
        http = sum(isinstance(u, str) and u.lower().startswith("http://") for u in urls)
        https = sum(isinstance(u, str) and u.lower().startswith("https://") for u in urls)
        saida[nome] = {
            "n": n,
            "explicito": explicito / n if n else None,
            "http": http / n if n else None,
            "https": https / n if n else None,
            "nenhum": (n - explicito) / n if n else None,
        }
    return saida


def medias_por_classe(frame: pd.DataFrame, chunk: int = 50_000) -> dict:
    """Média de cada uma das 41 características por classe (ANL-05).

    Extrai as características em blocos de `chunk` URLs, com a opção padrão
    (esquema mantido), e acumula soma e contagem por classe, para não manter o
    quadro de características inteiro em memória. Devolve
    `{"benigno": {"n", "medias": {característica: média}}, "phishing": ...}`.
    """
    import features

    nomes = list(features.FEATURE_NAMES)
    somas = {0: np.zeros(len(nomes)), 1: np.zeros(len(nomes))}
    contagens = {0: np.zeros(len(nomes)), 1: np.zeros(len(nomes))}
    n_por_classe = {0: 0, 1: 0}
    urls = frame["url"].tolist()
    rotulos = frame["label"].to_numpy()
    for inicio in range(0, len(urls), chunk):
        bloco = features.extract_features(urls[inicio : inicio + chunk])
        valores = bloco[nomes].to_numpy(dtype=np.float64)
        y = rotulos[inicio : inicio + chunk]
        for rotulo in (0, 1):
            sel = valores[y == rotulo]
            somas[rotulo] += np.nansum(sel, axis=0)
            contagens[rotulo] += np.sum(~np.isnan(sel), axis=0)
            n_por_classe[rotulo] += int(len(sel))
    saida = {}
    for nome, rotulo in (("benigno", 0), ("phishing", 1)):
        medias = {
            f: (float(somas[rotulo][j] / contagens[rotulo][j]) if contagens[rotulo][j] else None)
            for j, f in enumerate(nomes)
        }
        saida[nome] = {"n": n_por_classe[rotulo], "medias": medias}
    return saida


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def canonizar_colunas(urls, chunk: int = 100_000) -> dict[str, list]:
    """`url_norm`, eTLD+1 e host de cada URL, via a canonização que alimenta `splits.py`.

    Chama `data.url_canonicalization.canonicalize_url` diretamente em vez de
    `features.canonicalize_urls` de propósito: aquele devolve um DataFrame com
    as 8 colunas de canonização, e materializar 800.000 dicionários de 8
    entradas só para descartar 5 delas custa centenas de MB sem necessidade.
    A **lógica** de canonização é a mesma função, então o eTLD+1 medido aqui é
    bit-a-bit o mesmo que o particionador do Regime B usa. O `url_norm` é a
    chave da deduplicação (`splits.particionar_corpus`) e o host alimenta o
    diagnóstico REEX-13. Valor ausente vira `""` no eTLD+1 e no host, e `None`
    no `url_norm`.
    """
    total = len(urls)
    saida: dict[str, list] = {"url_norm": [], "etld_plus_one": [], "host": []}
    for inicio in range(0, total, chunk):
        for url in urls[inicio : inicio + chunk]:
            canon = canonicalize_url(url)
            etld = canon.get("etld_plus_one")
            host = canon.get("domain")
            norm = canon.get("url_norm")
            saida["url_norm"].append(norm if isinstance(norm, str) else None)
            saida["etld_plus_one"].append(etld if isinstance(etld, str) else "")
            saida["host"].append(host if isinstance(host, str) else "")
        if total > chunk:
            logger.info("canonizadas %d/%d URLs", min(inicio + chunk, total), total)
    return saida


def canonizar_etld1(urls, chunk: int = 100_000) -> list[str]:
    """eTLD+1 de cada URL (atalho de `canonizar_colunas`)."""
    return canonizar_colunas(urls, chunk=chunk)["etld_plus_one"]


def carregar_corpus_canonizado(
    nome: str, corpora_dir: Path | None = None
) -> pd.DataFrame:
    """Lê `corpora/<nome>/data.parquet` e anexa `etld_plus_one`, `url_norm` e `host`.

    Devolve só as colunas de que o diagnóstico precisa. As 41 features
    léxicas **não** são extraídas aqui: nada na carga depende delas, e extrair
    41 colunas de 2,55 M de linhas para depois ignorá-las custaria minutos e
    gigabytes (`medias_por_classe` as extrai em blocos, só onde precisa).
    """
    corpora_dir = corpora_dir or DEFAULT_CORPORA_DIR
    caminho = corpora_dir / nome / "data.parquet"
    if not caminho.exists():
        raise DiagnosticsError(
            f"corpus '{nome}' não encontrado em {_caminho_relativo(caminho)}. "
            f"Gere os parquet com: "
            f"venv/Scripts/python.exe -m src.corpora --raw-dir <dir> --out-dir corpora"
        )
    frame = pd.read_parquet(
        caminho, columns=["label", "url", "corpus", "split_oficial"]
    ).reset_index(drop=True)
    frame["url"] = frame["url"].astype("string")
    colunas = canonizar_colunas(frame["url"].tolist())
    frame["etld_plus_one"] = pd.array(colunas["etld_plus_one"], dtype="string")
    frame["url_norm"] = pd.array(colunas["url_norm"], dtype="string")
    frame["host"] = pd.array(colunas["host"], dtype="string")
    return frame


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


DEFINICOES = {
    "gap": "Diferença absoluta, em pontos percentuais, entre a fração de URLs "
           "phishing e a fração de URLs benignas que exibem um atributo "
           "estrutural. Medido sobre o split de treino distribuído pela fonte. "
           "Gap alto significa que o atributo identifica a origem do arquivo, "
           "não a classe.",
    "scheme": "A URL começa com 'http://', 'https://' ou 'ftp://'. Teste de "
              "prefixo, não de substring (evitar o bug B14).",
    "path": "Depois do host existe uma '/' e ao menos um caractere após ela. "
            "Barra final solta não conta como path.",
    "veredito": f"'contaminado' quando max(gap_scheme, gap_path) >= "
                f"{LIMIAR_VIES_P} pontos percentuais; 'limpo' caso contrário. "
                f"A distribuição medida é bimodal e qualquer limiar entre ~7 e "
                f"~59 p produz a mesma partição.",
    # ⚠️ Esta string vai para results/diagnostics.json e de lá pode ir para o
    # artigo. Até 06/08/2026 ela afirmava "lista de sufixos públicos em cache
    # versionado, RNF6" — o que é FALSO e está registrado na §13.4-16 do SDD:
    # o tldextract resolve a lista pela rede na primeira execução e a guarda em
    # ~/.cache/python-tldextract/, fora do repositório e por máquina. Corrigir a
    # causa (fixar o snapshot) mudaria valor de feature e exigiria rodada nova,
    # que não cabe no prazo; corrigir a AFIRMAÇÃO custa zero e é o que impede
    # que um número entre no artigo apoiado numa garantia que não existe.
    "dominio": "eTLD+1 obtido por data.url_canonicalization.canonicalize_url, "
               "pela lista de sufixos públicos versionada em config/psl, com a "
               "seção de domínios privados (PSL-01 a PSL-03). É a MESMA chave "
               "que splits.py usa para agrupar no Regime B: a Tabela 1 e o "
               "split contam 'domínio' do mesmo jeito. Linhas sem eTLD+1 "
               "extraível não entram nas contagens de domínio.",
    "psl_privada_contra_icann": "Por dataset: URLs cujo eTLD+1 muda entre a "
                                "PSL com a seção de domínios privados e só com "
                                "a seção ICANN, grupos ICANN afetados e grupos "
                                "resultantes (REEX-13).",
    "dominios_duas_classes": "Por dataset: eTLD+1 com URLs das duas classes, no "
                             "dataset bruto e no deduplicado do Regime B "
                             "(REEX-08).",
    "desvio_prevalencia": "Por regime, dataset e partição: prevalência de "
                          "phishing menos a do dataset deduplicado, em fração "
                          "(REEX-08).",
    "esquema_por_classe": "Por dataset e classe: fração de URLs com esquema "
                          "explícito, com http, com https e sem esquema "
                          "(ANL-05).",
    "medias_por_classe": "Por dataset e classe: média de cada uma das 41 "
                         "características, com o esquema mantido (ANL-05).",
    "contaminacao_dominios_pct": "Dos domínios eTLD+1 distintos do teste "
                                 "oficial, percentual que também aparece no "
                                 "treino oficial.",
    "contaminacao_linhas_pct": "Das linhas de teste com domínio válido, "
                               "percentual que pertence a um domínio presente "
                               "no treino oficial.",
    "exemplo": "Par treino/teste concreto do domínio sobreposto com mais "
               "linhas de teste, para citação na prosa do artigo. As duas URLs "
               "são DISTINTAS por construção: um par cujas URLs coincidissem "
               "ilustraria duplicação exata de URL, que é defeito diferente e "
               "mais forte do que o vazamento de domínio medido aqui. "
               "Determinístico (domínio com mais linhas de teste, desempate "
               "pelo nome; de cada lado a URL mais curta, desempate "
               "lexicográfico). Nulo se o domínio não admitir par distinto.",
    "sobreposicao_entre_corpora": "Por célula (treino=A, teste=B): percentual "
                                  "dos domínios do teste de B que também "
                                  "aparecem no treino de A, nas MESMAS "
                                  "partições que a matriz cruzada do artigo "
                                  "usa. Medido e declarado, nunca filtrado "
                                  "(SDD §4 D3, §11). A diagonal do regime "
                                  "'etld1' é 0,0% por construção; a do regime "
                                  "'oficial' não repete a Tabela 1 porque ali "
                                  "a validação já foi recortada do treino.",
    "sobreposicao_loco": "Por corpus retido: percentual dos domínios do teste "
                         "dele que também aparecem na UNIÃO dos treinos dos "
                         "outros três — que é o que o leave-one-corpus-out "
                         "(SDD §4 D3-b) de fato treina, nas mesmas partições "
                         "da matriz. Não se recupera da matriz par a par: "
                         "somar as três superestima (os conjuntos de domínio "
                         "se intersectam) e pegar a maior subestima. Não há "
                         "diagonal nem valor esperado de zero aqui: o corpus "
                         "retido não entra no próprio treino, mas os outros "
                         "três compartilham domínio com ele à vontade. Medido "
                         "e declarado, nunca filtrado (SDD §4 D3).",
}


def executar(
    config: dict | None = None,
    corpora_dir: Path | None = None,
    corpora: tuple[str, ...] = TODOS_OS_CORPORA,
    caracteristicas: bool = True,
) -> dict:
    """Roda o diagnóstico completo e devolve o dicionário de saída.

    Não grava nada — quem grava é `main`/`gravar`, para que os testes possam
    exercitar a medição sem tocar em `results/`.

    `caracteristicas=False` pula `medias_por_classe`, a única medição que
    extrai as 41 características (alguns minutos nos corpora grandes).
    """
    config = config or models.load_experiment_config()
    seed = int(config["seed"])
    test_size = float(config["split"]["test_size"])
    val_size = float(config["split"]["val_size"])
    val_size_oficial = float(config["split"]["val_size_oficial"])
    corpora_matriz = tuple(config["corpora"]["treino_avaliacao"])

    vies: dict = {}
    vazamento: dict = {}
    frames_matriz: dict[str, pd.DataFrame] = {}
    psl_privada: dict = {}
    duas_classes: dict = {}
    esquema: dict = {}
    medias: dict = {}

    for nome in corpora:
        logger.info("diagnosticando corpus %s", nome)
        frame = carregar_corpus_canonizado(nome, corpora_dir=corpora_dir)
        vies[nome] = medir_vies(frame)
        vazamento[nome] = medir_vazamento_split_oficial(frame)
        psl_privada[nome] = psl_privada_contra_icann(frame["host"])
        duas_classes[nome] = {
            "bruto": dominios_duas_classes(frame),
            "deduplicado_etld1": dominios_duas_classes(
                frame.iloc[dedup.deduplicar(frame, "etld1").posicoes]
            ),
        }
        esquema[nome] = esquema_por_classe(frame)
        if caracteristicas and nome in corpora_matriz:
            medias[nome] = medias_por_classe(frame)
        if nome in corpora_matriz:
            # Só os 4 de treino/avaliação seguem em memória para a matriz;
            # os 2 de demonstração de viés são liberados aqui.
            frames_matriz[nome] = frame
        else:
            del frame

    # A matriz 4×4 exige os 4 corpora de treino/avaliação (SDD §4 D2). Rodar
    # com `--corpora` reduzido é legítimo para inspeção, mas o arquivo que sai
    # daí NÃO cumpre o DoD da T2.3 e não pode alimentar o artigo — por isso
    # sai marcado como parcial em vez de silenciosamente incompleto (R5).
    faltando = [n for n in corpora_matriz if n not in frames_matriz]
    parcial = bool(faltando) or set(corpora) != set(TODOS_OS_CORPORA)

    entre_corpora: dict = {}
    loco_overlap: dict = {}
    prevalencia: dict = {}
    for regime in [] if faltando else config["regimes"]:
        logger.info("matriz de sobreposição entre corpora, regime '%s'", regime)
        quadros = {n: frames_matriz[n] for n in corpora_matriz}
        # Particionado UMA vez e passado às duas medições: elas têm de sair das
        # mesmas partições, senão a comparação que a §13.4-17 pede (união do
        # LOCO contra média da matriz) compara experimentos diferentes.
        indices = _particionar_para_diagnostico(
            quadros, regime, seed, test_size, val_size, val_size_oficial
        )
        matriz = matriz_sobreposicao_entre_corpora(
            quadros,
            regime=regime,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
            val_size_oficial=val_size_oficial,
            indices=indices,
        )
        entre_corpora[regime] = {"matriz": matriz, "resumo": _resumir_matriz(matriz)}
        prevalencia[regime] = {
            n: desvio_prevalencia(quadros[n], indices[n]) for n in corpora_matriz
        }

        logger.info("sobreposição do leave-one-corpus-out, regime '%s'", regime)
        por_retido = sobreposicao_loco(
            quadros,
            regime=regime,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
            val_size_oficial=val_size_oficial,
            indices=indices,
        )
        loco_overlap[regime] = {
            "por_corpus_retido": por_retido,
            "resumo": _resumir_loco(por_retido),
        }

    if faltando:
        logger.warning(
            "matriz de sobreposição entre corpora NÃO calculada: faltam %s. "
            "A saída sai marcada como parcial e não cumpre o DoD da T2.3.",
            faltando,
        )

    # Ordem de apresentação igual à das tabelas do artigo: viés crescente
    # (limpos primeiro, como na §3.2) e vazamento decrescente (o pior primeiro,
    # como na §3.3). Ordenação determinística, com o nome como critério de
    # desempate para que duas execuções gerem o mesmo arquivo (RNF1).
    vies_ordenado = dict(sorted(vies.items(), key=lambda kv: (kv[1]["gap_max_p"], kv[0])))
    vazamento_ordenado = dict(
        sorted(
            vazamento.items(),
            key=lambda kv: (-kv[1]["contaminacao_dominios_pct"], kv[0]),
        )
    )

    return {
        "run_id": evaluate.make_run_id(),
        "gerado_em": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parcial": parcial,
        # A `sobreposicao_entre_corpora` é medida SOBRE AS PARTIÇÕES do Regime B,
        # então este arquivo envelhece junto com o particionador — e ele é a fonte
        # das Tabelas T1 e T5. Sem a impressão digital, um `diagnostics.json` de
        # outra rodada convive com o `matrix.json` da atual e nada acusa: a
        # auditoria de 05/08/2026 encontrou exatamente isso, com o arquivo do
        # commit `1e564be` (04/08) medido sobre `GroupShuffleSplit`. Ver §13.4-18.
        #
        # Não é conferência automática — é rastro. Quem lê a T1 pode comparar com
        # o `config_fingerprint` do `run_meta.json` e ver se são a mesma rodada.
        "config_fingerprint": _fingerprint_da_config(),
        "parametros": {
            "seed": seed,
            "test_size": test_size,
            "val_size": val_size,
            "val_size_oficial": val_size_oficial,
            "limiar_vies_p": LIMIAR_VIES_P,
            "corpora_matriz": list(corpora_matriz),
        },
        "definicoes": DEFINICOES,
        "vies_construcao": vies_ordenado,
        "vazamento_split_oficial": vazamento_ordenado,
        "sobreposicao_entre_corpora": entre_corpora,
        "sobreposicao_loco": loco_overlap,
        "psl_privada_contra_icann": psl_privada,
        "dominios_duas_classes": duas_classes,
        "desvio_prevalencia": prevalencia,
        "esquema_por_classe": esquema,
        "medias_por_classe": medias,
    }


def _fingerprint_da_config() -> str | None:
    """Impressão digital da execução, ou `None` se não for obtenível.

    Importado aqui dentro, e não no topo, de propósito: `matrix.py` importa
    `models.py`, que puxa xgboost/lightgbm/catboost. Este módulo é o único que
    roda em máquina sem GPU por contrato (§0) e não deveria depender das três
    bibliotecas de treino só para carimbar um hash.
    """
    try:
        import matrix
        import models

        return matrix.config_fingerprint(models.load_experiment_config())
    except Exception as exc:  # noqa: BLE001
        logger.warning("config_fingerprint indisponivel no diagnostico", erro=str(exc))
        return None


def _caminho_relativo(caminho: Path) -> str:
    """Caminho relativo à raiz do repo (RNF3: zero caminho absoluto em relatório).

    Cai para o nome do arquivo quando o destino está fora do repositório
    (acontece só com `--out` apontando para outro lugar, em inspeção manual).
    """
    try:
        return caminho.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return caminho.name


def gravar(diagnostico: dict, out_path: Path | None = None) -> Path:
    """Grava o JSON em `results/diagnostics.json` (a ponte de SDD §7.1)."""
    out_path = out_path or DEFAULT_OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostico, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def _imprimir_tabelas(diagnostico: dict) -> None:
    """Ecoa as duas tabelas no terminal, para conferência a olho (RH3)."""
    print()
    print("TABELA 5 — viés de construção (gap em pontos percentuais, split de treino)")
    print(f"{'corpus':14s} {'n treino':>9s} {'n total':>9s} {'balanço':>9s} "
          f"{'scheme':>8s} {'path':>8s}  veredito")
    for nome, v in diagnostico["vies_construcao"].items():
        balanco = f"{v['balanco']['benigno_pct']:.0f}/{v['balanco']['phishing_pct']:.0f}"
        print(f"{nome:14s} {v['n_treino']:9d} {v['n_total']:9d} {balanco:>9s} "
              f"{v['scheme']['gap_p']:8.1f} {v['path']['gap_p']:8.1f}  {v['veredito']}")

    print()
    print("TABELA 1 — vazamento de domínio nos splits oficiais")
    print(f"{'corpus':14s} {'dom.treino':>11s} {'dom.teste':>10s} {'sobrep.':>8s} "
          f"{'% dom.':>7s} {'% linhas':>9s}")
    for nome, v in diagnostico["vazamento_split_oficial"].items():
        print(f"{nome:14s} {v['dominios_treino']:11d} {v['dominios_teste']:10d} "
              f"{v['dominios_sobrepostos']:8d} {v['contaminacao_dominios_pct']:7.1f} "
              f"{v['contaminacao_linhas_pct']:9.1f}")

    for regime, bloco in diagnostico["sobreposicao_entre_corpora"].items():
        print()
        print(f"MATRIZ 4×4 — sobreposição de domínio entre corpora, regime '{regime}' "
              f"(% dos domínios do teste da coluna presentes no treino da linha)")
        nomes = list(bloco["matriz"])
        # A barra invertida fica fora da f-string: dentro do `{}` ela só é
        # válida a partir do Python 3.12, e o alvo de lint do projeto é o 3.11.
        cabecalho = "treino \\ teste"
        print(f"{cabecalho:16s}" + "".join(f"{n[:10]:>12s}" for n in nomes))
        for treino in nomes:
            celulas = "".join(
                f"{bloco['matriz'][treino][teste]['shared_domain_pct']:12.2f}" for teste in nomes
            )
            print(f"{treino:16s}{celulas}")
        r = bloco["resumo"]
        print(f"  diagonal máx: {r['diagonal_max_pct']:.2f}%  ·  fora-diagonal: "
              f"{r['fora_diagonal_min_pct']:.2f}–{r['fora_diagonal_max_pct']:.2f}% "
              f"(média {r['fora_diagonal_media_pct']:.2f}%; por linha, média "
              f"{r['fora_diagonal_linhas_media_pct']:.2f}%)")

    # T4.10 (§13.4-17): o número que faltava ao lado da T6. Impresso junto da
    # média da matriz do mesmo regime, porque é essa comparação que importa —
    # a T6 carrega a contaminação da UNIÃO, não a de uma célula.
    for regime, bloco in (diagnostico.get("sobreposicao_loco") or {}).items():
        media_matriz = (
            diagnostico["sobreposicao_entre_corpora"][regime]["resumo"]
            ["fora_diagonal_media_pct"]
        )
        print()
        print(f"LOCO — contaminação da união de 3 corpora, regime '{regime}' "
              f"(% dos domínios do teste do retido presentes no treino dos outros)")
        print(f"{'corpus retido':16s}{'% dom.':>10s}{'% linhas':>10s}"
              f"{'dom.teste':>11s}{'compartilh.':>12s}")
        for retido, c in bloco["por_corpus_retido"].items():
            print(f"{retido:16s}{c['shared_domain_pct']:10.2f}"
                  f"{c['rows_with_shared_domain_pct']:10.2f}"
                  f"{c['test_domains']:11d}{c['shared_domains']:12d}")
        r = bloco["resumo"]
        print(f"  união: {r['min_pct']:.2f}–{r['max_pct']:.2f}% "
              f"(média {r['media_pct']:.2f}%)  ·  matriz par a par, mesmo regime: "
              f"média {media_matriz:.2f}%")
        print(f"  pior corpus retido: {r['corpus_retido_mais_contaminado']}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora-dir", type=Path, default=None,
        help="diretório dos parquet normalizados (default: corpora/ na raiz do repo)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="caminho do JSON de saída (default: results/diagnostics.json)",
    )
    parser.add_argument(
        "--corpora", nargs="+", default=list(TODOS_OS_CORPORA),
        help="subconjunto de corpora a diagnosticar (default: os 6)",
    )
    parser.add_argument(
        "--sem-caracteristicas", action="store_true",
        help="pula as médias das 41 características por classe (a medição mais lenta)",
    )
    args = parser.parse_args()

    diagnostico = executar(
        corpora_dir=args.corpora_dir,
        corpora=tuple(args.corpora),
        caracteristicas=not args.sem_caracteristicas,
    )
    caminho = gravar(diagnostico, out_path=args.out)
    _imprimir_tabelas(diagnostico)
    print()
    print(f"gravado: {_caminho_relativo(caminho)}")
    if diagnostico["parcial"]:
        print("ATENÇÃO: execução PARCIAL (subconjunto de corpora). Este arquivo não "
              "cumpre o DoD da T2.3 e não pode alimentar o artigo.")
    return 0


__all__ = [
    "LIMIAR_VIES_P",
    "TODOS_OS_CORPORA",
    "DEFINICOES",
    "DiagnosticsError",
    "tem_scheme",
    "tem_path",
    "medir_vies",
    "medir_vazamento_split_oficial",
    "matriz_sobreposicao_entre_corpora",
    "sobreposicao_loco",
    "psl_privada_contra_icann",
    "dominios_duas_classes",
    "desvio_prevalencia",
    "esquema_por_classe",
    "medias_por_classe",
    "canonizar_colunas",
    "canonizar_etld1",
    "carregar_corpus_canonizado",
    "executar",
    "gravar",
]


if __name__ == "__main__":
    sys.exit(main())
