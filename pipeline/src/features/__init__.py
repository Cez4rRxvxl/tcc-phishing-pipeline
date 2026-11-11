"""RF2/RF3 — canonização + extração de features léxicas (corrige B3 e B4).

Este pacote expõe, no caminho de import `features` (ou `src.features`,
dependendo de como o chamador monta `sys.path` — ambos funcionam, ver
nota técnica abaixo), o orquestrador previsto na arquitetura-alvo do SDD
(§7.1):

    [features.py]  canoniza URL  →  extrai 41 features léxicas

Nota técnica — por que isto está em `__init__.py` e não em `src/features.py`
-----------------------------------------------------------------------------
O SDD (§7.1) desenha `src/features.py` como um arquivo plano. Na prática
isso é inviável junto com o pacote `src/features/` que já existe (contém
`lexical_features.py`, reaproveitado com a correção de B4 — docs/diagnostico.md §3.5): Python
não permite um módulo `features.py` e um pacote `features/` no mesmo
diretório-pai sob o mesmo nome de import. Quando ambos existem, o pacote
sempre vence silenciosamente — `features.py` fica morto, nunca é
alcançado por `import features`, o que foi confirmado empiricamente antes
de decidir isto. Criar o arquivo teria sido escrever código morto sem
avisar ninguém.

Em vez disso, a orquestração pedida em RF2/RF3 mora aqui, em
`src/features/__init__.py`. O caminho de import (`features` /
`from features import build_feature_frame`) é exatamente o mesmo que o
SDD pretendia — só o arquivo físico dentro do pacote é diferente. Isso
preserva `src/features/lexical_features.py` importável como
`features.lexical_features`, do qual `src/pipelines/05_build_feature_matrix.py`
ainda depende.

Este pacote corrige **B3** (`05_build_feature_matrix.py:80` extraía
features de `df['url']` cru, ignorando os 596 LOC de
`src/data/url_canonicalization.py`): a partir de agora, toda extração via
`build_feature_frame()` passa primeiro pela canonização, que decide se a
URL é válida e fornece `etld_plus_one` (necessário para o split por
domínio, RF4).

Decisão importante de design — releia antes de mexer aqui
----------------------------------------------------------
`URLCanonicalizer.canonicalize_url()` produz `url_norm`, mas **não** é essa
string que alimenta o extrator de features léxicas. `url_norm` remove
deliberadamente o prefixo `www.`, portas padrão (`:80`/`:443`) e parâmetros
de tracking (utm_*, fbclid, gclid...), e reordena a query string — bom para
deduplicação (é para isso que existe), péssimo para features léxicas:
alimentar o extrator com `url_norm` zeraria permanentemente `has_www` e
`www_count`, e distorceria `query_params`/`ampersand_count`/`equal_count`
para qualquer URL com tracking params. Isso seria reintroduzir, de outra
forma, o mesmo tipo de bug que estamos corrigindo em B4 (uma feature morta
de forma sistemática e correlacionada com a origem/URL, não com o rótulo).

Por isso, `extract_features()` usa a URL como o *usuário a digitou/o feed
distribuiu*, apenas com scheme garantido — o que já é feito internamente e
corretamente por `features.lexical_features` (a correção de B4 em si). A
canonização (`canonicalize_url`) é usada aqui só para:

1. sinalizar URLs inválidas/sem host (`invalid_url`, `invalid_reason`) —
   linhas que não dá para extrair um domínio delas de forma alguma; e
2. fornecer `etld_plus_one`, que o split por domínio (RF4, `splits.py`)
   precisa para o split agrupado do Regime B.

Se essa decisão for revista, documente a mudança aqui e em SDD.md.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

# Segue a mesma convenção do resto do repositório (lexical_features.py,
# url_canonicalization.py, 05_build_feature_matrix.py): adiciona `src/` ao
# sys.path e importa os demais pacotes de `src/` por nome de topo (`data`,
# `utils`), em vez de assumir que `src` é importável como `src.data`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.lexical_features import (  # noqa: E402
    LexicalFeatureExtractor,
    get_feature_groups,
    get_feature_names,
)
from data.url_canonicalization import canonicalize_url  # noqa: E402
import psl  # noqa: E402
from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

FEATURE_NAMES = get_feature_names()
FEATURE_GROUPS = get_feature_groups()

# --- T3.2 — Auditoria de dtype ----------------------------------------------
#
# O mapa herdado de `artifacts/models/features.json` (pipeline legado)
# derivava o dtype de cada coluna a partir do min/max *observado naquela
# execução específica* (`05_build_feature_matrix.py::optimize_dtypes`).
# Isso é frágil por construção: o dtype de uma feature dependia de sorte —
# qual split calhou de ter a URL mais longa. Dois riscos reais foram
# medidos (ver relatório da tarefa, reproduzido em amostra real):
#   - domain_length (uint8, limite 255): máximo observado = 246-249
#     (97,6% do limite)
#   - special_char_count (uint8, limite 255): máximo observado = 784
#     — **já estourava** o uint8 (307% do limite; o pipeline legado com
#     certeza truncava isso silenciosamente via wraparound do numpy)
#   - percent_count (uint8, limite 255): máximo observado = 600 (235%) —
#     mesmo problema (URLs com muitos `%XX` de percent-encoding)
#
# Este pacote fixa o dtype por feature de forma *proativa e determinística*,
# em vez de inferi-lo por execução. Critério de promoção: qualquer contagem
# de caracteres derivada do texto inteiro da URL (dot/hyphen/underscore/
# percent/slash/equal/ampersand/question/hash/at/www) e derivados
# estruturais (subdomain_count, path_tokens, query_params) são, no limite,
# tão grandes quanto a própria URL — e `url_length` já é uint16 (até
# 65.535). Ou seja, teoricamente qualquer uma dessas contagens pode se
# aproximar do mesmo valor em uma URL adversarial (ex.: URL cheia de pontos
# ou de '%' para ofuscação) — e na prática, `special_char_count` e
# `percent_count` já ultrapassam 255 na amostra real testada. O custo de
# manter uint8 é nenhum ganho relevante de memória (< 20 MB a mais no
# corpus inteiro de ~1,8 M linhas) frente ao risco de reintroduzir
# silenciosamente um bug como B4/B10. Por isso, promovemos todas as
# contagens de caractere de uint8 -> uint16, e mantemos uint8 apenas para
# as 12 features genuinamente booleanas (0/1).
DTYPE_MAP: Dict[str, str] = {
    # Comprimentos — já eram uint16 no mapa legado, mantidos.
    "url_length": "uint16",
    "domain_length": "uint16",       # promovido de uint8 (máx observado 246-249, quase saturando)
    "path_length": "uint16",
    "query_length": "uint16",
    "fragment_length": "uint16",
    "max_token_length": "uint16",
    # Contagens de caracteres — promovidas de uint8 para uint16 (ver acima).
    "digit_count": "uint16",
    "letter_count": "uint16",
    "special_char_count": "uint16",  # promovido de uint8 (máx observado 784 — JÁ estourava uint8)
    "dot_count": "uint16",           # promovido de uint8 (máx observado 120, 47% do limite)
    "slash_count": "uint16",         # promovido de uint8 (máx observado 168, 66% do limite)
    "hyphen_count": "uint16",        # promovido de uint8
    "underscore_count": "uint16",    # promovido de uint8 (máx observado 162, 64% do limite)
    "percent_count": "uint16",       # promovido de uint8 (máx observado 600 — JÁ estourava uint8)
    "at_count": "uint16",            # promovido de uint8
    "question_count": "uint16",      # promovido de uint8 (máx observado 202, 79% do limite)
    "equal_count": "uint16",         # promovido de uint8
    "ampersand_count": "uint16",     # promovido de uint8
    "hash_count": "uint16",          # promovido de uint8
    # Contagens estruturais — mesmo raciocínio (limitadas pela contagem de
    # delimitadores, que agora também é uint16).
    "subdomain_count": "uint16",     # promovido de uint8
    "path_tokens": "uint16",         # promovido de uint8 (máx observado 166, 65% do limite)
    "query_params": "uint16",        # promovido de uint8
    "www_count": "uint16",           # promovido de uint8
    # Booleanas (0/1) — uint8 é suficiente e correto, mantidas.
    "has_ip": "uint8",
    "has_https": "uint8",
    "has_www": "uint8",
    "has_suspicious_tld": "uint8",
    "has_suspicious_extension": "uint8",
    "has_suspicious_keywords": "uint8",
    "has_url_shortener": "uint8",
    "has_double_slash": "uint8",
    "has_at_symbol": "uint8",
    "has_port": "uint8",
    "has_query": "uint8",
    "has_fragment": "uint8",
    # Ratios e entropia — float32, sem risco de overflow (intervalo de
    # valores é pequeno; float32 tem margem de sobra).
    "digit_letter_ratio": "float32",
    "special_char_ratio": "float32",
    "subdomain_ratio": "float32",
    "path_domain_ratio": "float32",
    "url_entropy": "float32",
    "domain_entropy": "float32",
}

assert set(DTYPE_MAP) == set(FEATURE_NAMES), (
    "DTYPE_MAP e get_feature_names() divergiram — alguma feature foi "
    "adicionada/removida em lexical_features.py sem atualizar a auditoria "
    "de dtype (T3.2) em src/features/__init__.py."
)

_UINT_LIMITS = {"uint8": 255, "uint16": 65535, "uint32": 4294967295}

# Colunas de canonização anexadas por `build_feature_frame` (não fazem
# parte das 41 features de treino — são metadados de proveniência/validade).
CANON_COLUMNS = [
    "url_norm", "domain", "etld_plus_one", "path", "query", "fragment",
    "invalid_url", "invalid_reason",
]


def canonicalize_urls(urls: Sequence[str]) -> pd.DataFrame:
    """Aplica `canonicalize_url` em lote. Ver `src/data/url_canonicalization.py`."""
    records = [canonicalize_url(u) for u in urls]
    if not records:
        return pd.DataFrame(columns=CANON_COLUMNS)
    return pd.DataFrame.from_records(records, columns=CANON_COLUMNS)


def extract_features(urls: Sequence[str], strip_scheme: bool = False) -> pd.DataFrame:
    """
    Extrai as 41 features léxicas de uma lista de URLs (RF3).

    `strip_scheme` (SENS-01) tira o prefixo `esquema://` das características
    de texto inteiro; ver `LexicalFeatureExtractor`.

    Corrige B4 internamente via `LexicalFeatureExtractor` (garante scheme
    antes de `urlparse`, sem o qual `domain_length` e outras ~7 features de
    domínio ficavam sistematicamente zeradas — ver
    `features/lexical_features.py`).

    URLs vazias, `None` ou malformadas nunca lançam exceção: caem no vetor
    de features "vazio" (todas as colunas zeradas/False), o mesmo contrato
    que `LexicalFeatureExtractor._get_empty_features()` já garantia.
    """
    extractor = LexicalFeatureExtractor(strip_scheme=strip_scheme)
    rows = []
    for url in urls:
        feats = extractor.extract_all_features(url)
        feats.update(extractor.extract_ratio_features(feats))
        rows.append(feats)

    if not rows:
        X = pd.DataFrame(columns=FEATURE_NAMES)
    else:
        X = pd.DataFrame(rows, columns=FEATURE_NAMES)

    return apply_dtypes(X)


def apply_dtypes(
    X: pd.DataFrame,
    dtype_map: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Aplica o mapa de dtype fixo e auditado (T3.2) — nunca infere dtype a
    partir do min/max observado na execução atual (é exatamente essa
    prática, no pipeline legado, que deixou domain_length e
    special_char_count a um passo de saturar sem que ninguém notasse).

    Args:
        X: DataFrame de features (colunas = subconjunto de FEATURE_NAMES).
        dtype_map: mapa alternativo; por padrão usa `DTYPE_MAP` auditado.
        strict: se True (padrão), levanta `OverflowError` caso algum valor
            observado exceda o limite do dtype-alvo, em vez de truncar
            silenciosamente (esse silêncio foi exatamente o mecanismo de
            B4/B10). Use `strict=False` só para inspeção exploratória.
    """
    dtype_map = dtype_map or DTYPE_MAP
    X = X.copy()

    for col, dtype in dtype_map.items():
        if col not in X.columns:
            continue

        limit = _UINT_LIMITS.get(dtype)
        if limit is not None and len(X) > 0:
            col_max = X[col].max()
            if pd.notna(col_max) and col_max > limit:
                if strict:
                    raise OverflowError(
                        f"Feature '{col}' tem valor máximo {col_max}, que excede "
                        f"o limite de '{dtype}' ({limit}). Promova o dtype em "
                        f"DTYPE_MAP (src/features/__init__.py) antes de prosseguir "
                        f"— não trunque silenciosamente (isso é B4/B10 de novo)."
                    )
                logger.warning(
                    f"Feature '{col}' excede o limite de '{dtype}'",
                    max_observed=float(col_max),
                    limit=limit,
                )
        X[col] = X[col].astype(dtype)

    return X


def build_feature_frame(
    df: pd.DataFrame, url_col: str = "url", strip_scheme: bool = False
) -> pd.DataFrame:
    """
    Orquestra canonização -> extração de features (corrige B3).

    Recebe um DataFrame com ao menos a coluna `url_col` (compatível com o
    contrato de `corpora.py`, SDD §7.3: `label`, `url`, `corpus`,
    `split_oficial`) e devolve um novo DataFrame com:

      - todas as colunas originais de `df`, preservadas e na mesma ordem
        de linhas (mesmo índice);
      - as colunas de canonização (`CANON_COLUMNS`): `url_norm`, `domain`,
        `etld_plus_one`, `path`, `query`, `fragment`, `invalid_url`,
        `invalid_reason`;
      - as 41 features léxicas (`FEATURE_NAMES`), já com dtype auditado
        (T3.2).

    `etld_plus_one` é o que `splits.py` (RF4) precisa para o
    split agrupado por domínio. `invalid_url` deixa explícito quais
    linhas não tinham host algum extraível — a decisão de descartá-las ou
    não é de quem consome este DataFrame, não deste módulo.
    """
    if url_col not in df.columns:
        raise KeyError(f"DataFrame não contém a coluna '{url_col}'")

    urls = df[url_col].tolist()

    canon_df = canonicalize_urls(urls)
    feat_df = extract_features(urls, strip_scheme=strip_scheme)

    out = pd.concat(
        [
            df.reset_index(drop=True),
            canon_df.reset_index(drop=True),
            feat_df.reset_index(drop=True),
        ],
        axis=1,
    )
    out.index = df.index

    n_invalid = int(out["invalid_url"].sum()) if len(out) else 0
    if n_invalid:
        logger.warning(
            "URLs inválidas encontradas durante a canonização",
            invalid_count=n_invalid,
            total=len(out),
            invalid_pct=round(100.0 * n_invalid / len(out), 2) if len(out) else 0.0,
        )

    return out


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "DTYPE_MAP",
    "CANON_COLUMNS",
    "LexicalFeatureExtractor",
    "canonicalize_urls",
    "extract_features",
    "apply_dtypes",
    "build_feature_frame",
    "URLS_CANARIO",
    "URLS_CANARIO_CANONIZACAO",
    "fingerprint_valores",
]


# ---------------------------------------------------------------------------
# Canária de valores (05/08/2026)
#
# O hash do "contrato" (nomes + dtypes) NÃO detecta mudança na *lógica* de
# extração. Comprovado ao corrigir a T3.4: os valores de `subdomain_count`
# mudaram em 10% das linhas reais e tanto `contrato_features_fingerprint()`
# quanto `matrix.config_fingerprint()` continuaram idênticos — o que faria a
# retomada (RH2) aceitar registros da rodada anterior como válidos e misturar
# duas rodadas na mesma tabela, sem rastro. É a mesma classe do B13: resultado
# errado em silêncio.
#
# A defesa é extrair um conjunto fixo de URLs e hashear os **valores**. Muda
# exatamente quando a saída muda, e não muda quando só comentário ou docstring
# mudam — que é o defeito de hashear o fonte. Sem passo manual para esquecer.
# ---------------------------------------------------------------------------

URLS_CANARIO: tuple[str, ...] = (
    "https://www.site.com/x",
    "https://site.com/www.algo",
    "https://site.com:8080/x",
    "https://a.b.bbc.co.uk/caminho?a=1&b=2#frag",
    "http://192.168.1.1/login.php",
    "http://user:pw@a.b.c.site.com.br/x",
    "site.com/sem-scheme",
    "https://xn--80ak6aa92e.com/%41%42",
    "https://bit.ly/abc",
    "https://muito-longo-" + "a" * 200 + ".tk/x",
    # Frente A (FEAT-04): IPv6 com porta, userinfo com IP e domínio privado da PSL.
    "http://[2001:db8::1]:8080/a",
    "http://user@192.168.0.1:8080/",
    "https://user.github.io/x",
)


# Canária da canonização (achado 4 do lote F). O extrator usa a URL crua, mas o
# `url_norm` é a chave da deduplicação (AD-057) e o `etld_plus_one` a do
# Regime B, e os dois vão para o cache. Exercita: `www`, ordem e tracking da
# query, fragmento, porta padrão, esquema ausente e a PSL (sufixo composto e
# domínio privado).
URLS_CANARIO_CANONIZACAO: tuple[str, ...] = (
    "https://WWW.Exemplo.com/a/?b=2&a=1&utm_source=x&fbclid=y#f",
    "http://www.site.co.uk:80/p",
    "http://site.com:443/x",
    "site.com/sem-esquema?z=1&y=2",
    "https://user.github.io/x?gclid=1",
    "https://a.b.bbc.co.uk/x?q=1",
    "https://www.exemplo.com/a?a=1&b=2",
)


def fingerprint_valores(strip_scheme: bool = False) -> str:
    """Hash dos **valores** que o extrator produz para `URLS_CANARIO`.

    Entra na impressão digital da execução (`matrix.config_fingerprint`) e no
    contrato do cache para que qualquer mudança de lógica de extração ou de
    canonização (`url_norm` e `etld_plus_one` de `URLS_CANARIO_CANONIZACAO`)
    invalide o livro-caixa da retomada. Inclui o sha256 da PSL versionada
    (PSL-03) e a opção `strip_scheme`. Custo: poucas dezenas de URLs.
    """
    import hashlib

    quadro = extract_features(URLS_CANARIO, strip_scheme=strip_scheme)
    canonico = quadro[list(FEATURE_NAMES)].to_csv(index=False, float_format="%.10g")
    canon = canonicalize_urls(URLS_CANARIO_CANONIZACAO)
    canonico += canon[["url_norm", "etld_plus_one"]].to_csv(index=False)
    canonico += f"|psl={psl.sha256()}|strip_scheme={int(strip_scheme)}"
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]
