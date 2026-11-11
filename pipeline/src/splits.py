"""RF4 / T2.4 — particionamento dos corpora nos dois regimes de split (SDD §4 D3).

Este módulo implementa os dois regimes que a matriz cruzada compara:

    Regime A ("oficial")  → usa o split que a fonte distribui (`split_oficial`).
                            Reproduz o protocolo da literatura, **com** o
                            vazamento de domínio medido em SDD §3.3 (45–80%).
    Regime B ("etld1")    → re-split por eTLD+1 com `StratifiedGroupKFold`.
                            Livre de vazamento por construção, e preservando
                            aproximadamente a prevalência de phishing nas três
                            partições (RF4 revisto em 05/08/2026 — ver §11).

A comparação entre os dois é a Tabela T3 do artigo (§7.4); o Regime B é a
Tabela T2, o resultado principal.

Contrato de saída
-----------------
As funções devolvem `SplitIndices`, que carrega **índices posicionais**
(`numpy.ndarray`, compatíveis com `df.iloc[...]`), não copias do DataFrame.
Isso é deliberado: `grambeddings` tem 800.000 linhas × ~50 colunas depois da
extração de features, e devolver três DataFrames materializados triplicaria o
pico de memória sem necessidade. Use `split.frames(df)` quando quiser os
recortes de fato.

Por que existe um conjunto de validação nos dois regimes
--------------------------------------------------------
SDD §4 D6 exige que as métricas secundárias (Precision, Recall, F1, FPR, MCC)
sejam avaliadas no limiar escolhido na **validação do corpus de treino** —
nunca no teste. Logo, todo regime precisa de três partições, não duas.

No Regime B as três saem do mesmo `StratifiedGroupKFold` encadeado, todas
disjuntas por domínio. O balanço de rótulo é **aproximado, nunca exato**: um
domínio com URLs das duas classes é indivisível por definição, então o desvio
residual segue reportado por partição em `split_summary` (`phishing_pct`).

No Regime A a validação é recortada do treino oficial de forma **aleatória
estratificada, e não por domínio**. Isso é intencional e não é um descuido:
o Regime A existe para reproduzir o protocolo da literatura tal como ele é.
Se a validação do Regime A fosse agrupada por domínio, o regime deixaria de
ser "o protocolo oficial" e passaria a ser um híbrido — e a queda medida na
Tabela T3 não poderia mais ser atribuída ao vazamento do split oficial,
porque parte da diferença viria da nossa própria intervenção.

Determinismo (RNF1)
-------------------
Tanto o `GroupShuffleSplit` quanto o `StratifiedGroupKFold` operam sobre os
grupos **únicos**, que o sklearn obtém via `np.unique` (portanto em ordem
canônica, ordenada). Com `random_state` fixo o resultado não depende da ordem
das linhas do DataFrame de entrada — duas execuções sobre o mesmo corpus
produzem exatamente as mesmas partições.

Nenhuma função deste módulo treina modelo algum (SDD §0).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

import dedup

# Seed do projeto. O valor efetivo vem de `config/experiment.yml`
# (`seed:`) — este default existe para uso interativo e nos testes.
DEFAULT_SEED = 42

# Proporções do Regime B. Também declaradas em `config/experiment.yml`.
DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_SIZE = 0.10

REGIMES = ("oficial", "etld1")

# Prefixo dos grupos sintéticos criados para linhas sem eTLD+1 extraível.
# Ver `group_keys`.
NO_ETLD1_PREFIX = "__sem_etld1__::"


class SplitError(RuntimeError):
    """Falha estrutural no particionamento — nunca deve ser silenciada."""


@dataclass(frozen=True)
class SplitIndices:
    """Índices posicionais das três partições de um corpus, em um regime.

    `train`/`val`/`test` são índices para `df.iloc[...]`, disjuntos e cobrindo
    exatamente as linhas elegíveis do corpus.
    """

    corpus: str
    regime: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise SplitError(f"regime desconhecido: {self.regime!r}. Válidos: {REGIMES}")

    @property
    def n_total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def frames(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Materializa as três partições como recortes de `df`."""
        return {
            "train": df.iloc[self.train],
            "val": df.iloc[self.val],
            "test": df.iloc[self.test],
        }

    def as_dict(self) -> dict:
        """Forma serializável (tamanhos, não os índices) para relatório."""
        return {
            "corpus": self.corpus,
            "regime": self.regime,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_total": self.n_total,
        }


# ---------------------------------------------------------------------------
# Chave de agrupamento
# ---------------------------------------------------------------------------


def group_keys(
    df: pd.DataFrame,
    etld_col: str = "etld_plus_one",
    url_col: str = "url",
) -> pd.Series:
    """Devolve a chave de agrupamento por domínio de cada linha.

    Normalmente é o `etld_plus_one` produzido pela canonização
    (`src/features/__init__.py::build_feature_frame`). Uma fração das linhas,
    porém, não tem eTLD+1 extraível — URL vazia, host malformado, lixo binário
    (ver os `dropped_*` de `corpora.py` e o `invalid_url` da canonização).

    Essas linhas **não podem** cair num grupo comum: um grupo único gigante
    chamado "sem domínio" seria indivisível pelo particionador e iria
    inteiro para uma das partições, além de juntar URLs sem nenhuma relação.
    Também não podem simplesmente virar um grupo por linha via hash do índice,
    porque então duas ocorrências da **mesma** URL inválida poderiam cair em
    partições diferentes — vazamento literal, do tipo que este módulo existe
    para impedir.

    A solução é agrupar pela própria URL: determinístico, sem colisão, e
    garante que URLs idênticas fiquem sempre juntas. O prefixo
    `NO_ETLD1_PREFIX` mantém essas chaves distinguíveis de domínios reais.
    """
    if etld_col not in df.columns:
        raise SplitError(
            f"coluna '{etld_col}' ausente. O split por domínio (RF4) opera sobre "
            f"o DataFrame já canonizado — chame features.build_feature_frame() antes."
        )
    if url_col not in df.columns:
        raise SplitError(f"coluna '{url_col}' ausente.")

    etld = df[etld_col].astype("object")
    fallback = NO_ETLD1_PREFIX + df[url_col].astype("string").fillna("").astype("object")
    keys = etld.where(etld.notna() & (etld != ""), fallback)
    return pd.Series(keys.to_numpy(dtype=object), index=df.index, name="group_key")


# ---------------------------------------------------------------------------
# Regime B — re-split por eTLD+1
# ---------------------------------------------------------------------------


def _group_shuffle_once(
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uma extração de `GroupShuffleSplit` sobre índices posicionais 0..n-1.

    Mantida apenas como referência do comportamento anterior ao RF4 revisto
    (05/08/2026) e para o caso de corpus de classe única, onde estratificar não
    faz sentido. O caminho normal do Regime B é `_grupo_estratificado_once`.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # `X` só precisa ter o número de linhas certo; `y` não é usado.
    dummy = np.zeros(len(groups), dtype=np.int8)
    left, right = next(gss.split(dummy, dummy, groups=groups))
    return left, right


def _grupo_estratificado_once(
    groups: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uma extração agrupada **e** estratificada, sobre índices posicionais.

    RF4 revisto em 05/08/2026 (§11). O `GroupShuffleSplit` sorteia domínios sem
    olhar o rótulo, e no ISCX isso produziu validação com 31,14% de phishing
    contra 56,41% no teste — 25,3 p.p. Como o limiar sai da validação (D6) e é
    transportado para as 4 células de teste da linha, o desvio contaminava as
    métricas com limiar de 24 dos 192 registros.

    O `StratifiedGroupKFold` distribui os **mesmos** grupos indivisíveis
    tentando preservar a prevalência, então a garantia de sobreposição zero não
    muda — muda a qualidade do sorteio. O balanço continua aproximado, nunca
    exato: um domínio com URLs das duas classes é indivisível por definição, e
    é por isso que `phishing_pct` segue reportado por partição.

    Limitação assumida: sendo K-fold, a fração é quantizada em `1/k`. Para os
    valores do projeto (0,20 → k=5; 0,125 → k=8) a quantização é exata. Um
    pedido que não caia perto de um recíproco inteiro **levanta erro** em vez
    de devolver silenciosamente uma partição de tamanho diferente do pedido.
    """
    k = max(2, int(round(1.0 / test_size)))
    realizado = 1.0 / k
    if abs(realizado - test_size) > 0.02:
        raise SplitError(
            f"test_size={test_size} não é aproximável por um K-fold: o mais "
            f"próximo é k={k}, que produz {realizado:.4f} "
            f"(desvio {abs(realizado - test_size):.4f} > 0,02). Use uma fração "
            f"próxima de 1/k inteiro (0,5; 0,333; 0,25; 0,2; 0,125; 0,1)."
        )

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    dummy = np.zeros(len(groups), dtype=np.int8)
    left, right = next(sgkf.split(dummy, labels, groups=groups))
    return left, right


def split_etld1(
    df: pd.DataFrame,
    corpus: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    etld_col: str = "etld_plus_one",
    url_col: str = "url",
    label_col: str = "label",
) -> SplitIndices:
    """Regime B — re-split por eTLD+1, livre de vazamento de domínio.

    Ignora `split_oficial` de propósito: o ponto do regime é **descartar** a
    partição da fonte, que é justamente o que vaza (SDD §3.3). Opera sobre o
    corpus inteiro (união do treino e do teste oficiais).

    `val_size` é a fração do corpus **inteiro**, não do treino — a conversão
    para a fração relativa do bloco treino+val é feita internamente.
    """
    if not 0 < test_size < 1:
        raise SplitError(f"test_size fora de (0,1): {test_size}")
    if not 0 <= val_size < 1:
        raise SplitError(f"val_size fora de [0,1): {val_size}")
    if test_size + val_size >= 1:
        raise SplitError(f"test_size + val_size >= 1: {test_size} + {val_size}")

    corpus = corpus or _infer_corpus_name(df)
    groups = group_keys(df, etld_col=etld_col, url_col=url_col).to_numpy()
    n = len(df)
    if n == 0:
        raise SplitError(f"corpus '{corpus}' está vazio")

    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise SplitError(
            f"corpus '{corpus}' tem apenas {n_groups} domínio(s) distinto(s) — "
            f"impossível formar treino/validação/teste disjuntos por domínio."
        )

    # RF4 (revisto em 05/08/2026): o particionamento é agrupado **e**
    # estratificado. Um corpus de classe única não admite estratificação — aí
    # o sorteio agrupado simples é o comportamento correto, não uma exceção.
    if label_col in df.columns:
        labels = df[label_col].to_numpy()
        estratificar = len(np.unique(labels)) > 1
    else:
        labels = None
        estratificar = False

    def _particionar(
        grupos: np.ndarray, rotulos: np.ndarray | None, frac: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if estratificar:
            return _grupo_estratificado_once(grupos, rotulos, frac, seed)
        return _group_shuffle_once(grupos, frac, seed)

    trainval_pos, test_pos = _particionar(groups, labels, test_size)

    if val_size > 0:
        # `val_size` é fração do total; dentro do bloco treino+val ela vale
        # val_size / (1 - test_size).
        rel_val = val_size / (1.0 - test_size)
        sub_train, sub_val = _particionar(
            groups[trainval_pos],
            labels[trainval_pos] if labels is not None else None,
            rel_val,
        )
        train_pos = trainval_pos[sub_train]
        val_pos = trainval_pos[sub_val]
    else:
        train_pos = trainval_pos
        val_pos = np.array([], dtype=trainval_pos.dtype)

    split = SplitIndices(
        corpus=corpus,
        regime="etld1",
        train=np.sort(train_pos),
        val=np.sort(val_pos),
        test=np.sort(test_pos),
    )
    _assert_partition_is_total(split, n, corpus)
    assert_no_domain_overlap(df, split, etld_col=etld_col, url_col=url_col)
    return split


# ---------------------------------------------------------------------------
# Regime A — split oficial da fonte
# ---------------------------------------------------------------------------


def split_official(
    df: pd.DataFrame,
    corpus: str | None = None,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    split_col: str = "split_oficial",
    label_col: str = "label",
) -> SplitIndices:
    """Regime A — o split que a fonte distribui, preservado tal como está.

    O teste é exatamente `split_oficial == "test"`. A validação é recortada do
    treino oficial de forma aleatória **estratificada por rótulo**, sem
    consciência de domínio — ver a nota no topo do módulo sobre por que
    agrupar aqui descaracterizaria o regime.

    `val_size` é a fração do **treino oficial** (diferente do Regime B, onde é
    fração do corpus inteiro): no Regime A o teste tem tamanho fixado pela
    fonte, então não há um "total" sobre o qual as três frações se somem.
    """
    if split_col not in df.columns:
        raise SplitError(
            f"coluna '{split_col}' ausente — o Regime A depende do split "
            f"distribuído pela fonte (ver corpora.py, SDD §7.3)."
        )
    if not 0 <= val_size < 1:
        raise SplitError(f"val_size fora de [0,1): {val_size}")

    corpus = corpus or _infer_corpus_name(df)
    split_values = df[split_col].to_numpy()
    train_pos = np.flatnonzero(split_values == "train")
    test_pos = np.flatnonzero(split_values == "test")

    unknown = set(pd.unique(split_values)) - {"train", "test"}
    if unknown:
        raise SplitError(
            f"corpus '{corpus}': valores inesperados em '{split_col}': {sorted(unknown)}. "
            f"O contrato de SDD §7.3 admite apenas 'train' e 'test'."
        )
    if len(train_pos) == 0 or len(test_pos) == 0:
        raise SplitError(
            f"corpus '{corpus}': split oficial degenerado "
            f"(train={len(train_pos)}, test={len(test_pos)})."
        )

    if val_size > 0:
        val_pos, train_pos = _stratified_holdout(
            df, train_pos, val_size, seed, label_col=label_col
        )
    else:
        val_pos = np.array([], dtype=train_pos.dtype)

    split = SplitIndices(
        corpus=corpus,
        regime="oficial",
        train=np.sort(train_pos),
        val=np.sort(val_pos),
        test=np.sort(test_pos),
    )
    _assert_partition_is_total(split, len(df), corpus)
    return split


def _stratified_holdout(
    df: pd.DataFrame,
    pool: np.ndarray,
    frac: float,
    seed: int,
    label_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta `frac` de `pool`, estratificado por rótulo, de forma determinística.

    Implementado à mão em vez de `train_test_split` porque precisamos operar
    sobre índices posicionais e garantir que cada classe contribua com pelo
    menos uma linha para a validação — sem isso, um corpus muito desbalanceado
    (PDRCNN é 97/3) poderia gerar uma validação de uma única classe, e a
    seleção de limiar sobre ela seria sem sentido.
    """
    if label_col not in df.columns:
        raise SplitError(f"coluna '{label_col}' ausente — necessária para estratificar.")

    labels = df[label_col].to_numpy()[pool]
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []

    for value in np.unique(labels):
        idx_of_class = pool[labels == value]
        n_pick = int(round(frac * len(idx_of_class)))
        n_pick = max(1, min(n_pick, len(idx_of_class) - 1))
        chosen = rng.choice(idx_of_class, size=n_pick, replace=False)
        picked.append(chosen)

    val_pos = np.sort(np.concatenate(picked))
    train_pos = np.sort(np.setdiff1d(pool, val_pos, assume_unique=True))
    if len(train_pos) == 0:
        raise SplitError("recorte de validação consumiu o treino inteiro")
    return val_pos, train_pos


# ---------------------------------------------------------------------------
# Fachada
# ---------------------------------------------------------------------------


def make_split(
    df: pd.DataFrame,
    regime: str,
    corpus: str | None = None,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
) -> SplitIndices:
    """Despacha para o regime pedido. `regime` ∈ {"oficial", "etld1"}."""
    if regime == "oficial":
        return split_official(df, corpus=corpus, val_size=val_size, seed=seed)
    if regime == "etld1":
        return split_etld1(
            df, corpus=corpus, test_size=test_size, val_size=val_size, seed=seed
        )
    raise SplitError(f"regime desconhecido: {regime!r}. Válidos: {REGIMES}")


def particionar_corpus(
    quadro: pd.DataFrame,
    regime: str,
    *,
    corpus: str,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
) -> tuple[SplitIndices, dict]:
    """Deduplica pelo `url_norm` e particiona, num único ponto (AD-057).

    Chama `dedup.deduplicar`, roda `make_split` sobre `quadro.iloc[posicoes]` e
    devolve as posições **do quadro original** (`posicoes[split.train]` etc.),
    junto com as contagens da deduplicação. As três partições são disjuntas e
    cobrem exatamente as posições mantidas pela dedup. No Regime A a validação
    sai do treino já deduplicado; no Regime B nenhum eTLD+1 aparece em duas
    partições (conferido por `make_split`).

    Levanta `SplitError` nomeando *dataset*, regime e partição se alguma
    partição ficar sem uma das duas classes: sem as duas, nem AUROC nem
    limiar fazem sentido.
    """
    resultado = dedup.deduplicar(quadro, regime)
    posicoes = resultado.posicoes
    elegiveis = quadro.iloc[posicoes]
    interno = make_split(
        elegiveis,
        regime,
        corpus=corpus,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )
    split = SplitIndices(
        corpus=corpus,
        regime=regime,
        train=posicoes[interno.train],
        val=posicoes[interno.val],
        test=posicoes[interno.test],
    )
    _assert_partition_is_total(split, len(posicoes), corpus)

    rotulos = quadro["label"].to_numpy()
    partes = [("train", split.train), ("test", split.test)]
    if val_size > 0:
        partes.append(("val", split.val))
    for nome, idx in partes:
        if set(np.unique(rotulos[idx]).tolist()) != {0, 1}:
            raise SplitError(
                f"dataset '{corpus}' / regime '{regime}': a partição '{nome}' "
                f"ficou sem uma das classes depois da deduplicação."
            )
    return split, dict(resultado.contagens)


# ---------------------------------------------------------------------------
# Verificação e diagnóstico
# ---------------------------------------------------------------------------


def _assert_partition_is_total(split: SplitIndices, n_rows: int, corpus: str) -> None:
    """As três partições precisam ser disjuntas e cobrir todas as linhas."""
    union = np.concatenate([split.train, split.val, split.test])
    if len(union) != n_rows:
        raise SplitError(
            f"corpus '{corpus}' / regime '{split.regime}': as partições somam "
            f"{len(union)} linhas, mas o corpus tem {n_rows} — houve perda ou "
            f"duplicação de linhas."
        )
    if len(np.unique(union)) != n_rows:
        raise SplitError(
            f"corpus '{corpus}' / regime '{split.regime}': linhas repetidas entre "
            f"partições."
        )


def domain_overlap(
    df: pd.DataFrame,
    split: SplitIndices,
    etld_col: str = "etld_plus_one",
    url_col: str = "url",
) -> dict[str, int]:
    """Conta domínios compartilhados entre cada par de partições.

    É a medida de SDD §3.3 aplicada às nossas próprias partições. No Regime B
    todos os valores têm de ser 0; no Regime A é esperado que não sejam — é
    exatamente o vazamento que o artigo denuncia.
    """
    keys = group_keys(df, etld_col=etld_col, url_col=url_col).to_numpy()
    sets = {
        "train": set(keys[split.train]),
        "val": set(keys[split.val]),
        "test": set(keys[split.test]),
    }
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def assert_no_domain_overlap(
    df: pd.DataFrame,
    split: SplitIndices,
    etld_col: str = "etld_plus_one",
    url_col: str = "url",
) -> None:
    """Falha alto se qualquer domínio aparecer em duas partições (DoD da T2.4)."""
    overlap = domain_overlap(df, split, etld_col=etld_col, url_col=url_col)
    offenders = {pair: count for pair, count in overlap.items() if count > 0}
    if offenders:
        raise SplitError(
            f"corpus '{split.corpus}' / regime '{split.regime}': vazamento de "
            f"domínio entre partições: {offenders}. O Regime B existe para que "
            f"isso seja impossível — não relaxe esta checagem."
        )


def cross_corpus_domain_overlap(
    train_df: pd.DataFrame,
    train_idx: Iterable[int],
    test_df: pd.DataFrame,
    test_idx: Iterable[int],
    etld_col: str = "etld_plus_one",
    url_col: str = "url",
) -> dict[str, float]:
    """Sobreposição de domínio **entre corpora distintos** (célula fora-diagonal).

    O Regime B elimina o vazamento *dentro* de um corpus, mas não diz nada
    sobre a fora-diagonal: nada impede que um domínio do treino de
    `grambeddings` reapareça no teste de `kaggle03` — os corpora foram
    construídos de forma independente e as fontes de phishing são as mesmas
    poucas feeds públicas. Esta função mede esse resíduo para que ele possa ser
    reportado na seção de Ameaças à Validade em vez de ficar implícito.

    Não é usada para filtrar nada: mexer na fora-diagonal mudaria o que a
    matriz mede (SDD §4 D3).
    """
    train_keys = group_keys(train_df, etld_col=etld_col, url_col=url_col).to_numpy()
    test_keys = group_keys(test_df, etld_col=etld_col, url_col=url_col).to_numpy()
    train_set = set(train_keys[np.asarray(list(train_idx), dtype=int)])
    test_selected = test_keys[np.asarray(list(test_idx), dtype=int)]
    test_set = set(test_selected)
    shared = train_set & test_set
    n_rows = len(test_selected)
    rows_hit = int(np.isin(test_selected, list(shared)).sum()) if shared else 0
    return {
        "test_domains": len(test_set),
        "shared_domains": len(shared),
        "shared_domain_pct": round(100.0 * len(shared) / len(test_set), 2) if test_set else 0.0,
        "test_rows": n_rows,
        "rows_with_shared_domain_pct": round(100.0 * rows_hit / n_rows, 2) if n_rows else 0.0,
    }


def union_domain_overlap(
    train_parts: Sequence[tuple[pd.DataFrame, Iterable[int]]],
    test_df: pd.DataFrame,
    test_idx: Iterable[int],
    etld_col: str = "etld_plus_one",
    url_col: str = "url",
) -> dict[str, float]:
    """Sobreposição de domínio entre a **união** de vários treinos e um teste.

    Por que não bastava `cross_corpus_domain_overlap` (T4.10, §13.4-17)
    ---------------------------------------------------------------------
    A função par-a-par mede uma célula da matriz: treino de A contra teste de
    B. O leave-one-corpus-out (D3-b) treina na **união de três** corpora, e a
    contaminação dele é a união de três sobreposições — que é maior que
    qualquer uma delas e menor que a soma, porque os três conjuntos de
    domínios se intersectam entre si. Medir par a par e somar superestima;
    pegar o maior subestima. Só dá para saber contando a união de verdade.

    Antes desta função o SDD só tinha **limites derivados** da matriz par a
    par: PhishStorm retido em [20,84%; 28,02%], ISCX em [10,03%; 18,19%]. A
    T6 apareceria ao lado da T2 (média 4,79%) carregando até ~6× mais
    contaminação, sem declarar qual.

    `train_parts` é uma lista de `(quadro, índices de treino)` — os mesmos
    `indices[(n, regime)].train` que `matrix.run_loco` concatena. Vale a
    doutrina do D3: **medir e declarar, nunca filtrar**.

    Com um único elemento em `train_parts`, devolve exatamente o mesmo que
    `cross_corpus_domain_overlap` nas chaves que as duas compartilham —
    travado por teste, para que a T6 e a T2 não passem a contar "domínio" de
    formas diferentes (foi esse tipo de divergência que produziu a correção da
    §3.3 registrada na §11 de 04/08).
    """
    if not train_parts:
        raise SplitError("union_domain_overlap exige ao menos um quadro de treino")

    train_set: set = set()
    for df, idx in train_parts:
        keys = group_keys(df, etld_col=etld_col, url_col=url_col).to_numpy()
        train_set |= set(keys[np.asarray(list(idx), dtype=int)])

    test_keys = group_keys(test_df, etld_col=etld_col, url_col=url_col).to_numpy()
    test_selected = test_keys[np.asarray(list(test_idx), dtype=int)]
    test_set = set(test_selected)
    shared = train_set & test_set
    n_rows = len(test_selected)
    rows_hit = int(np.isin(test_selected, list(shared)).sum()) if shared else 0
    return {
        "n_train_corpora": len(train_parts),
        "train_domains": len(train_set),
        "test_domains": len(test_set),
        "shared_domains": len(shared),
        "shared_domain_pct": round(100.0 * len(shared) / len(test_set), 2) if test_set else 0.0,
        "test_rows": n_rows,
        "rows_with_shared_domain_pct": round(100.0 * rows_hit / n_rows, 2) if n_rows else 0.0,
    }


def split_summary(
    df: pd.DataFrame,
    split: SplitIndices,
    label_col: str = "label",
    etld_col: str = "etld_plus_one",
    url_col: str = "url",
) -> dict:
    """Resumo auditável de um split: tamanhos, domínios, balanço e sobreposição.

    O balanço por partição é reportado, não imposto: o `StratifiedGroupKFold`
    embaralha domínios, e um domínio é indivisível — logo o regime eTLD+1 não
    pode estratificar por rótulo nem em princípio. O desvio realizado é um
    fato do corpus e entra no relatório.
    """
    keys = group_keys(df, etld_col=etld_col, url_col=url_col).to_numpy()
    labels = df[label_col].to_numpy() if label_col in df.columns else None

    out = split.as_dict()
    out["n_domains_total"] = int(len(np.unique(keys)))
    out["domain_overlap"] = domain_overlap(df, split, etld_col=etld_col, url_col=url_col)
    out["partitions"] = {}
    for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
        part: dict = {"n_rows": int(len(idx)), "n_domains": int(len(np.unique(keys[idx])))}
        if labels is not None and len(idx):
            n_pos = int((labels[idx] == 1).sum())
            part["n_phishing"] = n_pos
            part["n_benigno"] = int(len(idx) - n_pos)
            part["phishing_pct"] = round(100.0 * n_pos / len(idx), 2)
        out["partitions"][name] = part
    return out


def _infer_corpus_name(df: pd.DataFrame) -> str:
    """Nome do corpus a partir da coluna `corpus` (contrato §7.3), se houver."""
    if "corpus" in df.columns and len(df):
        values = pd.unique(df["corpus"].dropna())
        if len(values) == 1:
            return str(values[0])
        if len(values) > 1:
            raise SplitError(
                f"DataFrame mistura {len(values)} corpora ({sorted(map(str, values))}). "
                f"O split é por corpus — separe antes de particionar."
            )
    return "desconhecido"


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_VAL_SIZE",
    "REGIMES",
    "NO_ETLD1_PREFIX",
    "SplitError",
    "SplitIndices",
    "group_keys",
    "split_etld1",
    "split_official",
    "make_split",
    "particionar_corpus",
    "domain_overlap",
    "assert_no_domain_overlap",
    "cross_corpus_domain_overlap",
    "union_domain_overlap",
    "split_summary",
]


# ---------------------------------------------------------------------------
# Canária de particionamento (05/08/2026)
# ---------------------------------------------------------------------------


def fingerprint_particao() -> str:
    """Hash das partições que o Regime B produz sobre um corpus sintético fixo.

    Entra em `matrix.config_fingerprint` pela mesma razão que
    `features.fingerprint_valores`: a `experiment.yml` declara `test_size` e
    `val_size`, mas **não** o algoritmo de particionamento. Sem esta canária,
    trocar o `GroupShuffleSplit` pelo `StratifiedGroupKFold` deixaria a
    impressão digital idêntica e a retomada (RH2) aceitaria registros da
    rodada anterior como válidos. Ver §11.

    O corpus sintético é determinístico e pequeno; o custo é desprezível.

    Desde a deduplicação (AD-057) a canária passa por `particionar_corpus` nos
    dois regimes e o corpus ganha duplicatas e um conflito de rótulo, para que
    o valor também mude se a lógica da dedup mudar.
    """
    import hashlib

    n_dominios, por_dominio = 60, 5
    linhas = []
    for d in range(n_dominios):
        # Prevalência deliberadamente desigual entre domínios: é isso que
        # distingue um particionador estratificado de um sorteio cego.
        positivos = (d * 7) % (por_dominio + 1)
        for i in range(por_dominio):
            url = f"https://d{d}.exemplo{d % 3}.com/p{i}"
            linhas.append(
                {
                    "url": url,
                    "url_norm": url,
                    "etld_plus_one": f"exemplo-{d}.com",
                    "label": 1 if i < positivos else 0,
                    "split_oficial": "train" if d % 5 else "test",
                }
            )
    # Duplicatas exatas (domínios 0 e 7) e um conflito de rótulo (domínio 3:
    # mesma URL da linha original, rótulo oposto).
    original = list(linhas)
    for base in (original[0], original[1], original[7 * por_dominio]):
        linhas.append(dict(base))
    conflito = dict(original[3 * por_dominio])
    conflito["label"] = 1 - conflito["label"]
    linhas.append(conflito)
    df = pd.DataFrame(linhas)

    partes = []
    for regime in REGIMES:
        split, contagens = particionar_corpus(
            df, regime, corpus="canario", test_size=0.2, val_size=0.1, seed=0
        )
        partes.append(
            "|".join(
                ",".join(map(str, parte.tolist()))
                for parte in (split.train, split.val, split.test)
            )
        )
        partes.append(json.dumps(contagens, sort_keys=True))
    canonico = "#".join(partes)
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]
