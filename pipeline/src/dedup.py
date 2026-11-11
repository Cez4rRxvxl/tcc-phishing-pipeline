"""Deduplicação pelo `url_norm`, por regime (AD-057, REEX-10).

Decide quais linhas de um *dataset* entram no particionamento e conta o que
saiu. Não treina nada e não olha o eTLD+1: a chave é só o `url_norm`.

Regras, nesta ordem
-------------------
1. `url_norm` nulo (URL inválida) fica no *dataset* e não participa da
   deduplicação: sem chave não há duplicata a decidir.
2. `url_norm` que aparece com os dois rótulos em qualquer linha do *dataset*
   sai por inteiro, nos dois regimes, e é contado como conflito.
3. Regime B (`etld1`): fica a primeira ocorrência de cada `url_norm` no
   *dataset* inteiro. Regime A (`oficial`): fica a primeira ocorrência dentro
   de `split_oficial == "train"` e, separadamente, dentro de `"test"`.
4. Regime A: contam-se os `url_norm` que sobram tanto no treino quanto no
   teste (mesmo rótulo, pois o conflito já saiu).

"Primeira ocorrência" é a da ordem do parquet, que é determinística e não
depende da semente (DEDUP-04).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIMES = ("oficial", "etld1")


class DedupError(RuntimeError):
    """Entrada inválida para a deduplicação."""


@dataclass(frozen=True)
class ResultadoDedup:
    """Posições mantidas (ordenadas, sobre o quadro de entrada) e contagens."""

    posicoes: np.ndarray
    contagens: dict


def deduplicar(
    quadro: pd.DataFrame,
    regime: str,
    url_col: str = "url_norm",
    label_col: str = "label",
    split_col: str = "split_oficial",
) -> ResultadoDedup:
    """Aplica as regras do módulo e devolve as posições mantidas e as contagens.

    As posições indexam `quadro` por `iloc`, em ordem crescente.
    """
    if regime not in REGIMES:
        raise DedupError(f"regime desconhecido: {regime!r}. Válidos: {REGIMES}")
    for col in (url_col, label_col):
        if col not in quadro.columns:
            raise DedupError(f"coluna '{col}' ausente.")
    if regime == "oficial" and split_col not in quadro.columns:
        raise DedupError(f"coluna '{split_col}' ausente: o Regime A depende dela.")

    n = len(quadro)
    url = quadro[url_col].reset_index(drop=True)
    rotulo = quadro[label_col].reset_index(drop=True)
    tem_chave = url.notna().to_numpy()

    # Regra 2: conflito de rótulo, no dataset inteiro.
    n_rotulos = rotulo[tem_chave].groupby(url[tem_chave]).nunique()
    urls_conflito = n_rotulos.index[n_rotulos.to_numpy() > 1]
    em_conflito = tem_chave & url.isin(urls_conflito).to_numpy()

    # Regra 3: primeira ocorrência entre as linhas com chave e sem conflito.
    candidatas = tem_chave & ~em_conflito
    chaves = pd.DataFrame({"url": url})
    if regime == "oficial":
        chaves["split"] = quadro[split_col].reset_index(drop=True)
    repetida = chaves.duplicated(keep="first").to_numpy() & candidatas

    mantidas = ~em_conflito & ~repetida
    posicoes = np.flatnonzero(mantidas)

    contagens = {
        "n_antes": int(n),
        "n_conflito_linhas": int(em_conflito.sum()),
        "n_conflito_urls": int(len(urls_conflito)),
        "n_duplicatas_removidas": int(repetida.sum()),
        "n_final": int(len(posicoes)),
        "n_sem_url_norm": int((~tem_chave).sum()),
    }

    # Regra 4: URLs presentes no treino e no teste mantidos (só Regime A).
    if regime == "oficial":
        split = chaves["split"].to_numpy()
        urls_mantidas = url.to_numpy()
        no_treino = set(urls_mantidas[mantidas & tem_chave & (split == "train")])
        no_teste = set(urls_mantidas[mantidas & tem_chave & (split == "test")])
        contagens["n_repetidas_treino_teste"] = int(len(no_treino & no_teste))

    return ResultadoDedup(posicoes=posicoes, contagens=contagens)


__all__ = ["DedupError", "REGIMES", "ResultadoDedup", "deduplicar"]
