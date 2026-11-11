"""Persistência atômica dos escores por amostra e das partições (AD-068, REEX-03).

Cada ajuste (matriz, LOCO, repetição, sensibilidade, HPO fora da diagonal)
grava um parquet com uma linha por linha de teste, e cada rodada grava um
`particoes.parquet` com a partição e o grupo de cada linha elegível. Com isso a
filtragem de sobreposição e o *bootstrap* pareado rodam depois, sem treinar de
novo e sem reextrair características.

Formato das predições (`predicoes/<tipo>/<regime>__<treino>__<modelo>.parquet`)::

    test_corpus : category   # dataset de teste do bloco
    linha       : int32      # posição da linha no parquet do dataset de teste
    y_true      : int8
    score       : float64    # float32 poderia desfazer empates e mudar a AUROC

Formato das partições (`predicoes/particoes.parquet`)::

    corpus, regime, particao : category   # particao em {train, val, test}
    linha                    : int32
    grupo                    : string     # splits.group_keys (eTLD+1 ou reserva)

A gravação é atômica (PRED-04): escreve em `.tmp` e troca com `os.replace`, de
modo que o caminho final tem um arquivo completo ou nenhum. O sha256 do arquivo
vai para o livro-caixa e para o `run_meta.json`; `conferir` o confere na
retomada (PRED-02).

Os arquivos ficam fora do git (`pipeline/results/**/predicoes/`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import splits
from corpora import _sha256_file

SUBDIR = "predicoes"
ARQUIVO_PARTICOES = f"{SUBDIR}/particoes.parquet"

COLUNAS_PREDICAO = ["test_corpus", "linha", "y_true", "score"]
COLUNAS_PARTICAO = ["corpus", "regime", "linha", "particao", "grupo"]


class PredicaoError(RuntimeError):
    """Falha na gravação ou na leitura de predições e partições."""


def _gravar_atomico(destino: Path, escrever) -> str:
    """Chama `escrever(temporario)` e troca por `destino` com `os.replace`.

    Devolve o sha256 do arquivo final. Se a gravação falhar antes da troca, o
    temporário é apagado e o caminho final fica como estava (sem arquivo, ou
    com o arquivo completo anterior).
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    try:
        escrever(temporario)
        os.replace(temporario, destino)
    except BaseException:
        temporario.unlink(missing_ok=True)
        raise
    return _sha256_file(destino)


def _relativo(dir_rodada: Path, caminho: Path) -> str:
    return caminho.relative_to(dir_rodada).as_posix()


def gravar_predicoes(
    dir_rodada: str | Path,
    tipo: str,
    regime: str,
    treino: str,
    modelo: str,
    blocos: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    """Grava as predições de um ajuste e devolve `{"arquivo", "sha256"}`.

    `blocos` é uma lista de `(test_corpus, linhas, y_true, score)`: um bloco por
    *dataset* de teste (a matriz tem quatro por ajuste; o LOCO, um). `arquivo` é
    o caminho relativo a `dir_rodada`, com barras `/`.
    """
    if not blocos:
        raise PredicaoError(
            f"{tipo}/{regime}/{treino}/{modelo}: nenhum bloco de predições para gravar."
        )
    dir_rodada = Path(dir_rodada)
    partes = []
    for test_corpus, linhas, y_true, score in blocos:
        n = len(linhas)
        if len(y_true) != n or len(score) != n:
            raise PredicaoError(
                f"{tipo}/{regime}/{treino}/{modelo}: o bloco '{test_corpus}' tem "
                f"linhas={n}, y_true={len(y_true)} e score={len(score)}; os três "
                f"precisam ter o mesmo tamanho."
            )
        partes.append(
            pd.DataFrame(
                {
                    "test_corpus": np.full(n, test_corpus, dtype=object),
                    "linha": np.asarray(linhas, dtype=np.int32),
                    "y_true": np.asarray(y_true, dtype=np.int8),
                    "score": np.asarray(score, dtype=np.float64),
                }
            )
        )
    df = pd.concat(partes, ignore_index=True)
    df["test_corpus"] = df["test_corpus"].astype("category")

    destino = dir_rodada / SUBDIR / tipo / f"{regime}__{treino}__{modelo}.parquet"
    sha = _gravar_atomico(destino, lambda tmp: df.to_parquet(tmp, index=False))
    return {"arquivo": _relativo(dir_rodada, destino), "sha256": sha}


def conferir(dir_rodada: str | Path, ref: Mapping | None) -> bool:
    """Verdadeiro se o arquivo de `ref` existe e o sha256 bate com o registrado."""
    if not ref or "arquivo" not in ref or "sha256" not in ref:
        return False
    caminho = Path(dir_rodada) / ref["arquivo"]
    if not caminho.is_file():
        return False
    return _sha256_file(caminho) == ref["sha256"]


def ler(dir_rodada: str | Path, ref: Mapping) -> pd.DataFrame:
    """Lê o parquet de `ref`, conferindo antes o sha256 (PredicaoError se falhar)."""
    if not conferir(dir_rodada, ref):
        raise PredicaoError(
            f"predições ausentes ou adulteradas: {Path(dir_rodada) / str(ref.get('arquivo'))}"
        )
    return pd.read_parquet(Path(dir_rodada) / ref["arquivo"])


def _chave(par: tuple[str, str]) -> str:
    return f"{par[0]}/{par[1]}"


def gravar_particoes(
    dir_rodada: str | Path,
    indices: Mapping[tuple[str, str], splits.SplitIndices],
    quadros: Mapping[str, pd.DataFrame],
    contagens_dedup: Mapping[tuple[str, str], dict],
) -> dict:
    """Grava `predicoes/particoes.parquet` e devolve `{"arquivo", "sha256"}`.

    `indices` mapeia `(dataset, regime)` para o `SplitIndices` de
    `splits.particionar_corpus` (posições do quadro original); `quadros`, o
    `dataset` para o seu quadro, que precisa ter a coluna `linha` (posição no
    parquet do *dataset*) e as colunas de `splits.group_keys`. O arquivo tem
    uma linha por linha elegível de cada `(dataset, regime)`, na ordem das
    chaves de `indices` ordenadas e, dentro delas, treino, validação e teste.

    `contagens_dedup` (mesmas chaves) é gravada nos metadados do parquet, na
    chave `dedup`, para que o arquivo seja autocontido.
    """
    dir_rodada = Path(dir_rodada)
    partes = []
    for chave in sorted(indices):
        corpus, regime = chave
        split = indices[chave]
        if corpus not in quadros:
            raise PredicaoError(f"quadro do dataset '{corpus}' ausente em `quadros`.")
        quadro = quadros[corpus]
        if "linha" not in quadro.columns:
            raise PredicaoError(
                f"o quadro de '{corpus}' não tem a coluna 'linha' (identidade da "
                f"linha no parquet do dataset)."
            )
        linha = quadro["linha"].to_numpy()
        grupo = splits.group_keys(quadro).to_numpy()
        for nome, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
            n = len(idx)
            partes.append(
                pd.DataFrame(
                    {
                        "corpus": np.full(n, corpus, dtype=object),
                        "regime": np.full(n, regime, dtype=object),
                        "linha": linha[idx].astype(np.int32),
                        "particao": np.full(n, nome, dtype=object),
                        "grupo": grupo[idx].astype(object),
                    }
                )
            )
    if not partes:
        raise PredicaoError("nenhuma partição para gravar.")
    df = pd.concat(partes, ignore_index=True)
    for col in ("corpus", "regime", "particao"):
        df[col] = df[col].astype("category")
    df["grupo"] = df["grupo"].astype("string")

    import pyarrow as pa
    import pyarrow.parquet as pq

    tabela = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(tabela.schema.metadata or {})
    meta[b"dedup"] = json.dumps(
        {_chave(k): v for k, v in sorted(contagens_dedup.items())}, sort_keys=True
    ).encode("utf-8")
    tabela = tabela.replace_schema_metadata(meta)

    sha = _gravar_atomico(
        dir_rodada / ARQUIVO_PARTICOES, lambda tmp: pq.write_table(tabela, tmp)
    )
    return {"arquivo": ARQUIVO_PARTICOES, "sha256": sha}


__all__ = [
    "ARQUIVO_PARTICOES",
    "COLUNAS_PARTICAO",
    "COLUNAS_PREDICAO",
    "PredicaoError",
    "conferir",
    "gravar_particoes",
    "gravar_predicoes",
    "ler",
]
