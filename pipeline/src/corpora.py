"""Carga e normalização dos corpora públicos (SDD §5.1 RF1, §7.3, tarefas T2.1/T2.2).

Converte os 6 corpora brutos (formatos heterogêneos — separador TAB ou vírgula,
alfabetos de rótulo distintos) para o formato interno único definido em SDD §7.3:

    label          int8     0 = benigno, 1 = phishing
    url            string   URL como distribuída pela fonte
    corpus         string   identificador do corpus
    split_oficial  string   "train" | "test" (conforme a fonte distribui)

Este módulo é **apenas ingestão e normalização**. Não canoniza URL (isso é
tarefa de `features.py`, corrige B3/B4) e não treina nenhum modelo.

Uso (linha de comando):

    python -m src.corpora --raw-dir <dir com os CSVs brutos> --out-dir corpora

Gera `corpora/<nome>/data.parquet` para cada um dos 6 corpora e um
`corpora/manifest.json` com checksum SHA-256, contagem de linhas, contagem por
classe e contagem por split_oficial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Especificação de cada corpus: onde estão os arquivos brutos (relativo a
# raw_dir), qual separador usar e como mapear o rótulo bruto da fonte para o
# rótulo interno (0 = benigno, 1 = phishing).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitFile:
    relpath: str  # caminho relativo a raw_dir
    split: str  # "train" | "test"


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    separator: str  # "\t" ou ","
    label_map: dict  # {"rótulo bruto da fonte": 0 ou 1}
    files: tuple  # tuple[SplitFile, ...]


# Nomes de corpus conforme SDD §4 D2 / instrução da tarefa.
CORPUS_SPECS: dict[str, CorpusSpec] = {
    "iscx2016": CorpusSpec(
        name="iscx2016",
        separator="\t",
        label_map={"-1": 0, "1": 1},
        files=(
            SplitFile("ISCXURL2016_train.csv", "train"),
            SplitFile("ISCXURL2016_test.csv", "test"),
        ),
    ),
    "phishstorm": CorpusSpec(
        name="phishstorm",
        separator="\t",
        label_map={"-1": 0, "1": 1},
        files=(
            SplitFile("PhishStorm_train.csv", "train"),
            SplitFile("PhishStorm_test.csv", "test"),
        ),
    ),
    "pdrcnn": CorpusSpec(
        name="pdrcnn",
        separator="\t",
        label_map={"-1": 0, "1": 1},
        files=(
            SplitFile("PDRCNN_train.csv", "train"),
            SplitFile("PDRCNN_test.csv", "test"),
        ),
    ),
    "kaggle03": CorpusSpec(
        name="kaggle03",
        separator="\t",
        label_map={"-1": 0, "1": 1},
        files=(
            SplitFile("kaggle_03-siddharthkumar25_train.csv", "train"),
            SplitFile("kaggle03_test.csv", "test"),
        ),
    ),
    "sid321axn": CorpusSpec(
        name="sid321axn",
        separator="\t",
        label_map={"-1": 0, "1": 1},
        files=(
            SplitFile("kaggle_01-sid321axn_train.csv", "train"),
            SplitFile("kaggle01-sid321axn_test.csv", "test"),
        ),
    ),
    "grambeddings": CorpusSpec(
        name="grambeddings",
        separator=",",
        # ATENÇÃO: rótulo invertido em relação aos demais corpora.
        # Ver grambeddings/classes.txt: "1:Phish", "2:Legitimate".
        label_map={"2": 0, "1": 1},
        files=(
            SplitFile("grambeddings/train.csv", "train"),
            SplitFile("grambeddings/test.csv", "test"),
        ),
    ),
}


@dataclass
class LoadStats:
    """Contadores de qualidade da carga de um único arquivo bruto.

    Existem para tornar auditável qualquer divergência entre a contagem de
    linhas brutas do arquivo-fonte e a contagem de registros efetivamente
    aproveitados — nunca para esconder uma divergência.
    """

    raw_lines: int = 0
    records_out: int = 0
    reconstructed_lines: int = 0  # linhas brutas com >1 registro colado (ver nota abaixo)
    dropped_unparseable: int = 0  # conteúdo corrompido, sem como recuperar o registro
    dropped_bad_label: int = 0
    dropped_empty_url: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _split_glued_tab_line(line: str) -> list[tuple[str, str]] | None:
    """Reconstrói uma linha bruta TAB-separated em um ou mais registros (label, url).

    Achado de qualidade nos dados brutos (medido em PDRCNN: ~2,9% das linhas;
    PhishStorm_train: 2 linhas): a quebra de linha entre dois registros
    consecutivos por vezes está ausente na fonte, colando o registro N ao N+1
    sem separador — ex.: ``"1\\thttp://a/x.htm-1\\tsourcetune.com"`` são na
    verdade dois registros: ``(1, "http://a/x.htm")`` e ``(-1, "sourcetune.com")``.

    Como o alfabeto de rótulo bruto é fechado ({"-1", "1"}), essa colagem é
    reversível de forma determinística: cada campo "do meio" (produzido por um
    ``split("\\t")`` da linha) termina exatamente com o rótulo do próximo
    registro colado ao final da URL do registro atual.

    Retorna ``None`` quando a linha não pode ser interpretada dessa forma —
    caso observado em ``kaggle_01-sid321axn`` (13 linhas em treino, 5 em
    teste): bytes corrompidos que não terminam em rótulo válido. Essas linhas
    são descartadas e contadas em ``dropped_unparseable``, nunca inventadas.
    """
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    if len(parts) == 2:
        return [(parts[0], parts[1])]

    records: list[tuple[str, str]] = []
    current_label = parts[0]
    for mid in parts[1:-1]:
        if mid.endswith("-1"):
            url, next_label = mid[:-2], "-1"
        elif mid.endswith("1"):
            url, next_label = mid[:-1], "1"
        else:
            return None
        records.append((current_label, url))
        current_label = next_label
    records.append((current_label, parts[-1]))
    return records


def _read_tab_file(path: Path, label_map: dict) -> tuple[list[int], list[str], LoadStats]:
    stats = LoadStats()
    labels: list[int] = []
    urls: list[str] = []
    with path.open("r", encoding="utf-8", newline=None) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line == "":
                continue
            stats.raw_lines += 1
            recs = _split_glued_tab_line(line)
            if recs is None:
                stats.dropped_unparseable += 1
                continue
            if len(recs) > 1:
                stats.reconstructed_lines += 1
            for label_str, url in recs:
                url = url.strip()
                if label_str not in label_map:
                    stats.dropped_bad_label += 1
                    continue
                if not url:
                    stats.dropped_empty_url += 1
                    continue
                labels.append(label_map[label_str])
                urls.append(url)
    stats.records_out = len(labels)
    return labels, urls, stats


def _read_comma_file(path: Path, label_map: dict) -> tuple[list[int], list[str], LoadStats]:
    """Grambeddings: separador vírgula; URL pode conter vírgulas -> split
    apenas na PRIMEIRA vírgula da linha."""
    stats = LoadStats()
    labels: list[int] = []
    urls: list[str] = []
    with path.open("r", encoding="utf-8", newline=None) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line == "":
                continue
            stats.raw_lines += 1
            idx = line.find(",")
            if idx == -1:
                stats.dropped_unparseable += 1
                continue
            label_str, url = line[:idx], line[idx + 1 :].strip()
            if label_str not in label_map:
                stats.dropped_bad_label += 1
                continue
            if not url:
                stats.dropped_empty_url += 1
                continue
            labels.append(label_map[label_str])
            urls.append(url)
    stats.records_out = len(labels)
    return labels, urls, stats


def load_split_file(path: Path, spec: CorpusSpec) -> tuple[pd.DataFrame, LoadStats]:
    """Carrega um único arquivo (um split de um corpus) já no formato
    (label, url), sem as colunas `corpus`/`split_oficial` (adicionadas por
    `load_corpus`)."""
    if spec.separator == "\t":
        labels, urls, stats = _read_tab_file(path, spec.label_map)
    elif spec.separator == ",":
        labels, urls, stats = _read_comma_file(path, spec.label_map)
    else:
        raise ValueError(f"separador não suportado: {spec.separator!r}")
    df = pd.DataFrame({"label": labels, "url": urls})
    return df, stats


def load_corpus(name: str, raw_dir: Path) -> tuple[pd.DataFrame, dict[str, LoadStats]]:
    """Carrega um corpus (todos os seus splits) no formato interno único de
    SDD §7.3, com o DataFrame já contendo as 4 colunas do contrato."""
    if name not in CORPUS_SPECS:
        raise KeyError(f"corpus desconhecido: {name!r}. Válidos: {sorted(CORPUS_SPECS)}")
    spec = CORPUS_SPECS[name]
    frames = []
    stats_by_split: dict[str, LoadStats] = {}
    for split_file in spec.files:
        path = raw_dir / split_file.relpath
        if not path.exists():
            raise FileNotFoundError(
                f"arquivo bruto não encontrado para corpus '{name}': {path}"
            )
        df, stats = load_split_file(path, spec)
        df["corpus"] = name
        df["split_oficial"] = split_file.split
        frames.append(df)
        stats_by_split[split_file.split] = stats

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["label", "url", "corpus", "split_oficial"]]
    combined["label"] = combined["label"].astype("int8")
    combined["url"] = combined["url"].astype("string")
    combined["corpus"] = combined["corpus"].astype("string")
    combined["split_oficial"] = combined["split_oficial"].astype("string")
    return combined, stats_by_split


def load_all_corpora(
    raw_dir: Path,
) -> dict[str, tuple[pd.DataFrame, dict[str, LoadStats]]]:
    """Carrega os 6 corpora. Retorna {nome: (DataFrame, stats_por_split)}."""
    return {name: load_corpus(name, raw_dir) for name in CORPUS_SPECS}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def build_corpora(raw_dir: Path, out_dir: Path, nomes: Iterable[str] | None = None) -> dict:
    """Carrega os 6 corpora, grava `<out_dir>/<nome>/data.parquet` para cada
    um e retorna o dicionário de manifesto (não grava o manifest.json —
    isso é responsabilidade do chamador, ver `main`).

    `nomes` restringe a carga a um subconjunto de `CORPUS_SPECS` (usado por
    `scripts/reconstruir_corpora.py`, que só cobre os corpora cujos CSVs estão
    no repositório de origem); sem ele, são os 6."""
    manifest: dict = {}
    escolhidos = list(CORPUS_SPECS) if nomes is None else list(nomes)
    desconhecidos = [n for n in escolhidos if n not in CORPUS_SPECS]
    if desconhecidos:
        raise KeyError(f"corpus desconhecido: {desconhecidos}. Válidos: {sorted(CORPUS_SPECS)}")
    for name in escolhidos:
        logger.info("carregando corpus %s", name)
        df, stats_by_split = load_corpus(name, raw_dir)

        corpus_dir = out_dir / name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = corpus_dir / "data.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

        n_benign = int((df["label"] == 0).sum())
        n_phishing = int((df["label"] == 1).sum())
        n_train = int((df["split_oficial"] == "train").sum())
        n_test = int((df["split_oficial"] == "test").sum())

        manifest[name] = {
            "parquet_path": str(parquet_path.as_posix()),
            "sha256": _sha256_file(parquet_path),
            "n_rows": int(len(df)),
            "by_label": {"benigno_0": n_benign, "phishing_1": n_phishing},
            "by_split_oficial": {"train": n_train, "test": n_test},
            "load_stats_by_split": {
                split: stats.as_dict() for split, stats in stats_by_split.items()
            },
        }
        logger.info(
            "corpus %s: %d linhas (%d benigno / %d phishing; %d train / %d test)",
            name,
            len(df),
            n_benign,
            n_phishing,
            n_train,
            n_test,
        )
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Diretório com os CSVs brutos dos 6 corpora (não versionado no git)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("corpora"),
        help="Diretório de saída dos parquet normalizados (default: corpora/ na raiz do repo)",
    )
    args = parser.parse_args()

    manifest = build_corpora(args.raw_dir, args.out_dir)

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("manifest gravado em %s", manifest_path)


if __name__ == "__main__":
    main()
