"""Testes para src/corpora.py (SDD tarefas T2.1/T2.2).

Cobre:
- a lógica de reconstrução de linhas TAB coladas (`_split_glued_tab_line`),
  o ponto mais delicado do parser (ver docstring do módulo);
- o schema do formato interno único (SDD §7.3) nos 6 `corpora/<nome>/data.parquet`
  já gerados neste repositório;
- consistência do `corpora/manifest.json` (checksum e contagens) com os
  arquivos parquet reais.

Os testes de schema/manifest são pulados (não falham) se os `.parquet` ainda
não tiverem sido gerados localmente (eles não são versionados no git — ver
`.gitignore`). Para gerá-los:

    python -m src.corpora --raw-dir <dir com os CSVs brutos> --out-dir corpora
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.corpora import CORPUS_SPECS, _split_glued_tab_line

EXPECTED_CORPUS_NAMES = {
    "grambeddings",
    "kaggle03",
    "phishstorm",
    "iscx2016",
    "pdrcnn",
    "sid321axn",
}


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def corpora_dir(project_root: Path) -> Path:
    return project_root / "corpora"


@pytest.fixture(scope="module")
def manifest(corpora_dir: Path) -> dict:
    manifest_path = corpora_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"{manifest_path} não existe")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CORPUS_SPECS — inventário de corpora
# ---------------------------------------------------------------------------


class TestCorpusSpecs:
    def test_seis_corpora_com_nomes_do_sdd(self):
        assert set(CORPUS_SPECS.keys()) == EXPECTED_CORPUS_NAMES

    @pytest.mark.parametrize("name", sorted(EXPECTED_CORPUS_NAMES))
    def test_cada_spec_tem_train_e_test(self, name: str):
        splits = {sf.split for sf in CORPUS_SPECS[name].files}
        assert splits == {"train", "test"}

    def test_grambeddings_usa_virgula_e_rotulo_invertido(self):
        spec = CORPUS_SPECS["grambeddings"]
        assert spec.separator == ","
        # 2 = legítimo (benigno=0), 1 = phishing -- invertido em relação aos demais
        assert spec.label_map == {"2": 0, "1": 1}

    @pytest.mark.parametrize(
        "name", sorted(EXPECTED_CORPUS_NAMES - {"grambeddings"})
    )
    def test_demais_corpora_usam_tab_e_rotulo_menos1_1(self, name: str):
        spec = CORPUS_SPECS[name]
        assert spec.separator == "\t"
        assert spec.label_map == {"-1": 0, "1": 1}


# ---------------------------------------------------------------------------
# _split_glued_tab_line — reconstrução de linhas coladas por falta de \n
# ---------------------------------------------------------------------------


class TestSplitGluedTabLine:
    def test_linha_normal_um_registro(self):
        assert _split_glued_tab_line("-1\thttp://example.com/") == [
            ("-1", "http://example.com/")
        ]

    def test_linha_colada_terminando_em_menos_um(self):
        # Caso real observado em PDRCNN_train.csv
        line = "1\thttps://sdowneymd.com/.well-known/pki-validation/mkk/raiffeisen.hu/index.htm-1\tsourcetune.com"
        assert _split_glued_tab_line(line) == [
            ("1", "https://sdowneymd.com/.well-known/pki-validation/mkk/raiffeisen.hu/index.htm"),
            ("-1", "sourcetune.com"),
        ]

    def test_linha_colada_terminando_em_um_sem_hifen(self):
        # Rótulo colado "1" (phishing) sem hífen -- também observado em PDRCNN
        line = "1\thttp://exodus.mcdir.ru/1\thttp://plataformaeducativa.se.jalisco.gob.mx/elpunto/misc/msci/-1\tqc2.ca"
        assert _split_glued_tab_line(line) == [
            ("1", "http://exodus.mcdir.ru/"),
            ("1", "http://plataformaeducativa.se.jalisco.gob.mx/elpunto/misc/msci/"),
            ("-1", "qc2.ca"),
        ]

    def test_linha_sem_tab_retorna_none(self):
        assert _split_glued_tab_line("linha-sem-separador") is None

    def test_linha_corrompida_sem_rotulo_reconheciivel_retorna_none(self):
        # Caso real observado em kaggle_01-sid321axn: bytes corrompidos no
        # campo do meio, que não terminam em "-1" nem em "1".
        line = "1\t\x1c\x88garbage\x89\tqualquer.com"
        assert _split_glued_tab_line(line) is None

    def test_campo_unico_sem_tab_no_meio_e_none(self):
        assert _split_glued_tab_line("sotag") is None


# ---------------------------------------------------------------------------
# Schema do formato interno único (SDD §7.3), sobre os parquet já gerados
# ---------------------------------------------------------------------------


@pytest.fixture(params=sorted(EXPECTED_CORPUS_NAMES))
def corpus_df(request, corpora_dir: Path):
    name = request.param
    path = corpora_dir / name / "data.parquet"
    if not path.exists():
        pytest.skip(
            f"{path} não existe -- rode "
            "`python -m src.corpora --raw-dir <dir> --out-dir corpora` antes de testar"
        )
    return name, pd.read_parquet(path)


class TestSchemaContratoSDD:
    def test_colunas_do_contrato(self, corpus_df):
        _, df = corpus_df
        assert list(df.columns) == ["label", "url", "corpus", "split_oficial"]

    def test_labels_apenas_0_e_1(self, corpus_df):
        _, df = corpus_df
        assert set(df["label"].unique()) <= {0, 1}

    def test_label_dtype_e_inteiro_pequeno(self, corpus_df):
        _, df = corpus_df
        assert df["label"].dtype == "int8"

    def test_split_oficial_apenas_train_test(self, corpus_df):
        _, df = corpus_df
        assert set(df["split_oficial"].unique()) <= {"train", "test"}

    def test_coluna_corpus_e_consistente(self, corpus_df):
        name, df = corpus_df
        assert set(df["corpus"].unique()) == {name}

    def test_zero_urls_vazias_ou_nulas(self, corpus_df):
        _, df = corpus_df
        assert df["url"].isna().sum() == 0
        assert (df["url"].str.strip() == "").sum() == 0

    def test_corpus_nao_vazio(self, corpus_df):
        _, df = corpus_df
        assert len(df) > 0

    def test_ambos_splits_presentes(self, corpus_df):
        _, df = corpus_df
        counts = df["split_oficial"].value_counts()
        assert counts.get("train", 0) > 0
        assert counts.get("test", 0) > 0


# ---------------------------------------------------------------------------
# manifest.json — checksums e contagens
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_tem_os_6_corpora(self, manifest: dict):
        assert set(manifest.keys()) == EXPECTED_CORPUS_NAMES

    @pytest.mark.parametrize("name", sorted(EXPECTED_CORPUS_NAMES))
    def test_sha256_bate_com_arquivo_real(
        self, name: str, manifest: dict, corpora_dir: Path
    ):
        path = corpora_dir / name / "data.parquet"
        if not path.exists():
            pytest.skip(f"{path} não existe")
        assert manifest[name]["sha256"] == _sha256_file(path)

    @pytest.mark.parametrize("name", sorted(EXPECTED_CORPUS_NAMES))
    def test_contagens_do_manifest_batem_com_parquet(
        self, name: str, manifest: dict, corpora_dir: Path
    ):
        path = corpora_dir / name / "data.parquet"
        if not path.exists():
            pytest.skip(f"{path} não existe")
        df = pd.read_parquet(path)
        entry = manifest[name]

        assert entry["n_rows"] == len(df)
        assert entry["by_label"]["benigno_0"] == int((df["label"] == 0).sum())
        assert entry["by_label"]["phishing_1"] == int((df["label"] == 1).sum())
        assert entry["by_split_oficial"]["train"] == int(
            (df["split_oficial"] == "train").sum()
        )
        assert entry["by_split_oficial"]["test"] == int(
            (df["split_oficial"] == "test").sum()
        )
