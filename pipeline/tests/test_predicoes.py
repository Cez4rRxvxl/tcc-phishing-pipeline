"""T11: persistência atômica das predições e das partições (spec PRED-01, PRED-02, PRED-04).

Dados sintéticos pequenos. Nenhum teste treina modelo.

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import predicoes
import splits
from predicoes import PredicaoError

REPO_ROOT = Path(__file__).resolve().parents[2]

# Escores em que `float32` desfaria valores ou empates: a ida e volta tem de
# devolver exatamente os mesmos 64 bits (PRED-03: a AUROC recalculada depende
# de os empates serem os mesmos).
SCORES_A = np.array(
    [0.1 + 0.2, 1 / 3, np.nextafter(0.5, 1.0), 0.5, 1e-300, 1.0 - 2**-53, 0.0, 1.0]
)
SCORES_B = np.array([2 / 3, np.nextafter(0.5, 0.0), 0.5, 0.25])


def _blocos():
    return [
        ("kaggle03", np.array([5, 9, 2, 7, 11, 0, 3, 8], dtype=np.int64),
         np.array([1, 0, 1, 1, 0, 0, 1, 0]), SCORES_A),
        ("iscx2016", np.array([40, 41, 42, 43]),
         np.array([0, 0, 1, 1]), SCORES_B),
    ]


def _gravar(tmp_path, **kw):
    args = dict(tipo="matriz", regime="etld1", treino="grambeddings", modelo="mlp",
                blocos=_blocos())
    args.update(kw)
    return predicoes.gravar_predicoes(
        tmp_path, args["tipo"], args["regime"], args["treino"], args["modelo"],
        args["blocos"],
    )


class TestIdaEVolta:
    def test_preserva_linha_y_true_e_score_bit_a_bit(self, tmp_path):
        # Done when 1 (PRED-01).
        ref = _gravar(tmp_path)
        df = predicoes.ler(tmp_path, ref)
        esperado_linha = np.concatenate([b[1] for b in _blocos()])
        esperado_y = np.concatenate([b[2] for b in _blocos()])
        esperado_score = np.concatenate([b[3] for b in _blocos()])

        assert df["linha"].to_numpy().tolist() == esperado_linha.tolist()
        assert df["y_true"].to_numpy().tolist() == esperado_y.tolist()
        assert df["score"].dtype == np.float64
        assert (
            df["score"].to_numpy().view(np.int64).tolist()
            == esperado_score.view(np.int64).tolist()
        )

    def test_colunas_tipos_e_blocos_por_dataset_de_teste(self, tmp_path):
        df = predicoes.ler(tmp_path, _gravar(tmp_path))
        assert list(df.columns) == ["test_corpus", "linha", "y_true", "score"]
        assert df["linha"].dtype == np.int32
        assert df["y_true"].dtype == np.int8
        assert df["test_corpus"].astype(str).tolist() == ["kaggle03"] * 8 + ["iscx2016"] * 4

    def test_referencia_tem_caminho_relativo_e_sha256_do_arquivo(self, tmp_path):
        ref = _gravar(tmp_path)
        assert ref["arquivo"] == "predicoes/matriz/etld1__grambeddings__mlp.parquet"
        arquivo = tmp_path / ref["arquivo"]
        assert ref["sha256"] == hashlib.sha256(arquivo.read_bytes()).hexdigest()
        assert set(ref) == {"arquivo", "sha256"}

    def test_mesmas_entradas_dao_o_mesmo_sha256(self, tmp_path):
        a = _gravar(tmp_path / "a")
        b = _gravar(tmp_path / "b")
        assert a["sha256"] == b["sha256"]

    def test_bloco_com_tamanhos_diferentes_levanta_erro(self, tmp_path):
        ruim = [("kaggle03", np.array([1, 2, 3]), np.array([0, 1]), np.array([0.1, 0.2, 0.3]))]
        with pytest.raises(PredicaoError, match="kaggle03"):
            _gravar(tmp_path, blocos=ruim)
        assert not (tmp_path / "predicoes").exists() or not list(
            (tmp_path / "predicoes").rglob("*.parquet")
        )

    def test_sem_blocos_levanta_erro(self, tmp_path):
        with pytest.raises(PredicaoError):
            _gravar(tmp_path, blocos=[])


class TestGravacaoAtomica:
    def test_interrupcao_antes_do_replace_nao_deixa_arquivo_no_caminho_final(
        self, tmp_path, monkeypatch
    ):
        # Done when 2 (PRED-04).
        def replace_que_cai(origem, destino):
            raise KeyboardInterrupt("interrupcao simulada")

        monkeypatch.setattr(predicoes.os, "replace", replace_que_cai)
        with pytest.raises(KeyboardInterrupt):
            _gravar(tmp_path)

        final = tmp_path / "predicoes" / "matriz" / "etld1__grambeddings__mlp.parquet"
        assert not final.exists()
        # Nem temporário sobra.
        assert list((tmp_path / "predicoes").rglob("*")) == [
            tmp_path / "predicoes" / "matriz"
        ]

    def test_interrupcao_preserva_o_arquivo_completo_anterior(self, tmp_path, monkeypatch):
        ref = _gravar(tmp_path)
        antes = (tmp_path / ref["arquivo"]).read_bytes()

        def replace_que_cai(origem, destino):
            raise OSError("disco cheio simulado")

        monkeypatch.setattr(predicoes.os, "replace", replace_que_cai)
        outros = [("kaggle03", np.array([1]), np.array([1]), np.array([0.9]))]
        with pytest.raises(OSError):
            _gravar(tmp_path, blocos=outros)
        assert (tmp_path / ref["arquivo"]).read_bytes() == antes
        assert predicoes.conferir(tmp_path, ref)


class TestConferir:
    def test_verdadeiro_para_arquivo_intacto(self, tmp_path):
        assert predicoes.conferir(tmp_path, _gravar(tmp_path)) is True

    def test_falso_para_arquivo_ausente(self, tmp_path):
        # Done when 3 (PRED-02).
        ref = _gravar(tmp_path)
        (tmp_path / ref["arquivo"]).unlink()
        assert predicoes.conferir(tmp_path, ref) is False

    def test_falso_para_arquivo_alterado(self, tmp_path):
        ref = _gravar(tmp_path)
        arquivo = tmp_path / ref["arquivo"]
        arquivo.write_bytes(arquivo.read_bytes() + b"\x00")
        assert predicoes.conferir(tmp_path, ref) is False

    def test_falso_para_arquivo_regravado_com_outro_conteudo(self, tmp_path):
        ref = _gravar(tmp_path)
        outro = [("kaggle03", np.array([1]), np.array([1]), np.array([0.9]))]
        _gravar(tmp_path, blocos=outro)  # mesmo caminho, outro conteúdo
        assert predicoes.conferir(tmp_path, ref) is False

    @pytest.mark.parametrize("ref", [None, {}, {"arquivo": "x.parquet"}])
    def test_falso_para_referencia_ausente_ou_incompleta(self, tmp_path, ref):
        assert predicoes.conferir(tmp_path, ref) is False


class TestLer:
    def test_arquivo_alterado_levanta_erro_nomeando_o_arquivo(self, tmp_path):
        ref = _gravar(tmp_path)
        arquivo = tmp_path / ref["arquivo"]
        arquivo.write_bytes(arquivo.read_bytes() + b"\x00")
        with pytest.raises(PredicaoError, match="etld1__grambeddings__mlp.parquet"):
            predicoes.ler(tmp_path, ref)


class TestParticoes:
    @pytest.fixture
    def dados(self):
        # `linha` não é a posição: simula quadro de amostra, com identidade
        # própria da linha no parquet do dataset.
        quadro = pd.DataFrame(
            {
                "url": [f"http://x{i}.com/" for i in range(10)],
                "etld_plus_one": ["a.com", "a.com", "b.com", "c.com", "d.com",
                                  "d.com", "e.com", "f.com", "g.com", None],
                "label": [0, 1] * 5,
                "linha": np.arange(10, dtype=np.int32) + 100,
            }
        )
        indices = {
            ("ds", "etld1"): splits.SplitIndices(
                corpus="ds", regime="etld1",
                train=np.array([0, 1, 2, 3]), val=np.array([4, 5]), test=np.array([6, 7, 9]),
            ),
            ("ds", "oficial"): splits.SplitIndices(
                corpus="ds", regime="oficial",
                train=np.array([0, 2, 4]), val=np.array([6]), test=np.array([7, 8]),
            ),
        }
        contagens = {
            ("ds", "etld1"): {"n_antes": 12, "n_final": 10},
            ("ds", "oficial"): {"n_antes": 12, "n_final": 6, "n_repetidas_treino_teste": 1},
        }
        return indices, {"ds": quadro}, contagens

    def test_uma_linha_por_linha_elegivel_com_particao_linha_e_grupo(self, tmp_path, dados):
        indices, quadros, contagens = dados
        ref = predicoes.gravar_particoes(tmp_path, indices, quadros, contagens)
        df = pd.read_parquet(tmp_path / ref["arquivo"])

        assert list(df.columns) == ["corpus", "regime", "linha", "particao", "grupo"]
        b = df[df["regime"] == "etld1"]
        assert len(b) == 9
        assert b.set_index("linha")["particao"].astype(str).to_dict() == {
            100: "train", 101: "train", 102: "train", 103: "train",
            104: "val", 105: "val",
            106: "test", 107: "test", 109: "test",
        }
        a = df[df["regime"] == "oficial"]
        assert a.set_index("linha")["particao"].astype(str).to_dict() == {
            100: "train", 102: "train", 104: "train", 106: "val", 107: "test", 108: "test",
        }
        # O grupo é o de `splits.group_keys`: eTLD+1, ou o grupo de reserva
        # da URL quando não há eTLD+1 (linha 109).
        grupos = b.set_index("linha")["grupo"].to_dict()
        assert grupos[100] == "a.com" and grupos[104] == "d.com"
        assert grupos[109] == splits.NO_ETLD1_PREFIX + "http://x9.com/"

    def test_referencia_com_sha256_e_metadados_de_dedup(self, tmp_path, dados):
        import pyarrow.parquet as pq

        indices, quadros, contagens = dados
        ref = predicoes.gravar_particoes(tmp_path, indices, quadros, contagens)
        arquivo = tmp_path / ref["arquivo"]
        assert ref["arquivo"] == "predicoes/particoes.parquet"
        assert ref["sha256"] == hashlib.sha256(arquivo.read_bytes()).hexdigest()
        assert predicoes.conferir(tmp_path, ref)
        meta = json.loads(pq.read_schema(arquivo).metadata[b"dedup"])
        assert meta["ds/oficial"]["n_repetidas_treino_teste"] == 1
        assert meta["ds/etld1"]["n_final"] == 10

    def test_quadro_sem_coluna_linha_levanta_erro(self, tmp_path, dados):
        indices, quadros, contagens = dados
        quadros = {"ds": quadros["ds"].drop(columns=["linha"])}
        with pytest.raises(PredicaoError, match="linha"):
            predicoes.gravar_particoes(tmp_path, indices, quadros, contagens)
        assert not (tmp_path / "predicoes" / "particoes.parquet").exists()

    def test_interrupcao_nao_deixa_arquivo_de_particoes(self, tmp_path, dados, monkeypatch):
        indices, quadros, contagens = dados

        def replace_que_cai(origem, destino):
            raise OSError("simulada")

        monkeypatch.setattr(predicoes.os, "replace", replace_que_cai)
        with pytest.raises(OSError):
            predicoes.gravar_particoes(tmp_path, indices, quadros, contagens)
        assert list((tmp_path / "predicoes").glob("*")) == []


def test_git_ignora_os_diretorios_de_predicoes():
    # Done when 4 (PRED-04, AC 6) e `.gitignore` ganha `pipeline/raw/`.
    for caminho in (
        "pipeline/results/seed_07/predicoes/x.parquet",
        "pipeline/results/predicoes/x.parquet",
        "pipeline/results/repeticao_seed_42/predicoes/matriz/a.parquet",
        "pipeline/raw/arquivo.csv",
    ):
        r = subprocess.run(
            ["git", "check-ignore", "-q", caminho], cwd=REPO_ROOT, check=False
        )
        assert r.returncode == 0, caminho


def test_git_nao_ignora_os_resultados_versionados():
    for caminho in ("pipeline/results/matrix.json", "pipeline/results/seed_07/matrix.json"):
        r = subprocess.run(
            ["git", "check-ignore", "-q", caminho], cwd=REPO_ROOT, check=False
        )
        assert r.returncode == 1, caminho
