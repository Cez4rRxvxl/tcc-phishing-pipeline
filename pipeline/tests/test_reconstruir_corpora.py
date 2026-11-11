"""T20 (PROV-01): reconstrução dos corpora a partir dos CSVs brutos do GramBeddings.

Os testes usam CSVs sintéticos minúsculos e `baixar` substituído por monkeypatch:
nada vai à rede. O teste com os CSVs reais só roda onde `pipeline/raw/` existe
(a pasta é ignorada pelo git) e apenas lê, confere sha256 e escreve parquet num
diretório temporário; não treina nada.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import corpora

REPO_ROOT = Path(__file__).parent.parent
CAMINHO_SCRIPT = REPO_ROOT / "scripts" / "reconstruir_corpora.py"
FONTES_REAIS = REPO_ROOT / "corpora" / "fontes_brutas.json"
RAW_REAL = REPO_ROOT / "raw"


@pytest.fixture()
def script():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("reconstruir_corpora_sob_teste", CAMINHO_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


TREINO = [
    "-1\thttp://exemplo.com/a",
    "1\thttp://phish.net/login",
    "-1\thttp://outro.org/",
    # Linha com dois registros colados: a reconstrução determinística de corpora.py.
    "1\thttp://cola.com/x-1\tsemlinha.com/y",
]
TESTE = ["-1\thttp://t.com/z", "1\thttp://p.biz/q"]


def _sha(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


@pytest.fixture()
def ambiente(tmp_path):
    """CSVs sintéticos do iscx2016, JSON de fontes e parquets versionados coerentes."""
    raw = tmp_path / "raw"
    raw.mkdir()
    conteudo = {
        "ISCXURL2016_train.csv": ("\n".join(TREINO) + "\n").encode("utf-8"),
        "ISCXURL2016_test.csv": ("\n".join(TESTE) + "\n").encode("utf-8"),
    }
    arquivos = {}
    for nome, dados in conteudo.items():
        (raw / nome).write_bytes(dados)
        arquivos[nome] = {
            "corpus": "iscx2016",
            "url": f"https://exemplo.invalid/{nome}",
            "commit": "a47f863",
            "sha256": _sha(dados),
        }
    fontes = tmp_path / "fontes_brutas.json"
    fontes.write_text(
        json.dumps(
            {
                "arquivos": arquivos,
                "fora_do_repositorio": {
                    "grambeddings": {"origem_declarada": "https://exemplo.invalid/gram"}
                },
            }
        ),
        encoding="utf-8",
    )
    versionado = tmp_path / "corpora"
    manifesto = corpora.build_corpora(raw, versionado, nomes=["iscx2016"])
    (versionado / "manifest.json").write_text(json.dumps(manifesto), encoding="utf-8")
    return {"raw": raw, "fontes": fontes, "versionado": versionado, "conteudo": conteudo,
            "manifesto": manifesto}


def _argv(amb, *extra):
    return ["--raw-dir", str(amb["raw"]), "--fontes", str(amb["fontes"]),
            "--corpora-dir", str(amb["versionado"]), *extra]


def _sai_com_1(script, argv):
    with pytest.raises(SystemExit) as saida:
        script.main(argv)
    return saida.value.code


# ---------------------------------------------------------------------------
# Caminho feliz e proveniência
# ---------------------------------------------------------------------------


def test_reconstrucao_identica_sai_com_0(script, ambiente, capsys):
    assert script.main(_argv(ambiente)) == 0
    saida = capsys.readouterr().out
    assert "iscx2016: identico ao versionado" in saida
    # O Grambeddings fica fora e o script diz de onde ele vem.
    assert "grambeddings: fora do repositorio" in saida


def test_reconstruir_usa_corpora_build_corpora_so_nos_cobertos(script, ambiente, monkeypatch):
    chamadas = []
    original = corpora.build_corpora

    def espia(raw_dir, out_dir, nomes=None):
        chamadas.append(list(nomes))
        return original(raw_dir, out_dir, nomes=nomes)

    monkeypatch.setattr(script.corpora, "build_corpora", espia)
    assert script.main(_argv(ambiente)) == 0
    assert chamadas == [["iscx2016"]]


def test_build_corpora_com_nomes_restringe_e_rejeita_desconhecido(ambiente, tmp_path):
    manifesto = corpora.build_corpora(ambiente["raw"], tmp_path / "saida", nomes=["iscx2016"])
    assert list(manifesto) == ["iscx2016"]
    with pytest.raises(KeyError, match="desconhecido"):
        corpora.build_corpora(ambiente["raw"], tmp_path / "saida2", nomes=["nao_existe"])


# ---------------------------------------------------------------------------
# sha256: CSV adulterado
# ---------------------------------------------------------------------------


def test_csv_adulterado_sai_com_1_nomeando_o_arquivo(script, ambiente, capsys, monkeypatch):
    (ambiente["raw"] / "ISCXURL2016_test.csv").write_bytes(b"-1\thttp://adulterada.com/\n")

    def nao_deveria_construir(*a, **k):
        raise AssertionError("a reconstrucao nao pode rodar com sha256 divergente")

    monkeypatch.setattr(script.corpora, "build_corpora", nao_deveria_construir)

    assert _sai_com_1(script, _argv(ambiente)) == 1
    saida = capsys.readouterr().out
    assert "ISCXURL2016_test.csv" in saida
    assert "ISCXURL2016_train.csv" not in saida  # só o divergente é nomeado
    assert "sha256 divergente" in saida


# ---------------------------------------------------------------------------
# Arquivo ausente e download
# ---------------------------------------------------------------------------


def test_arquivo_ausente_sem_baixar_sai_com_1_nomeando_o_arquivo(script, ambiente, capsys):
    (ambiente["raw"] / "ISCXURL2016_train.csv").unlink()
    assert _sai_com_1(script, _argv(ambiente)) == 1
    saida = capsys.readouterr().out
    assert "ISCXURL2016_train.csv" in saida and "--baixar" in saida


def test_baixar_obtem_so_o_que_falta_e_confere_o_sha256(script, ambiente, monkeypatch):
    por_url = {
        f"https://exemplo.invalid/{nome}": dados for nome, dados in ambiente["conteudo"].items()
    }
    pedidos = []

    def falso_baixar(url):
        pedidos.append(url)
        return por_url[url]

    monkeypatch.setattr(script, "baixar", falso_baixar)
    (ambiente["raw"] / "ISCXURL2016_train.csv").unlink()

    assert script.main(_argv(ambiente, "--baixar")) == 0
    assert pedidos == ["https://exemplo.invalid/ISCXURL2016_train.csv"]
    assert list(ambiente["raw"].glob("*.tmp")) == []


def test_download_corrompido_e_recusado_pelo_sha256(script, ambiente, monkeypatch, capsys):
    monkeypatch.setattr(script, "baixar", lambda url: b"-1\thttp://corrompido.com/\n")
    (ambiente["raw"] / "ISCXURL2016_train.csv").unlink()
    assert _sai_com_1(script, _argv(ambiente, "--baixar")) == 1
    assert "ISCXURL2016_train.csv" in capsys.readouterr().out
    # O arquivo corrompido não chega ao caminho final nem deixa temporário.
    assert not (ambiente["raw"] / "ISCXURL2016_train.csv").exists()
    assert list(ambiente["raw"].glob("*.tmp")) == []


def test_falha_de_rede_sai_com_1_com_a_url_e_sem_arquivo_parcial(
    script, ambiente, monkeypatch, capsys
):
    def explode(url):
        raise OSError("sem rede")

    monkeypatch.setattr(script, "baixar", explode)
    (ambiente["raw"] / "ISCXURL2016_train.csv").unlink()

    assert _sai_com_1(script, _argv(ambiente, "--baixar")) == 1
    assert "https://exemplo.invalid/ISCXURL2016_train.csv" in capsys.readouterr().out
    assert not (ambiente["raw"] / "ISCXURL2016_train.csv").exists()
    assert list(ambiente["raw"].glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Comparação com os parquets versionados
# ---------------------------------------------------------------------------


def test_parquet_versionado_diferente_sai_com_1_nomeando_o_dataset(script, ambiente, capsys):
    caminho = ambiente["versionado"] / "iscx2016" / "data.parquet"
    df = pd.read_parquet(caminho)
    df.iloc[::-1].reset_index(drop=True).to_parquet(caminho, index=False)  # mesma linhas, outra ordem

    assert _sai_com_1(script, _argv(ambiente)) == 1
    saida = capsys.readouterr().out
    assert "iscx2016" in saida and "conteudo diferente" in saida


def test_manifesto_versionado_diferente_sai_com_1(script, ambiente, capsys):
    caminho = ambiente["versionado"] / "manifest.json"
    manifesto = json.loads(caminho.read_text(encoding="utf-8"))
    manifesto["iscx2016"]["n_rows"] += 1
    caminho.write_text(json.dumps(manifesto), encoding="utf-8")

    assert _sai_com_1(script, _argv(ambiente)) == 1
    assert "n_rows" in capsys.readouterr().out


def test_sha256_do_parquet_nao_entra_na_comparacao(script, ambiente):
    # O sha256 do parquet depende da versão do pyarrow: não pode reprovar.
    caminho = ambiente["versionado"] / "manifest.json"
    manifesto = json.loads(caminho.read_text(encoding="utf-8"))
    manifesto["iscx2016"]["sha256"] = "0" * 64
    caminho.write_text(json.dumps(manifesto), encoding="utf-8")
    assert script.main(_argv(ambiente)) == 0


def test_arquivos_do_fontes_diferentes_dos_de_corpus_specs_sao_recusados(script, ambiente, capsys):
    dados = json.loads(ambiente["fontes"].read_text(encoding="utf-8"))
    dados["arquivos"].pop("ISCXURL2016_test.csv")
    ambiente["fontes"].write_text(json.dumps(dados), encoding="utf-8")
    assert _sai_com_1(script, _argv(ambiente)) == 1
    assert "CORPUS_SPECS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# fontes_brutas.json versionado (critério 3 do Done when)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fontes():
    return json.loads(FONTES_REAIS.read_text(encoding="utf-8"))


class TestFontesVersionadas:
    def test_cada_arquivo_registra_url_commit_e_sha256(self, fontes):
        assert fontes["arquivos"]
        for nome, info in fontes["arquivos"].items():
            assert info["commit"] == "a47f863", nome
            assert info["url"].startswith(
                "https://raw.githubusercontent.com/fcdalgic/GramBeddings/a47f863"
            ), nome
            assert len(info["sha256"]) == 64 and set(info["sha256"]) <= set("0123456789abcdef"), nome
            assert info["url"].endswith(info["caminho_no_repositorio"]), nome

    def test_cobre_todos_os_arquivos_de_corpus_specs_menos_os_declarados_fora(self, fontes):
        esperados = {
            f.relpath: nome
            for nome, spec in corpora.CORPUS_SPECS.items()
            for f in spec.files
        }
        fora = {
            arquivo
            for info in fontes["fora_do_repositorio"].values()
            for arquivo in info["arquivos"]
        }
        assert set(fontes["arquivos"]) | fora == set(esperados)
        assert not set(fontes["arquivos"]) & fora
        for nome, info in fontes["arquivos"].items():
            assert info["corpus"] == esperados[nome]

    def test_a_origem_declarada_do_que_esta_fora_esta_registrada(self, fontes):
        gram = fontes["fora_do_repositorio"]["grambeddings"]
        assert gram["origem_declarada"].startswith("https://")
        assert set(gram["arquivos"]) == {"grambeddings/train.csv", "grambeddings/test.csv"}


# ---------------------------------------------------------------------------
# CSVs reais (só onde pipeline/raw existe)
# ---------------------------------------------------------------------------


def test_reconstrucao_com_os_csvs_reais_e_identica_aos_parquets_versionados(script, capsys):
    fontes = json.loads(FONTES_REAIS.read_text(encoding="utf-8"))
    if not all((RAW_REAL / nome).is_file() for nome in fontes["arquivos"]):
        pytest.skip("pipeline/raw/ sem os CSVs (rode scripts/reconstruir_corpora.py --baixar)")

    assert script.main([]) == 0
    saida = capsys.readouterr().out
    for nome in ("iscx2016", "phishstorm", "pdrcnn", "kaggle03", "sid321axn"):
        assert f"{nome}: identico ao versionado" in saida
