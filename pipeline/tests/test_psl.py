"""T3 (PSL-02, PSL-03): instancia unica de tldextract sobre a PSL versionada."""

import json
import shutil
import socket

import pytest

import psl  # pythonpath = ["src"] no pyproject


@pytest.fixture(autouse=True)
def _limpar_cache():
    psl.extrator.cache_clear()
    yield
    psl.extrator.cache_clear()


def test_github_io_secao_privada_e_icann():
    # PSL-02 AC6: user.github.io e o eTLD+1; sem a secao privada, github.io.
    assert psl.etld_mais_um("user.github.io") == "user.github.io"
    assert psl.etld_mais_um("user.github.io", privada=False) == "github.io"


def test_sufixo_composto():
    assert psl.etld_mais_um("a.b.bbc.co.uk") == "bbc.co.uk"


def test_ip_e_host_sem_sufixo_dao_none():
    assert psl.etld_mais_um("192.168.0.1") is None
    assert psl.etld_mais_um("localhost") is None
    assert psl.etld_mais_um("") is None


def test_sem_rede(monkeypatch):
    # PSL-03 AC5: construir e extrair com o socket bloqueado.
    def bloqueado(*args, **kwargs):
        raise OSError("rede bloqueada no teste")

    monkeypatch.setattr(socket, "socket", bloqueado)
    assert psl.etld_mais_um("a.b.bbc.co.uk") == "bbc.co.uk"
    assert psl.etld_mais_um("user.github.io") == "user.github.io"


def test_sha256_divergente_levanta_pslerror_com_o_nome_do_arquivo(tmp_path, monkeypatch):
    # PSL-03 AC4: arquivo adulterado interrompe, nomeando o arquivo.
    lista = tmp_path / "lista_adulterada.dat"
    shutil.copy(psl.CAMINHO_LISTA, lista)
    with lista.open("ab") as f:
        f.write(b"\nadulterado\n")
    meta = tmp_path / "psl_meta.json"
    shutil.copy(psl.CAMINHO_META, meta)
    monkeypatch.setattr(psl, "CAMINHO_LISTA", lista)
    monkeypatch.setattr(psl, "CAMINHO_META", meta)
    with pytest.raises(psl.PSLError, match="lista_adulterada.dat"):
        psl.extrator()


def test_duas_chamadas_devolvem_o_mesmo_objeto():
    assert psl.extrator() is psl.extrator()


def test_sha256_confere_com_o_metadado_versionado():
    esperado = json.loads(psl.CAMINHO_META.read_text(encoding="utf-8"))["sha256"]
    assert psl.sha256() == esperado
