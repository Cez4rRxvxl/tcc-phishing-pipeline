"""T2 (PSL-01): obtencao e versionamento da Public Suffix List. Sem rede nos testes.

O `download` e substituido por monkeypatch; o unico passo com rede do projeto
(`scripts/baixar_psl.py`) e exercitado aqui apenas nas funcoes puras e no
codigo de saida.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CAMINHO_SCRIPT = REPO_ROOT / "scripts" / "baixar_psl.py"
DIR_PSL = REPO_ROOT / "config" / "psl"


@pytest.fixture()
def baixar_psl():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("baixar_psl_sob_teste", CAMINHO_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


CONTEUDO = (
    "// ===BEGIN ICANN DOMAINS===\ncom\nuk\nco.uk\n"
    "// ===END ICANN DOMAINS===\n// ===BEGIN PRIVATE DOMAINS===\ngithub.io\n"
    "// ===END PRIVATE DOMAINS===\n"
).encode("utf-8")


def test_grava_lista_e_metadados_com_sha256_correto(baixar_psl, tmp_path, monkeypatch):
    # PSL-01 AC1: lista em config/psl com URL de origem, data do download e sha256.
    monkeypatch.setattr(baixar_psl, "baixar", lambda url: CONTEUDO)

    codigo = baixar_psl.main(["--destino", str(tmp_path)])

    assert codigo == 0
    dat = tmp_path / "public_suffix_list.dat"
    assert dat.read_bytes() == CONTEUDO
    meta = json.loads((tmp_path / "psl_meta.json").read_text(encoding="utf-8"))
    assert meta["url"] == "https://publicsuffix.org/list/public_suffix_list.dat"
    assert meta["sha256"] == hashlib.sha256(CONTEUDO).hexdigest()
    assert meta["bytes"] == len(CONTEUDO)
    assert meta["baixado_em"]  # data ISO, nao vazia
    assert sorted(p.name for p in tmp_path.iterdir()) == ["psl_meta.json", "public_suffix_list.dat"]


def test_falha_de_rede_sai_com_1_sem_deixar_dat_nem_temporario(baixar_psl, tmp_path, monkeypatch):
    # PSL-01 AC2: download falha -> codigo diferente de zero, sem arquivo parcial.
    def explode(url):
        raise OSError("sem rede")

    monkeypatch.setattr(baixar_psl, "baixar", explode)

    with pytest.raises(SystemExit) as saida:
        baixar_psl.main(["--destino", str(tmp_path)])

    assert saida.value.code == 1
    assert list(tmp_path.iterdir()) == []


def test_falha_antes_do_replace_nao_deixa_arquivo_no_caminho_final(baixar_psl, tmp_path, monkeypatch):
    # PSL-01 AC2: a troca e por temporario + os.replace; falha no replace nao deixa .dat nem .tmp.
    monkeypatch.setattr(baixar_psl, "baixar", lambda url: CONTEUDO)

    def replace_falha(origem, destino):
        raise OSError("disco cheio")

    monkeypatch.setattr(baixar_psl.os, "replace", replace_falha)

    with pytest.raises(SystemExit) as saida:
        baixar_psl.main(["--destino", str(tmp_path)])

    assert saida.value.code == 1
    assert list(tmp_path.iterdir()) == []


def test_arquivo_versionado_tem_secao_privada_e_sha256_do_metadado():
    # T2 Done when: o .dat versionado traz a secao privada e o sha256 bate com o metadado.
    dat = DIR_PSL / "public_suffix_list.dat"
    meta = json.loads((DIR_PSL / "psl_meta.json").read_text(encoding="utf-8"))
    dados = dat.read_bytes()
    assert b"// ===BEGIN PRIVATE DOMAINS===" in dados
    assert hashlib.sha256(dados).hexdigest() == meta["sha256"]
    assert meta["bytes"] == len(dados)
    assert meta["url"].startswith("https://publicsuffix.org/")
