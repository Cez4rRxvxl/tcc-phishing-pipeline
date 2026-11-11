"""PROV-01: reconstroi os corpora a partir dos CSVs brutos do GramBeddings.

    venv\\Scripts\\python.exe scripts\\reconstruir_corpora.py [--baixar] [--raw-dir pipeline/raw]

Le `corpora/fontes_brutas.json`, que registra, para cada CSV bruto, a URL no
repositorio https://github.com/fcdalgic/GramBeddings no commit `a47f863` e o
sha256. O script:

1. obtem cada CSV (de `--raw-dir`, ou baixa da URL com `--baixar` quando falta);
2. confere o sha256 de cada um e SAI COM 1 nomeando o arquivo que divergir;
3. roda `corpora.build_corpora` num diretorio temporario, so para os corpora
   cobertos;
4. compara cada dataset reconstruido com o parquet versionado em `corpora/` por
   `DataFrame.equals` (conteudo e ordem, nao o sha256 do parquet, que depende da
   versao do pyarrow) e confere as contagens do manifesto.

Nao treina nada. O Grambeddings nao esta na pasta `data/` do commit de origem
(so ha um ReadMe que remete ao site dos autores), entao fica fora da
reconstrucao; o JSON registra a origem declarada.

Codigos de saida: 0 reconstrucao identica; 1 arquivo ausente, sha256 divergente,
falha de download, ou dataset reconstruido diferente do versionado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import corpora  # noqa: E402

DIR_RAW = REPO_ROOT / "raw"
DIR_CORPORA = REPO_ROOT / "corpora"
ARQUIVO_FONTES = DIR_CORPORA / "fontes_brutas.json"
TIMEOUT_S = 120

# Campos do manifesto que dependem da maquina ou da versao do pyarrow.
CAMPOS_FORA_DA_COMPARACAO = ("parquet_path", "sha256")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def falhar(msg: str) -> None:
    print(f"RECONSTRUCAO ABORTADA: {msg}", flush=True)
    sys.exit(1)


def baixar(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resposta:  # noqa: S310
        return resposta.read()


def sha256_arquivo(caminho: Path) -> str:
    return corpora._sha256_file(caminho)


def ler_fontes(caminho: Path) -> dict:
    if not caminho.is_file():
        falhar(f"{caminho} nao existe.")
    fontes = json.loads(caminho.read_text(encoding="utf-8"))
    if not fontes.get("arquivos"):
        falhar(f"{caminho} nao registra nenhum arquivo bruto.")
    return fontes


def obter_arquivos(fontes: dict, raw_dir: Path, permitir_download: bool) -> None:
    """Garante que cada CSV de `fontes` exista em `raw_dir`; baixa os que faltam se permitido.

    O download vai para um temporario e e trocado por `os.replace`: uma falha
    nao deixa arquivo parcial no caminho final.
    """
    for nome, info in fontes["arquivos"].items():
        destino = raw_dir / nome
        if destino.is_file():
            continue
        if not permitir_download:
            falhar(
                f"arquivo bruto ausente: {destino}. Rode com --baixar para obter de "
                f"{info['url']}."
            )
        log(f"baixando {nome} de {info['url']}")
        try:
            dados = baixar(info["url"])
        except Exception as exc:  # rede, DNS, HTTP, timeout
            falhar(f"download de {nome} falhou ({info['url']}): {exc}")
        obtido = hashlib.sha256(dados).hexdigest()
        if obtido != info["sha256"]:
            falhar(
                f"sha256 divergente do registrado em fontes_brutas.json no download de "
                f"{nome} ({info['url']}): esperado {info['sha256']}, obtido {obtido}."
            )
        raw_dir.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_name(destino.name + ".tmp")
        try:
            tmp.write_bytes(dados)
            os.replace(tmp, destino)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def conferir_sha256(fontes: dict, raw_dir: Path) -> None:
    """Sai com 1 nomeando o primeiro arquivo cujo sha256 difere do registrado."""
    divergentes = []
    for nome, info in fontes["arquivos"].items():
        obtido = sha256_arquivo(raw_dir / nome)
        if obtido != info["sha256"]:
            divergentes.append((nome, info["sha256"], obtido))
    if divergentes:
        linhas = [f"  {n}: esperado {e}, obtido {o}" for n, e, o in divergentes]
        falhar("sha256 divergente do registrado em fontes_brutas.json:\n" + "\n".join(linhas))


def corpora_cobertos(fontes: dict) -> list[str]:
    """Nomes dos corpora cujos CSVs estao em `fontes`, na ordem de `CORPUS_SPECS`."""
    cobertos = {info["corpus"] for info in fontes["arquivos"].values()}
    desconhecidos = cobertos - set(corpora.CORPUS_SPECS)
    if desconhecidos:
        falhar(f"fontes_brutas.json cita corpus fora de CORPUS_SPECS: {sorted(desconhecidos)}")
    for nome in cobertos:
        esperados = {f.relpath for f in corpora.CORPUS_SPECS[nome].files}
        registrados = {k for k, v in fontes["arquivos"].items() if v["corpus"] == nome}
        if esperados != registrados:
            falhar(
                f"arquivos de '{nome}' em fontes_brutas.json ({sorted(registrados)}) "
                f"diferem dos de CORPUS_SPECS ({sorted(esperados)})."
            )
    return [n for n in corpora.CORPUS_SPECS if n in cobertos]


def comparar(nome: str, reconstruido: Path, versionado: Path,
             manifesto_novo: dict, manifesto_versionado: dict) -> list[str]:
    """Diferencas entre o dataset reconstruido e o versionado (vazia se identicos)."""
    problemas = []
    novo = pd.read_parquet(reconstruido / nome / "data.parquet")
    antigo_caminho = versionado / nome / "data.parquet"
    if not antigo_caminho.is_file():
        return [f"{nome}: parquet versionado ausente em {antigo_caminho}"]
    antigo = pd.read_parquet(antigo_caminho)
    if not novo.equals(antigo):
        problemas.append(
            f"{nome}: conteudo diferente do parquet versionado "
            f"({len(novo)} linhas reconstruidas, {len(antigo)} versionadas)"
        )
    ref = manifesto_versionado.get(nome)
    if ref is None:
        problemas.append(f"{nome}: ausente do manifest.json versionado")
    else:
        for campo, valor in manifesto_novo[nome].items():
            if campo in CAMPOS_FORA_DA_COMPARACAO:
                continue
            if ref.get(campo) != valor:
                problemas.append(f"{nome}: o campo '{campo}' do manifesto difere do versionado")
    return problemas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-dir", type=Path, default=DIR_RAW,
                    help="pasta dos CSVs brutos (padrao: pipeline/raw, fora do git)")
    ap.add_argument("--fontes", type=Path, default=ARQUIVO_FONTES,
                    help="JSON de proveniencia (padrao: corpora/fontes_brutas.json)")
    ap.add_argument("--corpora-dir", type=Path, default=DIR_CORPORA,
                    help="pasta dos parquets versionados (padrao: corpora/)")
    ap.add_argument("--baixar", action="store_true",
                    help="baixa os CSVs que faltarem em --raw-dir")
    args = ap.parse_args(argv)

    fontes = ler_fontes(args.fontes)
    obter_arquivos(fontes, args.raw_dir, args.baixar)
    conferir_sha256(fontes, args.raw_dir)
    log(f"sha256 conferido em {len(fontes['arquivos'])} arquivos")

    nomes = corpora_cobertos(fontes)
    manifesto_path = args.corpora_dir / "manifest.json"
    manifesto_versionado = (
        json.loads(manifesto_path.read_text(encoding="utf-8")) if manifesto_path.is_file() else {}
    )

    problemas: list[str] = []
    with tempfile.TemporaryDirectory(prefix="reconstruir_corpora_") as tmp:
        saida = Path(tmp)
        manifesto_novo = corpora.build_corpora(args.raw_dir, saida, nomes=nomes)
        for nome in nomes:
            achados = comparar(nome, saida, args.corpora_dir, manifesto_novo, manifesto_versionado)
            log(f"{nome}: {'identico ao versionado' if not achados else 'DIFERENTE'}")
            problemas += achados

    for nome, info in fontes.get("fora_do_repositorio", {}).items():
        log(f"{nome}: fora do repositorio de origem, nao reconstruido ({info['origem_declarada']})")

    if problemas:
        falhar("reconstrucao diferente do versionado:\n" + "\n".join(f"  {p}" for p in problemas))
    log(f"reconstrucao identica nos {len(nomes)} datasets cobertos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
