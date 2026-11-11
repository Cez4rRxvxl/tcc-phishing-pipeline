"""PSL-01: baixa a Public Suffix List e a versiona em config/psl/.

    venv\\Scripts\\python.exe scripts\\baixar_psl.py

Este e o unico passo do pipeline que usa a rede para a PSL. Em tempo de execucao
(canonizacao, caracteristicas, particionamento) a lista e lida so do arquivo, com a
secao de dominios privados, e o sha256 do arquivo e conferido contra o metadado.

Grava, em `config/psl/`:
    public_suffix_list.dat  a lista, byte a byte como baixada
    psl_meta.json           url, baixado_em (ISO, UTC), sha256, bytes

Falha sem deixar arquivo parcial: baixa para a memoria, grava temporarios e troca
por `os.replace`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

URL_PSL = "https://publicsuffix.org/list/public_suffix_list.dat"
DIR_PSL = REPO_ROOT / "config" / "psl"
NOME_LISTA = "public_suffix_list.dat"
NOME_META = "psl_meta.json"
TIMEOUT_S = 60

DESCRICAO_CLI = "Baixa a Public Suffix List e grava a lista e o metadado em config/psl/."


def log(msg: str = "") -> None:
    print(msg, flush=True)


def falhar(msg: str) -> None:
    print(f"PSL ABORTADA: {msg}", flush=True)
    sys.exit(1)


def baixar(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resposta:  # noqa: S310
        return resposta.read()


def _gravar_atomico(destino: Path, dados: bytes) -> Path:
    """Grava `dados` num temporario ao lado de `destino`; devolve o temporario."""
    tmp = destino.with_name(destino.name + ".tmp")
    tmp.write_bytes(dados)
    return tmp


def gravar(dados: bytes, url: str, destino: Path) -> dict:
    """Grava lista e metadado em `destino`. Em erro nao deixa temporario nem lista nova."""
    destino.mkdir(parents=True, exist_ok=True)
    meta = {
        "url": url,
        "baixado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(dados).hexdigest(),
        "bytes": len(dados),
    }
    texto_meta = (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    alvo_lista = destino / NOME_LISTA
    alvo_meta = destino / NOME_META
    tmps: list[Path] = []
    try:
        tmps.append(_gravar_atomico(alvo_lista, dados))
        tmps.append(_gravar_atomico(alvo_meta, texto_meta))
        os.replace(tmps[0], alvo_lista)
        os.replace(tmps[1], alvo_meta)
    except BaseException:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        raise
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=DESCRICAO_CLI)
    ap.add_argument("--destino", type=Path, default=DIR_PSL, help="diretorio de saida (padrao: config/psl)")
    args = ap.parse_args(argv)

    log(f"Baixando {URL_PSL}")
    try:
        dados = baixar(URL_PSL)
    except Exception as exc:  # rede, DNS, HTTP, timeout
        falhar(f"download de {URL_PSL} falhou: {exc}")
    if b"// ===BEGIN PRIVATE DOMAINS===" not in dados:
        falhar(f"{URL_PSL} nao trouxe a secao de dominios privados")
    try:
        meta = gravar(dados, URL_PSL, args.destino)
    except OSError as exc:
        falhar(f"gravacao em {args.destino} falhou: {exc}")
    log(f"PSL gravada: {meta['bytes']} bytes, sha256 {meta['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
