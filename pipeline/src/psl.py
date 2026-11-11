"""PSL-02 e PSL-03: instancia unica de `tldextract` lida da PSL versionada.

Todo eTLD+1 do pipeline (canonizacao, caracteristicas, diagnosticos) sai daqui.
A lista vem de `config/psl/public_suffix_list.dat`, com a secao de dominios
privados, e o sha256 do arquivo e conferido contra `config/psl/psl_meta.json`
antes de construir o extrator. Sem rede e sem cache em disco: o `file://` e
servido pelo `requests_file`, que o `tldextract` ja usa.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import tldextract

DIR_PSL = Path(__file__).resolve().parent.parent / "config" / "psl"
CAMINHO_LISTA = DIR_PSL / "public_suffix_list.dat"
CAMINHO_META = DIR_PSL / "psl_meta.json"


class PSLError(RuntimeError):
    """A PSL versionada esta ausente ou nao bate com o metadado."""


def _sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def sha256() -> str:
    """sha256 do arquivo da PSL versionada, conferido contra o metadado."""
    try:
        esperado = json.loads(CAMINHO_META.read_text(encoding="utf-8"))["sha256"]
        obtido = _sha256_arquivo(CAMINHO_LISTA)
    except (OSError, KeyError, ValueError) as exc:
        raise PSLError(f"PSL ilegivel em {CAMINHO_LISTA}: {exc}") from exc
    if obtido != esperado:
        raise PSLError(
            f"sha256 de {CAMINHO_LISTA} difere do metadado {CAMINHO_META}: "
            f"arquivo {obtido}, metadado {esperado}"
        )
    return obtido


@functools.cache
def extrator() -> tldextract.TLDExtract:
    """Instancia unica de `TLDExtract` sobre a PSL versionada, com a secao privada."""
    sha256()
    return tldextract.TLDExtract(
        cache_dir=None,
        suffix_list_urls=(CAMINHO_LISTA.as_uri(),),
        fallback_to_snapshot=False,
        include_psl_private_domains=True,
    )


def etld_mais_um(host: str, privada: bool = True) -> str | None:
    """eTLD+1 em minusculas, ou None para IP, host sem sufixo ou vazio.

    `privada=False` ignora a secao de dominios privados (diagnostico REEX-13).
    """
    if not host:
        return None
    r = extrator()(host, include_psl_private_domains=privada)
    if r.domain and r.suffix:
        return f"{r.domain}.{r.suffix}".lower()
    return None
