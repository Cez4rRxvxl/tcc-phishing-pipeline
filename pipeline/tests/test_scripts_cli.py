"""A interface de linha de comando dos scripts tem de sobreviver ao console cp1252.

Complementa `test_run_smoke.py::TestSaidaCompativelComOConsoleWindows`, que varre
as chamadas a `print`/`log`/`falhar`. Essa varredura não pegava o caminho
`argparse.ArgumentParser(description=__doc__)`, que transforma a **docstring** em
saída impressa — e as docstrings usam `←`, `→` e `§`, que não existem em cp1252.

Consequência medida em 05/08/2026, antes da correção: `run_all.py --help` e
`run_smoke.py --help` morriam com `UnicodeEncodeError` em vez de imprimir a
ajuda. Para o Lucas, que roda esses scripts numa máquina Windows e para quem
`--help` é o primeiro reflexo diante de um erro, isso se parece exatamente com
"o pipeline está quebrado".

A pendência 10 da §13.4 dizia que "docstrings e comentários estão livres". Era
verdade até `description=__doc__` existir. Estes testes travam a correção.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = ["run_all.py", "run_smoke.py", "run_hpo.py", "verify_environment.py"]


def _python() -> str:
    """O interpretador do venv, ou o corrente se a suíte já roda dentro dele."""
    venv = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_nao_estoura_em_console_cp1252(script):
    """`--help` tem de sair com código 0 e texto codificável em cp1252."""
    caminho = REPO_ROOT / "scripts" / script
    if not caminho.exists():
        pytest.skip(f"{script} não existe neste commit")

    proc = subprocess.run(
        [_python(), str(caminho), "--help"],
        capture_output=True, timeout=180, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"{script} --help saiu com {proc.returncode}.\n"
        f"stderr:\n{proc.stderr.decode('utf-8', errors='replace')}"
    )

    # O código de saída 0 já é a prova: se a ajuda tivesse caractere fora do
    # cp1252, o próprio `print` do argparse teria levantado UnicodeEncodeError
    # dentro do subprocesso e o retorno seria 1. Não reencodamos o texto aqui —
    # a primeira versão deste teste decodificava o stdout como UTF-8 e falhava
    # em `run_smoke.py` por causa do "á" de "máx.", que é cp1252 perfeitamente
    # válido. O erro estava no teste, não no script.
    assert proc.stdout, f"{script} --help nao imprimiu nada"
    proc.stdout.decode("cp1252")  # levanta se o console nao daria conta


@pytest.mark.parametrize("script", ["run_all.py", "run_smoke.py", "run_hpo.py"])
def test_nao_usa_docstring_como_descricao_do_argparse(script):
    """Trava a causa, não só o sintoma.

    Sem isto, alguém reintroduz `description=__doc__` numa refatoração e o
    defeito volta na próxima vez que a docstring ganhar um travessão.
    """
    caminho = REPO_ROOT / "scripts" / script
    if not caminho.exists():
        pytest.skip(f"{script} não existe neste commit")
    fonte = caminho.read_text(encoding="utf-8")
    assert "ArgumentParser(description=__doc__" not in fonte, (
        f"{script} voltou a usar `description=__doc__`. Isso faz a docstring "
        f"virar saída impressa e quebra `--help` no console cp1252 do Windows. "
        f"Use a constante DESCRICAO_CLI, em ASCII puro."
    )
