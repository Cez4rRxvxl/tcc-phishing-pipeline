"""T4.3 — execução completa: matriz cruzada 4x4, nos 2 regimes, com os 6 modelos.

    venv\\Scripts\\python.exe scripts\\run_all.py

RODE `scripts/run_smoke.py` ANTES. É o portão (SDD §8.2, RK5). Este script faz 48
ajustes sobre corpus completo; descobrir um defeito de fiação aqui custa horas, e
o smoke descobre em segundos.

ESTE SCRIPT TREINA MODELOS. Por SDD §0 ele não deve rodar na máquina do Julio.
A checagem é feita em código, não por memória: sem GPU detectada ele se recusa a
começar e exige `--permitir-cpu` explícito. Ver `checar_restricao_0`.

O laço em si NÃO está aqui
--------------------------
Ele vive em `src/matrix.py` e é compartilhado com o `run_smoke.py`. Isso é
deliberado: a §0 impede validar o laço real nesta máquina, então o portão precisa
exercitar exatamente o código que o Lucas roda. Os 48 ajustes do smoke são a
validação deste script. Se o laço fosse duplicado aqui, o smoke aprovaria um
caminho e o Lucas executaria outro — o risco RK5 intacto.

Este arquivo cuida do que é específico da execução real: corpus completo em vez de
amostra, hiperparâmetros sem os cortes do smoke, cache de features, livro-caixa
para retomada e a checagem da §0.

O que produz (SDD §7.3, RF6/RF7)
--------------------------------
    results/matrix.jsonl   ← livro-caixa append-only; é o que permite retomar
    results/matrix.json    ← consolidado, em ordem canônica
    results/matrix.csv     ← achatado, para o report.py (T5.2)
    results/curves.json    ← pontos ROC/PR em grade fixa, para a figura F1 (T4.6)
    results/splits.json    ← resumo auditável das partições
    results/run_meta.json  ← proveniência: commit, GPU, config, tempos (RNF4)
    results/loco.json      ← leave-one-corpus-out (D3-b), tabela T6
    results/loco.csv       ← idem, achatado
    results/loco_curves.json ← curvas ROC/PR do LOCO
    results/predicoes/       ← escores por ajuste e particoes.parquet (fora do git, AD-068)
    results/sessoes.jsonl    ← uma linha por sessao de execucao (tempo de parede)

Argumentos de rodada adicional (a frente A, AD-055): `--sem-esquema` grava em
results/sensibilidade_sem_esquema/ e `--repeticao` em results/repeticao_seed_42/.
O script recusa iniciar com codigo modificado ou nao rastreado (RUN-03).

48 ajustes = 2 regimes x 4 corpora de treino x 6 modelos.
192 avaliações = cada ajuste testado nos 4 corpora (diagonal + fora-diagonal).

Retomada (RH2): a unidade de trabalho é o ajuste. Cada um grava seus 4 registros
no `.jsonl` com `fsync` assim que termina; ao reiniciar, ajustes completos são
pulados. "Se cair no 30º, retoma do 30º" é literal. Detalhes em `src/matrix.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Raiz do repositório (monorepo, AD-004): `pipeline/` fica um nível abaixo.
GIT_ROOT = REPO_ROOT.parent
# Caminhos de código, relativos à raiz do repositório. Qualquer mudança neles
# sem commit reprova a rodada (RUN-03), e o último commit que os tocou é o
# `commit_codigo` do `run_meta.json` (AD-068).
CAMINHOS_CODIGO = (
    "pipeline/src",
    "pipeline/scripts",
    "pipeline/config",
    "pipeline/corpora",
    "pipeline/requirements.lock",
)

CORPORA_DIR = REPO_ROOT / "corpora"
OUT_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache" / "features"
LEDGER = OUT_DIR / "matrix.jsonl"
# Livro-caixa próprio do LOCO: seus ajustes têm 1 avaliação, os da matriz têm
# 4, e `read_ledger` conta avaliações por ajuste para decidir o que está
# completo. No mesmo arquivo, um ajuste do LOCO pareceria sempre incompleto.
LEDGER_LOCO = OUT_DIR / "loco.jsonl"
# Sessões de execução (RUN-02): uma linha por sessão, anexada num `finally`,
# para sobreviver a retomadas.
SESSOES_NOME = "sessoes.jsonl"

# Descricao curta do `--help`. NAO usar `__doc__` aqui: o console do
# Windows e cp1252 e a docstring tem setas/travessoes, entao `--help`
# morria com UnicodeEncodeError. A pendencia 10 da SDD 13.4 dizia que
# docstring era livre -- e era, ate `description=__doc__` transformar a
# docstring em saida impressa. Mantenha esta string em ASCII puro.
DESCRICAO_CLI = (
    "Execucao completa: matriz cruzada 4x4, 2 regimes, 6 modelos, mais o leave-one-corpus-out (SDD D3-b). Rode scripts/run_smoke.py antes -- e o portao."
)

_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# Progresso legível (RH3)
# ---------------------------------------------------------------------------


def log(msg: str = "") -> None:
    if msg:
        print(f"[{(time.perf_counter() - _T0) / 60:5.1f}min] {msg}", flush=True)
    else:
        print(flush=True)


def falhar(msg: str, *, dica: str = ""):
    print(f"\n{'=' * 74}", flush=True)
    print(f"EXECUCAO ABORTADA: {msg}", flush=True)
    if dica:
        print(f"\n{dica}", flush=True)
    print(f"{'=' * 74}", flush=True)
    sys.exit(1)


def silenciar_ruido() -> None:
    """Sobe os loggers de biblioteca para WARNING, mantendo o progresso legível.

    Ao contrário do smoke, **não** suprimimos `ConvergenceWarning`: numa execução
    real, um MLP que não converge em 200 iterações é informação, não ruído.
    """
    import logging

    for nome in ("data.url_canonicalization", "features.lexical_features", "features",
                 "data.sampling", "utils.io"):
        logging.getLogger(nome).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Proveniência da rodada (RUN-02, RUN-03, AD-068)
# ---------------------------------------------------------------------------


def arvore_suja(raiz: Path | None = None) -> list[str]:
    """Linhas do `git status --porcelain` para os caminhos de código.

    Modificação em arquivo rastreado e arquivo não rastreado reprovam; mudança
    só em `pipeline/results/` não aparece, porque esse caminho não está na lista.
    Se o `git` falhar, devolve uma linha explicando: sem saber o estado da
    árvore, a rodada não pode ser carimbada com um commit.
    """
    raiz = Path(raiz) if raiz is not None else GIT_ROOT
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", *CAMINHOS_CODIGO],
            capture_output=True, text=True, timeout=60, check=False, cwd=str(raiz),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return [f"git indisponivel: {exc}"]
    if proc.returncode != 0:
        return [f"git status falhou: {proc.stderr.strip()}"]
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def checar_arvore_limpa(raiz: Path | None = None) -> None:
    """RUN-03: recusa iniciar com código modificado ou não rastreado."""
    sujas = arvore_suja(raiz)
    if sujas:
        falhar(
            "ha modificacao ou arquivo nao rastreado nos caminhos de codigo.",
            dica=(
                "A rodada e carimbada com o commit do codigo (`commit_codigo`), e\n"
                "so vale se o codigo que roda e o do commit. Faca o commit (ou\n"
                "descarte a mudanca) e rode de novo. Caminhos conferidos:\n"
                f"    {' '.join(CAMINHOS_CODIGO)}\n\n"
                "Arquivos:\n" + "\n".join(f"    {ln}" for ln in sujas)
            ),
        )


def commit_codigo(raiz: Path | None = None) -> str | None:
    """Último commit que tocou os caminhos de código (hash completo), ou `None`."""
    raiz = Path(raiz) if raiz is not None else GIT_ROOT
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", *CAMINHOS_CODIGO],
            capture_output=True, text=True, timeout=60, check=False, cwd=str(raiz),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    saida = proc.stdout.strip()
    return saida if proc.returncode == 0 and saida else None


# `commit_codigo` da sessão em curso, fixado por `checar_commit_da_rodada` no
# início: o `run_meta.json` leva este, não o lido ao fim da fase da matriz.
COMMIT_SESSAO: str | None = None


def checar_commit_da_rodada(out_dir: Path | None = None, recomecar: bool = False) -> str:
    """Fixa o `commit_codigo` da sessão e recusa rodada mista (RUN-03, AD-068).

    Sem commit (git falhou), recusa: a rodada não pode ser carimbada. Se alguma
    sessão anterior da rodada registrou outro `commit_codigo`, recusa a
    retomada e manda usar `--recomecar`; com `recomecar` o livro-caixa antigo é
    arquivado e a conferência não se aplica. Sessões sem o campo (anteriores a
    esta conferência) não entram na comparação.
    """
    global COMMIT_SESSAO
    commit = commit_codigo()
    if commit is None:
        falhar(
            "nao foi possivel obter o commit do codigo (`git log` falhou).",
            dica="A rodada e carimbada com o `commit_codigo`; sem ele nao ha carimbo.",
        )
    if not recomecar:
        anteriores = sorted({s["commit_codigo"] for s in ler_sessoes(out_dir)
                             if s.get("commit_codigo") and s["commit_codigo"] != commit})
        if anteriores:
            falhar(
                "a rodada em andamento foi iniciada com outro commit do codigo.",
                dica=(
                    f"Commit desta sessao: {commit}\n"
                    f"Commit(s) registrado(s) na rodada: {', '.join(anteriores)}\n\n"
                    "Retomar misturaria resultados de codigos diferentes sob um so\n"
                    "carimbo. Rode de novo com `--recomecar` (arquiva o livro-caixa\n"
                    "antigo) ou volte ao commit registrado."
                ),
            )
    COMMIT_SESSAO = commit
    return commit


class ArgumentoInvalido(ValueError):
    """Combinação de argumentos que a rodada consolidada não aceita."""


class Saidas(NamedTuple):
    """Onde uma rodada grava e de onde lê o cache de características."""

    out_dir: Path
    cache_dir: Path
    ledger: Path
    ledger_loco: Path


def resolver_saidas(
    seed: int | None,
    seed_canonica: int,
    sem_esquema: bool = False,
    repeticao: bool = False,
    repo_root: Path | None = None,
) -> Saidas:
    """Diretórios de saída e de cache para as combinações de argumentos.

    ============================  ======================================  ===========================
    argumentos                    saída                                   cache
    ============================  ======================================  ===========================
    (nenhum)                      results/                                cache/features/
    --seed N                      results/seed_NN/                        cache/features/
    --sem-esquema                 results/sensibilidade_sem_esquema/      cache/features_sem_esquema/
    --sem-esquema --seed N        results/sensibilidade_sem_esquema/      cache/features_sem_esquema/
                                  seed_NN/
    --repeticao                   results/repeticao_seed_42/              cache/features/
    ============================  ======================================  ===========================

    `--repeticao` só vale com a semente canônica e não combina com
    `--sem-esquema`: a repetição existe para medir o ruído da GPU na rodada
    principal (REP-01).
    """
    raiz = Path(repo_root) if repo_root is not None else REPO_ROOT
    nao_canonica = seed is not None and seed != seed_canonica
    if repeticao and nao_canonica:
        raise ArgumentoInvalido(
            f"--repeticao exige a semente canonica ({seed_canonica}); recebi --seed {seed}."
        )
    if repeticao and sem_esquema:
        raise ArgumentoInvalido(
            "--repeticao nao combina com --sem-esquema: a repeticao e da rodada principal."
        )

    base = raiz / "results"
    cache = raiz / "cache" / "features"
    if sem_esquema:
        base = base / "sensibilidade_sem_esquema"
        cache = raiz / "cache" / "features_sem_esquema"
    elif repeticao:
        base = base / f"repeticao_seed_{seed_canonica}"
    if nao_canonica:
        base = base / f"seed_{seed:02d}"
    return Saidas(base, cache, base / "matrix.jsonl", base / "loco.jsonl")


def ler_sessoes(out_dir: Path | None = None) -> list[dict]:
    """Sessões já registradas em `sessoes.jsonl` (lista vazia se não há arquivo)."""
    caminho = (Path(out_dir) if out_dir is not None else OUT_DIR) / SESSOES_NOME
    if not caminho.exists():
        return []
    sessoes = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            sessoes.append(json.loads(linha))
    return sessoes


def tempo_parede_s(sessoes: list[dict]) -> float:
    """Soma dos segundos de parede de todas as sessões."""
    return round(sum(float(s["segundos"]) for s in sessoes), 1)


def registrar_sessao(inicio: datetime, fim: datetime, out_dir: Path | None = None,
                     commit: str | None = None) -> None:
    """Anexa a sessão ao `sessoes.jsonl` e atualiza o `run_meta.json`, se existir.

    Chamada num `finally`: uma sessão interrompida também conta. O
    `run_meta.json` ganha `sessoes` e `tempo_parede_s` (a soma, RUN-02). O
    `commit` é o `commit_codigo` da sessão: `checar_commit_da_rodada` o compara
    com o da próxima, para que uma retomada com código novo não vire rodada mista.
    """
    destino = Path(out_dir) if out_dir is not None else OUT_DIR
    destino.mkdir(parents=True, exist_ok=True)
    sessao = {
        "inicio": inicio.isoformat(),
        "fim": fim.isoformat(),
        "segundos": round((fim - inicio).total_seconds(), 1),
    }
    if commit is not None:
        sessao["commit_codigo"] = commit
    with (destino / SESSOES_NOME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(sessao, ensure_ascii=False) + "\n")
    meta_path = destino / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sessoes = ler_sessoes(destino)
        meta["sessoes"] = sessoes
        meta["tempo_parede_s"] = tempo_parede_s(sessoes)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Restrição operacional §0
# ---------------------------------------------------------------------------


def checar_restricao_0(permitir_cpu: bool, gpu) -> None:
    """SDD §0 travado em código: sem GPU, exige consentimento explícito.

    A restrição existe porque esta é a máquina de escrita do Julio (12 núcleos,
    sem GPU) e o treino pertence à máquina do Lucas. Uma regra que vive só num
    documento é uma regra que se esquece às 2 da manhã na véspera do prazo.

    A porta de saída (`--permitir-cpu`) atende a mitigação do RK1: se o Lucas
    ficar indisponível, ISCXURL2016 e PhishStorm rodam em CPU em minutos e dão uma
    matriz 2x2 parcial para escrever em cima. É consentimento explícito, não
    default.
    """
    if gpu.available or permitir_cpu:
        return
    falhar(
        "nenhuma GPU detectada, e SDD §0 proibe treinar na maquina de escrita.",
        dica=(
            "Se voce e o Lucas: confira o driver NVIDIA (`nvidia-smi -L`). Sem\n"
            "ele o XGBoost e o CatBoost cairiam para CPU e a execucao estouraria\n"
            "o orcamento de 2h do RNF5.\n\n"
            "Se voce e o Julio e quer mesmo rodar aqui (mitigacao do RK1 --\n"
            "matriz parcial nos corpora pequenos enquanto o Lucas nao roda):\n\n"
            "    venv\\Scripts\\python.exe scripts\\run_all.py --permitir-cpu \\\n"
            "        --corpora iscx2016,phishstorm\n\n"
            "Nao rode --permitir-cpu com grambeddings ou kaggle03: sao 800 mil e\n"
            "450 mil linhas, e e exatamente o que a secao 0 proibe."
        ),
    )


# ---------------------------------------------------------------------------
# Pré-requisitos (RH1)
# ---------------------------------------------------------------------------


def checar_dependencias() -> None:
    faltando = []
    for mod in ("numpy", "pandas", "pyarrow", "sklearn", "yaml", "tldextract",
                "xgboost", "lightgbm", "catboost"):
        try:
            __import__(mod)
        except ImportError as exc:
            faltando.append(f"{mod} ({exc})")
    if faltando:
        falhar(
            f"dependencias ausentes: {', '.join(faltando)}",
            dica="venv\\Scripts\\python.exe -m pip install -r requirements.lock",
        )


def checar_corpora(nomes: list[str]) -> dict:
    """Confere presença dos parquet e devolve o manifesto."""
    manifesto_path = CORPORA_DIR / "manifest.json"
    if not manifesto_path.exists():
        falhar(f"manifesto ausente: {manifesto_path}")
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))

    ausentes = [n for n in nomes if not (CORPORA_DIR / n / "data.parquet").exists()]
    if ausentes:
        falhar(
            f"corpora ausentes: {', '.join(ausentes)}",
            dica=("Os parquet sao versionados no git desde 04/08/2026 (SDD secao 12,\n"
                  "P5), entao um `git pull` deve traze-los. Se nao vier:\n"
                  "    venv\\Scripts\\python.exe -m src.corpora --raw-dir <dir> --out-dir corpora"),
        )
    for nome in nomes:
        if nome not in manifesto:
            falhar(f"corpus '{nome}' nao esta no manifesto")
    return manifesto


# ---------------------------------------------------------------------------
# Cache de features, validado por checksum
# ---------------------------------------------------------------------------

# Colunas mantidas no cache. `url` fica porque `splits.group_keys` a usa como
# grupo de reserva nas linhas sem eTLD+1. `url_norm`, `path`, `query`,
# `fragment`, `domain` e `invalid_reason` são descartadas: servem à validação da
# canonização, não ao treino, e custam centenas de MB em 800 mil linhas.
#
# `linha` (posição no parquet do dataset, int32) é a identidade da linha nas
# predições persistidas, e `url_norm` é a chave da deduplicação (AD-057): o
# cache guarda os dois para que nenhum dos dois dependa de RangeIndex nem de
# recanonizar.
COLUNAS_CACHE = ["linha", "label", "url", "url_norm", "corpus", "split_oficial",
                 "etld_plus_one", "invalid_url"]


def contrato_features_fingerprint(strip_scheme: bool = False) -> str:
    """Hash do contrato das 41 features: nomes, dtypes **e valores**.

    Os nomes e dtypes vêm da T3.2. A canária de valores
    (`features.fingerprint_valores`) entrou aqui em 05/08/2026, depois de a
    auditoria da rodada 2 comprovar que sem ela **o cache aceita as features da
    rodada anterior**:

        contrato so com nomes+dtypes, na main, depois da T3.4 = b79d8ff8214c265f
        o mesmo valor que results/run_meta.json da rodada 1 registrou

    O `preparar_corpus` valida o cache por SHA-256 do parquet de origem **e**
    por este hash. O parquet não mudou e o contrato não mudava, então um cache
    gerado antes da T3.4 passava — medido no `cache/features/iscx2016.parquet`
    de 04/08: **10,86% das linhas** com `subdomain_count` diferente do que o
    código produz hoje. O resultado seria uma rodada 2 treinada em features da
    rodada 1, estampada com a impressão digital da rodada 2. Mesma classe do
    B13, e pior que o defeito que as canárias foram criadas para fechar: o
    livro-caixa era recusado corretamente e o **dado** entrava errado.

    Por que não bastava mandar o Lucas usar `--sem-cache`: é o passo manual que
    a §13.5 rejeita explicitamente ao justificar por que as canárias existem.
    E o `run_smoke.py` não usa cache, então o portão passaria com as features
    certas enquanto o `run_all.py` carregaria as erradas — o portão não podia
    pegar isto.
    """
    import features
    import psl

    canonico = json.dumps(
        {
            "nomes": list(features.FEATURE_NAMES),
            "dtypes": features.DTYPE_MAP,
            "valores": features.fingerprint_valores(strip_scheme),
            "psl": psl.sha256(),
            "strip_scheme": bool(strip_scheme),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]


def preparar_corpus(nome: str, manifesto: dict, usar_cache: bool,
                    strip_scheme: bool = False):
    """Carrega o corpus, extrai as 41 features e devolve o DataFrame pronto.

    O cache guarda o SHA-256 do parquet de origem, o hash do contrato de
    features, o sha256 da PSL e a opção `strip_scheme`. Qualquer divergência
    recomputa. Um cache velho servindo resultado de código antigo é exatamente o
    tipo de defeito silencioso que este projeto já pagou caro (ver B13). Por
    isso a validação é por conteúdo, não por data de modificação.

    O quadro devolvido tem `linha` (posição no parquet do dataset) e `url_norm`.
    """
    import numpy as np
    import pandas as pd

    import features
    import psl

    cache_parquet = CACHE_DIR / f"{nome}.parquet"
    cache_meta = CACHE_DIR / f"{nome}.meta.json"
    sha_origem = manifesto[nome]["sha256"]
    contrato = contrato_features_fingerprint(strip_scheme)
    sha_psl = psl.sha256()

    if usar_cache and cache_parquet.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        motivo = None
        if meta.get("sha256_origem") != sha_origem:
            motivo = "corpus"
        elif meta.get("psl_sha256") != sha_psl:
            motivo = "PSL"
        elif meta.get("strip_scheme") != bool(strip_scheme):
            motivo = "opcao strip_scheme"
        elif meta.get("contrato") != contrato:
            motivo = "contrato de features"
        if motivo is None:
            quadro = pd.read_parquet(cache_parquet)
            log(f"  {nome:14s} {len(quadro):7d} linhas (cache)")
            return quadro
        log(f"  {nome:14s} cache invalidado ({motivo} mudou) -- recomputando")

    t0 = time.perf_counter()
    bruto = pd.read_parquet(
        CORPORA_DIR / nome / "data.parquet",
        columns=["label", "url", "corpus", "split_oficial"],
    )
    bruto["linha"] = np.arange(len(bruto), dtype=np.int32)
    quadro = features.build_feature_frame(bruto, strip_scheme=strip_scheme)
    quadro = quadro[COLUNAS_CACHE + list(features.FEATURE_NAMES)]

    if usar_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        quadro.to_parquet(cache_parquet, index=False, engine="pyarrow")
        cache_meta.write_text(
            json.dumps({"sha256_origem": sha_origem, "contrato": contrato,
                        "psl_sha256": sha_psl, "strip_scheme": bool(strip_scheme),
                        "n_rows": len(quadro)}, indent=2),
            encoding="utf-8",
        )

    log(f"  {nome:14s} {len(quadro):7d} linhas  "
        f"{quadro['etld_plus_one'].nunique():7d} dominios  "
        f"{int(quadro['invalid_url'].sum()):5d} URL invalida  "
        f"({time.perf_counter() - t0:.0f}s)")
    return quadro


# ---------------------------------------------------------------------------
# Gravação
# ---------------------------------------------------------------------------


def minutos_de_ajuste(registros: list[dict]) -> float:
    """Soma o `fit_seconds` de cada ajuste **uma vez**, em minutos.

    Um ajuste da matriz aparece em 4 registros — as 4 células de teste daquela
    linha reusam o mesmo modelo treinado e carregam o mesmo `fit_seconds` —,
    então somar o campo cru contaria cada treino 4 vezes. A chave do ajuste é
    `(regime, treino, modelo)`, a mesma que o `n_ajustes` usa. No LOCO cada
    ajuste tem 1 avaliação e a deduplicação não muda nada.
    """
    por_ajuste = {}
    for r in registros:
        por_ajuste[(r["regime"], r["train_corpus"], r["model"])] = r["fit_seconds"]
    return round(sum(por_ajuste.values()) / 60, 1)


def montar_meta_proveniencia(resumo_splits: dict, particoes: dict | None,
                             strip_scheme: bool, repeticao: bool) -> dict:
    """Campos de proveniência do `run_meta.json` que a frente A acrescenta (AD-068).

    `commit_codigo` (último commit que tocou o código), `psl_sha256`,
    `strip_scheme`, `repeticao`, `particoes` (caminho e sha256 do
    `particoes.parquet`), `dedup` (contagens por `dataset/regime`, tiradas do
    resumo de partições) e as sessões com o tempo de parede.
    """
    import psl

    sessoes = ler_sessoes()
    return {
        "commit_codigo": COMMIT_SESSAO or commit_codigo(),
        "psl_sha256": psl.sha256(),
        "strip_scheme": bool(strip_scheme),
        "repeticao": bool(repeticao),
        "particoes": particoes,
        "dedup": {k: v["dedup"] for k, v in sorted(resumo_splits.items())
                  if "dedup" in v},
        "sessoes": sessoes,
        "tempo_parede_s": tempo_parede_s(sessoes),
    }


def gravar(registros: list[dict], resumo_splits: dict, config: dict,
           fingerprint: str, gpu, nomes: list[str], *,
           particoes: dict | None = None, strip_scheme: bool = False,
           repeticao: bool = False) -> float:
    import evaluate
    import matrix

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registros = matrix.sort_records(registros)
    anexos = matrix.separar_anexos(registros, OUT_DIR)

    (OUT_DIR / "matrix.json").write_text(
        json.dumps(registros, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evaluate.records_to_dataframe(registros).to_csv(
        OUT_DIR / "matrix.csv", index=False, encoding="utf-8"
    )
    (OUT_DIR / "splits.json").write_text(
        json.dumps(resumo_splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "run_meta.json").write_text(
        json.dumps(
            {
                "commit": evaluate.git_commit_hash(),
                **montar_meta_proveniencia(
                    resumo_splits, particoes, strip_scheme, repeticao
                ),
                "run_ids": sorted({r["run_id"] for r in registros}),
                "config_fingerprint": fingerprint,
                "contrato_features": contrato_features_fingerprint(strip_scheme),
                "gpu": gpu.describe(),
                "gpu_disponivel": gpu.available,
                "corpora": nomes,
                "regimes": list(config["regimes"]),
                "n_registros": len(registros),
                "n_ajustes": len({(r["regime"], r["train_corpus"], r["model"])
                                  for r in registros}),
                # Custo. Até 06/08/2026 havia aqui um único campo
                # `minutos_desta_execucao`, e ele era gravado **antes** de o
                # LOCO começar (esta função roda antes do bloco do LOCO, de
                # propósito — ver o comentário na `main`). O campo dizia 15,6
                # min numa execução que levou ~50: subnotificava o custo pelo
                # tempo do LOCO inteiro, que é a fase mais cara. Nenhum número
                # científico dependia disso, mas a tabela de orçamento da
                # §13.7 e a verificação do RNF5 dependiam, e o único registro
                # do total verdadeiro ia para a tela e não para arquivo.
                # Agora: os dois campos de fase são gravados separadamente e o
                # `minutos_desta_execucao` é reescrito pelo
                # `atualizar_custo_com_loco()` quando o LOCO termina. Com
                # `--sem-loco` a execução acaba aqui e o valor já está certo.
                "minutos_desta_execucao": round((time.perf_counter() - _T0) / 60, 1),
                "minutos_fase_matriz": round((time.perf_counter() - _T0) / 60, 1),
                "minutos_fase_loco": None,
                # Derivados dos `fit_seconds` dos registros: é o custo de
                # treino puro, independente de relógio de parede, e portanto
                # a única parte do custo que se recupera de um `results/` já
                # gravado por uma versão antiga do script.
                "minutos_ajuste_matriz": minutos_de_ajuste(registros),
                "minutos_ajuste_loco": None,
                "python": platform.python_version(),
                "plataforma": platform.platform(),
                "seed": int(config["seed"]),
                "config": {
                    "seed": config["seed"],
                    "split": config["split"],
                    "avaliacao": config["avaliacao"],
                    "modelos": config["modelos"],
                },
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    for arq in ["matrix.json", "matrix.csv", *anexos, "splits.json", "run_meta.json"]:
        log(f"gravado: {OUT_DIR.name}/{arq}")
    return round((time.perf_counter() - _T0) / 60, 1)


def atualizar_custo_com_loco(registros_loco: list[dict],
                             minutos_fase_matriz: float) -> None:
    """Reescreve os campos de custo do `run_meta.json` depois do LOCO.

    Existe porque a `gravar()` roda antes do LOCO (para que a matriz esteja em
    disco se o LOCO cair) e, portanto, não tem como conhecer o custo total.
    Reabre o arquivo em vez de receber o dicionário para não mudar a ordem de
    gravação: se esta função falhar, o `run_meta.json` da matriz continua
    válido, só com o custo do LOCO ausente.
    """
    caminho = OUT_DIR / "run_meta.json"
    meta = json.loads(caminho.read_text(encoding="utf-8"))
    total = round((time.perf_counter() - _T0) / 60, 1)
    meta["minutos_desta_execucao"] = total
    meta["minutos_fase_matriz"] = minutos_fase_matriz
    # `total` e `minutos_fase_matriz` saem do mesmo `_T0`, no mesmo processo,
    # então a subtração é consistente por construção. O `max` existe para o
    # caso em que isso deixe de valer (alguém chamar esta função com um valor
    # lido de um `run_meta.json` de outra execução): duração negativa é pior
    # que duração zerada, porque parece um dado e não parece um defeito.
    meta["minutos_fase_loco"] = max(0.0, round(total - minutos_fase_matriz, 1))
    meta["minutos_ajuste_loco"] = minutos_de_ajuste(registros_loco)
    meta["minutos_ajuste_total"] = round(
        meta["minutos_ajuste_matriz"] + meta["minutos_ajuste_loco"], 1
    )
    caminho.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"custo registrado: {total:.1f} min no total "
        f"({minutos_fase_matriz:.1f} matriz + {meta['minutos_fase_loco']:.1f} LOCO), "
        f"{meta['minutos_ajuste_total']:.1f} min de ajuste")


def gravar_loco(registros: list[dict]) -> None:
    """Grava o leave-one-corpus-out (D3-b) em arquivos próprios.

    Separado da matriz porque são tabelas diferentes do artigo (T6 vs T2/T3) e
    porque misturá-los faria `matrix.json` conter linhas cujo `train_corpus`
    não é um corpus — o `report.py` teria de filtrar, e uma leitura desatenta
    contaria 240 células numa matriz de 192.
    """
    import evaluate
    import matrix

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registros = matrix.sort_records(registros)
    anexos = matrix.separar_anexos(registros, OUT_DIR, prefixo="loco_")

    (OUT_DIR / "loco.json").write_text(
        json.dumps(registros, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evaluate.records_to_dataframe(registros).to_csv(
        OUT_DIR / "loco.csv", index=False, encoding="utf-8"
    )
    for arq in ["loco.json", "loco.csv", *anexos]:
        log(f"gravado: {OUT_DIR.name}/{arq}")


def resumo_diagonal(registros: list[dict]) -> None:
    """Diagonal por regime — a leitura que interessa de imediato.

    Não elege campeão: ranquear por métrica única foi como B6/B7 nasceram. Só
    mostra o maior AUROC por célula, para conferência rápida.
    """
    log()
    log("AUROC na diagonal (desempenho na propria distribuicao):")
    por_regime: dict = {}
    for rec in registros:
        if not rec["diagonal"] or rec["metrics"]["auroc"] is None:
            continue
        por_regime.setdefault(rec["regime"], {}).setdefault(
            rec["train_corpus"], {})[rec["model"]] = rec["metrics"]["auroc"]

    for regime in sorted(por_regime):
        log(f"  regime '{regime}':")
        for corpus in sorted(por_regime[regime]):
            valores = por_regime[regime][corpus]
            melhor = max(valores, key=valores.get)
            log(f"    {corpus:14s} maior AUROC: {melhor:20s} {valores[melhor]:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRICAO_CLI)
    parser.add_argument("--permitir-cpu", action="store_true",
                        help="autoriza treinar sem GPU (SDD secao 0). Use so para a "
                             "matriz parcial do RK1.")
    parser.add_argument("--corpora", type=str, default=None,
                        help="subconjunto separado por virgula (ex.: iscx2016,phishstorm)")
    parser.add_argument("--sem-cache", action="store_true",
                        help="recomputa as features em vez de usar cache/features/")
    parser.add_argument("--recomecar", action="store_true",
                        help="arquiva o livro-caixa e comeca do zero")
    parser.add_argument("--sem-loco", action="store_true",
                        help="pula o leave-one-corpus-out (SDD D3-b). A matriz "
                             "principal nao e afetada.")
    parser.add_argument("--seed", type=int, default=None,
                        help="repete a execucao com outra seed, em results/seed_NN/. "
                             "Mede a variancia de PARTICAO e de ajuste; a seed do "
                             "experiment.yml continua sendo o resultado do artigo. "
                             "Reportar como faixa, NUNCA escolher a melhor.")
    parser.add_argument("--sem-esquema", action="store_true",
                        help="rodada de sensibilidade (SENS-01): tira o esquema das "
                             "caracteristicas de texto inteiro. Grava em "
                             "results/sensibilidade_sem_esquema/ com livro-caixa e "
                             "cache proprios.")
    parser.add_argument("--repeticao", action="store_true",
                        help="repete a semente canonica (REP-01) em "
                             "results/repeticao_seed_42/, com livro-caixa proprio. "
                             "Mede o ruido da GPU. Recusa --seed diferente da canonica.")
    args = parser.parse_args()

    print("=" * 74)
    print("EXECUCAO COMPLETA -- matriz cruzada 4x4, 2 regimes, 6 modelos (T4.3)")
    print("Rode scripts/run_smoke.py antes. Este script TREINA MODELOS.")
    print("=" * 74)

    # RUN-03: antes de qualquer outra coisa, e antes de carregar dados. A rodada
    # e carimbada com o commit do codigo; codigo sujo invalida o carimbo.
    checar_arvore_limpa()

    checar_dependencias()
    silenciar_ruido()

    import models

    try:
        config = models.load_experiment_config()
    except models.ModelConfigError as exc:
        falhar(str(exc))

    # ------------------------------------------------------------------ seed
    # Uma seed diferente reparticiona os corpora e reinicializa os estimadores,
    # então mede as duas variâncias que o bootstrap por célula NÃO mede: a da
    # partição e a do ajuste. É a resposta à objeção mais previsível do revisor
    # ("a matriz inteira é uma amostra de tamanho 1").
    #
    # Cada seed escreve num diretório próprio e tem livro-caixa próprio. Isso
    # não é só arrumação: a seed entra no `config_fingerprint`, então uma seed
    # nova num livro-caixa velho seria **recusada** — o que é a proteção certa,
    # mas pararia a execução em vez de rodar.
    #
    # ⚠️ O resultado do artigo é a seed do `experiment.yml`. As demais entram no
    # texto como FAIXA de variação, e escolher a melhor seria selecionar
    # resultado — a mesma classe de erro que o D5 proíbe nos hiperparâmetros.
    global OUT_DIR, CACHE_DIR, LEDGER, LEDGER_LOCO
    seed_canonica = int(config["seed"])
    try:
        saidas = resolver_saidas(
            args.seed, seed_canonica, sem_esquema=args.sem_esquema,
            repeticao=args.repeticao,
        )
    except ArgumentoInvalido as exc:
        falhar(str(exc))
    OUT_DIR, CACHE_DIR = saidas.out_dir, saidas.cache_dir
    LEDGER, LEDGER_LOCO = saidas.ledger, saidas.ledger_loco
    commit_sessao = checar_commit_da_rodada(OUT_DIR, recomecar=args.recomecar)

    if args.seed is not None and args.seed != seed_canonica:
        config["seed"] = int(args.seed)
        print()
        print(f"SEED {args.seed} (a canonica e {seed_canonica}). Saida em "
              f"{OUT_DIR.relative_to(REPO_ROOT).as_posix()}/.")
        print("Esta execucao mede VARIANCIA. O resultado do artigo continua")
        print(f"sendo a seed {seed_canonica}, em results/.")
        print("=" * 74)
    if args.sem_esquema:
        config["_strip_scheme"] = True
        print()
        print("SENSIBILIDADE SEM ESQUEMA (SENS-01). Saida em "
              f"{OUT_DIR.relative_to(REPO_ROOT).as_posix()}/.")
        print("Nao e o resultado do artigo: testa a hipotese de formato de distribuicao.")
        print("=" * 74)
    if args.repeticao:
        print()
        print("REPETICAO DA SEMENTE CANONICA (REP-01). Saida em "
              f"{OUT_DIR.relative_to(REPO_ROOT).as_posix()}/.")
        print("=" * 74)

    inicio_sessao = datetime.now(UTC)
    try:
        return _executar(args, config)
    finally:
        registrar_sessao(inicio_sessao, datetime.now(UTC), commit=commit_sessao)


def _executar(args, config: dict) -> int:
    """Corpo da rodada, depois das verificacoes e da resolucao dos diretorios."""
    import evaluate
    import matrix
    import models
    import predicoes

    nomes = list(config["corpora"]["treino_avaliacao"])
    if args.corpora:
        pedidos = [n.strip() for n in args.corpora.split(",") if n.strip()]
        invalidos = [n for n in pedidos if n not in nomes]
        if invalidos:
            falhar(
                f"corpora invalidos: {invalidos}. Os de treino/avaliacao sao {nomes}.",
                dica=("PDRCNN e sid321axn NAO podem treinar: tem vies de construcao\n"
                      "medido (gap de 94,9 p e 59,9 p) e a restricao R6 os barra."),
            )
        nomes = pedidos

    manifesto = checar_corpora(nomes)
    # Ordem crescente de tamanho (mitigação do RK1): os pequenos primeiro, para
    # que uma execução interrompida já entregue linhas de tabela.
    nomes = sorted(nomes, key=lambda n: manifesto[n]["n_rows"])

    gpu = models.detect_gpu()
    checar_restricao_0(args.permitir_cpu, gpu)
    if not gpu.available:
        log("AVISO: treinando em CPU por consentimento explicito (--permitir-cpu).")

    if args.recomecar:
        for livro in (LEDGER, LEDGER_LOCO, OUT_DIR / SESSOES_NOME):
            if livro.exists():
                destino = livro.with_suffix(".jsonl.antigo")
                livro.replace(destino)
                log(f"livro-caixa anterior arquivado em {OUT_DIR.name}/{destino.name}")

    fingerprint = matrix.config_fingerprint(config)
    run_id = evaluate.make_run_id()
    log(f"run_id: {run_id}")
    log(f"dispositivo: {gpu.describe()}")
    log(f"impressao digital da config: {fingerprint}")
    log(f"corpora (menor -> maior): {', '.join(nomes)}")
    log()

    log("preparando features (canonizacao + 41 features)...")
    quadros = {n: preparar_corpus(n, manifesto, usar_cache=not args.sem_cache,
                                  strip_scheme=args.sem_esquema)
               for n in nomes}

    try:
        resultado = matrix.run_matrix(
            config, quadros, gpu,
            run_id=run_id,
            fingerprint=fingerprint,
            ledger_path=LEDGER,
            log=log,
            dir_rodada=OUT_DIR,
        )
    except matrix.MatrixError as exc:
        falhar(str(exc))
    except evaluate.MetricsError as exc:
        falhar(f"metrica invalida: {exc}")
    except KeyboardInterrupt:
        log()
        log(f"interrompido. O livro-caixa {LEDGER.parent.name}/{LEDGER.name} guarda")
        log("o progresso; rodar de novo retoma do ponto em que parou (RH2).")
        return 130
    except Exception as exc:  # noqa: BLE001
        falhar(f"{type(exc).__name__}: {exc}", dica=traceback.format_exc())

    log()
    try:
        matrix.check_records(resultado.records, config, len(nomes))
    except matrix.MatrixError as exc:
        falhar(str(exc))
    log(f"conferencia: {len(resultado.records)} registros, sem duplicata, "
        f"metricas todas na faixa")

    # Partições e grupos de cada linha elegível: a filtragem offline lê daqui,
    # sem reextrair características (PRED-01, AD-068).
    particoes = predicoes.gravar_particoes(
        OUT_DIR, resultado.indices, quadros,
        matrix.contagens_dedup(resultado.split_summaries),
    )
    log(f"gravado: {OUT_DIR.name}/{particoes['arquivo']}")
    minutos_fase_matriz = gravar(
        resultado.records, resultado.split_summaries, config, fingerprint, gpu, nomes,
        particoes=particoes, strip_scheme=args.sem_esquema, repeticao=args.repeticao,
    )
    resumo_diagonal(resultado.records)

    # ---------------------------------------------------------------- LOCO
    # D3-b. Roda DEPOIS da matriz e grava em arquivos próprios, de propósito:
    # a matriz é o resultado principal (T2/T3) e não pode ficar refém de um
    # experimento acrescentado depois. Se o LOCO cair, o `results/` da matriz
    # já está gravado e a entrega continua válida.
    if not args.sem_loco:
        log()
        log("=" * 74)
        log("LEAVE-ONE-CORPUS-OUT (SDD D3-b): treina na uniao de 3, testa no 4o")
        log("=" * 74)
        try:
            loco = matrix.run_loco(
                config, quadros, gpu,
                run_id=run_id,
                fingerprint=fingerprint,
                ledger_path=LEDGER_LOCO,
                log=log,
                dir_rodada=OUT_DIR,
            )
        except (matrix.MatrixError, evaluate.MetricsError) as exc:
            falhar(f"leave-one-corpus-out: {exc}")
        except KeyboardInterrupt:
            log()
            log(f"LOCO interrompido. A matriz principal ja esta gravada em "
                f"{OUT_DIR.name}/;")
            log("rodar de novo retoma o LOCO do ponto em que parou (RH2).")
            return 130
        except Exception as exc:  # noqa: BLE001
            falhar(f"leave-one-corpus-out: {type(exc).__name__}: {exc}",
                   dica=traceback.format_exc())

        # A matriz é conferida por `check_records` e o LOCO não era — só imprimia
        # a contagem. Acrescentado em 05/08/2026 pela auditoria: o portão
        # (`run_smoke.py`) já fazia esta checagem, então a execução real era a
        # **menos** rigorosa das duas, ao contrário do que se pretendia. Um
        # ajuste faltante sairia como uma T6 de 47 linhas, sem ninguém reclamar.
        try:
            matrix.check_loco_records(loco.records, config, len(nomes))
        except matrix.MatrixError as exc:
            falhar(f"leave-one-corpus-out: {exc}")
        log(f"conferencia do LOCO: {len(loco.records)} avaliacoes, sem duplicata")

        gravar_loco(loco.records)
        log(f"LOCO: {loco.n_fits_executados} ajustes nesta execucao "
            f"({loco.n_fits_pulados} retomados), {len(loco.records)} avaliacoes")
        atualizar_custo_com_loco(loco.records, minutos_fase_matriz)

    total_min = (time.perf_counter() - _T0) / 60
    print()
    print("=" * 74)
    print(f"{resultado.n_fits_executados} ajustes nesta execucao "
          f"({resultado.n_fits_pulados} retomados do livro-caixa) - "
          f"{len(resultado.records)} avaliacoes - {total_min:.1f} min")
    if not gpu.available:
        print("Executado em CPU -- o orcamento de 2h do RNF5 pressupoe GPU.")
    print("EXECUCAO COMPLETA. Commite o diretorio results/ (SDD secao 8.3).")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
