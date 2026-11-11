"""Análises offline: filtragem da sobreposição de eTLD+1 e *bootstrap* pareado.

Roda depois da rodada, sobre as predições e as partições que o `run_all.py`
gravou em `results/**/predicoes/` (AD-068). Não treina nada, não extrai
características e não recanoniza: lê escores e grupos já persistidos. É o único
módulo que lê predições (o `make tabelas` lê só JSON versionado).

Duas análises:

1. `filtrar_sobreposicao` (FILT-01, REEX-06): para cada célula fora da diagonal
   da matriz e para cada célula do LOCO, nos dois regimes, remove do teste as
   linhas cujo eTLD+1 aparece no treino ou na validação do *dataset* de treino
   (no LOCO, da união das três fontes) e recalcula as métricas com o limiar
   original do registro. A diagonal não entra.
2. `bootstrap_pareado` (BOOT-01, EST-02): para cada (treino, teste, modelo) da
   matriz e do LOCO na semente canônica, reamostra, estratificado pelo rótulo,
   as linhas comuns aos testes dos dois regimes e devolve o Delta B menos A de
   AUROC, AUC-PR e F1 com o IC percentil.

Uso (da pasta `pipeline`)::

    venv/Scripts/python -m src.offline            # grava results/offline.json

Saída (`results/offline.json`, versionado)::

    {"gerado_em", "commit_codigo", "fontes": {arquivo: sha256},
     "filtragem": {rodada: {semente: [{regime, train_corpus, test_corpus,
                   model, n_teste, n_removidas, fracao_removida, metrics,
                   auroc_indefinida}]}},
     "bootstrap_pareado": [{regime_par, train_corpus, test_corpus, model,
                            n_intersecao, n_por_classe, delta, ic,
                            ic_indefinido}]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# `src/` no path, mesma convenção de `diagnostics.py`: funciona em
# `python -m src.offline` e em `import offline` (pytest usa `pythonpath = ["src"]`).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import evaluate  # noqa: E402
import predicoes  # noqa: E402
from corpora import _sha256_file  # noqa: E402

REPO_ROOT = _SRC_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "experiment.yml"
ARQUIVO_SAIDA = "offline.json"

# Igual a `matrix.LOCO_PREFIXO`; repetido aqui para não importar `matrix`
# (que importa `models` e as bibliotecas de treino) numa análise sem treino. Um
# teste confere a igualdade.
LOCO_PREFIXO = "uniao_sem_"

# Regimes do par do *bootstrap*: A é o split distribuído pela fonte, B o
# particionado por eTLD+1. O Delta é B menos A.
REGIME_A = "oficial"
REGIME_B = "etld1"
REGIME_PAR = "B-A"

# Menos linhas que isto numa das classes da interseção torna o IC indefinido.
MIN_POR_CLASSE = 30

METRICAS_PAREADAS = ("auroc", "auc_pr", "f1")


class OfflineError(RuntimeError):
    """Entrada inconsistente nas análises offline."""


# ---------------------------------------------------------------------------
# Leitura da rodada
# ---------------------------------------------------------------------------


def _ler_json(caminho: Path):
    return json.loads(caminho.read_text(encoding="utf-8"))


def _registros(dir_rodada: Path, nome: str) -> list[dict]:
    caminho = dir_rodada / nome
    return _ler_json(caminho) if caminho.is_file() else []


def _carregar_particoes(dir_rodada: Path) -> pd.DataFrame:
    caminho = dir_rodada / predicoes.ARQUIVO_PARTICOES
    if not caminho.is_file():
        raise OfflineError(f"partições ausentes: {caminho}")
    meta = dir_rodada / "run_meta.json"
    if meta.is_file():
        ref = _ler_json(meta).get("particoes")
        if ref and not predicoes.conferir(dir_rodada, ref):
            raise OfflineError(
                f"partições adulteradas (sha256 diferente do run_meta.json): {caminho}"
            )
    df = pd.read_parquet(caminho)
    for col in ("corpus", "regime", "particao", "grupo"):
        df[col] = df[col].astype(str)
    return df


class _Predicoes:
    """Leitor de predições com cache por arquivo (cada ajuste é lido uma vez)."""

    def __init__(self, dir_rodada: Path):
        self.dir_rodada = dir_rodada
        self._cache: dict[str, pd.DataFrame] = {}

    def bloco(self, rec: Mapping) -> pd.DataFrame:
        ref = rec.get("predicoes")
        if not ref:
            raise OfflineError(
                f"registro sem referência de predições: {rec['regime']}/"
                f"{rec['train_corpus']}/{rec['test_corpus']}/{rec['model']}"
            )
        arquivo = ref["arquivo"]
        if arquivo not in self._cache:
            self._cache[arquivo] = predicoes.ler(self.dir_rodada, ref)
        df = self._cache[arquivo]
        return df[df["test_corpus"].astype(str) == rec["test_corpus"]]


# ---------------------------------------------------------------------------
# Filtragem da sobreposição (FILT-01)
# ---------------------------------------------------------------------------


def _corpora_de_treino(rec: Mapping, corpora_do_regime: Sequence[str]) -> list[str]:
    """Os *datasets* cujo treino e validação o ajuste do registro viu."""
    treino = rec["train_corpus"]
    if treino.startswith(LOCO_PREFIXO):
        retido = treino[len(LOCO_PREFIXO):]
        return [c for c in corpora_do_regime if c != retido]
    return [treino]


def filtrar_sobreposicao(dir_rodada: str | Path) -> list[dict]:
    """Recalcula as métricas sem as linhas de teste cujo eTLD+1 o modelo viu.

    Para cada registro fora da diagonal da matriz e para cada registro do LOCO
    (`matrix.json` e `loco.json` de `dir_rodada`), nos dois regimes: os grupos
    vistos são os de `particao` em `{train, val}` do *dataset* de treino (ou das
    três fontes da união, no LOCO) em `particoes.parquet`, no mesmo regime. As
    linhas de teste com `grupo` nesse conjunto saem, e as métricas são
    recalculadas por `evaluate.evaluate_predictions` com o limiar do registro.

    Devolve, ordenada por (regime, treino, teste, modelo), uma lista de
    `{regime, train_corpus, test_corpus, model, n_teste, n_removidas,
    fracao_removida, n_restantes, metrics, auroc_indefinida}`. Quando a
    filtragem deixa o teste sem uma das classes, `auroc_indefinida` é verdadeiro
    e a AUROC vem `None`; sem nenhuma linha, `metrics` é `None`. Nada é
    interrompido por isso.
    """
    dir_rodada = Path(dir_rodada)
    particoes = _carregar_particoes(dir_rodada)

    vistos: dict[tuple[str, str], set] = {}
    grupo_da_linha: dict[tuple[str, str], pd.Series] = {}
    corpora_por_regime: dict[str, list[str]] = {}
    for (corpus, regime), parte in particoes.groupby(["corpus", "regime"], sort=True):
        corpora_por_regime.setdefault(regime, []).append(corpus)
        grupo_da_linha[(corpus, regime)] = pd.Series(
            parte["grupo"].to_numpy(), index=parte["linha"].to_numpy()
        )
        treino_val = parte[parte["particao"].isin(["train", "val"])]
        vistos[(corpus, regime)] = set(treino_val["grupo"])

    registros = [
        rec
        for rec in _registros(dir_rodada, "matrix.json")
        if not rec.get("diagonal", rec["train_corpus"] == rec["test_corpus"])
    ] + _registros(dir_rodada, "loco.json")
    registros.sort(key=_chave_registro)

    leitor = _Predicoes(dir_rodada)
    saida = []
    for rec in registros:
        regime = rec["regime"]
        corpora = corpora_por_regime.get(regime, [])
        grupos_vistos: set = set()
        for corpus in _corpora_de_treino(rec, corpora):
            if (corpus, regime) not in vistos:
                raise OfflineError(
                    f"partições sem o dataset '{corpus}' no regime '{regime}' "
                    f"(registro {rec['train_corpus']}/{rec['test_corpus']}/{rec['model']})."
                )
            grupos_vistos |= vistos[(corpus, regime)]

        bloco = leitor.bloco(rec)
        chave_teste = (rec["test_corpus"], regime)
        if chave_teste not in grupo_da_linha:
            raise OfflineError(
                f"partições sem o dataset de teste '{rec['test_corpus']}' no regime '{regime}'."
            )
        if "n_test" in rec and int(rec["n_test"]) != len(bloco):
            raise OfflineError(
                f"o registro {regime}/{rec['train_corpus']}/{rec['test_corpus']}/"
                f"{rec['model']} declara n_test={rec['n_test']}, mas as predições "
                f"têm {len(bloco)} linhas."
            )
        grupos_teste = grupo_da_linha[chave_teste].reindex(bloco["linha"].to_numpy())
        if grupos_teste.isna().any():
            raise OfflineError(
                f"há linhas de teste de '{rec['test_corpus']}' ({regime}) fora das "
                f"partições gravadas."
            )
        sobreposta = grupos_teste.isin(grupos_vistos).to_numpy()
        restantes = bloco[~sobreposta]

        n_teste = int(len(bloco))
        n_removidas = int(sobreposta.sum())
        if len(restantes) == 0:
            metrics, indefinida = None, True
        else:
            metrics, _ = evaluate.evaluate_predictions(
                restantes["y_true"].to_numpy(),
                restantes["score"].to_numpy(),
                threshold=float(rec["metrics"]["threshold"]),
            )
            indefinida = metrics["auroc"] is None
        saida.append(
            {
                "regime": regime,
                "train_corpus": rec["train_corpus"],
                "test_corpus": rec["test_corpus"],
                "model": rec["model"],
                "n_teste": n_teste,
                "n_removidas": n_removidas,
                "fracao_removida": (n_removidas / n_teste) if n_teste else 0.0,
                "n_restantes": int(len(restantes)),
                "metrics": metrics,
                "auroc_indefinida": bool(indefinida),
            }
        )
    return saida


def _chave_registro(rec: Mapping) -> tuple:
    return (rec["regime"], rec["train_corpus"], rec["test_corpus"], rec["model"])


# ---------------------------------------------------------------------------
# Bootstrap pareado do Delta entre regimes (BOOT-01)
# ---------------------------------------------------------------------------


def semente_da_celula(chave: Sequence[str], seed: int) -> int:
    """Semente do gerador de uma célula: hash estável da chave mais a semente.

    A chave é `(regime_par, treino, teste, modelo)`. O hash é o sha256 do texto
    da chave, então não depende da ordem em que as células são processadas nem
    da execução (RNF1, BOOT-01 critério 6).
    """
    resumo = hashlib.sha256("|".join(chave).encode("utf-8")).digest()
    return int.from_bytes(resumo[:8], "big") + int(seed)


class _Regime:
    """Escores de um regime sobre as linhas comuns, prontos para reamostrar.

    Guarda a ordem decrescente de escore e os grupos de escore empatado, de modo
    que AUROC e AUC-PR de uma reamostra saem de contagens por grupo
    (`evaluate._auroc_e_ap_de_grupos`), em O(n) por reamostra, sem reordenar.
    """

    def __init__(self, y: np.ndarray, score: np.ndarray, limiar: float):
        self.y = y.astype(np.float64)
        self.limiar = float(limiar)
        self.pred = (score >= self.limiar).astype(np.float64)
        self.ordem = np.argsort(-score, kind="stable")
        ss = score[self.ordem]
        novo = np.r_[True, ss[1:] != ss[:-1]]
        self.grupo_de = np.cumsum(novo) - 1
        self.n_grupos = int(self.grupo_de[-1]) + 1
        self.y_ord = self.y[self.ordem]

    def metricas(self, c: np.ndarray) -> tuple[float, float, float]:
        """AUROC, AUC-PR e F1 com a multiplicidade `c` de cada linha."""
        co = c[self.ordem]
        wp = np.bincount(self.grupo_de, weights=co * self.y_ord, minlength=self.n_grupos)
        wn = np.bincount(self.grupo_de, weights=co * (1.0 - self.y_ord), minlength=self.n_grupos)
        auroc, ap = evaluate._auroc_e_ap_de_grupos(wp, wn)
        tp = float(np.sum(c * self.y * self.pred))
        fp = float(np.sum(c * (1.0 - self.y) * self.pred))
        fn = float(np.sum(c * self.y * (1.0 - self.pred)))
        denominador = 2 * tp + fp + fn
        f1 = 2 * tp / denominador if denominador > 0 else 0.0
        return auroc, ap, f1


def _interseccao(bloco_a: pd.DataFrame, bloco_b: pd.DataFrame, rotulo: str):
    """Linhas comuns aos dois regimes, ordenadas por `linha`, com os dois escores."""
    for nome, bloco in (("A", bloco_a), ("B", bloco_b)):
        if bloco["linha"].duplicated().any():
            raise OfflineError(f"{rotulo}: linhas repetidas nas predições do regime {nome}.")
    junto = bloco_a.merge(bloco_b, on="linha", suffixes=("_a", "_b"))
    if (junto["y_true_a"] != junto["y_true_b"]).any():
        raise OfflineError(
            f"{rotulo}: o rótulo de uma linha comum difere entre os regimes."
        )
    junto = junto.sort_values("linha", kind="stable")
    return (
        junto["y_true_a"].to_numpy(),
        junto["score_a"].to_numpy(dtype=np.float64),
        junto["score_b"].to_numpy(dtype=np.float64),
    )


def _delta_de_celula(
    chave: tuple[str, str, str],
    rec_a: Mapping,
    rec_b: Mapping,
    leitor: _Predicoes,
    bootstrap_n: int,
    nivel: float,
    seed: int,
) -> dict:
    treino, teste, modelo = chave
    rotulo = f"{treino}/{teste}/{modelo}"
    y, s_a, s_b = _interseccao(leitor.bloco(rec_a), leitor.bloco(rec_b), rotulo)
    n = int(len(y))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    resultado = {
        "regime_par": REGIME_PAR,
        "train_corpus": treino,
        "test_corpus": teste,
        "model": modelo,
        "n_intersecao": n,
        "n_por_classe": {"positivos": int(len(pos)), "negativos": int(len(neg))},
        "delta": dict.fromkeys(METRICAS_PAREADAS),
        "ic": None,
        "ic_indefinido": bool(min(len(pos), len(neg)) < MIN_POR_CLASSE),
    }
    if n == 0:
        return resultado

    # Estimativa pontual: cada regime com o próprio limiar, pela mesma função
    # que produz as métricas das tabelas.
    pontuais = {}
    for nome, score, rec in (("A", s_a, rec_a), ("B", s_b, rec_b)):
        metricas, _ = evaluate.evaluate_predictions(
            y, score, threshold=float(rec["metrics"]["threshold"])
        )
        pontuais[nome] = metricas
    for m in METRICAS_PAREADAS:
        a, b = pontuais["A"][m], pontuais["B"][m]
        resultado["delta"][m] = None if a is None or b is None else float(b - a)

    if resultado["ic_indefinido"]:
        return resultado

    regime_a = _Regime(y, s_a, rec_a["metrics"]["threshold"])
    regime_b = _Regime(y, s_b, rec_b["metrics"]["threshold"])
    rng = np.random.default_rng(semente_da_celula((REGIME_PAR, treino, teste, modelo), seed))
    deltas = np.empty((bootstrap_n, len(METRICAS_PAREADAS)), dtype=np.float64)
    for i in range(bootstrap_n):
        # Reamostra positivos e negativos separadamente (estratificado pelo
        # rótulo) e aplica a MESMA reamostra aos dois regimes: o par é o que
        # torna o IC do Delta estreito onde a diferença é de fato pequena.
        sorteio_pos = pos[rng.integers(0, len(pos), size=len(pos))]
        sorteio_neg = neg[rng.integers(0, len(neg), size=len(neg))]
        c = (
            np.bincount(sorteio_pos, minlength=n) + np.bincount(sorteio_neg, minlength=n)
        ).astype(np.float64)
        deltas[i] = np.array(regime_b.metricas(c)) - np.array(regime_a.metricas(c))

    baixo = 100.0 * (1.0 - nivel) / 2.0
    alto = 100.0 - baixo
    resultado["ic"] = {
        m: [
            round(float(np.percentile(deltas[:, j], baixo)), 6),
            round(float(np.percentile(deltas[:, j], alto)), 6),
        ]
        for j, m in enumerate(METRICAS_PAREADAS)
    }
    return resultado


def bootstrap_pareado(dir_rodada: str | Path, config: Mapping) -> list[dict]:
    """IC do Delta (B menos A) de AUROC, AUC-PR e F1 por célula, sem treinar.

    Para cada (treino, teste, modelo) da matriz e do LOCO de `dir_rodada`, junta
    por `linha` as predições dos dois regimes, reamostra a interseção
    estratificada pelo rótulo `avaliacao.bootstrap_n` vezes (gerador
    `np.random.default_rng` semeado por `semente_da_celula`, a partir da semente
    do `config`) e devolve o Delta com o IC percentil de nível
    `avaliacao.bootstrap_nivel`. Cada regime usa o próprio limiar (o do seu
    registro). Com menos de 30 linhas de alguma classe na interseção, o IC é
    `None` e `ic_indefinido` é verdadeiro.

    A saída é ordenada por (treino, teste, modelo) e não depende da ordem dos
    registros nem da execução (BOOT-01 critério 6).
    """
    dir_rodada = Path(dir_rodada)
    avaliacao = config["avaliacao"]
    bootstrap_n = int(avaliacao["bootstrap_n"])
    nivel = float(avaliacao.get("bootstrap_nivel", evaluate.DEFAULT_BOOTSTRAP_NIVEL))
    seed = int(config["seed"])
    if bootstrap_n < 2:
        raise OfflineError(f"bootstrap_n tem de ser >= 2; recebi {bootstrap_n}")

    celulas: dict[tuple[str, str, str], dict[str, dict]] = {}
    for rec in _registros(dir_rodada, "matrix.json") + _registros(dir_rodada, "loco.json"):
        chave = (rec["train_corpus"], rec["test_corpus"], rec["model"])
        celulas.setdefault(chave, {})[rec["regime"]] = rec

    leitor = _Predicoes(dir_rodada)
    saida = []
    for chave in sorted(celulas):
        por_regime = celulas[chave]
        faltam = [r for r in (REGIME_A, REGIME_B) if r not in por_regime]
        if faltam:
            raise OfflineError(
                f"célula {'/'.join(chave)} sem o regime {faltam}: o par precisa dos dois."
            )
        saida.append(
            _delta_de_celula(
                chave, por_regime[REGIME_A], por_regime[REGIME_B],
                leitor, bootstrap_n, nivel, seed,
            )
        )
    return saida


# ---------------------------------------------------------------------------
# Rodadas, saída e CLI
# ---------------------------------------------------------------------------


def _carregar_config(caminho: Path) -> dict:
    config = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    for chave in ("seed", "avaliacao"):
        if chave not in config:
            raise OfflineError(f"{caminho}: chave obrigatória ausente: '{chave}'")
    return config


def _identificar_rodada(results_dir: Path, diretorio: Path, seed_canonica: int) -> tuple[str, int]:
    """Nome da rodada e semente, pelo caminho (convenção de `run_all.resolver_saidas`)."""
    rel = diretorio.relative_to(results_dir).parts if diretorio != results_dir else ()
    seed = seed_canonica
    if rel and rel[-1].startswith("seed_") and rel[-1][5:].isdigit():
        seed = int(rel[-1][5:])
        rel = rel[:-1]
    nome = rel[0] if rel else "principal"
    return nome, seed


def descobrir_rodadas(results_dir: Path, seed_canonica: int) -> list[tuple[str, int, Path]]:
    """Diretórios com partições gravadas: a principal, `seed_NN/`, a sensibilidade, a repetição."""
    candidatos = [results_dir]
    candidatos += sorted(results_dir.glob("seed_*"))
    for base in sorted(
        p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("seed_")
    ):
        candidatos.append(base)
        candidatos += sorted(base.glob("seed_*"))
    rodadas = []
    for diretorio in candidatos:
        if (diretorio / predicoes.ARQUIVO_PARTICOES).is_file():
            nome, seed = _identificar_rodada(results_dir, diretorio, seed_canonica)
            rodadas.append((nome, seed, diretorio))
    return rodadas


def _fontes(results_dir: Path, diretorio: Path) -> dict[str, str]:
    """sha256 de cada arquivo de predições e das partições lidos da rodada."""
    prefixo = "" if diretorio == results_dir else diretorio.relative_to(results_dir).as_posix() + "/"
    refs = {predicoes.ARQUIVO_PARTICOES: _sha256_file(diretorio / predicoes.ARQUIVO_PARTICOES)}
    for nome in ("matrix.json", "loco.json"):
        for rec in _registros(diretorio, nome):
            ref = rec.get("predicoes")
            if ref:
                refs[ref["arquivo"]] = ref["sha256"]
    return {prefixo + k: v for k, v in sorted(refs.items())}


def executar(
    results_dir: str | Path,
    config: Mapping,
    rodadas: Sequence[str] | None = None,
    gerado_em: str | None = None,
    log=None,
) -> dict:
    """Roda a filtragem em cada rodada e o *bootstrap* pareado da principal na semente canônica.

    `rodadas` são diretórios relativos a `results_dir` (`"."` é a principal);
    sem ela, descobrem-se os que têm `particoes.parquet`. Não grava nada.
    """
    log = log or (lambda _msg="": None)
    results_dir = Path(results_dir)
    seed_canonica = int(config["seed"])
    if rodadas:
        achadas = []
        for rel in rodadas:
            diretorio = (results_dir / rel).resolve()
            base = results_dir.resolve()
            nome, seed = _identificar_rodada(base, diretorio, seed_canonica)
            achadas.append((nome, seed, results_dir / rel if rel != "." else results_dir))
    else:
        achadas = descobrir_rodadas(results_dir, seed_canonica)
    if not achadas:
        raise OfflineError(
            f"nenhuma rodada com predições em {results_dir} "
            f"(procurei {predicoes.ARQUIVO_PARTICOES})."
        )

    # `commit_codigo` do offline.json: o das rodadas lidas, que têm de ser do mesmo
    # código (AD-068). Rodada sem o campo não entra; sem nenhum, fica `None`.
    commits = {
        c for _, _, d in achadas
        if (d / "run_meta.json").is_file()
        for c in [_ler_json(d / "run_meta.json").get("commit_codigo")] if c
    }
    if len(commits) > 1:
        raise OfflineError(
            "as rodadas lidas têm commit_codigo diferentes (AD-068): "
            f"{', '.join(sorted(commits))}. Analise rodadas do mesmo código."
        )

    filtragem: dict = {}
    fontes: dict = {}
    principal = None
    for nome, seed, diretorio in achadas:
        log(f"filtrando a sobreposição: {nome}, semente {seed} ({diretorio})")
        filtragem.setdefault(nome, {})[str(seed)] = filtrar_sobreposicao(diretorio)
        fontes.update(_fontes(results_dir, diretorio))
        if nome == "principal" and seed == seed_canonica:
            principal = diretorio

    pareado: list[dict] = []
    if principal is not None:
        log(f"bootstrap pareado: principal, semente {seed_canonica}")
        pareado = bootstrap_pareado(principal, config)

    return {
        "gerado_em": gerado_em or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parametros": {
            "seed": seed_canonica,
            "bootstrap_n": int(config["avaliacao"]["bootstrap_n"]),
            "bootstrap_nivel": float(
                config["avaliacao"].get("bootstrap_nivel", evaluate.DEFAULT_BOOTSTRAP_NIVEL)
            ),
            "regime_par": REGIME_PAR,
            "min_por_classe": MIN_POR_CLASSE,
        },
        "commit_codigo": next(iter(commits), None),
        "fontes": fontes,
        "filtragem": filtragem,
        "bootstrap_pareado": pareado,
    }


def gravar(saida: Mapping, results_dir: str | Path) -> Path:
    """Grava `offline.json` por temporário e `os.replace`."""
    destino = Path(results_dir) / ARQUIVO_SAIDA
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    try:
        temporario.write_text(
            json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporario, destino)
    except BaseException:
        temporario.unlink(missing_ok=True)
        raise
    return destino


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="pasta de resultados (default: results/ na raiz do pipeline)",
    )
    parser.add_argument(
        "--rodadas", nargs="+", default=None,
        help="diretórios das rodadas, relativos a --results-dir ('.' é a principal); "
             "sem isto, descobre os que têm predicoes/particoes.parquet",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="experiment.yml (só a semente e o bloco avaliacao são lidos)",
    )
    args = parser.parse_args(argv)

    try:
        config = _carregar_config(args.config)
        saida = executar(args.results_dir, config, rodadas=args.rodadas, log=print)
        destino = gravar(saida, args.results_dir)
    except (OfflineError, predicoes.PredicaoError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    print(f"gravado: {destino}")
    return 0


__all__ = [
    "ARQUIVO_SAIDA",
    "LOCO_PREFIXO",
    "OfflineError",
    "bootstrap_pareado",
    "descobrir_rodadas",
    "executar",
    "filtrar_sobreposicao",
    "gravar",
    "main",
    "semente_da_celula",
]


if __name__ == "__main__":
    sys.exit(main())
