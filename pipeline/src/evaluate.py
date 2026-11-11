"""RF6 / RF7 / T4.2 — métricas da matriz cruzada, com MCC correto (corrige B1).

═══════════════════════════════════════════════════════════════════════════
B1 — LEIA ANTES DE MEXER EM QUALQUER COISA NESTE ARQUIVO
═══════════════════════════════════════════════════════════════════════════

O bug que invalidou os resultados do repositório legado estava em
`src/pipelines/07c_multiobjective_eval.py:50` do pipeline descontinuado (o código
saiu do working tree; ver `docs/diagnostico.md` §3.1 e §3.5). Ele era exatamente isto:

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()   # numpy.int64!
    numerator   = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator

A linha do denominador parece certa e é a fórmula do livro. O defeito não está
na álgebra: está no **tipo**. `confusion_matrix().ravel()` devolve `int64`, e o
produto dos quatro fatores marginais cresce como O(n⁴). Para a matriz de
confusão real que o pipeline legado avaliou —
`tn=338389, fp=3433, fn=5457, tp=18165` — esse produto vale

    59.964.496.224.571.849.872  ≈ 6,0 × 10¹⁹

contra um teto de `int64` de 9,22 × 10¹⁸. O numpy **não** satura e **não**
avisa: ele dá a volta (wraparound). O produto virou 4,62 × 10¹⁸, a raiz virou
2.150.410.194 em vez de 7.743.717.735, e o MCC publicado saiu

    2,8497   (medido, reproduzido em 04/08/2026)

quando o valor correto é

    0,7914

MCC ∈ [-1, 1] por definição, então 2,85 é impossível — e ninguém percebeu.
Pior: nos outros três modelos o wraparound caiu em faixa negativa,
`np.sqrt` de negativo deu `nan`, e o "campeão" do artigo foi eleito pela
comparação `2.85 > nan` (bug B2). A conclusão principal do repositório era
consequência de um overflow de inteiro.

Três defesas, todas obrigatórias, todas cobertas por teste:

  1. **Aritmética exata.** `mcc_from_confusion` converte cada contagem com
     `int(...)`, o que transforma qualquer `numpy.int64` em `int` do Python —
     que é de precisão arbitrária e **não pode** estourar. O denominador é
     calculado como produto de quatro raízes em `float64`, nunca como raiz de
     um produto gigante.
  2. **Conferência independente.** `mcc()` usa
     `sklearn.metrics.matthews_corrcoef` (exigência de SDD §4 D6), e o teste
     de regressão prova que as duas rotas concordam na matriz de B1.
  3. **Validação de faixa.** `validate_metrics` rejeita qualquer métrica fora
     do seu intervalo matemático. Um MCC de 2,85 não teria chegado a um
     arquivo de resultado: teria parado a execução. É a defesa que faltava.

Nunca reimplemente MCC com aritmética de inteiros de largura fixa. Nunca
remova a validação de faixa para "deixar o relatório rodar".

═══════════════════════════════════════════════════════════════════════════

Definições usadas (SDD §4 D6)
-----------------------------
- **Primárias da matriz, independentes de limiar:** AUROC e AUC-PR. AUC-PR é
  a *average precision* (`average_precision_score`), não a área trapezoidal
  sob a curva PR interpolada — a interpolação da curva PR é otimista e não é
  uma estimativa consistente.
- **Secundárias, com limiar:** Precision, Recall, F1, FPR, MCC. O limiar é
  escolhido na **validação do corpus de treino** e transportado como está para
  todos os corpora de teste. A degradação disso fora da diagonal é resultado
  do artigo, não defeito da avaliação.
- **Convenção do limiar:** prevê phishing quando `p >= limiar`, coerente com
  `precision_recall_curve` do sklearn.
- **Calibração:** Brier (`brier_score_loss`) e ECE. O ECE aqui é o erro de
  calibração da classe positiva: `ece_bins` bins de largura igual sobre
  `p̂(phishing)`, comparando a média prevista com a frequência observada de
  positivos em cada bin, ponderada pela massa do bin (Guo et al., 2017).
  Sem recalibração no baseline — B8 (Brier 0,5257 com 96% de acurácia) só foi
  visível porque ninguém tinha recalibrado por cima.

Este módulo não treina nada e não escolhe campeão (SDD §0; B6/B7 nasceram de
ranquear por métrica única).
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Chaves do bloco `metrics` — contrato de SDD §7.3, na ordem do documento.
# RF7 exige esquema estável: não adicione nem remova chave aqui sem editar o
# SDD antes (report.py e as tabelas do artigo leem exatamente estas).
METRIC_KEYS = (
    "auroc", "auc_pr",
    "precision", "recall", "f1",
    "fpr", "mcc", "brier", "ece",
    "threshold",
    "tp", "fp", "tn", "fn",
)

# Faixa matemática de cada métrica. É isto que faz um MCC = 2,85 parar a
# execução em vez de virar linha de tabela.
METRIC_RANGES = {
    "auroc": (0.0, 1.0),
    "auc_pr": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "f1": (0.0, 1.0),
    "fpr": (0.0, 1.0),
    "mcc": (-1.0, 1.0),
    "brier": (0.0, 1.0),
    "ece": (0.0, 1.0),
    "threshold": (0.0, 1.0),
}

DEFAULT_ECE_BINS = 15
THRESHOLD_CRITERIA = ("f1", "youden")


class MetricsError(RuntimeError):
    """Métrica impossível ou entrada inválida. Nunca silenciar (ver B1/B2)."""


# ---------------------------------------------------------------------------
# MCC — o núcleo da correção de B1
# ---------------------------------------------------------------------------


def mcc_from_confusion(tn, fp, fn, tp) -> float:
    """MCC a partir das quatro contagens, em aritmética que não pode estourar.

    A conversão `int(...)` é a linha que corrige B1: qualquer `numpy.int64`,
    `numpy.uint32` ou similar vira `int` do Python, de precisão arbitrária.
    O numerador `tp*tn - fp*fn` passa a ser exato para qualquer tamanho de
    corpus.

    O denominador é `√(tp+fp) · √(tp+fn) · √(tn+fp) · √(tn+fn)`, e não
    `√((tp+fp)·(tp+fn)·(tn+fp)·(tn+fn))`. Matematicamente é o mesmo número;
    numericamente evita materializar um produto de ~10²⁰, que é onde o
    `int64` do B1 morreu e onde o `float64` começaria a perder dígitos.

    Denominador zero (alguma margem inteira nula — o modelo previu só uma
    classe, ou o teste tem só uma classe) devolve **0,0**, igual ao
    `sklearn.metrics.matthews_corrcoef`. Nunca `nan`: foi comparando com
    `nan` que B2 elegeu um campeão.
    """
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    if min(tn, fp, fn, tp) < 0:
        raise MetricsError(
            f"contagem negativa na matriz de confusão: "
            f"tn={tn}, fp={fp}, fn={fn}, tp={tp}"
        )

    numerator = tp * tn - fp * fn  # int do Python: exato, sem teto

    factors = (tp + fp, tp + fn, tn + fp, tn + fn)
    if any(f == 0 for f in factors):
        return 0.0

    denominator = 1.0
    for f in factors:
        denominator *= math.sqrt(f)

    value = numerator / denominator

    # Rede de segurança: MCC ∈ [-1, 1] por definição. Se esta linha disparar,
    # há um defeito novo — não relaxe a faixa, encontre o defeito.
    if not -1.0 - 1e-9 <= value <= 1.0 + 1e-9:
        raise MetricsError(
            f"MCC fora de [-1, 1]: {value!r} para tn={tn}, fp={fp}, fn={fn}, "
            f"tp={tp}. É a assinatura de B1 (overflow de inteiro) — leia o "
            f"cabeçalho de src/evaluate.py antes de mudar qualquer coisa."
        )
    return float(min(1.0, max(-1.0, value)))


def mcc(y_true: Sequence, y_pred: Sequence) -> float:
    """MCC pela rota exigida em SDD §4 D6: `sklearn.metrics.matthews_corrcoef`.

    Mantida ao lado de `mcc_from_confusion` de propósito: as duas rotas são
    independentes e o teste de regressão exige que concordem na matriz de B1.
    Uma discordância futura entre elas é sinal de defeito, e é melhor
    descobrir por teste do que por artigo publicado.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(matthews_corrcoef(y_true, y_pred))


# ---------------------------------------------------------------------------
# Contagens e verificações de entrada
# ---------------------------------------------------------------------------


def confusion_counts(y_true: Sequence, y_pred: Sequence) -> tuple[int, int, int, int]:
    """Devolve `(tn, fp, fn, tp)` como `int` do Python.

    `labels=[0, 1]` é obrigatório: sem ele, um vetor com uma única classe faz
    o sklearn devolver uma matriz 1×1 e o `ravel()` de quatro valores estoura
    com `ValueError` — ou, se alguém "consertar" com try/except, devolve
    contagens trocadas.

    O `int(...)` na saída fecha a porta do B1 na origem: nenhuma contagem sai
    daqui como `numpy.int64`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise MetricsError(
            f"y_true e y_pred têm formatos diferentes: {y_true.shape} vs {y_pred.shape}"
        )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _check_inputs(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.shape != y_score.shape:
        raise MetricsError(
            f"y_true e y_score têm formatos diferentes: {y_true.shape} vs {y_score.shape}"
        )
    if y_true.size == 0:
        raise MetricsError("conjunto de avaliação vazio")

    labels = set(np.unique(y_true).tolist())
    if not labels <= {0, 1}:
        raise MetricsError(
            f"rótulos fora de {{0, 1}}: {sorted(labels)}. O contrato de SDD §7.3 "
            f"é 0 = benigno, 1 = phishing — corpora com -1/1 ou 2/1 são "
            f"normalizados em corpora.py."
        )
    if not np.all(np.isfinite(y_score)):
        raise MetricsError(
            "y_score contém nan/inf — o modelo devolveu probabilidade inválida. "
            "Isso invalida Brier e ECE (ver B8) e não pode ser mediado."
        )
    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if lo < -1e-9 or hi > 1.0 + 1e-9:
        raise MetricsError(
            f"y_score fora de [0, 1] (min={lo}, max={hi}). Passe probabilidade "
            f"(predict_proba), não margem de decisão — Brier e ECE só fazem "
            f"sentido sobre probabilidade."
        )


# ---------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: Sequence,
    y_score: Sequence,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """ECE da classe positiva, com `n_bins` bins de largura igual.

    Para cada bin de `p̂(phishing)`: |média de p̂ − fração observada de
    positivos|, ponderada pela fração de exemplos no bin. Bins vazios não
    entram (peso zero), então o resultado fica em [0, 1] por construção.

    Escolhemos a variante "classe positiva" em vez da variante "confiança
    máxima" (`max(p, 1-p)` contra acerto do argmax) porque num detector binário
    desbalanceado a segunda é dominada pela classe majoritária e esconde
    exatamente a miscalibração que interessa reportar.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    _check_inputs(y_true.astype(np.int64), y_score)
    if n_bins < 1:
        raise MetricsError(f"n_bins deve ser >= 1: {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `np.digitize` com right=True põe p=0 no bin 0; corrigido pelo clip.
    bin_of = np.clip(np.digitize(y_score, edges[1:-1], right=True), 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_of == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (count / n) * abs(confidence - accuracy)
    return float(ece)


# ---------------------------------------------------------------------------
# Ponto de operação
# ---------------------------------------------------------------------------


def select_threshold(
    y_true: Sequence,
    y_score: Sequence,
    criterion: str = "f1",
) -> float:
    """Escolhe o limiar na **validação do corpus de treino** (SDD §4 D6).

    Nunca chame isto com dados de teste: o limiar tem de ser transportado do
    corpus de treino para todos os corpora de teste, porque a degradação desse
    transporte é um dos resultados do artigo.

    Critérios:
      - `"f1"` (padrão): maximiza F1 na validação. Empates são resolvidos para
        o **menor** limiar, isto é, a favor do recall — numa tarefa de detecção
        de phishing, deixar passar é pior que alarmar.
      - `"youden"`: maximiza J = TPR − FPR (índice de Youden).

    Determinístico: sem sorteio, sem varredura em grade arbitrária — usa os
    limiares candidatos que as próprias curvas do sklearn produzem.
    """
    if criterion not in THRESHOLD_CRITERIA:
        raise MetricsError(
            f"critério de limiar desconhecido: {criterion!r}. Válidos: {THRESHOLD_CRITERIA}"
        )
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    _check_inputs(y_true, y_score)

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "validação com uma única classe — limiar fica em 0,5 (neutro). "
            "Isso é sintoma de split degenerado, não de modelo."
        )
        return 0.5

    if criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        # precision/recall têm um ponto extra no fim (recall=0) sem limiar.
        precision, recall = precision[:-1], recall[:-1]
        denom = precision + recall
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
        best = int(np.argmax(f1))  # argmax pega o PRIMEIRO máximo; thresholds
        return float(thresholds[best])  # vêm em ordem crescente -> menor limiar

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # roc_curve prefixa um limiar `inf` (ponto "nada é positivo").
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thresholds[best], 0.0, 1.0))


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: Sequence,
    y_score: Sequence,
    threshold: float,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> tuple[dict, list[str]]:
    """Calcula o bloco `metrics` de SDD §7.3 e a lista de avisos.

    `threshold` vem de `select_threshold` sobre a validação do corpus de
    **treino** — este módulo não o re-escolhe no teste.

    Devolve `(metrics, warnings)`. `warnings` é a lista de métricas que não
    puderam ser calculadas e por quê; elas vêm como `None` no dicionário, em
    vez de `0`. Reportar `0` num AUROC indefinido seria criar um número falso —
    e foi exatamente comparar valores inválidos que produziu B2 e B7.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    _check_inputs(y_true, y_score)
    if not 0.0 <= float(threshold) <= 1.0:
        raise MetricsError(f"limiar fora de [0, 1]: {threshold!r}")

    warnings: list[str] = []
    y_pred = (y_score >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    both_classes = len(np.unique(y_true)) == 2
    if not both_classes:
        warnings.append(
            "conjunto de teste com uma única classe: AUROC e AUC-PR são "
            "indefinidos e vêm como null"
        )

    auroc = float(roc_auc_score(y_true, y_score)) if both_classes else None
    auc_pr = float(average_precision_score(y_true, y_score)) if both_classes else None

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0
        warnings.append("nenhum exemplo benigno no teste: FPR indefinido, reportado como 0")

    metrics = {
        "auroc": auroc,
        "auc_pr": auc_pr,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "mcc": mcc_from_confusion(tn, fp, fn, tp),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": expected_calibration_error(y_true, y_score, n_bins=ece_bins),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    if tuple(metrics) != METRIC_KEYS:
        raise MetricsError(
            f"esquema de métricas divergiu do contrato de SDD §7.3 (RF7).\n"
            f"esperado: {METRIC_KEYS}\nobtido:   {tuple(metrics)}"
        )

    validate_metrics(metrics)
    return metrics, warnings


def validate_metrics(metrics: dict) -> None:
    """Rejeita métrica fora da faixa matemática — a defesa que faltava em B1.

    Chamado automaticamente por `evaluate_predictions`. Chame também ao ler
    resultado de disco antes de gerar tabela: é a última barreira entre um
    número impossível e o PDF submetido.
    """
    missing = [k for k in METRIC_KEYS if k not in metrics]
    if missing:
        raise MetricsError(f"métricas ausentes no registro: {missing}")

    for name, (lo, hi) in METRIC_RANGES.items():
        value = metrics[name]
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MetricsError(f"métrica '{name}' não é numérica: {value!r}")
        if not math.isfinite(float(value)):
            raise MetricsError(f"métrica '{name}' não é finita: {value!r}")
        if not lo - 1e-9 <= float(value) <= hi + 1e-9:
            raise MetricsError(
                f"métrica '{name}' = {value!r} fora de [{lo}, {hi}].\n"
                f"Foi assim que B1 publicou MCC = 2,8497 (impossível) e B2 "
                f"elegeu campeão comparando 2,85 > nan. Não amplie a faixa: "
                f"encontre o defeito."
            )

    for name in ("tp", "fp", "tn", "fn"):
        value = metrics[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise MetricsError(f"contagem '{name}' inválida: {value!r}")

    if metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == 0:
        raise MetricsError("matriz de confusão vazia — nada foi avaliado")


# ---------------------------------------------------------------------------
# Registro de resultado (RF7, RNF4)
# ---------------------------------------------------------------------------


def git_commit_hash() -> str:
    """Hash curto do commit atual, para carimbar os artefatos (RNF4)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git indisponível para carimbar run_id: %s", exc)
    return "sem-git"


def make_run_id(timestamp: datetime | None = None) -> str:
    """`<commit-hash>-<timestamp>`, conforme SDD §7.3 e RNF4.

    Usa `datetime.now(timezone.utc)`, não `datetime.utcnow()` — o segundo está
    depreciado no Python 3.14 (pendência 6 de SDD §13.4).
    """
    ts = timestamp or datetime.now(UTC)
    return f"{git_commit_hash()}-{ts.strftime('%Y%m%dT%H%M%SZ')}"


def make_result_record(
    *,
    run_id: str,
    regime: str,
    train_corpus: str,
    test_corpus: str,
    model: str,
    metrics: dict,
    fit_seconds: float,
    warnings: Sequence[str] = (),
    extra: dict | None = None,
) -> dict:
    """Monta um registro no esquema de SDD §7.3.

    O bloco `metrics` fica com **exatamente** as 14 chaves do contrato (RF7);
    proveniência e avisos entram no nível do registro, nunca dentro de
    `metrics`, para que o esquema lido por `report.py` não mude de forma.
    """
    validate_metrics(metrics)
    if regime not in ("oficial", "etld1"):
        raise MetricsError(f"regime desconhecido: {regime!r}")
    if fit_seconds < 0:
        raise MetricsError(f"fit_seconds negativo: {fit_seconds!r}")

    record = {
        "run_id": run_id,
        "regime": regime,
        "train_corpus": train_corpus,
        "test_corpus": test_corpus,
        "model": model,
        "metrics": {k: metrics[k] for k in METRIC_KEYS},
        "fit_seconds": float(fit_seconds),
        "n_test": metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"],
        "diagonal": train_corpus == test_corpus,
        "warnings": list(warnings),
    }
    if extra:
        record.update(extra)
    return record


# Campos que **não** podem virar coluna do CSV: são estruturas grandes que
# viajam no registro para sobreviver à retomada (RH2) e saem em arquivo próprio.
# Ver `run_all._separar_anexos`.
CAMPOS_ANEXOS = ("curves", "importances")


def records_to_dataframe(records: Sequence[dict]):
    """Achata os registros para CSV (RF7), com uma coluna por métrica.

    Levanta se um anexo ainda estiver no registro. Sem esta trava, esquecer de
    separar as curvas produz um `matrix.csv` com uma coluna `curves` contendo
    101 pares de números por linha, serializados como texto — foi exatamente o
    que a auditoria de 05/08/2026 encontrou no `results/smoke/matrix.csv` (27
    colunas, 1,26 MB). O CSV é o arquivo que o `report.py` vai ler; uma coluna
    dessas passa silenciosamente e só aparece como estranheza na tabela final.
    """
    import pandas as pd

    rows = []
    for rec in records:
        presentes = [c for c in CAMPOS_ANEXOS if c in rec]
        if presentes:
            raise MetricsError(
                f"registro ainda carrega anexo {presentes} ao virar CSV. "
                f"Separe-os antes (ver run_all._separar_anexos): eles viajam no "
                f"registro so para sobreviver a retomada, e nao podem entrar no "
                f"matrix.csv que o report.py le."
            )
        row = {k: v for k, v in rec.items() if k not in ("metrics", "warnings")}
        row.update(rec["metrics"])
        row["warnings"] = "; ".join(rec.get("warnings", ()))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Curvas ROC/PR para a figura F1 (T4.6, 05/08/2026)
# ---------------------------------------------------------------------------

# Grade fixa de 101 pontos (0,00 a 1,00 de centésimo em centésimo).
GRADE_CURVA = np.round(np.linspace(0.0, 1.0, 101), 2)


def curve_points(
    y_true: Sequence,
    y_score: Sequence,
    grade: np.ndarray = GRADE_CURVA,
) -> dict | None:
    """Pontos das curvas ROC e PR reamostrados numa **grade fixa**.

    A F1 (§7.4) precisa das curvas, e nada no pipeline as guardava: o registro
    carrega métricas escalares e matriz de confusão, nunca o score por
    predição. Guardar os scores brutos não é opção — 192 células x até 160 mil
    predições são gigabytes, e o `results/` inteiro volta da máquina do Lucas
    por git (§8.3).

    A grade fixa resolve os dois problemas de uma vez: o arquivo fica pequeno
    e as curvas passam a ser **comparáveis entre células**, porque todas são
    avaliadas nos mesmos 101 valores de FPR (ROC) e de recall (PR). Sem isso,
    cada célula teria seus próprios limiares e sobrepor duas curvas exigiria
    reinterpolar na hora de plotar.

    Devolve `None` quando a curva é indefinida (teste de classe única), pela
    mesma razão que AUROC vem `None` e não 0 — ver a decisão de 04/08/2026.

    ═══════════════════════════════════════════════════════════════════════
    As duas curvas são reamostradas de formas DIFERENTES, e de propósito
    ═══════════════════════════════════════════════════════════════════════

    **ROC — interpolação linear.** É a convenção correta e não é aproximação:
    qualquer ponto no segmento entre dois pontos de operação da ROC é
    atingível de fato, randomizando entre os dois classificadores. Interpolar
    linearmente descreve um classificador que existe.

    **PR — envelope monótono (`max{p : r >= g}`), NÃO interpolação linear.**
    A curva PR é uma função em degraus, não monótona, e com recall repetido em
    massa: `precision_recall_curve` devolveu 2.802 pontos de recall repetido em
    4.001 num caso típico, e 3.952 em 4.001 a 3% de prevalência. Interpolar
    linearmente **atravessa** o degrau e inventa um trecho que nenhum limiar
    produz. Medido em 05/08/2026, antes desta correção:

        classificador perfeito         AP = 1,0000  ->  curva gravada 0,7525
        arvore com 3 niveis de score   AP = 1,0000  ->  curva gravada 0,9194
        97/3 com score arredondado     AP = 0,6538  ->  curva gravada 0,6740
        continuo, sem empate           AP = 0,7364  ->  curva gravada 0,7367

    Para um classificador perfeito a curva gravada **descia em reta de 1,0 até
    0,5** — a verdadeira é constante em 1,0. E esse é o regime que este projeto
    tem: a diagonal do ISCX dá 0,99999 com quatro modelos, e ensemble de árvore
    empata score em massa. Desvio máximo de precisão medido: 0,49.

    O envelope é a convenção clássica de *precisão interpolada* (Manning,
    Raghavan & Schütze, *IIR* §8.4; average precision interpolada do TREC;
    PASCAL VOC até 2009). Também é o que um leitor lê ao olhar a figura: "no
    melhor limiar que atinge recall r, a precisão é p".

    ⚠️ **Consequência que a legenda da F1 TEM de declarar (§7.4).** A área sob
    o envelope é **otimista** e não é a AUC-PR das tabelas. A AUC-PR reportada
    é a *average precision* (`average_precision_score`), pela razão declarada
    no topo deste módulo. Integrar a curva plotada dá um número diferente, e
    isso é propriedade da convenção, não defeito: são duas grandezas distintas.
    Não troque uma pela outra, e não some a área do envelope numa tabela.

    Nota histórica: a primeira versão desta função fazia
    `np.interp(grade, recall[argsort(recall)], precisao[...])`. Além de
    atravessar o degrau, `np.argsort` usa quicksort, que **não é estável** —
    a ordem entre recalls empatados não era propriedade garantida, e era o
    único ponto do pipeline cujo resultado não estava preso pela seed.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    _check_inputs(y_true, y_score)

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # `np.interp` exige x crescente; `roc_curve` já devolve fpr ordenado.
    tpr_na_grade = np.interp(grade, fpr, tpr)

    precisao, recall, _ = precision_recall_curve(y_true, y_score)
    precisao_na_grade = _envelope_precisao(precisao, recall, grade)

    return {
        "grade": [float(v) for v in grade],
        "roc_tpr": [round(float(v), 6) for v in tpr_na_grade],
        "pr_precision": [round(float(v), 6) for v in precisao_na_grade],
    }


def _envelope_precisao(
    precisao: np.ndarray, recall: np.ndarray, grade: np.ndarray
) -> np.ndarray:
    """`max{precisao : recall >= g}` para cada `g` da grade, sem laço em Python.

    `precision_recall_curve` devolve recall **decrescente**. Logo
    `recall[j] >= recall[i]` equivale a `j <= i`, e o envelope avaliado no
    próprio ponto `i` da curva é `max(precisao[0..i])` — um
    `np.maximum.accumulate` para frente. Falta então, para cada `g` da grade,
    achar o **último** ponto da curva cujo recall ainda alcança `g`.

    Não faz interpolação alguma: todo valor devolvido é uma precisão que algum
    limiar produziu de fato. Ver a nota longa em `curve_points`.
    """
    # Envelope nos pontos da curva.
    env = np.maximum.accumulate(precisao)

    # Para cada g, o último índice i com recall[i] >= g. Como `recall` é
    # decrescente, `recall[::-1]` é crescente: o menor k com rec_cresc[k] >= g
    # (o que `searchsorted(..., 'left')` devolve) corresponde a i = n-1-k.
    rec_cresc = recall[::-1]
    pos = len(recall) - 1 - np.searchsorted(rec_cresc, grade, side="left")
    # `recall[0]` é sempre 1,0 (limiar mínimo prevê tudo positivo), então
    # `pos` só sairia da faixa se a grade pedisse recall > 1. Cinturão e
    # suspensório: nunca indexar fora em vez de inventar valor.
    pos = np.clip(pos, 0, len(recall) - 1)

    return np.asarray(env[pos], dtype=np.float64)


# ---------------------------------------------------------------------------
# Intervalo de confiança por bootstrap (05/08/2026)
# ---------------------------------------------------------------------------

DEFAULT_BOOTSTRAP_N = 500
DEFAULT_BOOTSTRAP_NIVEL = 0.95


def _auroc_e_ap_de_grupos(wp: np.ndarray, wn: np.ndarray) -> tuple[float, float]:
    """AUROC e average precision a partir de contagens por grupo de score.

    `wp[g]` e `wn[g]` são os totais ponderados de positivos e de negativos do
    g-ésimo grupo de score empatado, com os grupos em ordem **decrescente de
    score**. As duas métricas não precisam de mais nada — é essa redução que
    torna o bootstrap viável aqui: a alternativa (chamar `roc_auc_score` por
    reamostra) reordena 160 mil scores a cada chamada e custaria ~13 min só nas
    células do Grambeddings.

    AUROC sai da forma de Mann-Whitney com meio-crédito para empate, que é a
    mesma convenção do `roc_auc_score`. A average precision sai de
    `AP = Σ (R_g − R_{g−1}) · P_g` sobre os grupos em ordem decrescente de
    score, que é literalmente a definição que o `average_precision_score`
    implementa — e o teste em `tests/test_evaluate.py` confere as duas contra o
    sklearn com `sample_weight`, na mesma doutrina de conferência independente
    que a defesa 2 do B1 estabeleceu para o MCC.
    """
    # Grupo sem peso algum não contribui para nenhuma das duas somas (e a
    # ordem dos que sobram não muda), então descartá-lo é exato — e barato: numa
    # reamostra com reposição, ~37% das linhas não são sorteadas. Só se descarta
    # quando os **dois** lados são zero: um grupo com wp=0 e wn>0 ainda mexe na
    # precisão dos grupos seguintes.
    if len(wp) > 64:
        vivos = np.flatnonzero(wp + wn)
        if len(vivos) < len(wp):
            wp, wn = wp[vivos], wn[vivos]

    cum_tp = np.cumsum(wp)
    cum_n = np.cumsum(wn)
    total_p = float(cum_tp[-1]) if len(cum_tp) else 0.0
    total_n = float(cum_n[-1]) if len(cum_n) else 0.0
    if total_p <= 0 or total_n <= 0:
        return float("nan"), float("nan")

    # AUROC, forma de Mann-Whitney com meio-crédito para empate. Os negativos
    # que vêm **depois** do grupo g são `total_n - cum_n[g]`, e os do próprio
    # grupo entram com metade do peso:
    #
    #     AUROC = Σ_g wp_g · (total_n − cum_n_g + wn_g/2) / (total_p · total_n)
    #
    # Escrito como produto interno para não materializar o vetor de termos: com
    # score contínuo há um grupo por linha e cada passo extra sobre 160 mil
    # elementos aparece no relógio.
    auroc = float(np.dot(wp, (total_n - cum_n) + 0.5 * wn) / (total_p * total_n))

    # AP = Σ_g (R_g − R_{g−1}) · P_g. Como `R_g − R_{g−1} = wp_g / total_p`, isto
    # é outro produto interno. Grupos iniciais sem peso teriam `cum_pred = 0` e
    # produziriam `0/0 = nan`, que contaminaria a soma em silêncio; em vez de
    # dividir com máscara (que é mais lento), corta-se o prefixo vazio — e ele
    # é exatamente um prefixo, porque `cum_pred` é não-decrescente.
    cum_pred = cum_tp + cum_n
    inicio = int(np.argmax(cum_pred > 0))
    ap = float(
        np.dot(wp[inicio:], cum_tp[inicio:] / cum_pred[inicio:]) / total_p
    )
    return auroc, ap


def _auroc_e_ap_ponderados(
    ys: np.ndarray,
    grupo_de: np.ndarray,
    n_grupos: int,
    pesos: np.ndarray,
) -> tuple[float, float]:
    """Ponte peso-por-linha → contagem-por-grupo. Existe para a conferência.

    O bootstrap não usa este caminho (ele sorteia direto no nível do grupo),
    mas o teste contra `roc_auc_score(..., sample_weight=)` precisa dele: é
    assim que se compara a fórmula deste módulo com a do sklearn sob pesos
    arbitrários, e não só sob os pesos que o bootstrap por acaso sorteia.
    """
    wp = np.bincount(grupo_de, weights=pesos * ys, minlength=n_grupos)
    wn = np.bincount(grupo_de, weights=pesos * (1.0 - ys), minlength=n_grupos)
    return _auroc_e_ap_de_grupos(wp, wn)


def bootstrap_ci(
    y_true: Sequence,
    y_score: Sequence,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    seed: int = 42,
    nivel: float = DEFAULT_BOOTSTRAP_NIVEL,
) -> dict | None:
    """IC percentil de AUROC e AUC-PR por bootstrap **estratificado**.

    Para que serve (§7.4)
    ---------------------
    A matriz é uma amostra de tamanho 1 por célula. Sem intervalo, a afirmação
    central da tese revista — "corrigir o vazamento quase não muda o número,
    média de 0,0084 de AUROC" — é uma asserção sem escala de comparação: o
    leitor não tem como saber se 0,0084 é pequeno *em relação a quê*. Este IC
    responde a variação **de amostragem do conjunto de teste**, que é a que
    domina nos corpora pequenos (o teste do ISCX tem ~4 mil linhas contra as
    ~160 mil do Grambeddings).

    Não confunda com a variação **de partição e de ajuste**, que é medida
    rodando a matriz inteira com outras seeds (`run_all.py --seed`). São duas
    incertezas diferentes e o artigo precisa das duas: um IC estreito numa
    célula não diz nada sobre a partição ter sido sorteada com sorte.

    Por que estratificado
    ---------------------
    Reamostra positivos e negativos **separadamente**, preservando a
    prevalência do conjunto de teste. É o padrão em análise de ROC (é o
    `stratified=TRUE` do `pROC::ci.auc`) e condiciona no que é propriedade do
    corpus, não da amostra. De quebra elimina a reamostra degenerada de classe
    única, em que AUROC seria indefinido. O bootstrap i.i.d. não estratificado
    também seria defensável e daria intervalo um pouco mais largo, por incluir
    a flutuação da prevalência; a escolha está registrada na §11.

    Determinismo (RNF1)
    -------------------
    O gerador é criado do zero a cada chamada, a partir de `seed`, e sorteia no
    espaço **já ordenado por score**. Logo o resultado não depende da ordem em
    que as células foram executadas — o que importa porque a retomada (RH2)
    executa os ajustes em ordem diferente e as duas execuções têm de dar o
    mesmo número.

    Devolve `None` em teste de classe única, pela mesma razão que AUROC vem
    `None` e não 0 (decisão de 04/08/2026).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    _check_inputs(y_true, y_score)

    if len(np.unique(y_true)) < 2:
        return None
    if n_resamples < 2:
        raise MetricsError(f"n_resamples tem de ser >= 2; recebi {n_resamples}")
    if not 0 < nivel < 1:
        raise MetricsError(f"nivel tem de estar em (0,1); recebi {nivel}")

    # Ordenação decrescente de score, estável para não depender do desempate
    # arbitrário de um quicksort (foi o defeito da primeira `curve_points`).
    ordem = np.argsort(-y_score, kind="stable")
    ys = y_true[ordem].astype(np.float64)
    ss = y_score[ordem]

    # Grupos de score empatado. As duas métricas dependem **apenas** do total
    # ponderado de positivos e de negativos por grupo: dentro de um empate as
    # linhas são intercambiáveis, tanto no meio-crédito da AUROC quanto no
    # acumulado da AP. Isso é o que permite reamostrar no nível do grupo.
    novo_grupo = np.r_[True, ss[1:] != ss[:-1]]
    grupo_de = np.cumsum(novo_grupo) - 1
    n_grupos = int(grupo_de[-1]) + 1

    pos = np.flatnonzero(ys == 1)
    neg = np.flatnonzero(ys == 0)
    g_pos, g_neg = grupo_de[pos], grupo_de[neg]
    contagem_pos = np.bincount(g_pos, minlength=n_grupos).astype(np.float64)
    contagem_neg = np.bincount(g_neg, minlength=n_grupos).astype(np.float64)
    p_pos = contagem_pos / contagem_pos.sum()
    p_neg = contagem_neg / contagem_neg.sum()

    # Duas rotas, mesma distribuição exata, custos opostos — a escolha é só de
    # desempenho e é função determinística dos dados, então o RNF1 não sofre.
    #
    #   sorteio por GRUPO  é O(n_grupos): imbatível quando o score empata muito,
    #                      que é o caso dos quatro ensembles de árvore (medido:
    #                      976 grupos em 160 mil linhas -> 0,29 s por célula).
    #   sorteio por LINHA  é O(n): necessário quando o score é quase contínuo,
    #                      caso da regressão logística e do MLP, em que
    #                      n_grupos ~ n e o multinomial passaria de 18 s.
    #
    # Sem esta escolha o bootstrap custaria ~10 min por seed; com ela, ~1,5 min.
    sortear_por_grupo = 2 * n_grupos <= len(ys)
    rng = np.random.default_rng(seed)

    aurocs = np.empty(n_resamples, dtype=np.float64)
    aps = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        if sortear_por_grupo:
            wp = rng.multinomial(len(pos), p_pos).astype(np.float64)
            wn = rng.multinomial(len(neg), p_neg).astype(np.float64)
        else:
            wp = np.bincount(
                g_pos[rng.integers(0, len(pos), size=len(pos))], minlength=n_grupos
            ).astype(np.float64)
            wn = np.bincount(
                g_neg[rng.integers(0, len(neg), size=len(neg))], minlength=n_grupos
            ).astype(np.float64)
        aurocs[i], aps[i] = _auroc_e_ap_de_grupos(wp, wn)

    baixo = 100.0 * (1.0 - nivel) / 2.0
    alto = 100.0 - baixo
    return {
        "auroc_lo": round(float(np.nanpercentile(aurocs, baixo)), 6),
        "auroc_hi": round(float(np.nanpercentile(aurocs, alto)), 6),
        "auc_pr_lo": round(float(np.nanpercentile(aps, baixo)), 6),
        "auc_pr_hi": round(float(np.nanpercentile(aps, alto)), 6),
        "n_resamples": int(n_resamples),
        "nivel": float(nivel),
    }


__all__ = [
    "METRIC_KEYS",
    "METRIC_RANGES",
    "DEFAULT_ECE_BINS",
    "DEFAULT_BOOTSTRAP_N",
    "DEFAULT_BOOTSTRAP_NIVEL",
    "THRESHOLD_CRITERIA",
    "MetricsError",
    "mcc_from_confusion",
    "mcc",
    "confusion_counts",
    "expected_calibration_error",
    "select_threshold",
    "evaluate_predictions",
    "validate_metrics",
    "git_commit_hash",
    "make_run_id",
    "make_result_record",
    "records_to_dataframe",
    "curve_points",
    "GRADE_CURVA",
    "bootstrap_ci",
]
