"""T4.4 / RF9 — PORTÃO DE ENTREGA. Valida o pipeline inteiro em amostra pequena.

    venv/Scripts/python.exe scripts/run_smoke.py

⛔ NÃO SIGA PARA `run_all.py` SE ESTE SCRIPT FALHAR (SDD §8.2, RK5).

O que ele exercita, de ponta a ponta, com amostra de no máximo 5.000 linhas por
corpus (o teto do DoD da T4.4):

    corpora/*.parquet  →  canonização + 41 features  →  2 regimes de split
        →  6 modelos  →  matriz cruzada 4×4  →  métricas validadas  →  JSON/CSV

Ou seja: a mesma fiação que `run_all.py` usará, com o orçamento de cada modelo
cortado pelos `smoke.overrides` de `config/experiment.yml`.

O que ele NÃO é
---------------
**Não é resultado científico.** Os hiperparâmetros são reduzidos ao mínimo
(40 árvores em vez de 400, MLP de uma camada de 32 em vez de [128, 64]) e a
amostra é ~0,6% do Grambeddings. Qualquer número que saia daqui é descartável.
Por isso a saída vai para `results/smoke/`, nunca para `results/` — para que
não exista a menor chance de um número de smoke virar linha de tabela do
artigo (R5).

Também **não viola** SDD §0: rodar o smoke em amostra ≤ 5k é explicitamente
permitido nesta máquina (CLAUDE.md, "Permitido aqui"). O `.fit()` de verdade,
sobre corpus completo, roda na máquina do Lucas.

Requisitos de handoff que este script cumpre
--------------------------------------------
RH1 — falha alto e claro se faltar dependência ou corpus (e diz qual).
RH3 — progresso legível linha a linha: o Lucas precisa ver que está andando.
RH5 — nenhuma pergunta interativa; código de saída 0 = passou, 1 = falhou.
RF9 — orçamento de 60 s, verificado e reportado no fim.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

# `scripts/` não é pacote: adiciona `src/` ao path, igual ao resto do projeto.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CORPORA_DIR = REPO_ROOT / "corpora"
OUT_DIR = REPO_ROOT / "results" / "smoke"


# ---------------------------------------------------------------------------
# Saída legível (RH3)
# ---------------------------------------------------------------------------

# Descricao curta do `--help`. NAO usar `__doc__` aqui: o console do
# Windows e cp1252 e a docstring tem setas/travessoes, entao `--help`
# morria com UnicodeEncodeError. A pendencia 10 da SDD 13.4 dizia que
# docstring era livre -- e era, ate `description=__doc__` transformar a
# docstring em saida impressa. Mantenha esta string em ASCII puro.
DESCRICAO_CLI = (
    "Portao de entrega: valida o pipeline inteiro em amostra pequena. Nenhum numero daqui pode entrar no artigo."
)

_T0 = time.perf_counter()


def silenciar_logs_de_biblioteca() -> None:
    """Deixa o progresso do smoke legível (RH3).

    `src/utils/logging.py` emite JSON estruturado em nível INFO — útil numa
    execução longa, ilegível aqui: só a inicialização dos extratores produz
    cinco linhas de JSON antes de qualquer trabalho. Subimos esses loggers para
    WARNING, o que mantém visível o que importa (URL inválida, feature
    saturando) e descarta o "initialized".
    """
    import logging
    import warnings

    for nome in ("data.url_canonicalization", "features.lexical_features", "features",
                 "data.sampling", "utils.io"):
        logging.getLogger(nome).setLevel(logging.WARNING)

    # O `ConvergenceWarning` do MLP é **consequência esperada** do override do
    # smoke: `max_iter: 30` contra os 200 da configuração real. Suprimir aqui é
    # correto porque o smoke não mede qualidade de ajuste; suprimir em
    # `run_all.py` seria esconder um problema de verdade — não copie esta linha
    # para lá.
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def log(msg: str = "") -> None:
    """Uma linha de progresso com o relógio desde o início."""
    if msg:
        print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)
    else:
        print(flush=True)


def falhar(msg: str, *, dica: str = "") -> NoReturn:  # noqa: F821
    """RH1: falha alto, clara e acionável."""
    print(f"\n{'=' * 74}", flush=True)
    print(f"SMOKE FALHOU: {msg}", flush=True)
    if dica:
        print(f"\n{dica}", flush=True)
    print(f"{'=' * 74}", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Pré-requisitos (RH1)
# ---------------------------------------------------------------------------


def checar_dependencias() -> None:
    """Importa tudo o que o pipeline usa, antes de gastar tempo com dados."""
    faltando = []
    for mod in ("numpy", "pandas", "pyarrow", "sklearn", "yaml", "tldextract",
                "xgboost", "lightgbm", "catboost"):
        try:
            __import__(mod)
        except ImportError as exc:
            faltando.append(f"{mod} ({exc})")
    if faltando:
        falhar(
            f"dependências ausentes: {', '.join(faltando)}",
            dica="Instale o ambiente pinado:\n"
                 "    venv\\Scripts\\python.exe -m pip install -r requirements.lock\n"
                 "e confira com:\n"
                 "    venv\\Scripts\\python.exe scripts\\verify_environment.py",
        )
    log("dependências: OK (9 bibliotecas)")


def checar_corpora(nomes: list[str]) -> None:
    """Confere que os parquet dos corpora de treino existem."""
    ausentes = [n for n in nomes if not (CORPORA_DIR / n / "data.parquet").exists()]
    if ausentes:
        falhar(
            f"corpora ausentes: {', '.join(ausentes)}",
            dica=f"Os parquet não são versionados no git (SDD §7.1, .gitignore).\n"
                 f"Esperados em {CORPORA_DIR.name}/<nome>/data.parquet.\n"
                 f"Gere-os a partir dos CSVs brutos:\n"
                 f"    venv\\Scripts\\python.exe -m src.corpora --raw-dir <dir> --out-dir corpora\n"
                 f"Ou peça o pacote de corpora ao Julio (SDD §12 P5).",
        )
    log(f"corpora: OK ({len(nomes)} parquet encontrados)")


# ---------------------------------------------------------------------------
# Preparação de dados
# ---------------------------------------------------------------------------


def amostrar_corpus(nome: str, n_max: int, seed: int, com_linha: bool = False):
    """Carrega um corpus e devolve uma amostra estratificada de ≤ `n_max` linhas.

    Com `com_linha=True` a amostra ganha a coluna `linha` (int32): a posição de
    cada linha no parquet do corpus, antes da amostragem. É a identidade que as
    predições persistidas usam (PRED-01), e não depende do índice da amostra.

    A estratificação é conjunta por (`split_oficial`, `label`): o Regime A
    precisa de linhas dos dois lados do split da fonte, e os dois regimes
    precisam das duas classes. Uma amostra aleatória simples de um corpus 97/3
    poderia deixar o teste oficial com uma classe só, e aí o smoke reprovaria
    por causa da amostragem, não por causa do pipeline.

    A amostra é a mesma para os dois regimes, de propósito: assim a diferença
    entre eles vem do split, não de terem visto dados diferentes.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(
        CORPORA_DIR / nome / "data.parquet",
        columns=["label", "url", "corpus", "split_oficial"],
    )
    if com_linha:
        df["linha"] = np.arange(len(df), dtype=np.int32)
    if len(df) <= n_max:
        return df.reset_index(drop=True)

    # Amostragem por estrato feita sobre índices posicionais, sem
    # `groupby().apply()`: no pandas 3 o `apply` recebe o grupo já **sem** as
    # colunas de agrupamento, e `label`/`split_oficial` desapareceriam do
    # resultado. Fazer no numpy é mais rápido, determinístico e não depende
    # dessa semântica.
    frac = n_max / len(df)
    rng = np.random.default_rng(seed)
    estratos = df.groupby(["split_oficial", "label"], observed=True).indices
    escolhidos = []
    for _, posicoes in sorted(estratos.items(), key=lambda kv: str(kv[0])):
        n_pick = min(len(posicoes), max(2, int(round(frac * len(posicoes)))))
        escolhidos.append(rng.choice(posicoes, size=n_pick, replace=False))

    idx = np.sort(np.concatenate(escolhidos))
    # `round` por estrato pode passar alguns registros do teto; corta o excesso
    # mantendo determinismo.
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, size=n_max, replace=False))
    return df.iloc[idx].reset_index(drop=True)


def preparar(nomes: list[str], n_max: int, seed: int) -> dict:
    """Amostra + canoniza + extrai features de cada corpus."""
    import features

    preparados = {}
    for nome in nomes:
        t0 = time.perf_counter()
        amostra = amostrar_corpus(nome, n_max, seed, com_linha=True)
        frame = features.build_feature_frame(amostra)
        n_invalidas = int(frame["invalid_url"].sum())
        preparados[nome] = frame
        log(
            f"  {nome:14s} {len(frame):5d} linhas  "
            f"{frame['etld_plus_one'].nunique():5d} domínios  "
            f"{int((frame['label'] == 1).sum()):5d} phishing  "
            f"{n_invalidas:3d} URL inválida  ({time.perf_counter() - t0:.1f}s)"
        )
        if len(frame) > n_max:
            falhar(
                f"amostra de '{nome}' tem {len(frame)} linhas, acima do teto de "
                f"{n_max} do DoD da T4.4"
            )
    return preparados


# ---------------------------------------------------------------------------
# O laço — delegado a src/matrix.py
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def contar_mlp_agrupado():
    """Conta os ajustes do MLP que chegaram ao `fit` com grupos (MLP-01, RUN-05).

    Troca `models.ajustar` por um espião que chama o original. O MLP agrupado
    levanta `RuntimeError` se o corte do sklearn não for usado exatamente uma
    vez, então um ajuste contado aqui é um ajuste que fez a validação agrupada.
    """
    import models

    contador = {"n": 0}
    original = models.ajustar

    def espiao(modelo, nome, X, y, grupos=None):
        if (
            nome == "mlp"
            and grupos is not None
            and isinstance(models._final_estimator(modelo), models.MLPValidacaoAgrupada)
        ):
            contador["n"] += 1
        return original(modelo, nome, X, y, grupos=grupos)

    models.ajustar = espiao
    try:
        yield contador
    finally:
        models.ajustar = original


def executar(
    config: dict, preparados: dict, dir_rodada: Path
) -> tuple[list[dict], dict, list[dict], int]:
    """Roda a matriz cruzada reduzida chamando o laço compartilhado.

    Devolve `(registros da matriz, resumo das partições, registros do LOCO,
    ajustes do MLP com validação agrupada)`. As predições vão para `dir_rodada`,
    um diretório temporário (RUN-05): o portão exercita a persistência sem
    deixar nada em `results/`.

    O laço vive em `src/matrix.py` e é **o mesmo** que `scripts/run_all.py`
    executa na máquina do Lucas. Isso é o ponto: a restrição §0 impede validar o
    laço real aqui, então o smoke precisa exercitar exatamente aquele código —
    senão o portão aprovaria um caminho e o Lucas rodaria outro, deixando o risco
    RK5 de pé. Os 48 ajustes abaixo são a validação do código de produção.

    A diferença entre os dois scripts é só argumento: aqui, amostra ≤ 5k e
    hiperparâmetros cortados; lá, corpus completo e hiperparâmetros reais.

    `ledger_path=None` de propósito: um portão sempre roda do zero. Retomar um
    smoke esconderia justamente a falha que ele existe para encontrar.
    """
    import evaluate
    import matrix
    import models

    gpu = models.detect_gpu()
    run_id = evaluate.make_run_id()

    log(f"run_id: {run_id}")
    log(f"dispositivo: {gpu.describe()}")
    log(f"limiar: critério '{config['avaliacao']['criterio_limiar']}' na validação "
        f"do corpus de treino")

    with contar_mlp_agrupado() as mlp_agrupado:
        try:
            resultado = matrix.run_matrix(
                config, preparados, gpu,
                run_id=run_id,
                ledger_path=None,
                log=log,
                extra_por_registro={"smoke": True},
                dir_rodada=dir_rodada,
            )
        except matrix.MatrixError as exc:
            falhar(str(exc))
        except evaluate.MetricsError as exc:
            falhar(f"métrica inválida: {exc}")
        except Exception as exc:  # noqa: BLE001 (o motivo tem de aparecer no log)
            falhar(f"{type(exc).__name__}: {exc}", dica=traceback.format_exc())

        # O portão exercita também o leave-one-corpus-out (D3-b). Sem isto, o LOCO
        # seria o único caminho de código que o Lucas roda sem o smoke ter tocado,
        # exatamente o RK5 que este módulo existe para eliminar. Os registros não
        # entram no consolidado do smoke: o que importa aqui é que o laço fia.
        log()
        log("exercitando o leave-one-corpus-out (D3-b)...")
        try:
            loco = matrix.run_loco(
                config, preparados, gpu,
                run_id=run_id,
                ledger_path=None,
                log=log,
                extra_por_registro={"smoke": True},
                dir_rodada=dir_rodada,
            )
        except matrix.MatrixError as exc:
            falhar(f"leave-one-corpus-out: {exc}")
        except evaluate.MetricsError as exc:
            falhar(f"leave-one-corpus-out, métrica inválida: {exc}")
        except Exception as exc:  # noqa: BLE001
            falhar(f"leave-one-corpus-out: {type(exc).__name__}: {exc}",
                   dica=traceback.format_exc())
        n_mlp_agrupado = mlp_agrupado["n"]

    esperado_loco = len(preparados) * len(config["regimes"]) * len(config["modelos"])
    if len(loco.records) != esperado_loco:
        falhar(
            f"LOCO produziu {len(loco.records)} registros, esperava {esperado_loco} "
            f"({len(preparados)} corpora x {len(config['regimes'])} regimes x "
            f"{len(config['modelos'])} modelos)"
        )
    log(f"LOCO: {len(loco.records)} avaliacoes, fiacao OK")

    return resultado.records, resultado.split_summaries, loco.records, n_mlp_agrupado


# ---------------------------------------------------------------------------
# Conferências
# ---------------------------------------------------------------------------


def conferir(registros: list[dict], config: dict, n_corpora: int) -> None:
    """Conferências estruturais (compartilhadas) + a sanidade própria do portão."""
    import matrix

    try:
        matrix.check_records(registros, config, n_corpora)
    except matrix.MatrixError as exc:
        falhar(str(exc))

    # Sanidade que só o portão faz: um modelo que não aprendeu NADA em nenhuma
    # célula da diagonal indica fiação quebrada, não hiperparâmetro curto.
    # AUROC 0,5 é o chute.
    diagonais = [r for r in registros if r["diagonal"]]
    aurocs = [r["metrics"]["auroc"] for r in diagonais if r["metrics"]["auroc"] is not None]
    if not aurocs:
        falhar("nenhum AUROC calculável na diagonal")
    if max(aurocs) < 0.60:
        falhar(
            f"melhor AUROC na diagonal = {max(aurocs):.3f}, próximo do chute (0,5).\n"
            f"Com features léxicas funcionando, a diagonal fica bem acima disso "
            f"mesmo com 40 árvores. Suspeite de features zeradas (B4), rótulo "
            f"invertido no corpus (grambeddings usa 2=legítimo, 1=phishing) ou "
            f"escala ausente."
        )
    log(f"melhor AUROC na diagonal: {max(aurocs):.3f} (sanidade OK)")


def problemas_predicoes(
    dir_rodada: Path, registros: list[dict], tipo: str, n_ajustes: int
) -> list[str]:
    """Conferência 1 (PRED-01, PRED-02, PRED-03): predições por ajuste.

    Devolve a lista de problemas (vazia se está tudo certo): número de parquets
    em `predicoes/<tipo>/`, referência com sha256 válido em todo registro,
    tamanho do bloco igual a `n_test` e AUROC recalculada do parquet igual à do
    registro (diferença menor que 1e-9).
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    import predicoes

    problemas: list[str] = []
    pasta = Path(dir_rodada) / predicoes.SUBDIR / tipo
    n_arquivos = len(list(pasta.glob("*.parquet"))) if pasta.is_dir() else 0
    if n_arquivos != n_ajustes:
        problemas.append(
            f"{tipo}: {n_arquivos} parquet(s) de predições, esperava {n_ajustes} (um por ajuste)"
        )

    lidos: dict[str, pd.DataFrame] = {}
    for rec in registros:
        chave = f"{rec['regime']}/{rec['train_corpus']}/{rec['test_corpus']}/{rec['model']}"
        ref = rec.get("predicoes")
        if not predicoes.conferir(dir_rodada, ref):
            problemas.append(f"{tipo}: {chave} sem predições válidas (ausentes ou sha256 diferente)")
            continue
        arquivo = ref["arquivo"]
        if arquivo not in lidos:
            lidos[arquivo] = predicoes.ler(dir_rodada, ref)
        df = lidos[arquivo]
        bloco = df[df["test_corpus"] == rec["test_corpus"]]
        if len(bloco) != rec["n_test"]:
            problemas.append(f"{tipo}: {chave} tem {len(bloco)} predições, n_test é {rec['n_test']}")
            continue
        auroc = rec["metrics"]["auroc"]
        if auroc is None or bloco["y_true"].nunique() < 2:
            continue
        recalculada = float(roc_auc_score(bloco["y_true"], bloco["score"]))
        if abs(recalculada - auroc) >= 1e-9:
            problemas.append(
                f"{tipo}: {chave} AUROC do registro {auroc!r} difere da recalculada "
                f"{recalculada!r}"
            )
    return problemas


def problemas_dedup(resumos: dict, n_linhas: dict, regimes: list[str]) -> list[str]:
    """Conferência 2 (DEDUP-02, DEDUP-04): contagens de dedup coerentes.

    Cada `dataset/regime` tem de trazer as contagens, `n_antes` igual ao
    tamanho do quadro, a conta fechando (`antes - conflito - duplicatas =
    final`) e as três partições somando `n_final`. No Regime A as URLs
    repetidas entre treino e teste também são contadas.
    """
    problemas: list[str] = []
    for corpus, n_antes in n_linhas.items():
        for regime in regimes:
            chave = f"{corpus}/{regime}"
            dedup = (resumos.get(chave) or {}).get("dedup")
            if not dedup:
                problemas.append(f"{chave}: resumo de partições sem contagens de dedup")
                continue
            if dedup["n_antes"] != n_antes:
                problemas.append(
                    f"{chave}: n_antes={dedup['n_antes']}, o quadro tem {n_antes} linhas"
                )
            conta = dedup["n_antes"] - dedup["n_conflito_linhas"] - dedup["n_duplicatas_removidas"]
            if conta != dedup["n_final"]:
                problemas.append(
                    f"{chave}: antes - conflito - duplicatas = {conta}, n_final = {dedup['n_final']}"
                )
            soma = sum(p["n_rows"] for p in resumos[chave]["partitions"].values())
            if soma != dedup["n_final"]:
                problemas.append(
                    f"{chave}: as partições somam {soma} linhas, n_final = {dedup['n_final']}"
                )
            if regime == "oficial" and "n_repetidas_treino_teste" not in dedup:
                problemas.append(f"{chave}: Regime A sem n_repetidas_treino_teste")
    return problemas


def problemas_mlp_agrupado(
    registros: list[dict], n_ajustes_agrupados: int
) -> list[str]:
    """Conferência 3 (MLP-01, MLP-02): MLP agrupado no Regime B e só nele.

    `registros` reúne os da matriz e os do LOCO. A proveniência do MLP tem de
    dizer `agrupada_etld1` no Regime B e `linha` no A, e `n_ajustes_agrupados`
    (ajustes que chegaram ao `fit` com grupos, vistos pelo espião) tem de ser
    igual ao número de ajustes do MLP no Regime B.
    """
    problemas: list[str] = []
    ajustes_b: set = set()
    for rec in registros:
        if rec["model"] != "mlp":
            continue
        esperado = "agrupada_etld1" if rec["regime"] == "etld1" else "linha"
        obtido = rec["provenance"].get("validacao_interna")
        if obtido != esperado:
            problemas.append(
                f"mlp {rec['regime']}/{rec['train_corpus']}: validacao_interna={obtido!r}, "
                f"esperava {esperado!r}"
            )
        if rec["regime"] == "etld1":
            ajustes_b.add(rec["train_corpus"])
    if n_ajustes_agrupados != len(ajustes_b):
        problemas.append(
            f"{n_ajustes_agrupados} ajuste(s) do MLP com validação agrupada, "
            f"esperava {len(ajustes_b)} (um por treino do Regime B, matriz e LOCO)"
        )
    return problemas


def conferir_frente_a(
    dir_rodada: Path, registros: list[dict], registros_loco: list[dict],
    resumo_splits: dict, preparados: dict, config: dict, n_mlp_agrupado: int,
) -> None:
    """As três conferências da frente A (RUN-05); qualquer problema reprova o portão."""
    n_corpora = len(preparados)
    n_modelos = len(config["modelos"])
    n_ajustes = len(config["regimes"]) * n_corpora * n_modelos
    problemas = (
        problemas_predicoes(dir_rodada, registros, "matriz", n_ajustes)
        + problemas_predicoes(dir_rodada, registros_loco, "loco", n_ajustes)
        + problemas_dedup(
            resumo_splits, {n: len(q) for n, q in preparados.items()}, list(config["regimes"])
        )
        + problemas_mlp_agrupado(registros + registros_loco, n_mlp_agrupado)
    )
    if problemas:
        falhar(
            f"{len(problemas)} problema(s) nas conferências de predições, dedup e MLP agrupado.",
            dica="\n".join(f"  - {p}" for p in problemas[:20]),
        )
    log(f"predições: {n_ajustes} ajustes da matriz e {n_ajustes} do LOCO, AUROC recalculada bate")
    n_dup = sum(v["dedup"]["n_duplicatas_removidas"] for v in resumo_splits.values())
    log(f"dedup: contagens coerentes em {len(resumo_splits)} partições ({n_dup} duplicatas removidas)")
    log(f"MLP agrupado: {n_mlp_agrupado} ajustes com validação por eTLD+1 no Regime B")


def gravar(registros: list[dict], resumo_splits: dict, config: dict) -> None:
    import evaluate
    import matrix as matrix_mod

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mesma separação de anexos que a execução real faz (T4.6). O portão tem de
    # exercitar **este** escritor, não um parecido: a auditoria de 05/08/2026
    # encontrou a separação só no `run_all.py`, e o resultado era um
    # `results/smoke/matrix.json` de 1,26 MB com as curvas dentro e um CSV com
    # uma coluna `curves` — ou seja, o único código que produz a fonte da F1
    # nunca passava pelo portão. É o RK5 pela porta dos fundos.
    anexos = matrix_mod.separar_anexos(registros, OUT_DIR)

    matriz = OUT_DIR / "matrix.json"
    with matriz.open("w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
        f.write("\n")

    csv = OUT_DIR / "matrix.csv"
    evaluate.records_to_dataframe(registros).to_csv(csv, index=False, encoding="utf-8")

    resumo = OUT_DIR / "splits.json"
    with resumo.open("w", encoding="utf-8") as f:
        json.dump(resumo_splits, f, ensure_ascii=False, indent=2)
        f.write("\n")

    aviso = OUT_DIR / "LEIA-ME.md"
    aviso.write_text(
        "# results/smoke/ — saída do portão, NÃO é resultado\n\n"
        "Gerado por `scripts/run_smoke.py` (T4.4). Os números aqui vêm de uma\n"
        "amostra de no máximo "
        f"{config['smoke']['sample_size']} linhas por corpus e de "
        "hiperparâmetros\ncortados ao mínimo (`smoke.overrides` em "
        "`config/experiment.yml`).\n\n"
        "**Nenhum número deste diretório pode entrar no artigo** (SDD R5). O\n"
        "resultado de verdade é produzido por `scripts/run_all.py` na máquina do\n"
        "Lucas e gravado em `results/`.\n",
        encoding="utf-8",
    )

    for path in (matriz, csv, resumo, aviso, *(OUT_DIR / a for a in anexos)):
        log(f"gravado: {path.relative_to(REPO_ROOT).as_posix()}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRICAO_CLI)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="teto de linhas por corpus (default: smoke.sample_size do config; máx. 5000)",
    )
    parser.add_argument(
        "--tempo-limite", type=float, default=None,
        help="orçamento em segundos (default: smoke.tempo_limite_s do config)",
    )
    args = parser.parse_args()

    print("=" * 74)
    print("SMOKE TEST — portão de entrega (SDD §8.2, T4.4)")
    print("Amostra pequena, hiperparâmetros reduzidos. NÃO é resultado científico.")
    print("=" * 74)

    checar_dependencias()
    t_import = time.perf_counter() - _T0
    silenciar_logs_de_biblioteca()

    import models

    try:
        config = models.load_experiment_config()
    except models.ModelConfigError as exc:
        falhar(str(exc))
    config = models.apply_smoke_overrides(config)

    n_max = args.sample_size or int(config["smoke"]["sample_size"])
    if n_max > 5000:
        falhar(
            f"--sample-size {n_max} excede o teto de 5.000 linhas do DoD da T4.4 "
            f"(e SDD §0 proíbe treino sobre corpus completo nesta máquina)"
        )
    limite = args.tempo_limite or float(config["smoke"]["tempo_limite_s"])
    nomes = list(config["corpora"]["treino_avaliacao"])

    checar_corpora(nomes)
    log(f"amostra: até {n_max} linhas por corpus · seed {config['seed']}")
    log()

    log("preparando corpora (canonização + 41 features)...")
    _t = time.perf_counter()
    preparados = preparar(nomes, n_max, int(config["seed"]))
    t_prep = time.perf_counter() - _t

    # T4.11 (§13.9): "matriz" escondia o particionamento, que medido é o maior
    # item isolado da execução — 27,4 s dos 60,3 s, mais que os 96 ajustes
    # somados. Separar as duas é o que permite ler a decomposição corretamente.
    # O zeramento existe para o caso de `main()` rodar duas vezes no mesmo
    # processo (um teste, por exemplo): o acumulador é de módulo.
    import matrix

    matrix.zerar_tempo_particionamento()
    _t = time.perf_counter()
    # As predições vão para um diretório temporário (RUN-05): o portão exercita
    # a persistência e a conferência sem deixar nada em `results/`.
    with tempfile.TemporaryDirectory(prefix="smoke_predicoes_") as pasta:
        dir_rodada = Path(pasta)
        registros, resumo_splits, registros_loco, n_mlp_agrupado = executar(
            config, preparados, dir_rodada
        )
        t_bloco = time.perf_counter() - _t
        t_split = matrix.tempo_particionamento_s()
        t_loop = t_bloco - t_split

        log()
        conferir(registros, config, len(nomes))
        conferir_frente_a(
            dir_rodada, registros, registros_loco, resumo_splits, preparados,
            config, n_mlp_agrupado,
        )
    # As referências apontavam para o diretório temporário, já apagado.
    for rec in registros:
        rec.pop("predicoes", None)
    gravar(registros, resumo_splits, config)

    total = time.perf_counter() - _T0
    n_fits = len(config["regimes"]) * len(nomes) * len(config["modelos"])
    print()
    print("=" * 74)
    print(f"{n_fits} fits · {len(registros)} avaliações · {total:.1f}s "
          f"(orçamento RF9: {limite:.0f}s)")
    print(f"  importar bibliotecas {t_import:5.1f}s · "
          f"preparar features {t_prep:5.1f}s · particionar {t_split:5.1f}s · "
          f"matriz {t_loop:5.1f}s")

    # O código de saída reflete a FIAÇÃO, não o relógio.
    #
    # Um estouro de orçamento com a fiação íntegra não é o mesmo defeito que uma
    # dependência faltando ou uma métrica impossível: a primeira versão deste
    # script devolvia 1 nos dois casos, o que bloquearia a entrega ao Lucas
    # porque o Spotify estava tocando. Medido em 04/08/2026: o mesmo commit levou
    # 17,5 s com a máquina ociosa e 65 s com OneDrive/Cursor/Spotify ativos —
    # mesmos 192 registros, bit a bit.
    #
    # A T4.11 (§13.9) fechou isso com perfil em 06/08/2026, e a decisão de deixar
    # o código de saída em 0 saiu reforçada: os MESMOS 48 ajustes custaram 8,9 s
    # numa execução e 25,9 s noutra — mesma seed, mesmos dados, mesmo commit —, e
    # a execução rodada sob `cProfile`, que só acrescenta custo, foi a rápida
    # (80,4 s, dentro do orçamento). Não há decomposição de fase que explique
    # isso: o que varia é o throughput da máquina, e todas as fases escalam
    # juntas. Um orçamento de relógio não distingue isso de defeito, e é por isso
    # que ele não decide o código de saída.
    #
    # O estouro continua **visível e alto**, com a decomposição acima para que
    # dê para atribuir a fase responsável. Falha de fiação sai por `falhar()`,
    # que devolve 1.
    if total > limite:
        print()
        print(f"AVISO: fiação OK, mas o orçamento do RF9 estourou "
              f"({total:.1f}s > {limite:.0f}s).")
        print("   Isto quase sempre é carga de máquina, e não defeito. Perfilado")
        print("   na T4.11 (SDD §13.9): com a máquina inteira o portão fecha em")
        print("   80,4s, e o MESMO trabalho já custou 2,9x mais com ela ocupada.")
        print("   'particionar' ser o maior item é esperado (StratifiedGroupKFold,")
        print("   decisão RF4), não é regressão. Feche o que estiver aberto e rode")
        print("   de novo; se o tempo não cair, avise o Julio antes de seguir.")
    print()
    print("SMOKE PASSOU. Pipeline validado de ponta a ponta.")
    print("Próximo passo (máquina do Lucas, com GPU): scripts/run_all.py")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
