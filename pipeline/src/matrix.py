"""T4.3 — o laço da matriz cruzada, compartilhado pelo smoke e pela execução real.

Por que este módulo existe (e por que NÃO é duplicado nos dois scripts)
----------------------------------------------------------------------
`scripts/run_smoke.py` é o portão de entrega e `scripts/run_all.py` é a execução
que produz os números do artigo. Se cada um tivesse sua própria cópia do laço, o
smoke validaria um código e o Lucas executaria outro — e o risco RK5 (
"bug descoberto tarde invalida uma rodada") ficaria intacto, porque a
restrição §0 impede validar o laço real nesta máquina.

Com o laço aqui, os **48 ajustes do smoke exercitam exatamente o caminho de
código que o Lucas vai rodar**: mesma partição, mesma seleção de limiar, mesma
avaliação, mesmo livro-caixa, mesma retomada. A única diferença entre os dois é o
que se passa como argumento — amostra e hiperparâmetros cortados no smoke, corpus
completo e hiperparâmetros reais na execução final.

Unidade de trabalho e retomada (RH2)
------------------------------------
A unidade é o **ajuste** (`regime` × `corpus de treino` × `modelo`), não a
avaliação: ajustar custa minutos, avaliar custa segundos. Cada ajuste grava seus
N registros de uma vez no livro-caixa `.jsonl`, com `fsync`. Ao retomar, todo
ajuste que já tenha os N corpora de teste registrados é pulado.

Este módulo chama `.fit()`. Quem o invoca é responsável por respeitar a §0.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import evaluate
import features
import models
import predicoes
import psl
import splits


class MatrixError(RuntimeError):
    """Falha estrutural no laço da matriz — nunca silenciada."""


@dataclass
class MatrixRun:
    """Resultado de uma passada pelo laço.

    `indices` guarda as partições `{(dataset, regime): SplitIndices}` com posições
    do quadro original, para `predicoes.gravar_particoes`.
    """

    records: list[dict] = field(default_factory=list)
    split_summaries: dict = field(default_factory=dict)
    n_fits_executados: int = 0
    n_fits_pulados: int = 0
    fit_seconds: list[float] = field(default_factory=list)
    indices: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Impressão digital da configuração
# ---------------------------------------------------------------------------


def config_fingerprint(config: dict) -> str:
    """Hash dos parâmetros que afetam o resultado numérico.

    Serve para impedir retomada suja: se a `experiment.yml` mudar no meio de uma
    execução, reaproveitar os registros antigos misturaria hiperparâmetros
    diferentes na mesma tabela — o que o SDD §4 D5 proíbe, e que não deixaria
    rastro visível na tabela final.

    Deliberadamente **não** inclui `cpu.n_jobs`, `gpu` nem `smoke.tempo_limite_s`:
    número de núcleos não muda o resultado (o determinismo está travado em
    `models._predicao_serial`) e o dispositivo já é registrado na proveniência de
    cada registro.
    """
    import hashlib

    # SENS-01: a rodada sem esquema usa `config["_strip_scheme"]`, no mesmo
    # molde do `_smoke`, para ter livro-caixa e impressão digital próprios.
    strip_scheme = bool(config.get("_strip_scheme", False))

    relevante = {
        "seed": config["seed"],
        "modelos": config["modelos"],
        "split": config["split"],
        "avaliacao": config["avaliacao"],
        "corpora": config["corpora"],
        "smoke": bool(config.get("_smoke", False)),
        # Canárias de comportamento (05/08/2026). A `experiment.yml` sozinha
        # não descreve o resultado: a lógica de extração e a de particionamento
        # também o determinam, e nenhuma das duas aparece no config. Sem isto,
        # corrigir a T3.4 ou trocar o particionador deixa a impressão digital
        # **idêntica**, e a retomada (RH2) aceita registros da rodada anterior
        # como válidos — duas rodadas na mesma tabela, sem rastro. Comprovado
        # em 05/08/2026 antes de existir esta defesa. Ver §11.
        "features": features.fingerprint_valores(strip_scheme),
        "particao": splits.fingerprint_particao(),
        # Frente A (PSL-03, SENS-01, MLP-01, DEDUP-01): a lista de sufixos
        # congelada, a opção sem esquema, a validação interna agrupada do MLP
        # no Regime B e a dedup pelo url_norm determinam o resultado e não
        # aparecem no `experiment.yml`.
        "psl": psl.sha256(),
        "strip_scheme": strip_scheme,
        "mlp_validacao": "agrupada_etld1_v1",
        "dedup": "url_norm_v1",
    }
    canonico = json.dumps(relevante, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Livro-caixa (RH2)
# ---------------------------------------------------------------------------


def read_ledger(
    path: Path,
    fingerprint: str,
    n_test_corpora: int,
    dir_rodada: Path | None = None,
    log=None,
) -> tuple[list[dict], set, int]:
    """Lê o livro-caixa: devolve (registros válidos, ajustes completos, nº truncadas).

    Levanta `MatrixError` se houver registros de outra configuração: reaproveitá-los
    produziria uma tabela que compara modelos ajustados com hiperparâmetros
    diferentes.

    Linhas truncadas (queda no meio da escrita) são descartadas com aviso — o
    ajuste correspondente é simplesmente refeito.

    Com `dir_rodada` (PRED-02), um ajuste completo cujo parquet de predições
    falta, foi alterado ou nunca foi registrado sai de `completos` e é refeito;
    `log` recebe uma linha por ajuste. Se um ajuste foi refeito, o livro-caixa
    guarda os registros velhos e os novos: vale o último de cada célula.
    """
    if not path.exists():
        return [], set(), 0

    registros: list[dict] = []
    estranhos: set[str] = set()
    truncadas = 0
    for linha in path.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            rec = json.loads(linha)
        except json.JSONDecodeError:
            truncadas += 1
            continue
        if rec.get("config_fingerprint") != fingerprint:
            estranhos.add(str(rec.get("config_fingerprint")))
            continue
        registros.append(rec)

    if estranhos:
        raise MatrixError(
            f"o livro-caixa {path.name} contém registros de outra configuração "
            f"(impressões digitais {sorted(estranhos)}; a atual é {fingerprint}).\n\n"
            f"Alguma coisa que afeta o resultado mudou depois que esses "
            f"registros foram gerados: os hiperparâmetros da experiment.yml, a "
            f"lógica de extração de features, ou o particionador. As três "
            f"entram na impressão digital desde 05/08/2026 — as duas últimas "
            f"por canária de comportamento, porque não aparecem no config.\n\n"
            f"Misturar as duas rodadas na mesma tabela é o que o SDD §4 D5 "
            f"proíbe, e não deixaria rastro visível na tabela final.\n\n"
            f"Se você está na rodada 2 (features da T3.4 corrigidas e split "
            f"estratificado), isto é ESPERADO: o livro-caixa é da rodada 1.\n\n"
            f"Escolha uma:\n"
            f"  (a) reverter a mudança e retomar; ou\n"
            f"  (b) arquivar e começar limpo:\n"
            f"      venv\\Scripts\\python.exe scripts\\run_all.py --recomecar"
        )

    # Vale o último registro de cada célula (um ajuste refeito reanexa os seus).
    ultimos: dict = {}
    for rec in registros:
        ultimos[(rec["regime"], rec["train_corpus"], rec["test_corpus"], rec["model"])] = rec
    registros = list(ultimos.values())

    por_ajuste: dict = {}
    for rec in registros:
        chave = (rec["regime"], rec["train_corpus"], rec["model"])
        por_ajuste.setdefault(chave, set()).add(rec["test_corpus"])
    completos = {k for k, v in por_ajuste.items() if len(v) >= n_test_corpora}

    if dir_rodada is not None:
        refs: dict = {}
        for rec in registros:
            chave = (rec["regime"], rec["train_corpus"], rec["model"])
            if chave in completos:
                refs.setdefault(chave, []).append(rec.get("predicoes"))
        for chave in sorted(refs):
            if not all(predicoes.conferir(dir_rodada, ref) for ref in refs[chave]):
                completos.discard(chave)
                if log:
                    log(f"  predições de {'/'.join(chave)} ausentes ou alteradas, refazendo")

    # Descarta registros de ajustes incompletos: eles serão refeitos, e manter
    # os parciais criaria duplicata no consolidado.
    registros = [
        r for r in registros
        if (r["regime"], r["train_corpus"], r["model"]) in completos
    ]
    return registros, completos, truncadas


def append_ledger(path: Path, novos: list[dict]) -> None:
    """Anexa e força a descarga no disco.

    Sem o `fsync`, uma queda de energia perderia ajustes que o buffer do sistema
    ainda não escreveu — e a retomada os refaria, o que é correto mas custa o
    tempo do ajuste de novo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in novos:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Partições
# ---------------------------------------------------------------------------


# Segundos acumulados dentro de `build_splits`, para a decomposição por fase.
#
# Existe por causa da T4.11 (§13.9): o portão rotulava como "matriz" um bloco
# que inclui o particionamento, e medido são **27,4 s dos 60,3 s** dessa fase —
# o maior item isolado da execução, acima da soma dos 96 ajustes. Com o rótulo
# antigo, a mensagem de diagnóstico mandava procurar em "preparar features",
# que é a menor das fases.
#
# ⚠️ É contador de diagnóstico e nada mais: não entra em métrica, em registro,
# em livro-caixa nem em impressão digital. Acumula entre chamadas de propósito
# (`run_matrix` e `run_loco` particionam cada um a sua vez); quem quiser medir
# uma execução isolada chama `zerar_tempo_particionamento()` antes.
_TEMPO_PARTICIONAMENTO_S = 0.0


def tempo_particionamento_s() -> float:
    """Segundos acumulados em `build_splits` desde o último zeramento."""
    return _TEMPO_PARTICIONAMENTO_S


def zerar_tempo_particionamento() -> None:
    """Zera o acumulador de `tempo_particionamento_s()`."""
    global _TEMPO_PARTICIONAMENTO_S
    _TEMPO_PARTICIONAMENTO_S = 0.0


def build_splits(config: dict, frames: dict, log=None) -> tuple[dict, dict]:
    """Particiona cada corpus nos dois regimes, uma única vez.

    Devolve `(indices, resumos)`. Calcular aqui e devolver os dois evita
    reparticionar 800 mil linhas só para gerar o relatório — foi um desperdício
    real na primeira versão deste código.
    """
    global _TEMPO_PARTICIONAMENTO_S
    _t_particao = time.perf_counter()
    try:
        seed = int(config["seed"])
        indices: dict = {}
        resumos: dict = {}
        for regime in config["regimes"]:
            for nome in frames:
                frame = frames[nome]
                val_size = (
                    config["split"]["val_size"] if regime == "etld1"
                    else config["split"]["val_size_oficial"]
                )
                # Dedup pelo url_norm e partição numa fachada só (AD-057); as
                # posições devolvidas indexam o quadro original.
                try:
                    split, contagens = splits.particionar_corpus(
                        frame, regime, corpus=nome, seed=seed,
                        test_size=config["split"]["test_size"], val_size=val_size,
                    )
                except splits.SplitError as exc:
                    raise MatrixError(
                        f"split falhou em {nome}/{regime}: {exc}"
                    ) from exc

                # O Regime B promete sobreposição zero (DoD da T2.4). Conferir na
                # hora custa segundos e é a diferença entre descobrir agora ou no
                # artigo.
                if regime == "etld1":
                    splits.assert_no_domain_overlap(frame, split)

                indices[(nome, regime)] = split
                resumo = splits.split_summary(frame, split)
                resumo["dedup"] = contagens
                resumos[f"{nome}/{regime}"] = resumo
                if log:
                    vaz = resumo["domain_overlap"]["train_test"]
                    log(f"  {nome:14s} {regime:8s} treino={len(split.train):7d} "
                        f"val={len(split.val):6d} teste={len(split.test):7d}  "
                        f"domínios compartilhados treino/teste={vaz}  "
                        f"dedup: -{contagens['n_duplicatas_removidas']} duplicatas, "
                        f"-{contagens['n_conflito_linhas']} em conflito")
        return indices, resumos
    finally:
        _TEMPO_PARTICIONAMENTO_S += time.perf_counter() - _t_particao


def contagens_dedup(resumos: dict) -> dict:
    """Contagens de dedup por `(dataset, regime)`, tiradas dos resumos de `build_splits`."""
    saida = {}
    for chave, resumo in resumos.items():
        corpus, regime = chave.split("/")
        saida[(corpus, regime)] = resumo["dedup"]
    return saida


def _linhas_do_quadro(frame, nome: str, exigir: bool) -> np.ndarray:
    """Coluna `linha` (posição no parquet do dataset) ou, sem persistência, a posição."""
    if "linha" in frame.columns:
        return frame["linha"].to_numpy()
    if exigir:
        raise MatrixError(
            f"o quadro de '{nome}' não tem a coluna 'linha' (posição da linha no "
            f"parquet do dataset), necessária para gravar as predições."
        )
    return np.arange(len(frame))


# ---------------------------------------------------------------------------
# O laço
# ---------------------------------------------------------------------------


def _executar_ajuste(
    *,
    nome_modelo: str,
    config: dict,
    gpu,
    X_tr,
    y_tr,
    X_val,
    y_val,
    testes: list[tuple],
    regime: str,
    rotulo_treino: str,
    run_id: str,
    fingerprint: str,
    criterio: str,
    ece_bins: int,
    extra_por_registro: dict | None,
    seed: int = 42,
    bootstrap_n: int = 0,
    bootstrap_nivel: float = evaluate.DEFAULT_BOOTSTRAP_NIVEL,
    grupos_tr=None,
    dir_rodada: Path | None = None,
    tipo: str = "matriz",
) -> tuple[list[dict], float, float, list[str]]:
    """Um ajuste: treina, escolhe o limiar e avalia em cada teste de `testes`.

    Este é **o** corpo do experimento, e existe como função única porque tanto
    a matriz 4x4 (`run_matrix`) quanto o leave-one-corpus-out (`run_loco`)
    precisam dele. Duplicá-lo reintroduziria o RK5 pela porta dos fundos: o
    smoke validaria um caminho e o Lucas rodaria outro.

    `testes` é uma lista de `(nome, X_teste, y_teste, linhas)`, em que `linhas`
    são as posições das linhas de teste no parquet do *dataset* de teste.

    `grupos_tr` (eTLD+1 de cada linha de `X_tr`, na mesma ordem) só é passado no
    Regime B: o MLP o usa para a validação interna agrupada (MLP-01). No
    Regime A o MLP é o nativo e a validação interna é por linha (MLP-02).

    Com `dir_rodada`, grava as predições do ajuste (PRED-01) em
    `predicoes/<tipo>/` e acrescenta `predicoes` (arquivo e sha256) a cada
    registro (PRED-02).

    `bootstrap_n = 0` desliga o IC — é o que o smoke usa, porque ali o número
    é descartável e o portão tem orçamento de tempo (RF9).
    """
    agrupada = regime == "etld1"
    modelo = models.build_model(
        nome_modelo, config, gpu=gpu, validacao_agrupada=agrupada
    )
    t0 = time.perf_counter()
    models.ajustar(modelo, nome_modelo, X_tr, y_tr, grupos=grupos_tr if agrupada else None)
    fit_seconds = time.perf_counter() - t0

    # Limiar SEMPRE na validação do corpus de TREINO (SDD §4 D6), e
    # transportado como está para todos os corpora de teste. A degradação
    # desse transporte é resultado do artigo.
    p_val = models.positive_class_proba(modelo, X_val)
    limiar = evaluate.select_threshold(y_val, p_val, criterion=criterio)

    # Importância de feature é propriedade do AJUSTE, não da célula de teste:
    # calculada uma vez e repetida nos N registros do ajuste. São 41 floats, e
    # o `run_all.gravar()` desduplica ao separar em `results/importances.json`.
    importancia = models.feature_importance(modelo, nome_modelo)

    novos: list[dict] = []
    resumo: list[str] = []
    blocos: list[tuple] = []
    proveniencia = models.model_provenance(
        nome_modelo, config, gpu=gpu, validacao_agrupada=agrupada
    )
    for nome_teste, X_te, y_te, linhas_te in testes:
        p_te = models.positive_class_proba(modelo, X_te)
        blocos.append((nome_teste, linhas_te, y_te, p_te))
        metrics, avisos = evaluate.evaluate_predictions(
            y_te, p_te, threshold=limiar, ece_bins=ece_bins
        )
        extra = {
            "config_fingerprint": fingerprint,
            "provenance": proveniencia,
            # T4.6 — pontos da F1. Ficam **dentro** do registro, e portanto
            # dentro do livro-caixa, para que a retomada (RH2) não devolva uma
            # execução com metade das curvas. O `run_all.gravar()` os separa em
            # `results/curves.json` antes de escrever a matriz, senão o
            # `matrix.json` que o artigo lê inflaria de ~250 KB para ~3 MB.
            "curves": evaluate.curve_points(y_te, p_te),
            "importances": importancia,
        }
        if bootstrap_n:
            # A semente do IC é fixa e **não** depende da ordem de execução:
            # a retomada (RH2) executa os ajustes em ordem diferente e as duas
            # execuções têm de dar exatamente o mesmo intervalo (RNF1).
            extra["ci"] = evaluate.bootstrap_ci(
                y_te, p_te,
                n_resamples=bootstrap_n,
                seed=seed,
                nivel=bootstrap_nivel,
            )
        if extra_por_registro:
            extra.update(extra_por_registro)
        novos.append(
            evaluate.make_result_record(
                run_id=run_id, regime=regime,
                train_corpus=rotulo_treino, test_corpus=nome_teste,
                model=nome_modelo, metrics=metrics,
                fit_seconds=fit_seconds, warnings=avisos, extra=extra,
            )
        )
        auroc = metrics["auroc"]
        resumo.append(
            f"{nome_teste[:5]}={auroc:.3f}" if auroc is not None
            else f"{nome_teste[:5]}=n/d"
        )

    if dir_rodada is not None:
        ref = predicoes.gravar_predicoes(
            dir_rodada, tipo, regime, rotulo_treino, nome_modelo, blocos
        )
        for rec in novos:
            rec["predicoes"] = dict(ref)
    return novos, fit_seconds, limiar, resumo


def run_matrix(
    config: dict,
    frames: dict,
    gpu,
    *,
    run_id: str,
    fingerprint: str | None = None,
    ledger_path: Path | None = None,
    log=None,
    extra_por_registro: dict | None = None,
    dir_rodada: Path | None = None,
) -> MatrixRun:
    """Executa a matriz cruzada completa: regimes × treino × modelos × teste.

    Args:
        config: `experiment.yml` já carregado (com ou sem os overrides do smoke).
        frames: `{nome_do_corpus: DataFrame com features e etld_plus_one}`.
        gpu: `models.GpuInfo` da detecção.
        run_id: carimbo de proveniência (RNF4).
        fingerprint: impressão digital da config; calculada se omitida.
        ledger_path: se dado, habilita retomada (RH2). `None` = execução limpa,
            sem livro-caixa — é o que o smoke usa, porque um portão deve sempre
            rodar do zero.
        log: função de progresso (RH3).
        extra_por_registro: campos extras a anexar a cada registro.
        dir_rodada: diretório da rodada. Se dado, cada ajuste grava suas
            predições em `predicoes/matriz/` (PRED-01), o registro leva o
            caminho e o sha256 (PRED-02) e a retomada refaz o ajuste cujo
            parquet falta ou foi alterado. Os quadros precisam da coluna
            `linha`. `None` = sem persistência de predições.

    Ordem dos corpora: a de `frames`. Quem chama decide (o `run_all.py` ordena do
    menor para o maior, mitigando o RK1).
    """
    log = log or (lambda _msg="": None)
    fingerprint = fingerprint or config_fingerprint(config)
    criterio = config["avaliacao"]["criterio_limiar"]
    ece_bins = int(config["avaliacao"]["ece_bins"])
    seed = int(config["seed"])
    bootstrap_n = int(config["avaliacao"].get("bootstrap_n", 0))
    bootstrap_nivel = float(
        config["avaliacao"].get("bootstrap_nivel", evaluate.DEFAULT_BOOTSTRAP_NIVEL)
    )
    nomes = list(frames)
    n_teste = len(nomes)

    resultado = MatrixRun()

    if dir_rodada is not None:
        for nome in nomes:
            _linhas_do_quadro(frames[nome], nome, exigir=True)

    if ledger_path is not None:
        registros, completos, truncadas = read_ledger(
            ledger_path, fingerprint, n_teste, dir_rodada=dir_rodada, log=log
        )
        resultado.records.extend(registros)
        resultado.n_fits_pulados = len(completos)
        if truncadas:
            log(f"  aviso: {truncadas} linha(s) truncada(s) no livro-caixa, descartada(s)")
        if completos:
            log(f"retomando: {len(completos)} ajuste(s) já registrado(s) serão pulados (RH2)")
    else:
        completos = set()

    log()
    log("particionando...")
    indices, resumos = build_splits(config, frames, log=log)
    resultado.split_summaries = resumos
    resultado.indices = indices

    # float32 basta para as 41 features (uint16 e ratios em [0,1]) e corta a
    # memória pela metade em relação ao float64 que o pandas usaria.
    matrizes = {
        nome: (
            frames[nome][list(features.FEATURE_NAMES)].to_numpy(dtype=np.float32),
            frames[nome]["label"].to_numpy(),
        )
        for nome in nomes
    }
    linhas_por_corpus = {
        nome: _linhas_do_quadro(frames[nome], nome, exigir=False) for nome in nomes
    }
    # Chave de grupo (eTLD+1 ou reserva) de cada linha, só para o MLP do Regime B.
    grupos_por_corpus = {
        nome: splits.group_keys(frames[nome]).to_numpy() for nome in nomes
    }

    total = len(config["regimes"]) * len(nomes) * len(models.MODEL_ORDER)
    feitos = resultado.n_fits_pulados

    for regime in config["regimes"]:
        log()
        log(f"=== REGIME '{regime}' " + "=" * 44)
        for treino in nomes:
            split_tr = indices[(treino, regime)]
            X, y = matrizes[treino]
            X_tr, y_tr = X[split_tr.train], y[split_tr.train]
            X_val, y_val = X[split_tr.val], y[split_tr.val]
            grupos_tr = (
                grupos_por_corpus[treino][split_tr.train] if regime == "etld1" else None
            )

            if len(np.unique(y_tr)) < 2:
                raise MatrixError(
                    f"treino de {treino}/{regime} ficou com uma única classe "
                    f"({len(y_tr)} linhas)"
                )

            for nome_modelo in models.MODEL_ORDER:
                if (regime, treino, nome_modelo) in completos:
                    continue

                testes = [
                    (
                        teste,
                        matrizes[teste][0][indices[(teste, regime)].test],
                        matrizes[teste][1][indices[(teste, regime)].test],
                        linhas_por_corpus[teste][indices[(teste, regime)].test],
                    )
                    for teste in nomes
                ]
                novos, fit_seconds, limiar, resumo_linha = _executar_ajuste(
                    nome_modelo=nome_modelo, config=config, gpu=gpu,
                    X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
                    testes=testes, regime=regime, rotulo_treino=treino,
                    run_id=run_id, fingerprint=fingerprint,
                    criterio=criterio, ece_bins=ece_bins,
                    seed=seed, bootstrap_n=bootstrap_n,
                    bootstrap_nivel=bootstrap_nivel,
                    extra_por_registro=extra_por_registro,
                    grupos_tr=grupos_tr, dir_rodada=dir_rodada, tipo="matriz",
                )
                resultado.fit_seconds.append(fit_seconds)

                # Um ajuste é atômico no livro-caixa: grava só com os N testes
                # prontos, senão a retomada veria um ajuste pela metade.
                if ledger_path is not None:
                    append_ledger(ledger_path, novos)
                resultado.records.extend(novos)
                resultado.n_fits_executados += 1
                feitos += 1

                eta = ""
                restam = total - feitos
                if resultado.fit_seconds and restam > 0:
                    medio = sum(resultado.fit_seconds) / len(resultado.fit_seconds)
                    eta = f"  ETA ~{medio * restam / 60:.0f}min"
                log(f"  [{feitos:2d}/{total}] {treino:14s} {nome_modelo:20s} "
                    f"ajuste={fit_seconds:6.2f}s lim={limiar:.3f}  AUROC "
                    + " ".join(resumo_linha) + eta)

    return resultado


# ---------------------------------------------------------------------------
# Leave-one-corpus-out (D3-b, 05/08/2026)
# ---------------------------------------------------------------------------

# Prefixo do rótulo de treino nos registros do LOCO. Distingue-os dos registros
# da matriz sem precisar de outro campo, e garante que `train_corpus` nunca é
# igual a `test_corpus` — logo `diagonal` sai False, que é o correto.
LOCO_PREFIXO = "uniao_sem_"


def run_loco(
    config: dict,
    frames: dict,
    gpu,
    run_id: str,
    fingerprint: str | None = None,
    ledger_path: Path | None = None,
    log=None,
    extra_por_registro: dict | None = None,
    dir_rodada: Path | None = None,
) -> MatrixRun:
    """Treina na união de N-1 corpora e testa no que ficou de fora (SDD D3-b).

    `dir_rodada` tem o mesmo papel que em `run_matrix`; as predições vão para
    `predicoes/loco/`. No Regime B o MLP recebe os grupos das três fontes da
    união, concatenados na ordem de `X_tr`.

    Responde à objeção mais previsível à matriz cruzada: *"e se treinar em
    várias fontes, resolve?"*. 4 combinações x 2 regimes x 6 modelos = 48
    ajustes, cada um com **uma** avaliação (o corpus retido).

    Duas decisões que valem registro:

    1. O teste é a **partição de teste** do corpus retido, não o corpus
       inteiro. O corpus retido nunca entra no treino, então usar tudo seria
       legítimo — mas usar a mesma partição da matriz torna T6 diretamente
       comparável com a fora-diagonal de T2. Comparar contra um conjunto de
       teste diferente esconderia a única coisa que a tabela existe para
       mostrar: se a união melhora ou não em relação a treinar num corpus só.
    2. Os hiperparâmetros continuam os do D5, sem reajuste. Reajustar aqui
       compararia "1 fonte com parâmetros do D5" contra "3 fontes com
       parâmetros ajustados", e a diferença deixaria de ser atribuível à união.

    O laço de ajuste é o mesmo `_executar_ajuste` que a matriz usa — não há
    caminho de código exclusivo do LOCO que o smoke não exercite (RK5).
    """
    log = log or (lambda _msg="": None)
    fingerprint = fingerprint or config_fingerprint(config)
    criterio = config["avaliacao"]["criterio_limiar"]
    ece_bins = int(config["avaliacao"]["ece_bins"])
    seed = int(config["seed"])
    bootstrap_n = int(config["avaliacao"].get("bootstrap_n", 0))
    bootstrap_nivel = float(
        config["avaliacao"].get("bootstrap_nivel", evaluate.DEFAULT_BOOTSTRAP_NIVEL)
    )
    nomes = list(frames)

    if len(nomes) < 2:
        raise MatrixError(
            f"leave-one-corpus-out exige ao menos 2 corpora; recebi {nomes}"
        )

    resultado = MatrixRun()

    if dir_rodada is not None:
        for nome in nomes:
            _linhas_do_quadro(frames[nome], nome, exigir=True)

    if ledger_path is not None:
        registros, completos, truncadas = read_ledger(
            ledger_path, fingerprint, 1, dir_rodada=dir_rodada, log=log
        )
        resultado.records.extend(registros)
        resultado.n_fits_pulados = len(completos)
        if truncadas:
            log(f"  aviso: {truncadas} linha(s) truncada(s) no livro-caixa, descartada(s)")
        if completos:
            log(f"retomando LOCO: {len(completos)} ajuste(s) pulado(s) (RH2)")
    else:
        completos = set()

    log()
    log("particionando (LOCO)...")
    indices, resumos = build_splits(config, frames, log=log)
    resultado.split_summaries = resumos
    resultado.indices = indices

    matrizes = {
        nome: (
            frames[nome][list(features.FEATURE_NAMES)].to_numpy(dtype=np.float32),
            frames[nome]["label"].to_numpy(),
        )
        for nome in nomes
    }
    linhas_por_corpus = {
        nome: _linhas_do_quadro(frames[nome], nome, exigir=False) for nome in nomes
    }
    grupos_por_corpus = {
        nome: splits.group_keys(frames[nome]).to_numpy() for nome in nomes
    }

    total = len(config["regimes"]) * len(nomes) * len(models.MODEL_ORDER)
    feitos = resultado.n_fits_pulados

    for regime in config["regimes"]:
        log()
        log(f"=== LOCO, REGIME '{regime}' " + "=" * 38)
        for retido in nomes:
            outros = [n for n in nomes if n != retido]
            rotulo = f"{LOCO_PREFIXO}{retido}"

            X_tr = np.concatenate(
                [matrizes[n][0][indices[(n, regime)].train] for n in outros]
            )
            y_tr = np.concatenate(
                [matrizes[n][1][indices[(n, regime)].train] for n in outros]
            )
            X_val = np.concatenate(
                [matrizes[n][0][indices[(n, regime)].val] for n in outros]
            )
            y_val = np.concatenate(
                [matrizes[n][1][indices[(n, regime)].val] for n in outros]
            )
            grupos_tr = (
                np.concatenate(
                    [grupos_por_corpus[n][indices[(n, regime)].train] for n in outros]
                )
                if regime == "etld1" else None
            )

            if len(np.unique(y_tr)) < 2:
                raise MatrixError(
                    f"treino da uniao sem '{retido}' ({regime}) ficou com uma "
                    f"unica classe ({len(y_tr)} linhas)"
                )
            if len(y_val) == 0:
                raise MatrixError(
                    f"validacao da uniao sem '{retido}' ({regime}) ficou vazia — "
                    f"o limiar do D6 sai dela e nao pode ser inventado"
                )

            split_te = indices[(retido, regime)]
            testes = [(
                retido,
                matrizes[retido][0][split_te.test],
                matrizes[retido][1][split_te.test],
                linhas_por_corpus[retido][split_te.test],
            )]

            log(f"  uniao sem {retido}: treino={len(y_tr)} val={len(y_val)} "
                f"teste={len(split_te.test)}")

            for nome_modelo in models.MODEL_ORDER:
                if (regime, rotulo, nome_modelo) in completos:
                    continue

                novos, fit_seconds, limiar, resumo = _executar_ajuste(
                    nome_modelo=nome_modelo, config=config, gpu=gpu,
                    X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
                    testes=testes, regime=regime, rotulo_treino=rotulo,
                    run_id=run_id, fingerprint=fingerprint,
                    criterio=criterio, ece_bins=ece_bins,
                    seed=seed, bootstrap_n=bootstrap_n,
                    bootstrap_nivel=bootstrap_nivel,
                    extra_por_registro=extra_por_registro,
                    grupos_tr=grupos_tr, dir_rodada=dir_rodada, tipo="loco",
                )
                resultado.fit_seconds.append(fit_seconds)
                if ledger_path is not None:
                    append_ledger(ledger_path, novos)
                resultado.records.extend(novos)
                resultado.n_fits_executados += 1
                feitos += 1

                log(f"  [{feitos:2d}/{total}] {rotulo:22s} {nome_modelo:20s} "
                    f"ajuste={fit_seconds:6.2f}s lim={limiar:.3f}  AUROC "
                    + " ".join(resumo))

    return resultado


# ---------------------------------------------------------------------------
# Conferência e consolidação
# ---------------------------------------------------------------------------


def check_records(records: list[dict], config: dict, n_corpora: int) -> None:
    """Confere completude, ausência de duplicata e faixa de toda métrica."""
    esperado = len(config["regimes"]) * n_corpora * len(config["modelos"]) * n_corpora
    if len(records) != esperado:
        raise MatrixError(
            f"esperados {esperado} registros, obtidos {len(records)}. "
            f"A execução é retomável: rode de novo para completar o que falta (RH2)."
        )

    vistos = set()
    for rec in records:
        evaluate.validate_metrics(rec["metrics"])
        chave = (rec["regime"], rec["train_corpus"], rec["test_corpus"], rec["model"])
        if chave in vistos:
            raise MatrixError(f"registro duplicado: {chave}")
        vistos.add(chave)

    if not any(r["diagonal"] for r in records):
        raise MatrixError("nenhuma célula na diagonal — laço errado")
    if all(r["diagonal"] for r in records) and n_corpora > 1:
        raise MatrixError("nenhuma célula fora da diagonal — laço errado")


def separar_anexos(registros: list[dict], out_dir: Path, prefixo: str = "") -> list[str]:
    """Tira dos registros o que é volumoso e grava em arquivo próprio.

    Curvas (T4.6) e importâncias de feature viajam **dentro** do registro para
    sobreviverem à retomada (RH2) — senão uma execução retomada devolveria
    metade das curvas. Mas não podem ficar no `matrix.json`/`matrix.csv`, que
    são os arquivos que o `report.py` e o artigo leem: seriam megabytes de
    números que nenhuma tabela consome, contra os ~250 KB das métricas.

    O IC por bootstrap **fica** no registro de propósito: são 6 números por
    célula e é exatamente o que as tabelas T2/T3/T6 precisam ler.

    Vive neste módulo, e não no `run_all.py`, pela mesma razão que o laço vive
    aqui: o portão tem de exercitar o mesmo código da execução real. A auditoria
    de 05/08/2026 encontrou a separação só no `run_all.py`, e por isso o
    `results/smoke/matrix.json` saía com as curvas dentro e o CSV com uma coluna
    `curves` — o portão não validava o escritor que produz a fonte da F1.
    """
    curvas: dict = {}
    importancias: dict = {}
    for rec in registros:
        pontos = rec.pop("curves", None)
        if pontos is not None:
            chave = (f"{rec['regime']}|{rec['train_corpus']}|"
                     f"{rec['test_corpus']}|{rec['model']}")
            # A grade é a mesma em toda célula — gravada uma vez, no topo.
            curvas[chave] = {k: v for k, v in pontos.items() if k != "grade"}
        # Importância é propriedade do AJUSTE, não da célula: vem repetida nos N
        # registros do mesmo ajuste e aqui é desduplicada.
        imp = rec.pop("importances", None)
        if imp is not None:
            importancias[f"{rec['regime']}|{rec['train_corpus']}|{rec['model']}"] = imp

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefixo}curves.json").write_text(
        json.dumps({"grade": evaluate.GRADE_CURVA.tolist(), "celulas": curvas},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    escritos = [f"{prefixo}curves.json"]
    if importancias:
        (out_dir / f"{prefixo}importances.json").write_text(
            json.dumps(
                {"features": list(features.FEATURE_NAMES), "ajustes": importancias},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        escritos.append(f"{prefixo}importances.json")
    return escritos


def check_loco_records(records: list[dict], config: dict, n_corpora: int) -> None:
    """Mesma conferência do `check_records`, na forma do LOCO (D3-b).

    O LOCO tem **uma** avaliação por ajuste, não N, e o `train_corpus` é
    `uniao_sem_<corpus>` em vez de um corpus — logo a aritmética do
    `check_records` não serve e ele não podia ser reaproveitado.

    Existe porque a auditoria de 05/08/2026 achou o LOCO sem conferência alguma
    na execução real, enquanto o portão (`run_smoke.py`) já conferia: a execução
    que produz o número do artigo era a menos rigorosa das duas. Um ajuste
    faltante viraria uma T6 de 47 linhas sem ninguém reclamar.

    A faixa de cada métrica já é validada por `evaluate.evaluate_predictions`,
    que chama `validate_metrics` no momento do cálculo — a defesa do B1 alcança
    o LOCO. O que faltava era completude e duplicata.
    """
    esperado = len(config["regimes"]) * n_corpora * len(config["modelos"])
    if len(records) != esperado:
        raise MatrixError(
            f"esperadas {esperado} avaliacoes no LOCO "
            f"({len(config['regimes'])} regimes x {n_corpora} corpora retidos x "
            f"{len(config['modelos'])} modelos), obtidas {len(records)}. "
            f"A execucao e retomavel: rode de novo para completar o que falta (RH2)."
        )

    vistos = set()
    for rec in records:
        evaluate.validate_metrics(rec["metrics"])
        chave = (rec["regime"], rec["train_corpus"], rec["model"])
        if chave in vistos:
            raise MatrixError(f"registro do LOCO duplicado: {chave}")
        vistos.add(chave)
        if not rec["train_corpus"].startswith(LOCO_PREFIXO):
            raise MatrixError(
                f"registro do LOCO com train_corpus inesperado: "
                f"{rec['train_corpus']!r} (deveria comecar com {LOCO_PREFIXO!r})"
            )
        if rec["diagonal"]:
            raise MatrixError(
                f"registro do LOCO marcado como diagonal: {chave}. O corpus "
                f"retido nunca entra no treino, entao nenhuma celula do LOCO "
                f"pode ser diagonal — se uma for, a uniao esta vazando."
            )


def sort_records(records: list[dict]) -> list[dict]:
    """Ordem canônica, independente da ordem de execução.

    Necessário porque uma retomada executa os ajustes em ordem diferente; sem
    isso, dois resultados idênticos apareceriam como mudança no `git diff`.
    """
    ordem_modelo = {m: i for i, m in enumerate(models.MODEL_ORDER)}
    return sorted(
        records,
        key=lambda r: (r["regime"], r["train_corpus"], r["test_corpus"],
                       ordem_modelo.get(r["model"], 99)),
    )


__all__ = [
    "MatrixError",
    "MatrixRun",
    "config_fingerprint",
    "read_ledger",
    "append_ledger",
    "build_splits",
    "contagens_dedup",
    "tempo_particionamento_s",
    "zerar_tempo_particionamento",
    "run_matrix",
    "run_loco",
    "LOCO_PREFIXO",
    "check_records",
    "check_loco_records",
    "sort_records",
]
