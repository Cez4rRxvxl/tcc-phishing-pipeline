"""T5.2 — a ponte entre `results/` e o PDF: métricas viram `latex/tabelas/*.tex`.

Por que este módulo existe
--------------------------
A R5 do SDD proíbe número digitado à mão no artigo: **todo** valor impresso tem
de sair de `results/`. Sem este módulo o artigo teria de ser preenchido a dedo a
partir dos JSON, e a primeira vez que uma rodada mudasse um número o texto
ficaria em silêncio desatualizado — que é a forma mais barata de publicar um
resultado errado.

O `artigo.tex` (hoje em `docs/artigo/`) já carregava as tabelas por `\\input`, protegidas por
`\\IfFileExists`: enquanto o arquivo não existe, aparece um aviso emoldurado no
PDF em vez de um número inventado. Os nomes de arquivo são contrato, fixados na
contrato com o `artigo.tex`, e mudá-los quebra o artigo em silêncio.

Convenções de formatação, e por que são estas
----------------------------------------------
- **Vírgula decimal.** O artigo é em português (CLAUDE.md) e a convenção é a
  da ABNT. Números saem de `_num()`, nunca de f-string solta.
- **`booktabs`.** O `artigo.tex` carrega o pacote e as tabelas usam
  `\\toprule`/`\\midrule`/`\\bottomrule`. Sem `\\hline` e sem linha vertical.
- **Agregação por célula = média dos 6 modelos.** Isto **não** é escolha deste
  módulo: é a convenção que reproduz, em 4 casas decimais, os 0,9741 e 0,6856
  que a §13.8 já publicava antes de o módulo existir. Ver §11, 06/08/2026.
- **A T2 não traz IC.** O bootstrap reamostra o teste de *um* modelo, e a média
  de seis intervalos não é o intervalo da média. A T2 traz a faixa entre as 3
  seeds, que é bem definida para qualquer estatística. O IC segue reportado
  onde é válido. Decisão do Julio em 06/08/2026, §11 e §7.4.

O que este módulo NÃO faz
--------------------------
Não calcula métrica nenhuma. Ele lê, agrega e formata. Qualquer conta que
mereça teste vive em `evaluate.py` ou `diagnostics.py` — aqui só entra
aritmética que um revisor confere de cabeça (média, mínimo, máximo, diferença).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# `src/` no path, mesma convenção do `diagnostics.py`: funciona tanto em
# `python -m src.report` quanto em `import report` (pytest com `src` no path).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import analises  # noqa: E402
import estatistica  # noqa: E402

# REPO_ROOT é a raiz do pipeline (`pipeline/`). Tabelas e figuras vão para o
# `latex/` da monografia, na raiz do repositório do TCC II (TCC_ROOT).
REPO_ROOT = Path(__file__).resolve().parent.parent
TCC_ROOT = REPO_ROOT.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_TABLES_DIR = TCC_ROOT / "latex" / "tabelas"

# As seeds da rodada 2 (§11, 05/08/2026). A 42 é a canônica: o resultado do
# artigo é o dela, e as outras duas entram como FAIXA. Escolher a melhor seria
# selecionar resultado — a mesma classe de erro que o D5 proíbe.
SEED_CANONICA = 42
SUBDIRS_SEEDS = {42: "", 7: "seed_07", 13: "seed_13"}

# Rodadas adicionais da reexecução consolidada (AD-055, AD-057): a sensibilidade
# sem esquema repete a estrutura da principal (semente 42 na base e `seed_NN/`
# dentro), e a repetição é só a semente 42.
SUBDIR_SENSIBILIDADE = "sensibilidade_sem_esquema"
SUBDIR_REPETICAO = "repeticao_seed_42"

# Ordem de apresentação dos corpora nas tabelas da matriz. Fixa e explícita
# para que duas execuções gerem o mesmo `.tex` (RNF1) e para que a T2, a T3 e a
# T6 possam ser lidas em paralelo sem o leitor reconferir a ordem das colunas.
ORDEM_CORPORA = ("grambeddings", "kaggle03", "phishstorm", "iscx2016")

# Nomes como o artigo os cita. O JSON usa a chave curta do pipeline; a tabela
# usa a grafia dos autores originais, que é a que o leitor vai procurar na
# bibliografia.
ROTULO_CORPUS = {
    "grambeddings": "Grambeddings",
    "kaggle03": "Kaggle-03",
    "phishstorm": "PhishStorm",
    "iscx2016": "ISCX-URL2016",
    "pdrcnn": "PDRCNN",
    "sid321axn": "Kaggle-01",
}

# Abreviados de propósito: a T3 é por modelo (24 linhas x 7 colunas) e cada
# caractere aqui é largura de tabela. "Reg. logística" é inequívoco no contexto
# de uma tabela cujos outros cinco nomes são de biblioteca.
ROTULO_MODELO = {
    "logistic_regression": "Reg. logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
}


class ReportError(RuntimeError):
    """Falha ao montar uma tabela — sempre alto, nunca tabela pela metade."""


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------


def _num(valor, casas: int = 4) -> str:
    """Número no formato do artigo: vírgula decimal, ou travessão se ausente.

    `None` vira `--` (travessão LaTeX) e **não** 0. É a mesma doutrina do
    `evaluate.py`: AUROC indefinido reportado como zero criaria um número falso
    que atravessaria a tabela até o PDF, e foi comparar valor inválido que
    produziu os bugs B2 e B7.
    """
    if valor is None:
        return "--"
    texto = f"{valor:.{casas}f}"
    # Negativo que arredonda a zero sai sem sinal: `-0,0000` na T3 por modelo
    # leria como queda onde não há nenhuma nas casas impressas.
    if texto.startswith("-") and not texto.strip("-0."):
        texto = texto[1:]
    return texto.replace(".", ",")


def _pct(valor, casas: int = 2) -> str:
    return _num(valor, casas)


def _milhar(n: int) -> str:
    """Inteiro com separador de milhar do português: ponto.

    Existe como função porque a primeira versão aplicava `.replace(",", ".")`
    na **linha inteira** da tabela, o que trocava também a vírgula decimal que
    o `_num()` acabara de produzir: `78,51` virava `78.51` no PDF. Formatar
    cada número na sua função impede que uma substituição pegue a outra.
    """
    return f"{int(n):,}".replace(",", ".")


def _escapar(texto: str) -> str:
    """Escapa o que o LaTeX interpretaria. Nome de corpus pode ter `_`."""
    for de, para in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                     ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        texto = texto.replace(de, para)
    return texto


def _rotulo(nome: str, mapa: dict) -> str:
    return _escapar(mapa.get(nome, nome))


def _faixa(valores, casas: int = 4, sep: str = "--") -> str:
    """`[mín--máx]` de uma faixa entre seeds; vazio quando há uma seed só.

    ⚠️ `casas` tem de ser o **mesmo** do valor pontual que a faixa acompanha.
    Com 4 casas no ponto e 3 na faixa, o arredondamento põe o ponto **fora** do
    próprio intervalo — medido no primeiro PDF gerado: `0,0028 [0,003--0,004]`
    e `-0,0071 [-0,007--0,006]`. Um revisor lê isso como inconsistência, e com
    razão.

    `sep` existe porque o travessão duplo colide com o sinal de menos: a faixa
    `[-0,011--0,010]` é ilegível. Colunas que podem ter valor negativo passam
    `sep=";~"`; as que são sempre positivas ficam com o travessão, que é a
    convenção tipográfica de intervalo.
    """
    limpos = [v for v in valores if v is not None]
    if len(limpos) < 2:
        return ""
    lo, hi = min(limpos), max(limpos)
    return f"[{_num(lo, casas)}{sep}{_num(hi, casas)}]"


def _celula_media(valores, casas: int = 4, sep: str = "--") -> str:
    r"""Média das sementes com a faixa embaixo, em `\tiny`; a média só, com uma semente.

    ANL-03: as tabelas principais trazem, ao lado da semente 42, a média das
    sementes e a faixa entre elas. A faixa vai numa segunda linha da célula
    (`\shortstack`) para não alargar a coluna: com as duas na mesma linha as
    tabelas de 12 e 13 colunas não cabem na largura do texto. Valor ausente
    sai da média; sem nenhum valor a célula é `--`, como em `_num`.
    """
    limpos = [v for v in valores if v is not None]
    media = _num(_media(limpos), casas) if limpos else _num(None)
    faixa = _faixa(limpos, casas, sep)
    if not faixa:
        return media
    return rf"\shortstack[r]{{{media}\\{{\tiny {faixa}}}}}"


def _min_a_max(valores, casas: int = 4) -> str:
    """`mín a máx` de uma faixa entre **modelos**. Sem colchete, de propósito.

    No artigo o colchete tem um significado só, declarado uma vez no parágrafo
    de convenções da Metodologia: *faixa entre as três repetições com partições
    sorteadas de forma independente*, que mede incerteza de **partição**. A T3
    encolhida (T5.9) precisa publicar uma faixa entre os seis **modelos**, que
    é outra coisa — usar colchete aqui faria o leitor treinado pela convenção
    ler incerteza de partição onde está escrito espalhamento entre modelos.

    Por isso a forma é `mín a máx`, e o cabeçalho da coluna diz `(6 mod.)`.
    """
    limpos = [v for v in valores if v is not None]
    if not limpos:
        return "--"
    if len(limpos) == 1:
        return _num(limpos[0], casas)
    return f"{_num(min(limpos), casas)} a {_num(max(limpos), casas)}"


def _cabecalho(origem: str) -> str:
    """Comentário de proveniência no topo de cada `.tex` gerado.

    Existe para que quem abrir o `.tex` saiba que editá-lo à mão é inútil (a
    próxima execução sobrescreve) e de qual arquivo de `results/` ele saiu —
    a rastreabilidade que a R5 exige.
    """
    return (
        "% GERADO POR src/report.py (T5.2). NAO EDITE A MAO.\n"
        f"% Fonte: {origem}\n"
        "% Regenerar: venv/Scripts/python.exe -m src.report\n"
    )


def _fonte(*arquivos: str) -> str:
    r"""`\fonte{...}` visível sob o float, com os arquivos de `results/` de origem.

    AD-039: o cabeçalho `% Fonte:` só se lê no `.tex`; a NBR 14724 pede a fonte
    impressa abaixo da tabela ou da figura. Os nomes vão em `\texttt` com o
    caminho a partir da raiz do repositório, que é o que o leitor encontra.
    """
    # `\allowbreak` depois de cada `/`: sem ele o caminho em `\texttt` não
    # quebra de linha e a fonte com três arquivos estoura a largura do texto.
    nomes = [
        r"\texttt{"
        + f"pipeline/results/{_escapar(a)}".replace("/", r"/\allowbreak{}")
        + "}"
        for a in arquivos
    ]
    lista = nomes[0] if len(nomes) == 1 else ", ".join(nomes[:-1]) + " e " + nomes[-1]
    return rf"\fonte{{Elaboração própria, a partir de {lista}}}"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def _ler_json(caminho: Path):
    if not caminho.exists():
        raise ReportError(
            f"arquivo ausente: {caminho.name}. Rode o pipeline (scripts/run_all.py) "
            f"ou, no caso do diagnostics.json, `python -m src.diagnostics`."
        )
    with caminho.open(encoding="utf-8") as f:
        return json.load(f)


def _ler_json_opcional(caminho: Path):
    """Como `_ler_json`, mas arquivo ausente devolve `None` (rodada adicional)."""
    return _ler_json(caminho) if caminho.exists() else None


def _carregar_rodada(base: Path) -> dict | None:
    """Matriz, LOCO e `run_meta` de uma rodada adicional; `None` se não existe.

    A rodada existe quando há `matrix.json`. `loco.json` e `run_meta.json`
    ausentes viram lista e dicionário vazios: a rodada pode estar no meio.
    """
    if not (base / "matrix.json").exists():
        return None
    return {
        "matriz": _ler_json(base / "matrix.json"),
        "loco": _ler_json_opcional(base / "loco.json") or [],
        "run_meta": _ler_json_opcional(base / "run_meta.json") or {},
    }


# Campos que só um `run_meta.json` do esquema novo (AD-068) tem; distinguem a
# rodada nova da anterior à reexecução consolidada.
_CAMPOS_META_AD068 = frozenset({"psl_sha256", "sessoes", "particoes", "dedup"})


def conferir_commits(dados: dict) -> None:
    """RUN-03, AD-068b: todas as rodadas presentes têm de ser do mesmo código.

    Compara o `commit_codigo` do `run_meta.json` de cada rodada carregada (as
    sementes da principal, as da sensibilidade e a repetição). Levanta
    `ReportError` listando o *commit* de cada rodada quando há mais de um
    valor. Rodada de `run_meta.json` antigo (anterior à reexecução consolidada,
    sem nenhum campo do AD-068) não entra na comparação: não há o que conferir
    nela. Rodada de esquema novo sem `commit_codigo`, ou com ele nulo (o `git`
    falhou), reprova: não se sabe de que código ela veio.

    Entram também o `hpo.json`, o `hpo_fora_diagonal.json` (esquema novo quando
    têm `psl_sha256`) e o `offline.json` (esquema novo quando tem `fontes`).
    """
    metas: dict[str, dict] = {}
    marcadores: dict[str, frozenset] = {}
    for seed, meta in sorted(dados.get("run_meta_por_seed", {}).items()):
        metas[f"principal, semente {seed}"] = meta or {}
    for seed, rodada in sorted(dados.get("sensibilidade_por_seed", {}).items()):
        metas[f"sensibilidade sem esquema, semente {seed}"] = rodada["run_meta"]
    if dados.get("repeticao"):
        metas["repetição da semente 42"] = dados["repeticao"]["run_meta"]
    for nome, chave, marcador in (
        ("hpo.json", "hpo", _CAMPOS_META_AD068),
        ("hpo_fora_diagonal.json", "hpo_fora_diagonal", _CAMPOS_META_AD068),
        ("offline.json", "offline", frozenset({"fontes"})),
    ):
        if isinstance(dados.get(chave), dict):
            metas[nome] = dados[chave]
            marcadores[nome] = marcador
    sem_commit = [
        nome for nome, meta in metas.items()
        if not meta.get("commit_codigo")
        and marcadores.get(nome, _CAMPOS_META_AD068) & set(meta)
    ]
    if sem_commit:
        raise ReportError(
            "rodada(s) de esquema novo sem commit_codigo (AD-068): "
            f"{', '.join(sem_commit)}. Sem o commit não há como saber de que código "
            "vieram os números; refaça a rodada."
        )
    por_rodada = {nome: meta.get("commit_codigo") for nome, meta in metas.items()}
    registrados = {nome: c for nome, c in por_rodada.items() if c}
    if len(set(registrados.values())) > 1:
        lista = "; ".join(f"{nome}: {c}" for nome, c in registrados.items())
        raise ReportError(
            "as rodadas foram geradas por commits de código diferentes "
            f"(commit_codigo, AD-068): {lista}. Os números de uma tabela têm de "
            "vir de um único commit; refaça as rodadas divergentes."
        )


def _predicoes_registradas(registros: list, run_meta: dict) -> dict[str, str]:
    """`{arquivo: sha256}` das predições dos registros e das partições do `run_meta`."""
    refs = {}
    for rec in registros:
        ref = rec.get("predicoes")
        if ref:
            refs[ref["arquivo"]] = ref["sha256"]
    particoes = run_meta.get("particoes")
    if particoes:
        refs[particoes["arquivo"]] = particoes["sha256"]
    return refs


def conferir_fontes_offline(dados: dict) -> None:
    """O `offline.json` tem de ter lido as predições das rodadas carregadas.

    O mapa `fontes` do `offline.json` guarda o sha256 de cada arquivo de
    predições que a filtragem e o *bootstrap* leram, com o caminho a partir de
    `results/`. Para cada rodada carregada que aparece nele, o mapa tem de ser
    igual ao que o `matrix.json`, o `loco.json` e o `run_meta.json` da rodada
    registram; senão o `offline.json` é de outra execução e `ReportError` nomeia
    o arquivo e os caminhos divergentes. Rodada que o `offline.json` não leu
    (ou que não está carregada) não é conferida.
    """
    offline = dados.get("offline")
    if not isinstance(offline, dict) or not isinstance(offline.get("fontes"), dict):
        return
    rodadas: dict[str, tuple[list, dict]] = {}
    for seed, sub in SUBDIRS_SEEDS.items():
        if seed in dados.get("matriz_por_seed", {}):
            rodadas[f"{sub}/" if sub else ""] = (
                list(dados["matriz_por_seed"][seed])
                + list(dados.get("loco_por_seed", {}).get(seed) or []),
                dados.get("run_meta_por_seed", {}).get(seed) or {},
            )
    for seed, rodada in dados.get("sensibilidade_por_seed", {}).items():
        sub = SUBDIRS_SEEDS[seed]
        rodadas[f"{SUBDIR_SENSIBILIDADE}/" + (f"{sub}/" if sub else "")] = (
            list(rodada["matriz"]) + list(rodada["loco"]), rodada["run_meta"],
        )
    if dados.get("repeticao"):
        rep = dados["repeticao"]
        rodadas[f"{SUBDIR_REPETICAO}/"] = (
            list(rep["matriz"]) + list(rep["loco"]), rep["run_meta"],
        )

    lidas: dict[str, dict[str, str]] = {}
    for caminho, sha in offline["fontes"].items():
        corte = caminho.find("predicoes/")
        prefixo = caminho[:corte] if corte >= 0 else None
        if prefixo in rodadas:
            lidas.setdefault(prefixo, {})[caminho[corte:]] = sha
    problemas = []
    for prefixo, fontes in sorted(lidas.items()):
        registradas = _predicoes_registradas(*rodadas[prefixo])
        for arquivo in sorted(set(fontes) | set(registradas)):
            if fontes.get(arquivo) != registradas.get(arquivo):
                problemas.append(prefixo + arquivo)
    if problemas:
        raise ReportError(
            "o offline.json não confere com as rodadas carregadas (sha256 das "
            f"predições em `fontes` divergente ou ausente): {', '.join(problemas[:5])}"
            + (f" e mais {len(problemas) - 5}" if len(problemas) > 5 else "")
            + ". Rode `python -m src.offline` de novo sobre estes resultados."
        )


def carregar(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict:
    """Lê tudo que as tabelas consomem, com as 3 seeds separadas.

    Falta de seed extra **não** é erro: `results/seed_07/` e `seed_13/` só
    existem depois que o Lucas roda os passos 5 do handoff. Sem elas as tabelas
    saem sem a faixa, e a legenda deixa de prometê-la — melhor que quebrar.

    O mesmo vale para as rodadas da frente A (REEX): a sensibilidade sem
    esquema (`sensibilidade_sem_esquema/`, com `seed_NN/` dentro), a repetição
    da semente 42 (`repeticao_seed_42/`), `hpo_fora_diagonal.json`,
    `offline.json`, o `splits.json` da semente canônica e o `manifest.json` dos
    *corpora*. Ausentes, viram `None` ou dicionário vazio, e as tabelas e macros
    que dependem delas não saem. Com as rodadas lidas, `conferir_commits`
    recusa `commit_codigo` divergentes.
    """
    results_dir = Path(results_dir)
    dados: dict = {
        # Guardado para o Apêndice D, que lê o `requirements.lock` no diretório
        # pai de `results/` (a raiz do pipeline), sem caminho absoluto no código.
        "results_dir": results_dir,
        "diagnostics": _ler_json(results_dir / "diagnostics.json"),
        "hpo": _ler_json(results_dir / "hpo.json"),
        "importances": _ler_json(results_dir / "importances.json"),
        "loco_importances": _ler_json(results_dir / "loco_importances.json"),
        "matriz_por_seed": {},
        "loco_por_seed": {},
        "run_meta_por_seed": {},
        "importances_por_seed": {},
    }
    for seed, sub in SUBDIRS_SEEDS.items():
        base = results_dir / sub if sub else results_dir
        if not (base / "matrix.json").exists():
            continue
        dados["matriz_por_seed"][seed] = _ler_json(base / "matrix.json")
        dados["loco_por_seed"][seed] = _ler_json(base / "loco.json")
        dados["run_meta_por_seed"][seed] = _ler_json(base / "run_meta.json")
        # As importâncias das 3 seeds entraram na T5.5a por necessidade
        # medida, não por simetria: o achado mais forte da T7 na seed 42 --
        # o ISCX não compartilha NENHUMA das 5 primeiras características com
        # os outros corpora -- vira 1 na seed 7 e 2 na seed 13. Publicá-lo
        # como "nenhuma" seria o absoluto que o PC5 proíbe, e só as três
        # seeds mostram isso.
        dados["importances_por_seed"][seed] = _ler_json(base / "importances.json")

    if SEED_CANONICA not in dados["matriz_por_seed"]:
        raise ReportError(
            f"a seed canônica {SEED_CANONICA} não foi encontrada em {results_dir} — "
            f"ela é o resultado do artigo (§7.4) e nenhuma tabela sai sem ela"
        )

    base_sens = results_dir / SUBDIR_SENSIBILIDADE
    dados["sensibilidade_por_seed"] = {}
    for seed, sub in SUBDIRS_SEEDS.items():
        rodada = _carregar_rodada(base_sens / sub if sub else base_sens)
        if rodada is not None:
            dados["sensibilidade_por_seed"][seed] = rodada
    dados["repeticao"] = _carregar_rodada(results_dir / SUBDIR_REPETICAO)
    dados["hpo_fora_diagonal"] = _ler_json_opcional(results_dir / "hpo_fora_diagonal.json")
    dados["offline"] = _ler_json_opcional(results_dir / "offline.json")
    # Resumo das partições da semente canônica, com as contagens de dedup
    # (`splits.json` anterior à deduplicação não traz o campo `dedup`).
    dados["splits"] = _ler_json_opcional(results_dir / "splits.json")
    # O manifesto fica em `corpora/`, ao lado de `results/` na raiz do pipeline.
    dados["manifest"] = _ler_json_opcional(results_dir.parent / "corpora" / "manifest.json")

    conferir_commits(dados)
    conferir_fontes_offline(dados)
    return dados


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------


def media_por_celula(registros: list, regime: str, metrica: str) -> dict:
    """`{(treino, teste): média dos 6 modelos}` para um regime.

    A média sobre os modelos é a convenção que reproduz os números que a §13.8
    publicou (§11, 06/08/2026) — não é escolha deste módulo. Métrica `None`
    (teste de classe única) é **excluída** da média em vez de virar zero, pela
    razão de sempre: zero é um número falso.
    """
    acumulado: dict = {}
    for r in registros:
        if r["regime"] != regime:
            continue
        valor = r["metrics"].get(metrica)
        if valor is None:
            continue
        acumulado.setdefault((r["train_corpus"], r["test_corpus"]), []).append(valor)
    return {k: sum(v) / len(v) for k, v in acumulado.items() if v}


def _celulas_por_seed(matriz_por_seed: dict, regime: str, metrica: str) -> dict:
    return {
        seed: media_por_celula(registros, regime, metrica)
        for seed, registros in matriz_por_seed.items()
    }


def _diagonal_por_modelo(registros: list) -> dict:
    """`{(corpus, modelo, regime): registro}` para as células da diagonal.

    A T3 é a única tabela do artigo que **não** agrega os modelos, e a razão é
    medida, não estética: no ISCX a AUROC do regime oficial é `1,0000` exato em
    4 dos 6 modelos. A média por corpus dá -0,0027 e se lê como "queda
    desprezível"; o que há de fato é teto de métrica. Ver §11, 06/08/2026.
    """
    return {
        (r["train_corpus"], r["model"], r["regime"]): r
        for r in registros
        if r["diagonal"]
    }


def _media(valores) -> float | None:
    limpos = [v for v in valores if v is not None]
    return sum(limpos) / len(limpos) if limpos else None


def medias_da_coluna(registros: list, regime: str, metrica: str) -> dict:
    """Por corpus de **teste**: média na diagonal e média fora dela.

    Devolve `{corpus_teste: (diagonal, fora_diagonal)}`. É a leitura por
    coluna da matriz, e existe porque a T6 compara três coisas no mesmo
    corpus de teste: treinar nele (diagonal), treinar em **um** outro
    (fora-diagonal) e treinar na **união** dos outros três (LOCO). Sem fixar
    o corpus de teste a comparação não fecha — cada corpus tem uma partição
    de teste de tamanho e prevalência diferentes.
    """
    diagonal: dict = {}
    fora: dict = {}
    for r in registros:
        if r["regime"] != regime:
            continue
        valor = r["metrics"].get(metrica)
        if valor is None:
            continue
        alvo = diagonal if r["train_corpus"] == r["test_corpus"] else fora
        alvo.setdefault(r["test_corpus"], []).append(valor)
    return {
        c: (_media(diagonal.get(c, [])), _media(fora.get(c, [])))
        for c in set(diagonal) | set(fora)
    }


def media_por_modelo(registros: list, regime: str, metrica: str) -> dict:
    """`{(modelo, na_diagonal): média das células}` para um regime.

    É a leitura da calibração (Cap. 6, seção 6.7): cada modelo tem 4 células na
    diagonal e 12 fora dela, e a pergunta é se o mesmo modelo calibra pior
    quando o dataset de teste muda. Métrica `None` sai da média, como em
    `media_por_celula`.
    """
    acumulado: dict = {}
    for r in registros:
        if r["regime"] != regime:
            continue
        valor = r["metrics"].get(metrica)
        if valor is None:
            continue
        chave = (r["model"], r["train_corpus"] == r["test_corpus"])
        acumulado.setdefault(chave, []).append(valor)
    return {k: _media(v) for k, v in acumulado.items()}


def medias_loco(registros: list, regime: str, metrica: str) -> dict:
    """Por corpus retido: média dos 6 modelos treinados na união dos outros 3."""
    acumulado: dict = {}
    for r in registros:
        if r["regime"] != regime:
            continue
        valor = r["metrics"].get(metrica)
        if valor is None:
            continue
        acumulado.setdefault(r["test_corpus"], []).append(valor)
    return {c: _media(v) for c, v in acumulado.items()}


def _postos(valores: dict, features: list) -> dict:
    """Posto 1..n de cada característica dentro de **um** ajuste, por |valor|.

    O valor absoluto não é detalhe: a regressão logística grava coeficiente
    **com sinal**, e o que a T7 pergunta é "o modelo se apoia nisto?", não
    "para que lado". Ordenar pelo valor com sinal poria as características
    que empurram para *benigno* no fim da lista como se fossem irrelevantes.

    A ordenação é estável e o desempate segue a ordem de `features`, que é
    fixa em `config/features.json` — sem isso duas execuções poderiam
    trocar duas características empatadas e o `.tex` deixaria de regenerar
    byte a byte (RNF1). Empate é comum: o LightGBM grava contagem de splits,
    e características não usadas empatam todas em zero.
    """
    ordenado = sorted(features, key=lambda f: -abs(valores[f]))
    return {f: i for i, f in enumerate(ordenado, start=1)}


def posto_medio(importances: dict, regime: str, corpus: str) -> dict:
    """`{feature: posto médio}` sobre os modelos que expõem importância.

    ⚠️ **Posto, e nunca média dos valores.** Medido em `importances.json`:
    os cinco modelos gravam em escalas incompatíveis — coeficiente com sinal
    na regressão logística (−5,13 a +4,09), soma 1 no Random Forest e no
    XGBoost, soma 100 no CatBoost e **contagem de splits** no LightGBM
    (soma ~24.800). Uma média entre eles seria dominada pelo LightGBM por
    três ordens de grandeza e não mediria coisa nenhuma. Ver §11, 07/08/2026.

    São **5** modelos e não 6: o MLP não expõe importância por
    característica (§13.8), o que faz a T7 cobrir 40 dos 48 ajustes.
    """
    features = importances["features"]
    soma = {f: 0 for f in features}
    n = 0
    for chave, valores in importances["ajustes"].items():
        r, c, _modelo = chave.split("|")
        if r != regime or c != corpus:
            continue
        postos = _postos(valores, features)
        for f in features:
            soma[f] += postos[f]
        n += 1
    if not n:
        raise ReportError(
            f"nenhum ajuste com importância para {regime}|{corpus} — "
            f"importances.json tem {len(importances['ajustes'])} ajustes"
        )
    return {f: v / n for f, v in soma.items()}


def top_caracteristicas(importances: dict, regime: str, corpus: str, n: int) -> list:
    """As `n` características de menor posto médio, em ordem."""
    media = posto_medio(importances, regime, corpus)
    return sorted(media, key=lambda f: media[f])[:n]


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------


def tabela1_vazamento(dados: dict) -> str:
    """T1: vazamento de domínio no corte de referência dos 6 benchmarks."""
    vaz = dados["diagnostics"]["vazamento_split_oficial"]
    linhas = []
    for nome, v in vaz.items():
        linhas.append(
            f"{_rotulo(nome, ROTULO_CORPUS)} & {_milhar(v['dominios_treino'])} & "
            f"{_milhar(v['dominios_teste'])} & {_milhar(v['dominios_sobrepostos'])} & "
            f"{_pct(v['contaminacao_dominios_pct'])} & "
            f"{_pct(v['contaminacao_linhas_pct'])} \\\\"
        )
    corpo = "\n".join(linhas)
    return _cabecalho("results/diagnostics.json :: vazamento_split_oficial") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Vazamento de domínio no corte de referência dos seis \emph{{datasets}}. Domínio
é o eTLD+1, pela mesma chave que o Regime B usa para agrupar. A contaminação
por linha excede a por domínio: os compartilhados são os que mais reaparecem.}}
\label{{tab:vazamento}}
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Dataset & Dom.\ treino & Dom.\ teste & Sobrep. & \% dom. & \% linhas \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
{_fonte("diagnostics.json")}
\end{{table}}
"""


def tabela5_vies(dados: dict) -> str:
    """T5 — gap de viés de construção nos 6 corpora."""
    vies = dados["diagnostics"]["vies_construcao"]
    limiar = dados["diagnostics"]["parametros"]["limiar_vies_p"]
    linhas = []
    for nome, v in vies.items():
        veredito = "contaminado" if v["veredito"] == "contaminado" else "limpo"
        marca = r"\textbf{" + veredito + "}" if veredito == "contaminado" else veredito
        linhas.append(
            f"{_rotulo(nome, ROTULO_CORPUS)} & {_milhar(v['n_total'])} & "
            f"{_pct(v['balanco']['phishing_pct'], 1)} & "
            f"{_pct(v['scheme']['gap_p'], 1)} & {_pct(v['path']['gap_p'], 1)} & "
            f"{marca} \\\\"
        )
    corpo = "\n".join(linhas)
    return _cabecalho("results/diagnostics.json :: vies_construcao") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Viés de construção nos seis \emph{{datasets}}. O \emph{{gap}} é a diferença
absoluta entre as frações de URLs da classe positiva e benignas com o
atributo. Veredito ``contaminado'' quando algum \emph{{gap}} atinge
{_num(limiar, 0 if float(limiar).is_integer() else 1)} pontos percentuais; a distribuição medida é bimodal.}}
\label{{tab:vies}}
\small
\begin{{tabular}}{{lrrrrl}}
\toprule
Dataset & Linhas & \% positiva & Gap \emph{{scheme}} & Gap \emph{{path}} & Veredito \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
{_fonte("diagnostics.json")}
\end{{table}}
"""


def tabela2_matriz(dados: dict) -> str:
    """T2 — matriz cruzada 4x4 no regime eTLD+1, média dos 6 modelos.

    Sem IC, e a legenda diz por quê: a célula é média sobre modelos, e a média
    de seis intervalos de bootstrap não é o intervalo da média (§11, 06/08).
    """
    por_seed_auroc = _celulas_por_seed(dados["matriz_por_seed"], "etld1", "auroc")
    por_seed_aucpr = _celulas_por_seed(dados["matriz_por_seed"], "etld1", "auc_pr")
    seeds = sorted(por_seed_auroc)

    def bloco(por_seed, treino):
        celulas = []
        for teste in ORDEM_CORPORA:
            ponto = por_seed[SEED_CANONICA].get((treino, teste))
            entre = [por_seed[s].get((treino, teste)) for s in seeds]
            faixa = _faixa(entre, 3)
            celulas.append(f"{_num(ponto, 3)} {faixa}".strip())
        return " & ".join(celulas)

    linhas = []
    for treino in ORDEM_CORPORA:
        rot = _rotulo(treino, ROTULO_CORPUS)
        linhas.append(
            rf"\multirow{{2}}{{*}}{{{rot}}} & AUROC & {bloco(por_seed_auroc, treino)} \\"
        )
        linhas.append(rf" & AUC-PR & {bloco(por_seed_aucpr, treino)} \\")
        linhas.append(r"\addlinespace")
    corpo = "\n".join(linhas[:-1])
    cols = " & ".join(_rotulo(c, ROTULO_CORPUS) for c in ORDEM_CORPORA)
    n_seeds = len(seeds)

    return _cabecalho("results/matrix.json + results/seed_*/matrix.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Matriz cruzada no Regime B: linha = dataset de treino, coluna =
dataset de teste. Cada célula é a \textbf{{média dos seis modelos}} na
semente canônica (42), seguida da faixa \mbox{{[mín--máx]}} entre as
{n_seeds} sementes ({", ".join(str(s) for s in seeds)}). A faixa mede a
incerteza de \emph{{partição}}, que fora da diagonal domina: em 95 das 192
células ela excede a largura do intervalo de confiança por \emph{{bootstrap}}.
\textbf{{O intervalo de confiança não é reportado aqui}} porque a célula agrega
seis modelos, e a média de seis intervalos não é o intervalo da média; ele
aparece na Tabela~\ref{{tab:diagonal}}, por modelo, onde é válido. Nenhuma
célula fora da diagonal deve ser lida pontualmente; o que se afirma é o
padrão por linha.}}
\label{{tab:matriz}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrr}}
\toprule
Treino & Métrica & \multicolumn{{4}}{{c}}{{Teste}} \\
\cmidrule(l){{3-6}}
 & & {cols} \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}}}
{_fonte("matrix.json", "seed_*/matrix.json")}
\end{{table}}
"""


# Critério de AUROC saturada no Regime A: 1,0000 depois do arredondamento a 4
# casas, que é o que a tabela imprime. É o teto de métrica que faz a queda da
# diagonal no ISCX parecer desprezível (§13.4-20).
LIMIAR_SATURACAO = 0.99995


def tabela3_diagonal(dados: dict) -> str:
    """T3: Regime A contra Regime B na diagonal, uma linha por modelo e dataset.

    Volta às 24 linhas (6 modelos x 4 datasets) por decisão dos autores em
    01/10/2026 (design do Cap. 6, CAP6-15). A monografia não tem o teto de
    página do artigo, e a forma por modelo é a que deixa o leitor conferir a
    separação por dataset em vez de recebê-la pronta.

    A coluna do critério antigo (|ΔAUROC| na largura do IC) e a contagem dela no
    subtítulo de cada bloco saíram (A2, TAB-02): o critério passou a ser o IC
    pareado da `tab:bootstrap-pareado`. O subtítulo traz só a contagem de modelos
    com AUROC saturada no Regime A (TAB-03). A média por dataset continua
    **proibida**: no ISCX ela dá -0,0027 e leria como "queda desprezível" o que
    é teto de métrica.

    A faixa das colunas de Δ é entre as **sementes**, e não entre os modelos
    como na versão de 4 linhas: com um ponto por modelo, a faixa entre sementes
    é bem definida e é a convenção do colchete no resto do texto. O IC vai entre
    parênteses, e não em colchetes, pela mesma convenção.
    """
    seeds = sorted(dados["matriz_por_seed"])
    por_seed = {
        s: _diagonal_por_modelo(dados["matriz_por_seed"][s]) for s in seeds
    }
    canonica = por_seed[SEED_CANONICA]

    def delta(seed, corpus, modelo, metrica):
        a = por_seed[seed].get((corpus, modelo, "oficial"))
        b = por_seed[seed].get((corpus, modelo, "etld1"))
        if not a or not b:
            return None
        va, vb = a["metrics"].get(metrica), b["metrics"].get(metrica)
        return None if va is None or vb is None else vb - va

    def com_faixa(corpus, modelo, metrica):
        ponto = delta(SEED_CANONICA, corpus, modelo, metrica)
        entre = [delta(s, corpus, modelo, metrica) for s in seeds]
        # Faixa em `\scriptsize`: com 24 linhas e 7 colunas, é o que faz a
        # tabela caber na largura do texto sem `\resizebox`.
        faixa = _faixa(entre, 4, r";\,")
        return f"{_num(ponto)} {{\\scriptsize {faixa}}}" if faixa else _num(ponto)

    # ANL-03: com mais de uma semente, a média e a faixa da AUROC de cada regime
    # entram ao lado da semente 42; com uma só, as colunas não existem.
    com_media = len(seeds) > 1
    n_colunas = 6 + 2 * com_media

    def auroc_entre_sementes(corpus, modelo, regime):
        registros = [por_seed[s].get((corpus, modelo, regime)) for s in seeds]
        return [r["metrics"].get("auroc") for r in registros if r]

    blocos = []
    for corpus in ORDEM_CORPORA:
        linhas = []
        n_modelos = saturados = 0
        for modelo in ROTULO_MODELO:
            a = canonica.get((corpus, modelo, "oficial"))
            b = canonica.get((corpus, modelo, "etld1"))
            if not a or not b:
                continue
            n_modelos += 1
            saturados += a["metrics"]["auroc"] >= LIMIAR_SATURACAO
            ci = b.get("ci") or {}
            tem_ic = "auroc_lo" in ci and "auroc_hi" in ci
            ic = (
                rf"{{\scriptsize ({_num(ci['auroc_lo'])};\,{_num(ci['auroc_hi'])})}}"
                if tem_ic else "--"
            )
            medias = (
                f" & {_celula_media(auroc_entre_sementes(corpus, modelo, 'oficial'))} & "
                f"{_celula_media(auroc_entre_sementes(corpus, modelo, 'etld1'))}"
                if com_media else ""
            )
            linhas.append(
                f"{_rotulo(modelo, ROTULO_MODELO)} & {_num(a['metrics']['auroc'])} & "
                f"{_num(b['metrics']['auroc'])} & {ic} & "
                f"{com_faixa(corpus, modelo, 'auroc')} & "
                f"{com_faixa(corpus, modelo, 'f1')}"
                f"{medias} \\\\"
            )
        # O teto de métrica do bloco, contado: é a leitura que a seção 6.3 faz, e
        # o leitor a confere nas linhas logo abaixo.
        cabecalho_bloco = (
            rf"\multicolumn{{{n_colunas}}}{{l}}{{\textbf{{{_rotulo(corpus, ROTULO_CORPUS)}}}: "
            rf"AUROC saturada no Regime A em {saturados} de {n_modelos}}} \\"
        )
        blocos.append("\n".join([cabecalho_bloco, *linhas]))
    corpo = "\n\\midrule\n".join(blocos)

    # A legenda só promete a faixa e a média quando elas aparecem (ANL-03).
    if com_media:
        complemento = (
            f". Faixa das {len(seeds)} sementes entre colchetes; Média: das "
            f"{len(seeds)} sementes, faixa abaixo"
        )
        cabecalho_medias = r" & \multicolumn{2}{c}{Média}"
        cmidrule_medias = r" \cmidrule(lr){7-8}"
        colunas_medias = " & A & B"
        formato = "lrrcrrrr"
        # Com as duas colunas de média a tabela passa da largura do texto, e a
        # redução à largura fica só nesse caso. Medido em `make pdf` (10/11/2026):
        # com a coluna do critério antigo eram 115 pt a mais; sem ela, 89 pt
        # (Overfull de 89,4 pt sem o `\resizebox`), então o `\resizebox` fica (A2, TAB-02).
        abre, fecha = "\\resizebox{\\textwidth}{!}{%\n", "}"
    else:
        complemento = ""
        cabecalho_medias = ""
        cmidrule_medias = ""
        colunas_medias = ""
        formato = "lrrcrr"
        abre, fecha = "", ""

    return _cabecalho("results/matrix.json + results/seed_*/matrix.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Diagonal da matriz por modelo, semente {SEED_CANONICA}: AUROC nos Regimes A e B,
IC de 95\% por \emph{{bootstrap}} da célula no Regime B e
$\Delta$ = Regime B $-$ Regime A{complemento}.}}
\label{{tab:diagonal}}
% `\footnotesize`; o `\resizebox` entra só com as colunas de média das
% sementes (ANL-03), que passam a tabela da largura do texto.
\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
{abre}\begin{{tabular}}{{{formato}}}
\toprule
 & \multicolumn{{2}}{{c}}{{AUROC, regime}} & & &{cabecalho_medias} \\
\cmidrule(lr){{2-3}}{cmidrule_medias}
Modelo & A & B & IC (B) & $\Delta$AUROC & $\Delta$F1{colunas_medias} \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}{fecha}
{_fonte("matrix.json", "seed_*/matrix.json")}
\end{{table}}
"""


def tabela4_ablacao(dados: dict) -> str:
    """T4 — ablação de hiperparâmetros: seco contra ajustado, por modelo.

    A T4 subiu de P2 para P1 em 06/08/2026 (§7.4) porque deixou de ser "e se
    a gente ajustasse?" e virou a **defesa empírica do D5**: ela mede o que
    se compra ajustando hiperparâmetro, e a resposta é ≤ 0,0012 de AUROC em
    cinco modelos e **prejuízo** no sexto. É também o argumento concreto
    contra copiar `hpo.json` para a `experiment.yml`, que o `CLAUDE.md`
    proíbe ao Lucas.

    O resultado do HPO entra **em todos os modelos, como resultado e não como
    nota de rodapé** (§11, 06/08): selecionar quais modelos mostrar seria
    escolher resultado pela via que o D5 proíbe. A frase de perda da legenda é
    DERIVADA dos dados: sai quando algum delta de teste é <= 0, nomeia o de
    maior perda ("o único que" com uma perda, "um dos que" com mais) e só diz
    "após melhorar na validação" quando o delta de validação desse modelo é
    positivo. Sem perda nenhuma, a legenda não traz frase de perda.
    """
    hpo = dados["hpo"]
    modelos = hpo["modelos"]

    linhas = []
    deltas = {}
    deltas_val = {}
    for nome in ROTULO_MODELO:  # ordem fixa do artigo, não a do JSON
        v = modelos.get(nome)
        if v is None:
            continue
        d_val = v["auroc_val_ajustado"] - v["auroc_val_seco"]
        d_teste = v["auroc_teste_ajustado"] - v["auroc_teste_seco"]
        deltas[nome] = d_teste
        deltas_val[nome] = d_val
        seco, ajust = v["fit_seconds_seco"], v["fit_seconds_ajustado"]
        razao = ajust / seco if seco else None

        # "Melhor tentativa" é a leitura compacta da trajetória da busca que
        # a §7.4 pede: em que ponto das tentativas o melhor valor de
        # validação apareceu. Uma coluna, em vez de 30 números por modelo.
        traj = v.get("trajetoria_val_melhor_ate_agora") or []
        melhor = traj.index(traj[-1]) + 1 if traj else None

        linhas.append(
            f"{_rotulo(nome, ROTULO_MODELO)} & "
            f"{_num(v['auroc_val_seco'])} & {_num(d_val, 5)} & "
            f"{_num(v['auroc_teste_seco'])} & {_num(d_teste, 5)} & "
            f"{melhor if melhor else '--'} & "
            f"{_num(seco, 1)} & {_num(ajust, 1)} & "
            f"{_num(razao, 1)} \\\\"
        )
    corpo = "\n".join(linhas)

    perdeu = {k: v for k, v in deltas.items() if v <= 0}
    pior = min(perdeu, key=perdeu.get) if perdeu else None
    # "o único" é asserção de contagem: se um dia dois modelos piorarem, a
    # frase tem de deixar de dizer isso sozinha, e não por alguém reler a
    # legenda. Mesma doutrina do PC5 contra absoluto não medido.
    quantificador = "o único que" if len(perdeu) == 1 else "um dos que"
    # Cabe nas 40 palavras da legenda (D-C2). O nome do modelo é DERIVADO
    # (nunca digitado) e a palavra "piora" só é impressa quando existe perda
    # (os testes `test_a_t4_imprime_todos_os_modelos_com_o_delta_no_corpo` e
    # `test_a_t4_frase_de_perda_cobre_os_tres_ramos` travam os três ramos). O
    # trecho "após melhorar na validação" também é derivado: só sai quando o
    # delta de validação do modelo de maior perda é positivo.
    apos_val = " após melhorar na validação" if pior and deltas_val[pior] > 0 else ""
    frase_perda = (
        f" O {_rotulo(pior, ROTULO_MODELO)} é {quantificador} "
        f"piora no teste{apos_val}."
        if pior else ""
    )

    return _cabecalho("results/hpo.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Ablação de hiperparâmetros em {_rotulo(hpo["corpus"], ROTULO_CORPUS)},
Regime B: a configuração fixa de
\texttt{{config/experiment.yml}}, idêntica entre os \emph{{datasets}}, contra a melhor
de {hpo["n_trials"]} tentativas.{frase_perda}}}
\label{{tab:ablacao}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{Validação}} & \multicolumn{{2}}{{c}}{{Teste}} & & \multicolumn{{3}}{{c}}{{Tempo de ajuste}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(l){{7-9}}
Modelo & AUROC & $\Delta$ & AUROC & $\Delta$ & Tent.\ da melhor val. & fixa (s) & ajust.\ (s) & razão \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}}}
{_fonte("hpo.json")}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# Apêndice B: hiperparâmetros (spec 8, T6)
# ---------------------------------------------------------------------------

# Rótulo da família como o Cap. 4 a nomeia. O `run_meta.json` grava a chave
# curta do pipeline (`rede_neural`), que não cabe numa tabela impressa.
ROTULO_FAMILIA = {
    "linear": "Linear",
    "bagging": r"\emph{Bagging}",
    "boosting": r"\emph{Boosting}",
    "rede_neural": "Rede neural",
}

ROTULO_REGIME = {"oficial": "Regime A", "etld1": "Regime B"}


def _valor_codigo(valor) -> str:
    """Valor de parâmetro como o código o lê (Python), sem arredondar.

    A tabela B1 é a configuração executada, não um resultado: `1.0` fica `1.0`
    e `null` do JSON vira `None`, que é o que o estimador recebeu.
    """
    if valor is None:
        return "None"
    if isinstance(valor, bool):
        return "True" if valor else "False"
    if isinstance(valor, (list, tuple)):
        itens = ", ".join(_valor_codigo(i) for i in valor)
        return f"({itens},)" if len(valor) == 1 else f"({itens})"
    return str(valor)


def _valor_ajustado(valor) -> str:
    """Valor de `params_ajustados` como o código o lê, com reais em 4 algarismos
    significativos (`43.35`, `4.502e-05`). Inteiro, `None`, texto e tupla ficam
    como estão. Ponto decimal, como na B1: o par `chave = valor` é código."""
    if isinstance(valor, float):
        return f"{valor:.4g}"
    return _valor_codigo(valor)


def apendiceB1_configuracao(dados: dict) -> str:
    """Apêndice B, etapa 1: parâmetros fixos dos seis modelos (`config.modelos`)."""
    meta = dados["run_meta_por_seed"].get(SEED_CANONICA)
    if meta is None:
        raise ReportError(f"run_meta.json da semente {SEED_CANONICA} ausente")
    modelos = meta["config"]["modelos"]

    linhas = []
    for nome in ROTULO_MODELO:  # ordem fixa do artigo, não a do JSON
        cfg = modelos.get(nome)
        if cfg is None:
            raise ReportError(f"config.modelos de run_meta.json não tem {nome}")
        pares = ", ".join(
            r"\texttt{" + _escapar(f"{chave} = {_valor_codigo(valor)}") + "}"
            for chave, valor in cfg["params"].items()
        )
        familia = ROTULO_FAMILIA.get(cfg["familia"], _escapar(cfg["familia"]))
        padronizacao = "sim" if cfg["escala"] else "não"
        linhas.append(
            f"{_rotulo(nome, ROTULO_MODELO)} & {familia} & {padronizacao} & {pares} \\\\"
        )
    corpo = "\n".join(linhas)

    return _cabecalho("results/run_meta.json :: config.modelos") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Configuração fixa dos seis modelos na etapa 1, idêntica em todos os
\emph{{datasets}}, regimes e sementes. Valores como gravados, sem arredondamento.}}
\label{{tab:apB-configuracao}}
\footnotesize
\begin{{tabular}}{{lll>{{\raggedright\arraybackslash}}p{{0.46\textwidth}}}}
\toprule
Modelo & Família & Padronização & Parâmetros \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
{_fonte("run_meta.json")}
\end{{table}}
"""


def apendiceB2_hpo(dados: dict) -> str:
    """Apêndice B, etapa 2: a melhor configuração de cada modelo (`params_ajustados`)."""
    hpo = dados["hpo"]
    modelos = hpo["modelos"]

    linhas = []
    for nome in ROTULO_MODELO:
        v = modelos.get(nome)
        if v is None:
            raise ReportError(f"hpo.json não tem {nome}")
        pares = ", ".join(
            r"\texttt{" + _escapar(f"{chave} = {_valor_ajustado(valor)}") + "}"
            for chave, valor in v["params_ajustados"].items()
        )
        linhas.append(f"{_rotulo(nome, ROTULO_MODELO)} & {pares} \\\\")
    corpo = "\n".join(linhas)

    regime = ROTULO_REGIME.get(hpo["regime"], _escapar(hpo["regime"]))
    return _cabecalho("results/hpo.json :: modelos[*].params_ajustados") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Melhor configuração de cada modelo no ajuste da etapa 2, em
{_rotulo(hpo["corpus"], ROTULO_CORPUS)}, {regime}, entre {hpo["n_trials"]} tentativas.
Reais com quatro algarismos significativos.}}
\label{{tab:apB-hpo}}
\footnotesize
\begin{{tabular}}{{l>{{\raggedright\arraybackslash}}p{{0.72\textwidth}}}}
\toprule
Modelo & Melhor configuração \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
{_fonte("hpo.json")}
\end{{table}}
"""


def tabela6_loco(dados: dict) -> str:
    """T6 — leave-one-corpus-out: treino na união de 3, teste no 4º.

    Responde à objeção mais previsível do revisor ("então junte as fontes"),
    que é a razão do D3-b. As três colunas de AUROC são o mesmo corpus de
    teste medido sob três origens de treino, e é por isso que a tabela é
    lida por linha e não por coluna.

    ⚠️ **As duas últimas colunas não são enfeite.** O LOCO treina na união
    de três corpora, então sua contaminação residual é a união de três
    sobreposições e chega a ser 2,5x a do par a par (§13.4-17). Publicar o
    ganho sem declará-la deixaria em aberto a leitura alternativa mais fácil
    — que o ganho vem de vazamento —, e a decisão do D3 é **medir e
    declarar, nunca filtrar**: filtrar mudaria o que a matriz mede.
    """
    seeds = sorted(dados["loco_por_seed"])
    coluna = {
        s: medias_da_coluna(dados["matriz_por_seed"][s], "etld1", "auroc")
        for s in seeds
    }
    loco = {
        s: medias_loco(dados["loco_por_seed"][s], "etld1", "auroc") for s in seeds
    }
    sobrep = dados["diagnostics"]["sobreposicao_loco"]["etld1"]
    por_retido = sobrep["por_corpus_retido"]

    def ganho(seed, corpus):
        _diag, fora = coluna[seed].get(corpus, (None, None))
        lo = loco[seed].get(corpus)
        return None if fora is None or lo is None else lo - fora

    # ANL-03: com mais de uma semente, a média e a faixa de cada coluna de AUROC
    # entram ao lado da semente 42; com uma só, as colunas não existem.
    com_media = len(seeds) > 1

    def medias_da_linha(fora_s, loco_s, ganho_s, diag_s):
        if not com_media:
            return ""
        return (
            f"{_celula_media(fora_s)} & {_celula_media(loco_s)} & "
            f"{_celula_media(ganho_s, 4, ';' + chr(92) + ',')} & {_celula_media(diag_s)} & "
        )

    linhas = []
    for corpus in ORDEM_CORPORA:
        diag, fora = coluna[SEED_CANONICA].get(corpus, (None, None))
        lo = loco[SEED_CANONICA].get(corpus)
        g = [ganho(s, corpus) for s in seeds]
        ov = por_retido.get(corpus, {})
        medias = medias_da_linha(
            [coluna[s].get(corpus, (None, None))[1] for s in seeds],
            [loco[s].get(corpus) for s in seeds],
            g,
            [coluna[s].get(corpus, (None, None))[0] for s in seeds],
        )
        linhas.append(
            f"{_rotulo(corpus, ROTULO_CORPUS)} & {_num(fora)} & {_num(lo)} & "
            f"{_num(ganho(SEED_CANONICA, corpus))} & "
            f"{_num(diag)} & {medias}{_pct(ov.get('shared_domain_pct'))} & "
            f"{_pct(ov.get('rows_with_shared_domain_pct'))} \\\\"
        )
    corpo = "\n".join(linhas)

    def media_seed(fonte, seed, idx=None):
        vals = [
            fonte[seed][c][idx] if idx is not None else fonte[seed][c]
            for c in ORDEM_CORPORA
            if c in fonte[seed]
        ]
        return _media(vals)

    m_fora = [media_seed(coluna, s, 1) for s in seeds]
    m_loco = [media_seed(loco, s) for s in seeds]
    m_diag = [media_seed(coluna, s, 0) for s in seeds]
    m_ganho = [
        _media([ganho(s, c) for c in ORDEM_CORPORA]) for s in seeds
    ]
    canon = seeds.index(SEED_CANONICA)
    resumo = sobrep["resumo"]
    media_linha = (
        rf"\textbf{{Média}} & {_num(m_fora[canon])} & {_num(m_loco[canon])} & "
        rf"{_num(m_ganho[canon])} & {_num(m_diag[canon])} & "
        rf"{medias_da_linha(m_fora, m_loco, m_ganho, m_diag)}"
        rf"{_pct(resumo['media_pct'])} & {_pct(resumo['linhas_media_pct'])} \\"
    )
    if com_media:
        complemento = f" Média das {len(seeds)} sementes e faixa à direita."
        grupos = (
            r"\multicolumn{4}{c}{AUROC, semente 42} & \multicolumn{4}{c}{AUROC, média das sementes} & "
            r"\multicolumn{2}{c}{Sobrep.\ residual} \\"
            "\n" r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(l){10-11}"
        )
        colunas = r"1$\to$1 & União & Ganho & Diagonal & 1$\to$1 & União & Ganho & Diagonal & \% dom. & \% linhas"
        formato = "lrrrrrrrrrr"
        # Com as colunas de média a tabela passa da largura do texto (medido na
        # compilação: 190 pt a mais); a redução à largura fica só nesse caso.
        abre, fecha = "\\resizebox{\\textwidth}{!}{%\n", "}"
    else:
        complemento = ""
        grupos = (
            r"\multicolumn{4}{c}{AUROC (média dos seis modelos)} & "
            r"\multicolumn{2}{c}{Sobrep.\ residual} \\"
            "\n" r"\cmidrule(lr){2-5} \cmidrule(l){6-7}"
        )
        colunas = r"1$\to$1 & União & Ganho & Diagonal & \% dom. & \% linhas"
        formato = "lrrrrrr"
        abre, fecha = "", ""

    return _cabecalho(
        "results/loco.json + results/matrix.json + results/seed_*/ + "
        "results/diagnostics.json :: sobreposicao_loco"
    ) + rf"""
\begin{{table}}[htb]
\centering
\caption{{\emph{{Leave-one-corpus-out}} no Regime B, por \emph{{dataset}} de teste:
``1$\to$1'' treina em outro \emph{{dataset}}, ``União'' nos três, ``Diagonal'' no
próprio. As duas últimas colunas são a contaminação residual, medida, nunca filtrada.{complemento}}}
\label{{tab:loco}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
{abre}\begin{{tabular}}{{{formato}}}
\toprule
 & {grupos}
Dataset de teste & {colunas} \\
\midrule
{corpo}
\midrule
{media_linha}
\bottomrule
\end{{tabular}}{fecha}
{_fonte("loco.json", "matrix.json", "diagnostics.json")}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# Precisão, revocação e taxa de falsos positivos (RUN-01, REEX-04)
# ---------------------------------------------------------------------------

REGIME_A = "oficial"
REGIME_B = "etld1"
# Sufixo de macro por regime: o nome de comando só aceita letras.
SUFIXO_MACRO_REGIME = {REGIME_A: "A", REGIME_B: "B"}

# (chave em `metrics`, cabeçalho da coluna, prefixo da macro)
METRICAS_LIMIAR = (
    ("precision", "Precisão", "precisao"),
    ("recall", "Revocação", "revocacao"),
    ("fpr", "TFP", "tfp"),
)


def _rotulo_treino(treino: str) -> str:
    """Rótulo do dataset de treino; a união do LOCO sai como "União"."""
    return "União" if treino.startswith("uniao_sem_") else _rotulo(treino, ROTULO_CORPUS)


def _grupos_de_celulas(celulas: dict) -> dict:
    """`{(treino, teste): valor}` em média das células da diagonal e das de fora."""
    return {
        "diagonal": _media([v for (tr, te), v in celulas.items() if tr == te]),
        "fora_diagonal": _media([v for (tr, te), v in celulas.items() if tr != te]),
    }


def _medias_limiar(dados: dict, regime: str, metrica: str, semente: int = SEED_CANONICA) -> dict:
    """Médias de uma métrica de limiar de uma semente (a canônica): diagonal, fora e LOCO.

    A célula é a média dos seis modelos, a convenção das demais tabelas; o grupo
    é a média das células. O LOCO é a média dos quatro datasets retidos.
    """
    matriz = media_por_celula(dados["matriz_por_seed"][semente], regime, metrica)
    loco = medias_loco(dados["loco_por_seed"][semente], regime, metrica)
    grupos = _grupos_de_celulas(matriz)
    grupos["loco"] = _media(list(loco.values()))
    return grupos


def tabela10_limiar(dados: dict) -> str:
    """Tabela 10: precisão, revocação e TFP por célula da matriz e do LOCO (RUN-01).

    Responde ao REEX-04: as métricas dependentes de limiar, nos dois regimes,
    com o limiar da validação do dataset de treino (D6). Cada valor é a média
    dos seis modelos da célula, a mesma convenção da AUROC. Traz as 16 células
    da matriz, as 4 do LOCO e as médias dos três grupos, as mesmas das macros.
    """
    # ANL-03: com mais de uma semente, a média e a faixa de cada valor entram ao
    # lado da semente 42, numa segunda faixa de seis colunas; com uma só, não
    # existem e a legenda não as promete. A tabela de 13 colunas é reduzida à
    # largura do texto (como a da ablação).
    sementes = _sementes_presentes(dados)
    com_media = len(sementes) > 1
    pares = [(regime, chave) for regime in (REGIME_A, REGIME_B) for chave, _c, _p in METRICAS_LIMIAR]
    celulas = {
        (s, regime, chave): media_por_celula(dados["matriz_por_seed"][s], regime, chave)
        for s in sementes
        for regime, chave in pares
    }
    loco = {
        (s, regime, chave): medias_loco(dados["loco_por_seed"][s], regime, chave)
        for s in sementes
        for regime, chave in pares
    }
    medias = {
        (s, regime, chave): _medias_limiar(dados, regime, chave, s)
        for s in sementes
        for regime, chave in pares
    }

    def valores(escolha):
        """Semente 42 e, havendo mais de uma semente, a média com a faixa."""
        partes = [_num(escolha(SEED_CANONICA, regime, chave)) for regime, chave in pares]
        if com_media:
            partes += [
                _celula_media([escolha(s, regime, chave) for s in sementes])
                for regime, chave in pares
            ]
        return " & ".join(partes)

    linhas_matriz = [
        f"{_rotulo(tr, ROTULO_CORPUS)} $\\to$ {_rotulo(te, ROTULO_CORPUS)} & "
        + valores(lambda s, r, c, tr=tr, te=te: celulas[(s, r, c)].get((tr, te)))
        + " \\\\"
        for tr in ORDEM_CORPORA
        for te in ORDEM_CORPORA
    ]
    linhas_loco = [
        f"União $\\to$ {_rotulo(te, ROTULO_CORPUS)} & "
        + valores(lambda s, r, c, te=te: loco[(s, r, c)].get(te))
        + " \\\\"
        for te in ORDEM_CORPORA
    ]
    linhas_media = [
        rf"\textbf{{{titulo}}} & "
        + valores(lambda s, r, c, g=grupo: medias[(s, r, c)][g])
        + " \\\\"
        for titulo, grupo in (
            ("Média da diagonal", "diagonal"),
            ("Média fora da diagonal", "fora_diagonal"),
            ("Média do LOCO", "loco"),
        )
    ]
    cabecalhos = " & ".join(rotulo for _k, rotulo, _p in METRICAS_LIMIAR)

    if com_media:
        legenda = (
            rf"Precisão, revocação e taxa de falsos positivos (TFP) por célula da matriz e do "
            rf"\emph{{leave-one-corpus-out}} (LOCO), nos dois regimes: semente 42 e média das "
            rf"{len(sementes)} sementes com a faixa. Média dos seis modelos, limiar da validação."
        )
        grupos = (
            r" & \multicolumn{3}{c}{Regime A} & \multicolumn{3}{c}{Regime B} & "
            r"\multicolumn{3}{c}{Média, Regime A} & \multicolumn{3}{c}{Média, Regime B} \\"
            "\n"
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(l){11-13}"
        )
        colunas = " & ".join([cabecalhos] * 4)
        formato = "l" + "r" * 12
        abre, fecha = "\\resizebox{\\textwidth}{!}{%\n", "}"
    else:
        legenda = (
            r"Precisão, revocação e taxa de falsos positivos (TFP) por célula da matriz e do "
            r"\emph{leave-one-corpus-out} (LOCO), nos dois regimes, na semente 42. Cada valor é "
            r"a média dos seis modelos, com o limiar escolhido na validação."
        )
        grupos = (
            r" & \multicolumn{3}{c}{Regime A} & \multicolumn{3}{c}{Regime B} \\"
            "\n"
            r"\cmidrule(lr){2-4} \cmidrule(l){5-7}"
        )
        colunas = " & ".join([cabecalhos] * 2)
        formato = "lrrrrrr"
        abre, fecha = "", ""
    return _cabecalho("results/matrix.json + results/loco.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{{legenda}}}
\label{{tab:limiar}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
{abre}\begin{{tabular}}{{{formato}}}
\toprule
{grupos}
Treino $\to$ teste & {colunas} \\
\midrule
{chr(10).join(linhas_matriz)}
\midrule
{chr(10).join(linhas_loco)}
\midrule
{chr(10).join(linhas_media)}
\bottomrule
\end{{tabular}}{fecha}
{_fonte("matrix.json", "loco.json")}
\end{{table}}
"""


def macros_limiar(dados: dict) -> list[str]:
    """Macros da média de precisão, revocação e TFP por regime e grupo (RUN-01)."""
    macros = []
    for regime in (REGIME_A, REGIME_B):
        sufixo = SUFIXO_MACRO_REGIME[regime]
        for chave, _cabecalho_coluna, prefixo in METRICAS_LIMIAR:
            grupos = _medias_limiar(dados, regime, chave)
            macros += [
                _macro(f"{prefixo}DiagonalRegime{sufixo}", _num(grupos["diagonal"])),
                _macro(f"{prefixo}ForaDiagonalRegime{sufixo}",
                       _num(grupos["fora_diagonal"])),
                _macro(f"{prefixo}LocoRegime{sufixo}", _num(grupos["loco"])),
            ]
    return macros


# ---------------------------------------------------------------------------
# Apêndice C: tabelas completas da matriz e do LOCO (spec 8, T8)
# ---------------------------------------------------------------------------

# (chave em `metrics`, rótulo impresso, sufixo do arquivo). A ordem é a das
# tabelas no apêndice.
METRICAS_APENDICE_C = (
    ("auroc", "AUROC", "auroc"),
    ("auc_pr", "AUC-PR", "aucpr"),
    ("f1", "F1", "f1"),
    ("mcc", "MCC", "mcc"),
)

# Regime do JSON (`oficial`, `etld1`) para o sufixo do arquivo e do rótulo.
# O nome do regime na legenda vem de `ROTULO_REGIME`: "oficial" não vai ao papel.
SUFIXO_REGIME = {"oficial": "a", "etld1": "b"}


def _sep_faixa_apendice_c(metrica: str) -> str:
    r"""Separador da faixa: `;\,` no MCC, único com valor negativo (AD-041)."""
    return r";\," if metrica == "mcc" else "--"


def _valores_por_seed(registros_por_seed: dict, regime: str, metrica: str, chave) -> dict:
    """`{semente: {chave(registro): valor}}` de um regime, sem agregar modelos.

    O Apêndice C é a única publicação que não agrega os seis modelos nem os
    pares: cada célula é um registro. Por isso a chave identifica o registro, e
    o valor `None` (métrica indefinida) fica `None` em vez de virar zero.
    """
    return {
        semente: {
            chave(r): r["metrics"].get(metrica)
            for r in registros
            if r["regime"] == regime
        }
        for semente, registros in registros_por_seed.items()
    }


def _celula_apendice_c(por_semente: dict, chave, metrica: str) -> str:
    r"""Valor da semente 42 e, ao lado, a faixa das sementes disponíveis, no mesmo corpo.

    As mesmas 4 casas no ponto e na faixa, para o ponto nunca ficar fora dela
    (ver `_faixa`). O valor não vai em negrito nem na diagonal: o negrito alarga
    a coluna inteira e a tabela deixaria de caber na página paisagem.
    """
    ponto = _num(por_semente[SEED_CANONICA].get(chave), 4)
    faixa = _faixa(
        [por_semente[s].get(chave) for s in sorted(por_semente)],
        4,
        _sep_faixa_apendice_c(metrica),
    )
    return f"{ponto} {faixa}" if faixa else ponto


def _frase_da_faixa(dados_por_semente: dict) -> str:
    """Trecho da legenda que diz de onde vêm o valor e a faixa de cada célula."""
    # Na ordem em que a Seção 4.8 as cita (42, 7 e 13), não na ordem numérica.
    sementes = [s for s in SUBDIRS_SEEDS if s in dados_por_semente]
    if len(sementes) < 2:
        return f"Valor da semente {SEED_CANONICA}."
    lista = ", ".join(str(s) for s in sementes[:-1]) + f" e {sementes[-1]}"
    return (
        f"Valor da semente {SEED_CANONICA} e, entre colchetes, "
        f"faixa das sementes {lista}."
    )


def apendiceC_matriz(dados: dict, regime: str, metrica: str) -> str:
    """Apêndice C: uma métrica da matriz cruzada, 16 pares por 6 modelos.

    Linhas agrupadas pelo *dataset* de treino, e a diagonal (treino e teste no
    mesmo *dataset*) com o nome do teste em negrito. Colunas: os seis modelos, na
    ordem do artigo. Página paisagem, em `\\scriptsize`: a tabela só é escalada
    (`\\resizebox`) se passar da largura da página.
    """
    rotulos = {chave: nome for chave, nome, _ in METRICAS_APENDICE_C}
    if metrica not in rotulos:
        raise ReportError(f"métrica {metrica!r} fora do Apêndice C")
    if regime not in SUFIXO_REGIME:
        raise ReportError(f"regime {regime!r} fora do Apêndice C")
    sufixo = next(s for chave, _, s in METRICAS_APENDICE_C if chave == metrica)
    letra = SUFIXO_REGIME[regime]

    por_semente = _valores_por_seed(
        dados["matriz_por_seed"], regime, metrica,
        lambda r: (r["train_corpus"], r["test_corpus"], r["model"]),
    )

    blocos = []
    for treino in ORDEM_CORPORA:
        linhas = []
        for teste in ORDEM_CORPORA:
            diagonal = treino == teste
            rotulo = _rotulo(teste, ROTULO_CORPUS)
            if diagonal:
                rotulo = rf"\textbf{{{rotulo}}}"
            # O treino aparece só na primeira linha do bloco: uma coluna a mais
            # custa menos altura que uma linha de cabeçalho por bloco.
            rotulo_treino = _rotulo(treino, ROTULO_CORPUS) if not linhas else ""
            celulas = [
                _celula_apendice_c(por_semente, (treino, teste, modelo), metrica)
                for modelo in ROTULO_MODELO
            ]
            linhas.append(
                f"{rotulo_treino} & {rotulo} & " + " & ".join(celulas) + r" \\"
            )
        blocos.append("\n".join(linhas))
    corpo = "\n\\midrule\n".join(blocos)
    cols = " & ".join(_rotulo(m, ROTULO_MODELO) for m in ROTULO_MODELO)

    legenda = (
        f"{rotulos[metrica]} no {ROTULO_REGIME[regime]}. "
        f"{_frase_da_faixa(por_semente)} Negrito: diagonal."
    )

    return _cabecalho("results/matrix.json + results/seed_*/matrix.json") + rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{legenda}}}
\label{{tab:apC-{letra}-{sufixo}}}
\scriptsize
\setlength{{\abovecaptionskip}}{{2pt}}
\setlength{{\belowcaptionskip}}{{2pt}}
\setlength{{\tabcolsep}}{{1.5pt}}
\renewcommand{{\arraystretch}}{{0.75}}
% Sem redução: a tabela só é escalada se passar da largura da página paisagem
% (o MCC, com valores negativos, passa por poucos pontos).
\setbox0=\hbox{{%
\begin{{tabular}}{{llrrrrrr}}
\toprule
Treino & Teste & {cols} \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}}}%
\ifdim\wd0>\linewidth\resizebox{{\linewidth}}{{!}}{{\copy0}}\else\copy0\fi
{_fonte("matrix.json", "seed_*/matrix.json")}
\end{{table}}
"""


def apendiceC_loco(dados: dict, regime: str) -> str:
    """Apêndice C: o LOCO de um regime, 4 *datasets* retidos por 4 métricas.

    Cada bloco é um *dataset* retido para o teste, com uma linha por métrica e
    uma coluna por modelo, a mesma forma da matriz: 16 linhas por 6 modelos. A
    forma com uma linha por modelo (24 linhas) não cabe duas por página
    paisagem. O treino é a união dos outros três *datasets*, de modo que a
    chave do registro é o *dataset* retido (de teste) e o modelo.
    """
    if regime not in SUFIXO_REGIME:
        raise ReportError(f"regime {regime!r} fora do Apêndice C")
    letra = SUFIXO_REGIME[regime]

    por_metrica = {
        metrica: _valores_por_seed(
            dados["loco_por_seed"], regime, metrica,
            lambda r: (r["test_corpus"], r["model"]),
        )
        for metrica, _, _ in METRICAS_APENDICE_C
    }

    blocos = []
    for retido in ORDEM_CORPORA:
        linhas = []
        for metrica, nome, _ in METRICAS_APENDICE_C:
            celulas = [
                _celula_apendice_c(por_metrica[metrica], (retido, modelo), metrica)
                for modelo in ROTULO_MODELO
            ]
            # O dataset de teste só na primeira linha do bloco, como na matriz.
            teste = _rotulo(retido, ROTULO_CORPUS) if not linhas else ""
            linhas.append(f"{teste} & {nome} & " + " & ".join(celulas) + r" \\")
        blocos.append("\n".join(linhas))
    corpo = "\n\\midrule\n".join(blocos)
    cols = " & ".join(_rotulo(m, ROTULO_MODELO) for m in ROTULO_MODELO)

    legenda = (
        f"LOCO no {ROTULO_REGIME[regime]}. {_frase_da_faixa(por_metrica['auroc'])}"
    )

    return _cabecalho("results/loco.json + results/seed_*/loco.json") + rf"""
\begin{{table}}[!htbp]
\centering
\caption{{{legenda}}}
\label{{tab:apC-loco-{letra}}}
\scriptsize
\setlength{{\abovecaptionskip}}{{2pt}}
\setlength{{\belowcaptionskip}}{{2pt}}
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{0.75}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
Teste & Métrica & {cols} \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
{_fonte("loco.json", "seed_*/loco.json")}
\end{{table}}
"""


def _gerador_matriz(regime: str, metrica: str):
    return lambda dados: apendiceC_matriz(dados, regime, metrica)


def _gerador_loco(regime: str):
    return lambda dados: apendiceC_loco(dados, regime)


# Nomes dos dez arquivos do Apêndice C, no mesmo laço que os gera: oito da
# matriz (regime x métrica) e dois do LOCO (regime).
TABELAS_APENDICE_C = {
    **{
        f"apendiceC-matriz-{SUFIXO_REGIME[regime]}-{sufixo}.tex": _gerador_matriz(regime, metrica)
        for regime in SUFIXO_REGIME
        for metrica, _, sufixo in METRICAS_APENDICE_C
    },
    **{
        f"apendiceC-loco-{SUFIXO_REGIME[regime]}.tex": _gerador_loco(regime)
        for regime in SUFIXO_REGIME
    },
}


# ---------------------------------------------------------------------------
# Apêndice D: ambiente de execução (spec 8, T10)
# ---------------------------------------------------------------------------

# Bibliotecas do Apêndice D, na ordem impressa: (nome no `requirements.lock`,
# nome como o texto o escreve). O nome do lock é comparado em minúsculas.
BIBLIOTECAS_APENDICE_D = (
    ("scikit-learn", "scikit-learn"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("catboost", "CatBoost"),
    ("optuna", "Optuna"),
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("scipy", "SciPy"),
    ("tldextract", "tldextract"),
)


def _texttt_quebravel(texto: str, bloco: int = 8) -> str:
    r"""`\texttt` de um identificador, com quebra de linha permitida.

    Hash e identificador de execução não têm ponto de quebra natural: sem o
    `\allowbreak` depois de `-` e a cada `bloco` caracteres, a célula estoura a
    largura do texto (`Overfull \hbox`, gate 1).
    """
    partes = []
    for i, c in enumerate(texto):
        partes.append(_escapar(c))
        if i + 1 < len(texto) and (c in "-_" or (i + 1) % bloco == 0):
            partes.append(r"\allowbreak{}")
    return r"\texttt{" + "".join(partes) + "}"


def _igual_em_todas(valores: dict, nome: str) -> str:
    """O valor único de um campo que tem de ser o mesmo em todas as execuções.

    A tabela diz "iguais em todas" por construção; se um campo divergir, o
    erro é aqui, alto, e não uma tabela que afirma o que os dados negam.
    """
    distintos = sorted({str(v) for v in valores.values()})
    if len(distintos) != 1:
        raise ReportError(f"{nome} difere entre as execuções: {distintos}")
    return distintos[0]


def apendiceD1_execucao(dados: dict) -> str:
    """Apêndice D: as execuções que geraram os resultados.

    Uma linha por execução: a matriz e o LOCO nas três sementes e o ajuste de
    hiperparâmetros da etapa 2. Python, plataforma e GPU, que são os mesmos,
    vão em linhas comuns. O tempo de busca do ajuste não foi gravado: o
    `hpo.json` tem só os tempos de treino de cada modelo, que não são o tempo
    de busca, e por isso a célula diz "não registrado" em vez de somá-los.
    """
    metas = {
        s: dados["run_meta_por_seed"][s]
        for s in SUBDIRS_SEEDS
        if s in dados["run_meta_por_seed"]
    }
    if SEED_CANONICA not in metas:
        raise ReportError(f"run_meta.json da semente {SEED_CANONICA} ausente")
    hpo = dados["hpo"]

    def tempo(meta) -> str:
        minutos = meta.get("minutos_ajuste_total")
        return "não registrado" if minutos is None else _num(minutos, 1)

    linhas = []
    for semente, meta in metas.items():
        ids = ", ".join(meta["run_ids"])
        linhas.append(
            f"Matriz e LOCO, semente {semente} & {_texttt_quebravel(meta['commit'])} & "
            f"{_texttt_quebravel(ids)} & "
            f"{_texttt_quebravel(meta['config_fingerprint'])} & {tempo(meta)} \\\\"
        )
    linhas.append(
        f"Ajuste de hiperparâmetros & {_texttt_quebravel(hpo['commit'])} & "
        f"{_texttt_quebravel(hpo['run_id'])} & "
        f"{_texttt_quebravel(hpo['config_fingerprint'])} & não registrado \\\\"
    )

    python = _igual_em_todas({s: m["python"] for s, m in metas.items()}, "python")
    plataforma = _igual_em_todas(
        {s: m["plataforma"] for s, m in metas.items()}, "plataforma"
    )
    gpu = _igual_em_todas(
        {**{s: m["gpu"] for s, m in metas.items()}, "hpo": hpo["gpu"]}, "gpu"
    )
    corpo = "\n".join(linhas)
    comuns = "\n".join(
        [
            rf"Python (matriz e LOCO) & \multicolumn{{4}}{{l}}{{{_escapar(python)}}} \\",
            rf"Plataforma (matriz e LOCO) & \multicolumn{{4}}{{l}}{{{_escapar(plataforma)}}} \\",
            rf"GPU (todas) & \multicolumn{{4}}{{l}}{{{_escapar(gpu)}}} \\",
        ]
    )

    return _cabecalho(
        "results/run_meta.json + results/seed_*/run_meta.json + results/hpo.json"
    ) + rf"""
\begin{{table}}[htb]
\centering
\caption{{Execuções que geraram os resultados e, abaixo, o que é comum a elas.}}
\label{{tab:apD-execucao}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lllll}}
\toprule
Execução & \emph{{Commit}} & Identificador & \emph{{Fingerprint}} & Treino (min) \\
\midrule
{corpo}
\midrule
{comuns}
\bottomrule
\end{{tabular}}
{_fonte("run_meta.json", "seed_*/run_meta.json", "hpo.json")}
\end{{table}}
"""


def _pinos_do_lock(texto: str) -> tuple[str, dict]:
    """`(Python exigido, {nome em minúsculas: versão})` do `requirements.lock`."""
    exigido = None
    pinos: dict = {}
    for linha in texto.splitlines():
        m = re.match(r"#\s*Python exigido:\s*(.+?)\s*$", linha)
        if m:
            exigido = m.group(1)
        m = re.match(r"([A-Za-z0-9_.\-]+)==(\S+)\s*$", linha)
        if m:
            pinos[m.group(1).lower().replace("_", "-")] = m.group(2)
    if exigido is None:
        raise ReportError("requirements.lock sem a linha 'Python exigido'")
    return exigido, pinos


def apendiceD2_versoes(dados: dict) -> str:
    """Apêndice D: as versões das bibliotecas, lidas do `requirements.lock`.

    O arquivo está na raiz do pipeline, o diretório pai de `results/`. Pino que
    falta é erro: a tabela não sai com uma biblioteca a menos.
    """
    lock = Path(dados["results_dir"]).parent / "requirements.lock"
    if not lock.exists():
        raise ReportError(f"arquivo ausente: {lock.name} (ao lado de results/)")
    exigido, pinos = _pinos_do_lock(lock.read_text(encoding="utf-8"))

    entradas = [("Python exigido", _escapar(exigido))]
    for chave, nome in BIBLIOTECAS_APENDICE_D:
        versao = pinos.get(chave)
        if versao is None:
            raise ReportError(f"requirements.lock não fixa a versão de {chave}")
        entradas.append((nome, f"\\texttt{{{_escapar(versao)}}}"))
    # Duas metades lado a lado: 10 entradas em 5 linhas, para o apêndice caber
    # em uma página. A metade esquerda vem primeiro, na ordem de `BIBLIOTECAS_APENDICE_D`.
    meio = (len(entradas) + 1) // 2
    linhas = []
    for i in range(meio):
        esquerda = " & ".join(entradas[i])
        direita = " & ".join(entradas[i + meio]) if i + meio < len(entradas) else " & "
        linhas.append(f"{esquerda} & {direita} \\\\")
    corpo = "\n".join(linhas)

    return _cabecalho("requirements.lock (raiz do pipeline)") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Versão de Python exigida e versões das bibliotecas principais, fixadas no
arquivo \texttt{{requirements.lock}}.}}
\label{{tab:apD-versoes}}
\footnotesize
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{ll@{{\hspace{{2.5em}}}}ll}}
\toprule
Componente & Versão & Componente & Versão \\
\midrule
{corpo}
\bottomrule
\end{{tabular}}
\fonte{{Elaboração própria, a partir de \texttt{{pipeline/\allowbreak{{}}requirements.lock}}}}
\end{{table}}
"""


# Quantas características entram na T7 por corpus. Cinco, e não dez: o
# orçamento de página é o risco mais apertado do projeto (§13.4-23) e a
# separação entre os corpora já é visível nas cinco primeiras.
TOP_CARACTERISTICAS = 5


def tabela7_importancias(dados: dict) -> str:
    """T7: as 5 primeiras características por dataset e o compartilhamento.

    No Cap. 6 (T4, 01/10/2026) as 5 linhas de posto voltaram; o histórico do
    corte do artigo fica abaixo, porque explica a faixa da linha "Compart.".

    Encolhida na T5.9 (D-C3, 08/08/2026) para **a linha de compartilhamento**,
    que é a única que a Discussão usa. Eram 5 postos x 4 datasets mais a linha
    de compartilhamento, com legenda de 133 palavras, e custava ~0,35 página.

    ⚠️ **O que se perde é real e está declarado**: os nomes das cinco
    características por dataset saem do artigo (`has_https` no topo do
    Kaggle-03, `domain_length` no do ISCX eram ilustrativos). A afirmação da
    Discussão não depende deles, e o roteiro de revisão registra
    `results/importances.json` como o caminho de volta na câmera-ready.

    ⚠️ A célula traz **faixa entre as 3 seeds** e não um número só, e a razão é
    medida: na seed 42 o ISCX não compartilha nenhuma das cinco com os outros
    datasets, mas na seed 7 compartilha uma e na seed 13 duas. Publicar
    "nenhuma" seria o absoluto que o PC5 proíbe — e seria falso em dois terços
    das execuções. Com a tabela reduzida a esta linha, a garantia que o
    `test_a_t7_nao_afirma_que_o_iscx_nao_compartilha_nenhuma` trava deixa de
    ser um detalhe do rodapé e passa a ser **a tabela inteira**.
    """
    por_seed = dados["importances_por_seed"] or {
        SEED_CANONICA: dados["importances"]
    }
    seeds = sorted(por_seed)

    tops = {
        s: {
            c: top_caracteristicas(por_seed[s], "etld1", c, TOP_CARACTERISTICAS)
            for c in ORDEM_CORPORA
        }
        for s in seeds
    }

    def compartilhadas(seed, corpus):
        outros = set()
        for c in ORDEM_CORPORA:
            if c != corpus:
                outros |= set(tops[seed][c])
        return len(set(tops[seed][corpus]) & outros)

    def celula(corpus):
        # Faixa suprimida quando as 3 seeds coincidem: `3 [3--3]` é ruído, e
        # a coluna é estreita. O ponto continua sendo o da seed canônica.
        entre = [compartilhadas(s, corpus) for s in seeds]
        ponto = compartilhadas(SEED_CANONICA, corpus)
        faixa = "" if min(entre) == max(entre) else _faixa(entre, 0)
        return f"{ponto} {faixa}".strip()

    corpo = " & ".join(celula(c) for c in ORDEM_CORPORA)

    # As linhas de posto voltaram no Cap. 6 (T4, decisão dos autores de
    # 01/10/2026): os nomes por dataset são o que a seção 6.6 lê, e a
    # monografia não tem o teto de página do artigo. Semente canônica, como o
    # ponto da linha "Compart.".
    postos = "\n".join(
        f"{i} & "
        + " & ".join(
            rf"\texttt{{{_escapar(tops[SEED_CANONICA][c][i - 1])}}}"
            for c in ORDEM_CORPORA
        )
        + r" \\"
        for i in range(1, TOP_CARACTERISTICAS + 1)
    )

    return _cabecalho("results/importances.json + results/seed_*/importances.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{As {TOP_CARACTERISTICAS} características de menor posto médio por \emph{{dataset}} de
treino (Regime B, semente {SEED_CANONICA}) e, na linha Compart., quantas delas reaparecem
entre as {TOP_CARACTERISTICAS} de algum outro \emph{{dataset}}; colchetes: faixa entre as
{len(seeds)} sementes.}}
\label{{tab:importancias}}
% `\scriptsize` e `\tabcolsep` de 4pt: em `\footnotesize` os nomes em
% `\texttt` estouravam a largura do texto em 61pt (medido em 01/10/2026).
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{c{"l" * len(ORDEM_CORPORA)}}}
\toprule
Posto & {" & ".join(_rotulo(c, ROTULO_CORPUS) for c in ORDEM_CORPORA)} \\
\midrule
{postos}
\midrule
Compart.\ & {corpo} \\
\bottomrule
\end{{tabular}}
{_fonte("importances.json", "seed_*/importances.json")}
\end{{table}}
"""



def tabela9_calibracao(dados: dict) -> str:
    """T9: escore de Brier e ECE por modelo, na diagonal e fora dela (Cap. 6).

    Regime B, semente canônica. Cada célula do modelo é a média das suas 4
    células na diagonal ou das 12 fora dela; a linha de média é a média dos 6
    modelos, que coincide com a dos macros `\\brierDiagonal` e vizinhos porque
    toda célula tem os 6 modelos. A faixa entre sementes vai só na média, como
    no resto do texto: por modelo, ela repetiria a mesma leitura 24 vezes.

    O número de bins do ECE vem da configuração gravada em `run_meta.json`, e
    não deste módulo: a legenda não pode prometer uma conta que a avaliação
    não fez.
    """
    seeds = sorted(dados["matriz_por_seed"])
    por_seed = {
        s: {
            met: media_por_modelo(dados["matriz_por_seed"][s], "etld1", met)
            for met in ("brier", "ece")
        }
        for s in seeds
    }
    canon = por_seed[SEED_CANONICA]
    colunas = [("brier", True), ("brier", False), ("ece", True), ("ece", False)]

    linhas = []
    for modelo in ROTULO_MODELO:
        celulas = [_num(canon[met].get((modelo, na_diag))) for met, na_diag in colunas]
        linhas.append(f"{_rotulo(modelo, ROTULO_MODELO)} & " + " & ".join(celulas) + r" \\")
    corpo = "\n".join(linhas)

    def media(seed, met, na_diag):
        return _media([por_seed[seed][met].get((m, na_diag)) for m in ROTULO_MODELO])

    # A faixa vai numa linha própria, logo abaixo da média: ao lado do ponto,
    # as quatro colunas estouravam a largura do texto em 41pt (01/10/2026).
    pontos, faixas = [], []
    for met, na_diag in colunas:
        entre = [media(s, met, na_diag) for s in seeds]
        pontos.append(_num(media(SEED_CANONICA, met, na_diag)))
        faixas.append(_faixa(entre, 4))
    linha_media = (
        r"\textbf{Média} & " + " & ".join(pontos) + r" \\" + "\n"
        + r"\quad faixa & " + " & ".join(faixas) + r" \\"
    )

    n_bins = dados["run_meta_por_seed"][SEED_CANONICA]["config"]["avaliacao"]["ece_bins"]

    return _cabecalho("results/matrix.json + results/seed_*/matrix.json + results/run_meta.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Calibração por modelo no Regime B, semente {SEED_CANONICA}: escore de Brier e ECE
({n_bins} \emph{{bins}} de largura igual), médias das células da diagonal e de fora dela, sem
recalibração. Colchetes: faixa entre as {len(seeds)} sementes.}}
\label{{tab:calibracao}}
\footnotesize
\begin{{tabular}}{{lrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{Brier}} & \multicolumn{{2}}{{c}}{{ECE}} \\
\cmidrule(lr){{2-3}} \cmidrule(l){{4-5}}
Modelo & Diagonal & Fora & Diagonal & Fora \\
\midrule
{corpo}
\midrule
{linha_media}
\bottomrule
\end{{tabular}}
{_fonte("matrix.json", "seed_*/matrix.json", "run_meta.json")}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# E4: Friedman e Nemenyi (T8 e F3)
# ---------------------------------------------------------------------------
#
# A conta é do `estatistica.py`; aqui só se formata. O protocolo (16 blocos do
# Regime B, AUROC, alfa 0,05) foi prometido no Cap. 4 e está fixado lá, nas
# constantes do módulo. A semente canônica decide e as outras duas entram como
# faixa e como contagem de sementes em que cada par difere, pela mesma doutrina
# do resto deste arquivo: escolher a semente que dá o resultado mais limpo seria
# selecionar resultado.


def friedman_por_seed(dados: dict) -> dict:
    """`{seed: resultado de estatistica.comparar}` para cada seed disponível."""
    try:
        return {
            seed: estatistica.comparar(registros)
            for seed, registros in sorted(dados["matriz_por_seed"].items())
        }
    except ValueError as exc:
        raise ReportError(f"Friedman: {exc}") from exc


def _sementes_por_par(por_seed: dict) -> dict:
    """`{frozenset(par): nº de seeds em que o par difere pelo Nemenyi}`."""
    contagem: dict = {}
    for res in por_seed.values():
        for par in res["pares_significativos"]:
            contagem[frozenset(par)] = contagem.get(frozenset(par), 0) + 1
    return contagem


def _p_tex(p: float) -> str:
    """p-valor com 4 casas; abaixo de 0,0001 vira desigualdade, nunca `0,0000`."""
    return r"$<$ 0,0001" if p < 1e-4 else _num(p, 4)


def tabela8_friedman(dados: dict) -> str:
    """T8: posto médio por modelo, faixa entre seeds e pares que diferem."""
    por_seed = friedman_por_seed(dados)
    canon = por_seed[SEED_CANONICA]
    contagem = _sementes_por_par(por_seed)
    n_seeds = len(por_seed)
    medios = canon["postos_medios"]
    ordem = sorted(canon["modelos"], key=lambda m: (medios[m], m))

    linhas = []
    for m in ordem:
        faixa = _faixa([r["postos_medios"][m] for r in por_seed.values()], 4)
        outros = sorted(
            (o for o in canon["modelos"] if frozenset((m, o)) in contagem),
            key=lambda o: (medios[o], o),
        )
        difere = ", ".join(
            f"{_rotulo(o, ROTULO_MODELO)} ({contagem[frozenset((m, o))]})" for o in outros
        ) or "nenhum"
        linhas.append(
            f"{_rotulo(m, ROTULO_MODELO)} & {_num(medios[m], 4)} & {faixa} & {difere} \\\\"
        )
    corpo = "\n".join(linhas)
    gl1, gl2 = canon["gl_ff"]
    return _cabecalho("results/matrix.json + results/seed_*/matrix.json (estatistica.py)") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Posto médio de cada modelo nos {canon["n_blocos"]} blocos da matriz cruzada no Regime B
(AUROC, semente {SEED_CANONICA}), com a faixa entre as {n_seeds} sementes, e os modelos dos quais
difere pelo teste de Nemenyi (entre parênteses, em quantas sementes).}}
\label{{tab:friedman}}
\small
\begin{{tabular}}{{lrr>{{\raggedright\arraybackslash}}p{{5.4cm}}}}
\toprule
Modelo & Posto médio & Faixa & Difere de \\
\midrule
{corpo}
\midrule
\multicolumn{{4}}{{l}}{{Friedman: $\chi^2_F$ = {_num(canon["qui2"], 2)}, {canon["gl"]} g.l., $p$ = {_p_tex(canon["p"])}.}} \\
\multicolumn{{4}}{{l}}{{Iman e Davenport: $F_F$ = {_num(canon["ff"], 2)}, ({gl1}, {gl2}) g.l., $p$ = {_p_tex(canon["p_ff"])}.}} \\
\multicolumn{{4}}{{l}}{{Nemenyi: $q_{{\alpha}}$ = {_num(canon["q_alfa"], 3)} e diferença crítica (CD) = {_num(canon["cd"], 3)}, com $\alpha$ = {_num(canon["alfa"], 2)}.}} \\
\bottomrule
\end{{tabular}}
{_fonte("matrix.json", "seed_*/matrix.json")}
\end{{table}}
"""


def figura3_diferenca_critica(dados: dict) -> str:
    """F3: diagrama de diferença crítica (Demšar, 2006), em TikZ.

    Sai como `.tex` e não como PDF do matplotlib para nascer no `make tabelas`
    (o EST-01 pede tabela e diagrama pelo mesmo alvo) e ter a fonte do texto.
    Os segmentos usam `to` e não o operador de caminho de dois hifens, que o
    gate 5 confundiria com travessão se a figura saísse de `tabelas/`.

    Leitura: eixo dos postos com o melhor à esquerda; cada modelo pende do seu
    posto médio; a barra grossa liga modelos que NÃO diferem entre si; o
    segmento acima do eixo tem o comprimento da CD.
    """
    canon = friedman_por_seed(dados)[SEED_CANONICA]
    medios = canon["postos_medios"]
    k = canon["k_modelos"]
    cd = canon["cd"]
    ordem = sorted(canon["modelos"], key=lambda m: (medios[m], m))

    def x(v: float) -> str:
        return f"{v:.4f}"

    cmds = [rf"\draw (1,0) to ({k},0);"]
    for r in range(1, k + 1):
        cmds.append(rf"\draw ({r},0) to ({r},0.15) node[above] {{{r}}};")
    cmds.append(rf"\draw[thick] (1,1.2) to ({x(1 + cd)},1.2);")
    cmds.append(rf"\draw (1,1.05) to (1,1.35);")
    cmds.append(rf"\draw ({x(1 + cd)},1.05) to ({x(1 + cd)},1.35);")
    cmds.append(
        rf"\node[above] at ({x(1 + cd / 2)},1.2) {{CD = {_num(cd, 3)}}};"
    )

    grupos = canon["grupos"]
    for j, grupo in enumerate(grupos):
        y = -0.35 - 0.3 * j
        lo = min(medios[m] for m in grupo) - 0.04
        hi = max(medios[m] for m in grupo) + 0.04
        cmds.append(rf"\draw[line width=2.5pt] ({x(lo)},{y:.2f}) to ({x(hi)},{y:.2f});")

    topo = -0.35 - 0.3 * len(grupos)
    metade = (k + 1) // 2
    for i, m in enumerate(ordem):
        lado_esq = i < metade
        nivel = i if lado_esq else k - 1 - i
        y = topo - 0.55 * (nivel + 1)
        r = medios[m]
        rot = f"{_rotulo(m, ROTULO_MODELO)} ({_num(r, 4)})"
        if lado_esq:
            cmds.append(
                rf"\draw ({x(r)},0) to ({x(r)},{y:.2f}) to (0.8,{y:.2f}) "
                rf"node[left] {{{rot}}};"
            )
        else:
            cmds.append(
                rf"\draw ({x(r)},0) to ({x(r)},{y:.2f}) to ({k + 0.2:.1f},{y:.2f}) "
                rf"node[right] {{{rot}}};"
            )
    desenho = "\n".join(cmds)
    return _cabecalho("results/matrix.json (estatistica.py)") + rf"""
\begin{{figure}}[htb]
\centering
\caption{{Diagrama de diferença crítica dos seis modelos no Regime B (AUROC, semente
{SEED_CANONICA}). Eixo de postos médios, melhor à esquerda; a barra grossa liga modelos sem
diferença significativa pelo pós-teste de Nemenyi.}}
\label{{fig:diferenca-critica}}
\begin{{tikzpicture}}[x=1.55cm,y=1cm,font=\footnotesize]
{desenho}
\end{{tikzpicture}}
{_fonte("matrix.json")}
\end{{figure}}
"""


# Nome de macro só aceita letras: a chave do pipeline vira sufixo fixo, e não
# um "melhor modelo" derivado, porque na seed 42 o último posto é um empate
# (Reg. logística e Random Forest) e um `\friedmanPiorModelo` escolheria um dos
# dois em silêncio.
SUFIXO_MACRO_MODELO = {
    "logistic_regression": "Logistica",
    "random_forest": "RandomForest",
    "xgboost": "Xgboost",
    "lightgbm": "Lightgbm",
    "catboost": "Catboost",
    "mlp": "Mlp",
}


def _macros_friedman(dados: dict) -> list[str]:
    por_seed = friedman_por_seed(dados)
    canon = por_seed[SEED_CANONICA]
    contagem = _sementes_por_par(por_seed)
    n_seeds = len(por_seed)
    k = canon["k_modelos"]
    macros = [
        _macro("friedmanBlocos", str(canon["n_blocos"]), "celulas da matriz no Regime B"),
        _macro("friedmanModelos", str(k)),
        _macro("friedmanQui", _num(canon["qui2"], 2), f"seed {SEED_CANONICA}"),
        _macro("friedmanGl", str(canon["gl"])),
        _macro("friedmanP", _num(canon["p"], 4)),
        _macro("friedmanFF", _num(canon["ff"], 2), "Iman e Davenport"),
        _macro("friedmanPFF", _num(canon["p_ff"], 4)),
        _macro("friedmanQuiMin", _num(min(r["qui2"] for r in por_seed.values()), 2),
               f"entre as {n_seeds} seeds"),
        _macro("friedmanQuiMax", _num(max(r["qui2"] for r in por_seed.values()), 2)),
        _macro("friedmanPMax", _num(max(r["p"] for r in por_seed.values()), 4),
               "o maior p entre as seeds"),
        _macro("friedmanSementesRejeita",
               str(sum(r["rejeita_h0"] for r in por_seed.values())),
               f"de {n_seeds} seeds com H0 rejeitada"),
        _macro("friedmanAlfa", _num(canon["alfa"], 2)),
        _macro("nemenyiQ", _num(canon["q_alfa"], 3)),
        _macro("nemenyiCD", _num(canon["cd"], 3)),
        _macro("nemenyiParesTotal", str(k * (k - 1) // 2)),
        _macro("nemenyiPares", str(len(canon["pares_significativos"])),
               f"pares que diferem na seed {SEED_CANONICA}"),
        _macro("nemenyiParesTodasSementes",
               str(sum(1 for n in contagem.values() if n == n_seeds)),
               f"pares que diferem nas {n_seeds} seeds"),
    ]
    for modelo in sorted(canon["modelos"], key=lambda m: SUFIXO_MACRO_MODELO[m]):
        sufixo = SUFIXO_MACRO_MODELO[modelo]
        valores = [r["postos_medios"][modelo] for r in por_seed.values()]
        macros += [
            _macro(f"posto{sufixo}", _num(canon["postos_medios"][modelo], 4)),
            _macro(f"posto{sufixo}Min", _num(min(valores), 4)),
            _macro(f"posto{sufixo}Max", _num(max(valores), 4)),
        ]
    return macros


# ---------------------------------------------------------------------------
# Números soltos para a prosa (`numeros.tex`)
# ---------------------------------------------------------------------------
#
# Este arquivo é a resposta ao PC3, e ele mudou de status em 07/08/2026: até
# então "número medido não entra na prosa" era regra sem alternativa, porque
# só tabela e figura liam de `results/` e **frase não lia**. A T5.4 conviveu
# com isso remetendo a `\ref{}`; a T5.5 não convive, porque Resultados é
# feito de número. Decisão do Julio, §11 de 07/08/2026.
#
# ⚠️ A regra NÃO afrouxou. Continua proibido digitar número medido no `.tex`.
# O que existe agora é a segunda saída que a §11 de 06/08 já registrava como
# "adiada, não descartada": o valor vem daqui, gerado, ou vem por `\ref{}`.
# Macro nova se acrescenta AQUI, com teste — nunca no `artigo.tex`.
#
# Por que `\newcommand` e não `\providecommand`: um nome já definido tem de
# derrubar a compilação. `\providecommand` silenciaria a colisão e o artigo
# imprimiria o valor de outra pessoa.


def _macro(nome: str, valor: str, comentario: str = "") -> str:
    linha = f"\\newcommand{{\\{nome}}}{{{valor}}}"
    return f"{linha:<52}% {comentario}" if comentario else linha


# ---------------------------------------------------------------------------
# Macros da frente A: proveniência, dedup, PSL e prevalência (T24)
# ---------------------------------------------------------------------------
#
# Cada gerador devolve a lista de linhas `\newcommand` (vazia quando o dado não
# existe, como num `results/` anterior à reexecução consolidada) e nunca falha
# por ausência: macro sem fonte simplesmente não sai.

# Sufixo de macro por dataset: o nome de comando só aceita letras, e as chaves
# `kaggle03` e `sid321axn` têm dígitos.
SUFIXO_MACRO_CORPUS = {
    "grambeddings": "Grambeddings",
    "kaggle03": "KaggleTres",
    "phishstorm": "Phishstorm",
    "iscx2016": "Iscx",
    "pdrcnn": "Pdrcnn",
    "sid321axn": "KaggleUm",
}
ORDEM_TODOS_CORPORA = (*ORDEM_CORPORA, "pdrcnn", "sid321axn")

# (campo de `dedup`, prefixo da macro)
CAMPOS_DEDUP = (
    ("n_antes", "dedupAntes"),
    ("n_duplicatas_removidas", "dedupDuplicatas"),
    ("n_conflito_urls", "dedupConflitoUrls"),
    ("n_conflito_linhas", "dedupConflitoLinhas"),
    ("n_sem_url_norm", "dedupSemUrlNorm"),
    ("n_final", "dedupFinal"),
    ("n_repetidas_treino_teste", "dedupRepetidasTreinoTeste"),
)

# (campo de `load_stats_by_split`, prefixo da macro): linhas descartadas por motivo.
MOTIVOS_DESCARTE = (
    ("dropped_bad_label", "descarteRotuloInvalido"),
    ("dropped_empty_url", "descarteUrlVazia"),
    ("dropped_unparseable", "descarteUrlInvalida"),
)

PARTICOES_PREVALENCIA = (
    ("train", "Treino"), ("val", "Validacao"), ("test", "Teste"),
)


def _corpora_na_ordem(chaves) -> list[str]:
    """Os datasets presentes em `chaves`, na ordem fixa de apresentação."""
    return [c for c in ORDEM_TODOS_CORPORA if c in chaves]


def macros_dedup(dados: dict) -> list[str]:
    """Contagens de dedup por dataset e regime (DEDUP-04), de `splits.json`."""
    splits_json = dados.get("splits") or {}
    macros = []
    for corpus in _corpora_na_ordem({k.split("/")[0] for k in splits_json}):
        for regime in (REGIME_A, REGIME_B):
            contagens = (splits_json.get(f"{corpus}/{regime}") or {}).get("dedup")
            if not contagens:
                continue
            sufixo = f"{SUFIXO_MACRO_CORPUS[corpus]}Regime{SUFIXO_MACRO_REGIME[regime]}"
            macros += [
                _macro(f"{prefixo}{sufixo}", _milhar(contagens[campo]))
                for campo, prefixo in CAMPOS_DEDUP
                if campo in contagens
            ]
    return macros


def macros_descartes(dados: dict) -> list[str]:
    """Linhas descartadas por motivo e por dataset (PROV-04), de `manifest.json`.

    Soma as partições de `load_stats_by_split`; `descarteTotal` é a soma dos
    três motivos.
    """
    manifest = dados.get("manifest") or {}
    macros = []
    for corpus in _corpora_na_ordem(manifest):
        por_particao = manifest[corpus].get("load_stats_by_split")
        if not por_particao:
            continue
        sufixo = SUFIXO_MACRO_CORPUS[corpus]
        total = 0
        for campo, prefixo in MOTIVOS_DESCARTE:
            soma = sum(int(p.get(campo, 0)) for p in por_particao.values())
            total += soma
            macros.append(_macro(f"{prefixo}{sufixo}", _milhar(soma)))
        macros.append(_macro(f"descarteTotal{sufixo}", _milhar(total)))
    return macros


def macros_psl(dados: dict) -> list[str]:
    """Efeito da seção privada da PSL no agrupamento (PSL-04, REEX-13)."""
    psl_json = (dados.get("diagnostics") or {}).get("psl_privada_contra_icann") or {}
    macros = []
    for corpus in _corpora_na_ordem(psl_json):
        sufixo = SUFIXO_MACRO_CORPUS[corpus]
        d = psl_json[corpus]
        macros += [
            _macro(f"pslUrls{sufixo}", _milhar(d["n_urls"])),
            _macro(f"pslUrlsMudam{sufixo}", _milhar(d["n_urls_mudam"]),
                   "URLs cujo eTLD+1 muda com a secao privada"),
            _macro(f"pslGruposIcannAfetados{sufixo}", _milhar(d["n_grupos_icann_afetados"])),
            _macro(f"pslGruposResultantes{sufixo}", _milhar(d["n_grupos_resultantes"])),
        ]
    return macros


def macros_prevalencia(dados: dict) -> list[str]:
    """Domínios com as duas classes e desvio de prevalência (DEDUP-05, REEX-08).

    A prevalência e o desvio saem em pontos percentuais (a fração vezes 100);
    o desvio é o da partição menos o do dataset deduplicado.
    """
    diag = dados.get("diagnostics") or {}
    macros = []
    duas = diag.get("dominios_duas_classes") or {}
    for corpus in _corpora_na_ordem(duas):
        sufixo = SUFIXO_MACRO_CORPUS[corpus]
        macros += [
            _macro(f"dominiosDuasClassesBruto{sufixo}",
                   _milhar(duas[corpus]["bruto"]["n_dominios_duas_classes"])),
            _macro(f"dominiosDuasClassesDedup{sufixo}",
                   _milhar(duas[corpus]["deduplicado_etld1"]["n_dominios_duas_classes"])),
        ]
    desvio = diag.get("desvio_prevalencia") or {}
    for regime in (REGIME_A, REGIME_B):
        por_corpus = desvio.get(regime) or {}
        for corpus in _corpora_na_ordem(por_corpus):
            d = por_corpus[corpus]
            sufixo = f"{SUFIXO_MACRO_CORPUS[corpus]}Regime{SUFIXO_MACRO_REGIME[regime]}"
            macros.append(
                _macro(f"prevalenciaDataset{sufixo}", _num(d["prevalencia_dataset"] * 100, 2))
            )
            for chave, nome in PARTICOES_PREVALENCIA:
                desv = d[chave]["desvio"]
                macros.append(
                    _macro(f"prevalenciaDesvio{nome}{sufixo}",
                           _num(None if desv is None else desv * 100, 2))
                )
    return macros


# ---------------------------------------------------------------------------
# Sensibilidade sem esquema, repetição da semente 42 e HPO fora da diagonal (T25)
# ---------------------------------------------------------------------------

# Sufixos só de letras, pela mesma razão de `SUFIXO_MACRO_CORPUS`.
SUFIXO_MACRO_SEMENTE = {42: "QuarentaEDois", 7: "Sete", 13: "Treze"}
SUFIXO_MACRO_METRICA = {
    "auroc": "Auroc", "auc_pr": "Aucpr", "precision": "Precisao",
    "recall": "Revocacao", "f1": "FUm", "fpr": "Tfp", "mcc": "Mcc",
    "brier": "Brier", "ece": "Ece",
}
ROTULO_METRICA = {
    "auroc": "AUROC", "auc_pr": "AUC-PR", "precision": "Precisão",
    "recall": "Revocação", "f1": "F1", "fpr": "TFP", "mcc": "MCC",
    "brier": "Brier", "ece": "ECE",
}
ROTULO_REGIME_CURTO = {REGIME_A: "Regime A", REGIME_B: "Regime B"}
SUFIXO_MACRO_GRUPO = {"diagonal": "Diagonal", "fora_diagonal": "ForaDiagonal"}
ROTULO_GRUPO = {"diagonal": "diagonal", "fora_diagonal": "fora da diagonal"}


def _nd(valor, casas: int = 4) -> str:
    """Como `_num`, mas valor indefinido sai "n/d" (não disponível)."""
    return "n/d" if valor is None else _num(valor, casas)


def _sensibilidade(dados: dict) -> dict | None:
    """Diferenças da rodada sem esquema contra a principal; `None` sem a rodada."""
    sens = dados.get("sensibilidade_por_seed") or {}
    if not sens:
        return None
    sementes = [s for s in sens if s in dados["matriz_por_seed"]]
    if not sementes:
        return None
    try:
        return analises.diferencas_sensibilidade(
            {s: dados["matriz_por_seed"][s] for s in sementes},
            {s: sens[s]["matriz"] for s in sementes},
        )
    except analises.AnalisesError as exc:
        raise ReportError(f"sensibilidade sem esquema: {exc}") from exc


def macros_sensibilidade(dados: dict) -> list[str]:
    """AUROC média por regime nas duas rodadas e diferença (SENS-02, REEX-05).

    Por regime e grupo (diagonal, fora da diagonal): a média das sementes da
    rodada principal e da sem esquema, a diferença média (sem esquema menos
    principal) e a diferença de cada semente.
    """
    resultado = _sensibilidade(dados)
    if resultado is None:
        return []
    macros = []
    for regime in (REGIME_A, REGIME_B):
        if regime not in resultado:
            continue
        r = SUFIXO_MACRO_REGIME[regime]
        for grupo in analises.GRUPOS:
            g = SUFIXO_MACRO_GRUPO[grupo]
            media = resultado[regime]["media"][grupo]
            macros += [
                _macro(f"semEsquemaAurocPrincipal{g}Regime{r}", _num(media["principal"])),
                _macro(f"semEsquemaAuroc{g}Regime{r}", _num(media["sem_esquema"])),
                _macro(f"semEsquemaDif{g}Regime{r}", _num(media["diferenca"])),
            ]
            for semente, por_grupo in resultado[regime]["por_semente"].items():
                macros.append(
                    _macro(f"semEsquemaDif{g}Regime{r}Semente{SUFIXO_MACRO_SEMENTE[semente]}",
                           _num(por_grupo[grupo]["diferenca"]))
                )
    return macros


def tabela11_sensibilidade(dados: dict) -> str | None:
    """Tabela 11: AUROC média sem o esquema e diferença para a principal (SENS-02).

    `None` quando a rodada de sensibilidade não existe. A coluna da diferença
    traz a média das sementes e, havendo mais de uma, a faixa entre elas.
    """
    resultado = _sensibilidade(dados)
    if resultado is None:
        return None
    linhas = []
    n_sementes = 0
    for regime in (REGIME_A, REGIME_B):
        if regime not in resultado:
            continue
        por_semente = resultado[regime]["por_semente"]
        n_sementes = len(por_semente)
        for grupo in analises.GRUPOS:
            media = resultado[regime]["media"][grupo]
            faixa = _faixa([p[grupo]["diferenca"] for p in por_semente.values()], 4, ";~")
            linhas.append(
                f"{ROTULO_REGIME_CURTO[regime]}, {ROTULO_GRUPO[grupo]} & "
                f"{_num(media['principal'])} & {_num(media['sem_esquema'])} & "
                f"{_num(media['diferenca'])} {faixa}".rstrip() + " \\\\"
            )
    complemento = ", com a faixa entre as sementes" if n_sementes > 1 else ""
    return _cabecalho(
        "results/matrix.json + results/sensibilidade_sem_esquema/"
    ) + rf"""
\begin{{table}}[htb]
\centering
\caption{{AUROC média da rodada principal e da rodada sem o esquema nas características de
texto inteiro, por regime, e a diferença (sem esquema menos principal){complemento}.}}
\label{{tab:sensibilidade}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrr}}
\toprule
Regime e células & Principal & Sem esquema & Diferença \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte("matrix.json", "sensibilidade_sem_esquema/matrix.json")}
\end{{table}}
"""


def _repeticao(dados: dict) -> dict | None:
    """Maior diferença por métrica entre a semente 42 e a repetição; `None` sem ela."""
    repeticao = dados.get("repeticao")
    if not repeticao:
        return None
    principal = (
        dados["matriz_por_seed"][SEED_CANONICA] + dados["loco_por_seed"][SEED_CANONICA]
    )
    try:
        return analises.diferencas_repeticao(
            principal, repeticao["matriz"] + repeticao["loco"]
        )
    except analises.AnalisesError as exc:
        raise ReportError(f"repetição da semente 42: {exc}") from exc


def macros_repeticao(dados: dict) -> list[str]:
    """Maior diferença absoluta por métrica, e onde ocorre (REP-01, REEX-12)."""
    resultado = _repeticao(dados)
    if resultado is None:
        return []
    macros = [_macro("repeticaoCelulas", _milhar(resultado["n_celulas"]),
                     "celulas da matriz e do LOCO comparadas")]
    for metrica, sufixo in SUFIXO_MACRO_METRICA.items():
        d = resultado["metricas"].get(metrica)
        if d is None:
            macros.append(_macro(f"repeticaoMax{sufixo}", "n/d"))
            continue
        macros += [
            _macro(f"repeticaoMax{sufixo}", _num(d["diferenca_max"], 5)),
            _macro(f"repeticaoModelo{sufixo}", _rotulo(d["model"], ROTULO_MODELO)),
            _macro(f"repeticaoRegime{sufixo}", ROTULO_REGIME_CURTO[d["regime"]]),
            _macro(f"repeticaoTreino{sufixo}", _rotulo_treino(d["train_corpus"])),
            _macro(f"repeticaoTeste{sufixo}", _rotulo(d["test_corpus"], ROTULO_CORPUS)),
        ]
    return macros


def tabela12_repeticao(dados: dict) -> str | None:
    """Tabela 12: maior diferença por métrica entre as duas execuções (REP-01)."""
    resultado = _repeticao(dados)
    if resultado is None:
        return None
    linhas = []
    for metrica, rotulo in ROTULO_METRICA.items():
        d = resultado["metricas"].get(metrica)
        if d is None:
            linhas.append(f"{rotulo} & n/d & n/d & n/d & n/d \\\\")
            continue
        celula = (
            f"{_rotulo_treino(d['train_corpus'])} $\\to$ "
            f"{_rotulo(d['test_corpus'], ROTULO_CORPUS)}"
        )
        linhas.append(
            f"{rotulo} & {_num(d['diferenca_max'], 5)} & "
            f"{_rotulo(d['model'], ROTULO_MODELO)} & "
            f"{ROTULO_REGIME_CURTO[d['regime']]} & {celula} \\\\"
        )
    return _cabecalho(
        "results/matrix.json + results/loco.json + results/repeticao_seed_42/"
    ) + rf"""
\begin{{table}}[htb]
\centering
\caption{{Maior diferença absoluta por métrica entre as duas execuções da semente 42, sobre as
células da matriz e do \emph{{leave-one-corpus-out}}, com o modelo e a célula em que ocorre.}}
\label{{tab:repeticao}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrlll}}
\toprule
Métrica & Maior diferença & Modelo & Regime & Treino $\to$ teste \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte("matrix.json", "loco.json", "repeticao_seed_42/matrix.json")}
\end{{table}}
"""


def _deltas_hpo_fora(dados: dict) -> dict | None:
    """`{modelo: {dataset de teste: AUROC ajustada menos AUROC seca}}` (HPO-02).

    `None` sem `hpo_fora_diagonal.json`. Diferença indefinida (AUROC `None` em
    uma das duas configurações) fica `None`.
    """
    hpo = dados.get("hpo_fora_diagonal")
    if not hpo:
        return None
    saida: dict = {}
    for modelo in [m for m in ROTULO_MODELO if m in hpo["modelos"]]:
        auroc: dict = {}
        for a in hpo["modelos"][modelo]["avaliacoes"]:
            auroc[(a["configuracao"], a["test_corpus"])] = a["metrics"].get("auroc")
        saida[modelo] = {}
        for corpus in ORDEM_CORPORA:
            seco, ajustado = auroc.get(("seco", corpus)), auroc.get(("ajustado", corpus))
            saida[modelo][corpus] = (
                None if seco is None or ajustado is None else ajustado - seco
            )
    return saida


def macros_hpo_fora(dados: dict) -> list[str]:
    """Diferença de AUROC entre a configuração ajustada e a fixa, por modelo e teste."""
    deltas = _deltas_hpo_fora(dados)
    if deltas is None:
        return []
    macros = []
    for modelo, por_corpus in deltas.items():
        for corpus, delta in por_corpus.items():
            macros.append(
                _macro(f"hpoForaDelta{SUFIXO_MACRO_MODELO[modelo]}"
                       f"{SUFIXO_MACRO_CORPUS[corpus]}", _nd(delta, 5))
            )
    todos = [d for por_corpus in deltas.values() for d in por_corpus.values() if d is not None]
    if todos:
        macros += [
            _macro("hpoForaDeltaMax", _num(max(todos), 5), "o maior ganho fora da diagonal"),
            _macro("hpoForaDeltaMin", _num(min(todos), 5), "a maior perda fora da diagonal"),
        ]
    return macros


def tabela13_hpo_fora(dados: dict) -> str | None:
    """Tabela 13: ganho de AUROC da configuração ajustada, por modelo e teste (HPO-02)."""
    deltas = _deltas_hpo_fora(dados)
    if deltas is None:
        return None
    linhas = [
        f"{_rotulo(modelo, ROTULO_MODELO)} & "
        + " & ".join(_nd(por_corpus[c]) for c in ORDEM_CORPORA)
        + " \\\\"
        for modelo, por_corpus in deltas.items()
    ]
    colunas = " & ".join(_rotulo(c, ROTULO_CORPUS) for c in ORDEM_CORPORA)
    return _cabecalho("results/hpo_fora_diagonal.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Diferença de AUROC entre a configuração ajustada e a fixa de cada modelo, treinado no
Grambeddings (Regime B, semente 42) e avaliado no teste de cada \emph{{dataset}}.}}
\label{{tab:hpo-fora}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Modelo & {colunas} \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte("hpo_fora_diagonal.json")}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# Filtragem da sobreposição e bootstrap pareado, de `offline.json` (T26)
# ---------------------------------------------------------------------------


def _tipo_de_celula(treino: str, teste: str) -> str:
    """`diagonal`, `loco` (treino na união) ou `fora_diagonal`."""
    if treino.startswith("uniao_sem_"):
        return "loco"
    return "diagonal" if treino == teste else "fora_diagonal"


def _filtragem_da_semente(dados: dict, seed: int) -> list[dict] | None:
    """Registros filtrados da rodada principal numa semente; `None` sem eles."""
    offline = dados.get("offline")
    if not offline:
        return None
    registros = (offline.get("filtragem") or {}).get("principal", {}).get(str(seed))
    return registros or None


def _filtragem_canonica(dados: dict) -> list[dict] | None:
    """Registros filtrados da rodada principal na semente canônica; `None` sem eles."""
    return _filtragem_da_semente(dados, SEED_CANONICA)


def _celulas_filtradas(registros: list[dict]) -> dict:
    """`{(regime, treino, teste): {"fracao", "auroc"}}`: média dos modelos por célula.

    `auroc` é a média das AUROC definidas; sem nenhuma (todo modelo com uma só
    classe depois da filtragem, ou sem linha), a célula fica `None`: o "n/d".
    """
    agrupado: dict = {}
    for r in registros:
        agrupado.setdefault((r["regime"], r["train_corpus"], r["test_corpus"]), []).append(r)
    celulas = {}
    for chave, lista in agrupado.items():
        aurocs = [
            r["metrics"]["auroc"] for r in lista
            if r.get("metrics") and r["metrics"].get("auroc") is not None
        ]
        celulas[chave] = {
            "fracao": _media([r["fracao_removida"] for r in lista]),
            "auroc": _media(aurocs),
        }
    return celulas


def tabela14_filtragem(dados: dict) -> str | None:
    """Tabela 14: linhas removidas pela filtragem e AUROC antes e depois (FILT-01).

    `None` sem `offline.json`. A AUROC original é a da matriz e do LOCO na
    semente 42 (média dos seis modelos); a filtrada vem do `offline.json`.
    Célula sem AUROC definida depois da filtragem sai "n/d".
    """
    registros = _filtragem_canonica(dados)
    if registros is None:
        return None
    filtradas = _celulas_filtradas(registros)
    original = {
        regime: {
            **media_por_celula(dados["matriz_por_seed"][SEED_CANONICA], regime, "auroc"),
            **media_por_celula(dados["loco_por_seed"][SEED_CANONICA], regime, "auroc"),
        }
        for regime in (REGIME_A, REGIME_B)
    }

    def linha(rotulo: str, treino: str, teste: str) -> str:
        partes = []
        for regime in (REGIME_A, REGIME_B):
            f = filtradas.get((regime, treino, teste), {"fracao": None, "auroc": None})
            fracao = None if f["fracao"] is None else f["fracao"] * 100
            partes += [
                _nd(fracao, 2),
                _nd(original[regime].get((treino, teste))),
                _nd(f["auroc"]),
            ]
        return f"{rotulo} & " + " & ".join(partes) + " \\\\"

    linhas_matriz = [
        linha(f"{_rotulo(tr, ROTULO_CORPUS)} $\\to$ {_rotulo(te, ROTULO_CORPUS)}", tr, te)
        for tr in ORDEM_CORPORA
        for te in ORDEM_CORPORA
        if tr != te
    ]
    linhas_loco = [
        linha(f"União $\\to$ {_rotulo(te, ROTULO_CORPUS)}", f"uniao_sem_{te}", te)
        for te in ORDEM_CORPORA
    ]
    colunas = r"Removidas (\%) & Original & Filtrada"
    return _cabecalho("results/offline.json + results/matrix.json + results/loco.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Fração de linhas de teste removidas por compartilhar eTLD+1 com o treino ou a
validação, e AUROC média dos seis modelos antes e depois da remoção, nos dois regimes, na
semente 42. n/d indica AUROC indefinida.}}
\label{{tab:filtragem}}
\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
 & \multicolumn{{3}}{{c}}{{Regime A}} & \multicolumn{{3}}{{c}}{{Regime B}} \\
\cmidrule(lr){{2-4}} \cmidrule(l){{5-7}}
Treino $\to$ teste & {colunas} & {colunas} \\
\midrule
{chr(10).join(linhas_matriz)}
\midrule
{chr(10).join(linhas_loco)}
\bottomrule
\end{{tabular}}
{_fonte("offline.json", "matrix.json", "loco.json")}
\end{{table}}
"""


SEMENTES_EXTRAS = (7, 13)


def _macros_filtragem_do_regime(do_regime: dict, r: str, sufixo_semente: str) -> list[str]:
    """Fração removida e AUROC filtrada, fora da diagonal e no LOCO, de um regime.

    `sufixo_semente` vazio dá os nomes da semente 42; `SementeSete` e
    `SementeTreze` dão os das outras duas. A regra é a mesma nas três (FIL-02).
    """
    macros = []
    for grupo, sufixo in (("fora_diagonal", "ForaDiagonal"), ("loco", "Loco")):
        doc = [v for (_r, tr, te), v in do_regime.items() if _tipo_de_celula(tr, te) == grupo]
        if not doc:
            continue
        fracao = _media([v["fracao"] for v in doc])
        macros += [
            _macro(f"filtragemRemovida{sufixo}Regime{r}{sufixo_semente}",
                   _nd(None if fracao is None else fracao * 100, 2),
                   "pontos percentuais de linhas de teste removidas"),
            _macro(f"filtragemAuroc{sufixo}Regime{r}{sufixo_semente}",
                   _nd(_media([v["auroc"] for v in doc]))),
        ]
    return macros


def _filtragem_das_sementes_extras(dados: dict) -> dict[int, list[dict]]:
    """Registros filtrados das sementes 7 e 13 com rodada carregada (FIL-01, FIL-04).

    Rodada da semente carregada em `matriz_por_seed` sem o bloco da semente no
    `offline.json` é erro: o `offline.json` é de outra execução, ou o
    `make offline` não rodou sobre ela. Sem a rodada carregada, a semente não
    entra e não há erro.
    """
    extras = {}
    for seed in SEMENTES_EXTRAS:
        if seed not in dados.get("matriz_por_seed", {}):
            continue
        registros = _filtragem_da_semente(dados, seed)
        if registros is None:
            raise ReportError(
                f"a rodada da semente {seed} está carregada, mas o offline.json não traz "
                f"a filtragem da semente {seed}. Rode `make offline` sobre estes resultados."
            )
        extras[seed] = registros
    return extras


def macros_filtragem(dados: dict) -> list[str]:
    """Médias da filtragem por regime: linhas removidas, AUROC e células indefinidas.

    A semente 42 sai sem sufixo. As sementes 7 e 13, quando a rodada está
    carregada, saem com `SementeSete` e `SementeTreze`, mais a AUROC original
    do LOCO (a coluna "Original" da `tab:filtragem`, agregada como a filtrada).
    """
    registros = _filtragem_canonica(dados)
    if registros is None:
        return []
    celulas = _celulas_filtradas(registros)
    macros = []
    for regime in (REGIME_A, REGIME_B):
        r = SUFIXO_MACRO_REGIME[regime]
        do_regime = {k: v for k, v in celulas.items() if k[0] == regime}
        macros += _macros_filtragem_do_regime(do_regime, r, "")
        indefinidas = sum(1 for v in do_regime.values() if v["auroc"] is None)
        macros.append(
            _macro(f"filtragemCelulasIndefinidasRegime{r}", str(indefinidas),
                   "celulas sem AUROC definida depois da filtragem")
        )
    extras = _filtragem_das_sementes_extras(dados)
    for seed, registros_seed in extras.items():
        sufixo_semente = f"Semente{SUFIXO_MACRO_SEMENTE[seed]}"
        celulas_seed = _celulas_filtradas(registros_seed)
        for regime in (REGIME_A, REGIME_B):
            r = SUFIXO_MACRO_REGIME[regime]
            do_regime = {k: v for k, v in celulas_seed.items() if k[0] == regime}
            macros += _macros_filtragem_do_regime(do_regime, r, sufixo_semente)
            original_loco = media_por_celula(
                dados.get("loco_por_seed", {}).get(seed) or [], regime, "auroc")
            macros.append(
                _macro(f"locoAurocOriginalRegime{r}{sufixo_semente}",
                       _nd(_media(list(original_loco.values()))),
                       "AUROC original (sem filtragem), media das celulas do LOCO")
            )
    macros += _macros_sementes_eleva_iscx(dados, registros, extras)
    return macros


# Células com teste no ISCX-URL2016: o treino e o nome da macro. O LOCO é "Uniao".
CELULAS_TESTE_ISCX = (
    ("grambeddings", "Grambeddings"),
    ("kaggle03", "KaggleTres"),
    ("phishstorm", "Phishstorm"),
    ("uniao_sem_iscx2016", "Uniao"),
)


def _macros_sementes_eleva_iscx(
    dados: dict, registros_canonica: list[dict], extras: dict[int, list[dict]],
) -> list[str]:
    """Em quantas sementes a AUROC média filtrada fica acima da original (FIL-03).

    Uma macro por célula com teste no ISCX-URL2016 (três da matriz e o LOCO) e
    por regime. A comparação é por semente, célula a célula (L-041): a semente
    com AUROC filtrada indefinida não conta como "acima" e vai no comentário.
    """
    por_semente = {SEED_CANONICA: registros_canonica, **extras}
    macros = []
    filtradas = {s: _celulas_filtradas(reg) for s, reg in sorted(por_semente.items())}
    for regime in (REGIME_A, REGIME_B):
        r = SUFIXO_MACRO_REGIME[regime]
        originais = {
            s: {
                **media_por_celula(dados["matriz_por_seed"][s], regime, "auroc"),
                **media_por_celula((dados.get("loco_por_seed") or {}).get(s) or [],
                                   regime, "auroc"),
            }
            for s in filtradas
        }
        for treino, nome in CELULAS_TESTE_ISCX:
            chave = (treino, "iscx2016")
            acima = indefinidas = 0
            for s, celulas in filtradas.items():
                filtrada = celulas.get((regime, *chave), {}).get("auroc")
                original = originais[s].get(chave)
                if filtrada is None:
                    indefinidas += 1
                elif original is not None and filtrada > original:
                    acima += 1
            comentario = f"de {len(filtradas)} sementes com AUROC filtrada acima da original"
            if indefinidas:
                comentario += f"; {indefinidas} sementes com AUROC indefinida"
            macros.append(_macro(f"filtragemSementesEleva{nome}IscxRegime{r}", str(acima), comentario))
    return macros


def _celulas_pareadas(bootstrap: list[dict]) -> dict:
    """`{(treino, teste): [registros dos modelos]}` do bootstrap pareado."""
    agrupado: dict = {}
    for e in bootstrap:
        agrupado.setdefault((e["train_corpus"], e["test_corpus"]), []).append(e)
    return agrupado


def _ic_exclui_zero(e: dict) -> bool:
    lo, hi = e["ic"]["auroc"]
    return lo > 0 or hi < 0


def tabela15_bootstrap_pareado(dados: dict) -> str | None:
    """Tabela 15: Delta de AUROC entre regimes por célula, com o IC pareado (BOOT-01).

    `None` sem `offline.json` ou sem bootstrap. O Delta é B menos A. Como a
    média de seis intervalos não é o intervalo da média, a tabela traz o Delta
    médio e a faixa entre os modelos, e conta quantos modelos têm IC que exclui
    zero; o IC de cada modelo fica no `offline.json`. Célula sem IC definido em
    nenhum modelo sai "n/d".
    """
    offline = dados.get("offline") or {}
    bootstrap = offline.get("bootstrap_pareado") or []
    if not bootstrap:
        return None
    celulas = _celulas_pareadas(bootstrap)
    nivel = (offline.get("parametros") or {}).get("bootstrap_nivel")
    nivel_texto = "" if nivel is None else rf" de {_num(nivel * 100, 0)}\%"

    def linha(rotulo: str, treino: str, teste: str) -> str:
        lista = celulas.get((treino, teste))
        if not lista:
            return f"{rotulo} & n/d & n/d & n/d & n/d \\\\"
        deltas = [e["delta"]["auroc"] for e in lista if e["delta"]["auroc"] is not None]
        definidos = [e for e in lista if not e["ic_indefinido"] and e["ic"]]
        exclui = (
            f"{sum(_ic_exclui_zero(e) for e in definidos)}/{len(definidos)}"
            if definidos else "n/d"
        )
        faixa = _min_a_max(deltas) if deltas else "n/d"
        return (
            f"{rotulo} & {_milhar(max(e['n_intersecao'] for e in lista))} & "
            f"{_nd(_media(deltas))} & {faixa} & {exclui} \\\\"
        )

    linhas_matriz = [
        linha(f"{_rotulo(tr, ROTULO_CORPUS)} $\\to$ {_rotulo(te, ROTULO_CORPUS)}", tr, te)
        for tr in ORDEM_CORPORA
        for te in ORDEM_CORPORA
    ]
    linhas_loco = [
        linha(f"União $\\to$ {_rotulo(te, ROTULO_CORPUS)}", f"uniao_sem_{te}", te)
        for te in ORDEM_CORPORA
    ]
    return _cabecalho("results/offline.json :: bootstrap_pareado") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Diferença de AUROC entre os regimes (B menos A) por célula, na semente 42: média dos seis
modelos, mínimo e máximo, e quantos modelos têm IC{nivel_texto} pelo \emph{{bootstrap}} pareado que
exclui zero. n/d indica IC indefinido.}}
\label{{tab:bootstrap-pareado}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Treino $\to$ teste & Interseção & Média & Mínimo a máximo & IC exclui zero \\
\midrule
{chr(10).join(linhas_matriz)}
\midrule
{chr(10).join(linhas_loco)}
\bottomrule
\end{{tabular}}
{_fonte("offline.json")}
\end{{table}}
"""


def macros_bootstrap_pareado(dados: dict) -> list[str]:
    """Delta de AUROC médio por grupo de células e contagem dos ICs (BOOT-01)."""
    offline = dados.get("offline") or {}
    bootstrap = offline.get("bootstrap_pareado") or []
    if not bootstrap:
        return []
    celulas = _celulas_pareadas(bootstrap)
    macros = [
        _macro("bootstrapAjustes", str(len(bootstrap)), "pares celula e modelo com Delta"),
        _macro("bootstrapInterseccaoMin", _milhar(min(e["n_intersecao"] for e in bootstrap))),
        _macro("bootstrapInterseccaoMax", _milhar(max(e["n_intersecao"] for e in bootstrap))),
    ]
    definidos = [e for e in bootstrap if not e["ic_indefinido"] and e["ic"]]
    macros += [
        _macro("bootstrapIcIndefinidos", str(len(bootstrap) - len(definidos))),
        _macro("bootstrapIcDefinidos", str(len(definidos))),
        _macro("bootstrapIcExcluiZero", str(sum(_ic_exclui_zero(e) for e in definidos))),
    ]
    # Modelos da diagonal cujo IC pareado exclui zero, por dataset: a mesma regra
    # da coluna "IC exclui zero" da `tab:bootstrap-pareado` (numerador do k/n).
    for corpus in ORDEM_CORPORA:
        lista = celulas.get((corpus, corpus))
        if not lista:
            continue
        com_ic = [e for e in lista if not e["ic_indefinido"] and e["ic"]]
        macros.append(
            _macro(f"bootstrapModelosIcExcluiZeroDiagonal{SUFIXO_MACRO_CORPUS[corpus]}",
                   str(sum(_ic_exclui_zero(e) for e in com_ic)),
                   "modelos (de 6) com IC pareado do Delta AUROC que exclui zero, semente 42")
        )
    parametros = offline.get("parametros") or {}
    if "bootstrap_n" in parametros:
        macros.append(_macro("bootstrapReamostras", _milhar(parametros["bootstrap_n"])))
    for grupo, sufixo in (
        ("diagonal", "Diagonal"), ("fora_diagonal", "ForaDiagonal"), ("loco", "Loco"),
    ):
        por_celula = [
            _media([e["delta"]["auroc"] for e in lista])
            for (tr, te), lista in celulas.items()
            if _tipo_de_celula(tr, te) == grupo
        ]
        macros.append(
            _macro(f"bootstrapDeltaAurocMedio{sufixo}", _nd(_media(por_celula)),
                   "Regime B menos Regime A, media das celulas")
        )
    return macros


# ---------------------------------------------------------------------------
# Análises da F04: variação, união contra fontes, degeneradas e efeito (T30)
# ---------------------------------------------------------------------------
#
# O cálculo mora em `analises.py`; aqui só se carrega, se agrega entre sementes
# e se formata. A semente 42 é a das análises inferenciais (AD-063); a média
# das sementes presentes acompanha, e a afirmação qualitativa vale pela média.

CAMPOS_DECOMPOSICAO = (
    ("treino", "Treino"), ("teste", "Teste"), ("modelo", "Modelo"), ("resto", "Resto"),
)
CAMPOS_UNIAO = (
    ("loco", "Loco"), ("melhor", "Melhor"), ("pior", "Pior"),
    ("media", "Media"), ("media_sem_iscx", "MediaSemIscx"),
)
GRUPOS_EFEITO = (("diagonal", "Diagonal"), ("loco", "Loco"))
ROTULO_SEMENTE = {42: "Semente 42", 7: "Semente 7", 13: "Semente 13"}


def _sementes_presentes(dados: dict) -> list[int]:
    return [s for s in SUBDIRS_SEEDS if s in dados["matriz_por_seed"]]


def _decomposicao(dados: dict) -> dict:
    """`{regime: {"por_semente": {semente: frações}, "media": frações}}`.

    Regime sem AUROC fora da diagonal ou sem variância em alguma semente fica
    de fora: sem macro e sem linha, nunca um valor inventado.
    """
    saida = {}
    for regime in (REGIME_A, REGIME_B):
        por_semente = {}
        try:
            for s in _sementes_presentes(dados):
                por_semente[s] = analises.decompor_variancia(dados["matriz_por_seed"][s], regime)
        except analises.AnalisesError:
            continue
        if not por_semente:
            continue
        saida[regime] = {
            "por_semente": por_semente,
            "media": {c: _media([d[c] for d in por_semente.values()]) for c, _ in CAMPOS_DECOMPOSICAO},
        }
    return saida


def macros_decomposicao(dados: dict) -> list[str]:
    """Fração da soma de quadrados da AUROC explicada por fator (ANL-01).

    Por regime: a média das sementes (`...Media`) e a de cada semente, em
    pontos percentuais.
    """
    macros = []
    for regime, d in _decomposicao(dados).items():
        r = SUFIXO_MACRO_REGIME[regime]
        for campo, nome in CAMPOS_DECOMPOSICAO:
            macros.append(_macro(f"decomp{nome}Regime{r}Media", _pct(100 * d["media"][campo]),
                                 "pontos percentuais, media das sementes"))
            for semente, frac in d["por_semente"].items():
                macros.append(
                    _macro(f"decomp{nome}Regime{r}Semente{SUFIXO_MACRO_SEMENTE[semente]}",
                           _pct(100 * frac[campo]))
                )
        macros += [
            _macro(f"decompValoresRegime{r}", str(d["por_semente"][next(iter(d["por_semente"]))]["n"]),
                   "AUROC fora da diagonal por regime e semente"),
            _macro(f"decompSementesRegime{r}", str(len(d["por_semente"]))),
        ]
    return macros


def tabela16_decomposicao(dados: dict) -> str | None:
    """Tabela 16: fração da variação da AUROC fora da diagonal por fator (ANL-01)."""
    decomposicao = _decomposicao(dados)
    if not decomposicao:
        return None
    linhas = []
    n_sementes = 0
    for regime, d in decomposicao.items():
        n_sementes = max(n_sementes, len(d["por_semente"]))
        linhas.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{ROTULO_REGIME_CURTO[regime]}}}}} \\")
        for semente, frac in d["por_semente"].items():
            linhas.append(
                f"{ROTULO_SEMENTE[semente]} & "
                + " & ".join(_pct(100 * frac[c]) for c, _ in CAMPOS_DECOMPOSICAO)
                + f" & {frac['n']} \\\\"
            )
        if len(d["por_semente"]) > 1:
            n_valores = d["por_semente"][next(iter(d["por_semente"]))]["n"]
            linhas.append(
                "Média das sementes & "
                + " & ".join(_pct(100 * d["media"][c]) for c, _ in CAMPOS_DECOMPOSICAO)
                + f" & {n_valores} \\\\"
            )
    media = ", por semente e na média" if n_sementes > 1 else ""
    return _cabecalho("results/matrix.json (e seed_*/matrix.json)") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Fração (\%) da soma de quadrados da AUROC fora da diagonal explicada pelo treino, pelo
teste e pelo modelo{media}. O resto é um menos a soma (frações marginais, pode ser negativo).}}
\label{{tab:variacao}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
 & Treino & Teste & Modelo & Resto & AUROC (n) \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte("matrix.json")}
\end{{table}}
"""


def _uniao_planos(dados: dict) -> dict:
    """`{"semente": {regime: {teste: {campo: AUROC}}}, "media": {...}}` da união contra fontes.

    `semente` é a canônica (42) e `media` a das sementes presentes com LOCO.
    Vazio sem LOCO na semente canônica.
    """
    por_semente = {}
    for s in _sementes_presentes(dados):
        loco = dados["loco_por_seed"].get(s)
        if not loco:
            continue
        por_semente[s] = {
            regime: analises.uniao_contra_fontes(dados["matriz_por_seed"][s], loco, regime)
            for regime in (REGIME_A, REGIME_B)
        }
    if SEED_CANONICA not in por_semente:
        return {}

    def valor(entrada, campo):
        v = entrada[campo]
        return v["auroc"] if isinstance(v, dict) else v

    planos = {"semente": {}, "media": {}, "fontes": {}}
    for regime in (REGIME_A, REGIME_B):
        testes = sorted({t for s in por_semente.values() for t in s[regime]})
        planos["semente"][regime] = {
            t: {c: valor(e, c) for c, _ in CAMPOS_UNIAO}
            for t, e in por_semente[SEED_CANONICA][regime].items()
        }
        planos["fontes"][regime] = {
            t: (e["melhor"]["treino"], e["pior"]["treino"])
            for t, e in por_semente[SEED_CANONICA][regime].items()
        }
        planos["media"][regime] = {
            t: {
                c: _media([valor(s[regime][t], c) for s in por_semente.values() if t in s[regime]])
                for c, _ in CAMPOS_UNIAO
            }
            for t in testes
        }
    return planos


def _abaixo(plano: dict, campo: str) -> int:
    """Em quantos datasets de teste o LOCO fica abaixo da referência `campo`."""
    return sum(
        1 for v in plano.values()
        if v["loco"] is not None and v[campo] is not None and v["loco"] < v[campo]
    )


def macros_uniao(dados: dict) -> list[str]:
    """AUROC do LOCO contra a melhor, a pior e a média das fontes isoladas (ANL-02).

    Por regime e dataset de teste, na semente 42 e na média das sementes
    (`...MediaSementes`), mais quantos datasets têm o LOCO abaixo de cada
    referência e quais fontes são a melhor e a pior.
    """
    planos = _uniao_planos(dados)
    if not planos:
        return []
    macros = []
    for regime in (REGIME_A, REGIME_B):
        r = SUFIXO_MACRO_REGIME[regime]
        for sufixo, plano in (("", planos["semente"][regime]), ("MediaSementes", planos["media"][regime])):
            for teste in ORDEM_CORPORA:
                if teste not in plano:
                    continue
                c = SUFIXO_MACRO_CORPUS[teste]
                for campo, nome in CAMPOS_UNIAO:
                    macros.append(_macro(f"uniao{nome}{c}Regime{r}{sufixo}", _nd(plano[teste][campo])))
            macros.append(_macro(f"uniaoDatasetsRegime{r}{sufixo}", str(len(plano))))
            for campo, nome in CAMPOS_UNIAO[1:]:
                macros.append(_macro(f"uniaoAbaixo{nome}Regime{r}{sufixo}", str(_abaixo(plano, campo)),
                                     "datasets de teste com o LOCO abaixo da referencia"))
        for teste in ORDEM_CORPORA:
            if teste not in planos["fontes"][regime]:
                continue
            melhor, pior = planos["fontes"][regime][teste]
            c = SUFIXO_MACRO_CORPUS[teste]
            macros += [
                _macro(f"uniaoMelhorFonte{c}Regime{r}", _rotulo(melhor, ROTULO_CORPUS)),
                _macro(f"uniaoPiorFonte{c}Regime{r}", _rotulo(pior, ROTULO_CORPUS)),
            ]
    return macros


def tabela17_uniao(dados: dict) -> str | None:
    """Tabela 17: AUROC do LOCO contra as fontes isoladas, na semente 42 (ANL-02)."""
    planos = _uniao_planos(dados)
    if not planos:
        return None
    linhas = []
    for regime in (REGIME_A, REGIME_B):
        linhas.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{ROTULO_REGIME_CURTO[regime]}}}}} \\")
        for teste in ORDEM_CORPORA:
            plano = planos["semente"][regime].get(teste)
            if plano is None:
                continue
            linhas.append(
                f"{_rotulo(teste, ROTULO_CORPUS)} & "
                + " & ".join(_nd(plano[c]) for c, _ in CAMPOS_UNIAO)
                + " \\\\"
            )
    return _cabecalho("results/matrix.json + results/loco.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{AUROC do \emph{{leave-one-corpus-out}} (LOCO) e das fontes isoladas por dataset de teste,
nos dois regimes, na semente 42: melhor, pior e média das fontes, e média sem o ISCX-URL2016.
Média dos seis modelos.}}
\label{{tab:uniao-fontes}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Dataset de teste & LOCO & Melhor fonte & Pior fonte & Média das fontes & Média sem ISCX \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte("matrix.json", "loco.json")}
\end{{table}}
"""


def _composicao(dados: dict) -> dict:
    """Composição da união da semente canônica; vazia sem `splits.json` ou sem `n_train`."""
    splits = {k: v for k, v in (dados.get("splits") or {}).items() if "n_train" in v}
    return analises.composicao_uniao(splits) if splits else {}


def macros_composicao(dados: dict) -> list[str]:
    """Composição da união do LOCO por dataset de origem, depois da dedup (ANL-02)."""
    macros = []
    for regime, retidos in _composicao(dados).items():
        if regime not in SUFIXO_MACRO_REGIME:
            continue
        r = SUFIXO_MACRO_REGIME[regime]
        for retido in _corpora_na_ordem(retidos):
            c = SUFIXO_MACRO_CORPUS.get(retido)
            if c is None:
                continue
            macros.append(_macro(f"composicaoTotalRetido{c}Regime{r}",
                                 _milhar(retidos[retido]["total"]),
                                 "linhas de treino da uniao"))
            for fonte in _corpora_na_ordem(retidos[retido]["fontes"]):
                f = SUFIXO_MACRO_CORPUS.get(fonte)
                if f is None:
                    continue
                fracao = retidos[retido]["fontes"][fonte]["fracao"]
                macros.append(
                    _macro(f"composicaoRetido{c}Fonte{f}Regime{r}",
                           _nd(None if fracao is None else fracao * 100, 2),
                           "pontos percentuais das linhas de treino da uniao")
                )
    return macros


def tabela18_composicao(dados: dict) -> str | None:
    """Tabela 18: composição da união do LOCO por dataset de origem (ANL-02)."""
    composicao = _composicao(dados)
    if not composicao:
        return None
    linhas = []
    for regime in (REGIME_A, REGIME_B):
        retidos = composicao.get(regime)
        if not retidos:
            continue
        linhas.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{ROTULO_REGIME_CURTO[regime]}}}}} \\")
        for retido in _corpora_na_ordem(retidos):
            fontes = retidos[retido]["fontes"]
            celulas = []
            for origem in ORDEM_CORPORA:
                if origem == retido:
                    celulas.append("retido")
                elif origem in fontes and fontes[origem]["fracao"] is not None:
                    celulas.append(_num(fontes[origem]["fracao"] * 100, 2))
                else:
                    celulas.append("n/d")
            linhas.append(
                f"União sem {_rotulo(retido, ROTULO_CORPUS)} & "
                f"{_milhar(retidos[retido]['total'])} & " + " & ".join(celulas) + " \\\\"
            )
    cabecalho = " & ".join(_rotulo(c, ROTULO_CORPUS) for c in ORDEM_CORPORA)
    return _cabecalho("results/splits.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Composição do treino da união de cada LOCO, depois da deduplicação, nos dois regimes:
linhas de treino e fração (\%) de cada dataset de origem.}}
\label{{tab:composicao-uniao}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
União & Linhas & {cabecalho} \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte("splits.json")}
\end{{table}}
"""


def _degeneradas_do_grupo(registros: list, regime: str) -> tuple[list, int]:
    """Ajustes degenerados e total de ajustes com taxa definida, de um regime."""
    do_regime = [r for r in registros if r["regime"] == regime and r["train_corpus"] != r["test_corpus"]]
    total = sum(1 for r in do_regime if analises.taxa_predicao_positiva(r) is not None)
    return analises.celulas_degeneradas(do_regime), total


def macros_degeneradas(dados: dict) -> list[str]:
    """Ajustes e células degenerados fora da diagonal e no LOCO, por semente (ANL-04).

    Os macros sem sufixo de semente são os da semente 42. Um ajuste é
    degenerado com taxa de predição positiva abaixo de 0,05 ou acima de 0,95.
    """
    macros = []
    for semente in _sementes_presentes(dados):
        sufixo = "" if semente == SEED_CANONICA else f"Semente{SUFIXO_MACRO_SEMENTE[semente]}"
        grupos = (
            ("ForaDiagonal", dados["matriz_por_seed"][semente]),
            ("Loco", dados["loco_por_seed"].get(semente) or []),
        )
        for regime in (REGIME_A, REGIME_B):
            r = SUFIXO_MACRO_REGIME[regime]
            for nome, registros in grupos:
                lista, total = _degeneradas_do_grupo(registros, regime)
                if not total:
                    continue
                celulas = {(d["train_corpus"], d["test_corpus"]) for d in lista}
                macros += [
                    _macro(f"degenerados{nome}Regime{r}{sufixo}", str(len(lista)),
                           "ajustes com taxa de predicao positiva fora de 0,05 a 0,95"),
                    _macro(f"degeneradosAjustes{nome}Regime{r}{sufixo}", str(total)),
                    _macro(f"degeneradasCelulas{nome}Regime{r}{sufixo}", str(len(celulas)),
                           "celulas com ao menos um ajuste degenerado"),
                ]
    return macros


def tabela19_degeneradas(dados: dict) -> str | None:
    """Tabela 19: F1, MCC e taxa de predição positiva por célula fora da diagonal (ANL-04)."""
    matriz = dados["matriz_por_seed"][SEED_CANONICA]
    loco = dados["loco_por_seed"].get(SEED_CANONICA) or []
    if not any(analises.taxa_predicao_positiva(r) is not None for r in matriz):
        return None

    def taxas(registros, regime):
        acumulado: dict = {}
        for r in registros:
            taxa = analises.taxa_predicao_positiva(r)
            if r["regime"] == regime and taxa is not None:
                acumulado.setdefault((r["train_corpus"], r["test_corpus"]), []).append(taxa)
        return {k: _media(v) for k, v in acumulado.items()}

    def por_regime(registros, regime):
        lista, _total = _degeneradas_do_grupo(registros, regime)
        contagem: dict = {}
        for d in lista:
            chave = (d["train_corpus"], d["test_corpus"])
            contagem[chave] = contagem.get(chave, 0) + 1
        modelos: dict = {}
        for r in registros:
            if r["regime"] == regime:
                chave = (r["train_corpus"], r["test_corpus"])
                modelos[chave] = modelos.get(chave, 0) + 1
        return {
            "f1": media_por_celula(registros, regime, "f1"),
            "mcc": media_por_celula(registros, regime, "mcc"),
            "taxa": taxas(registros, regime),
            "deg": contagem,
            "n": modelos,
        }

    calculado = {
        (grupo, regime): por_regime(registros, regime)
        for grupo, registros in (("matriz", matriz), ("loco", loco))
        for regime in (REGIME_A, REGIME_B)
    }

    def linha(rotulo, grupo, chave):
        partes = []
        for regime in (REGIME_A, REGIME_B):
            c = calculado[(grupo, regime)]
            n = c["n"].get(chave)
            partes += [
                _nd(c["f1"].get(chave)), _nd(c["mcc"].get(chave)), _nd(c["taxa"].get(chave)),
                "n/d" if not n else f"{c['deg'].get(chave, 0)}/{n}",
            ]
        return f"{rotulo} & " + " & ".join(partes) + " \\\\"

    linhas_matriz = [
        linha(f"{_rotulo(tr, ROTULO_CORPUS)} $\\to$ {_rotulo(te, ROTULO_CORPUS)}", "matriz", (tr, te))
        for tr in ORDEM_CORPORA for te in ORDEM_CORPORA if tr != te
    ]
    linhas_loco = [
        linha(f"União $\\to$ {_rotulo(te, ROTULO_CORPUS)}", "loco", (f"uniao_sem_{te}", te))
        for te in ORDEM_CORPORA
    ] if loco else []
    colunas = "F1 & MCC & Taxa & Degen."
    meio = "\\midrule\n" + "\n".join(linhas_loco) + "\n" if linhas_loco else ""
    return _cabecalho("results/matrix.json + results/loco.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{F1, MCC e taxa de predição positiva por célula fora da diagonal e do LOCO, nos dois
regimes, na semente 42 (média dos seis modelos). Degen. conta os modelos com taxa abaixo de 0,05
ou acima de 0,95.}}
\label{{tab:degeneradas}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
 & \multicolumn{{4}}{{c}}{{Regime A}} & \multicolumn{{4}}{{c}}{{Regime B}} \\
\cmidrule(lr){{2-5}} \cmidrule(l){{6-9}}
Treino $\to$ teste & {colunas} & {colunas} \\
\midrule
{chr(10).join(linhas_matriz)}
{meio}\bottomrule
\end{{tabular}}
{_fonte("matrix.json", "loco.json")}
\end{{table}}
"""


def _efeito(dados: dict) -> dict | None:
    matriz = dados["matriz_por_seed"][SEED_CANONICA]
    loco = dados["loco_por_seed"].get(SEED_CANONICA)
    if not loco:
        return None
    return analises.metricas_efeito(dados.get("offline") or {}, matriz, loco)


def macros_efeito(dados: dict) -> list[str]:
    """Erro de ordenação por regime, razão e Delta F1 com IC por modelo (ANL-06).

    Na semente 42, por célula da diagonal e do LOCO e na média de cada grupo.
    """
    efeito = _efeito(dados)
    if efeito is None:
        return []
    macros = []
    for grupo, g in GRUPOS_EFEITO:
        celulas = efeito[grupo]
        for teste in ORDEM_CORPORA:
            e = celulas.get(teste)
            if e is None:
                continue
            c = SUFIXO_MACRO_CORPUS[teste]
            macros += [
                _macro(f"efeitoErroA{g}{c}", _nd(e["erro_a"]), "1 menos AUROC no Regime A"),
                _macro(f"efeitoErroB{g}{c}", _nd(e["erro_b"]), "1 menos AUROC no Regime B"),
                _macro(f"efeitoRazao{g}{c}", _nd(e["razao"], 2), "erro do Regime B sobre o do A"),
            ]
            if e["por_modelo"]:
                macros += [
                    _macro(f"efeitoDeltaFUm{g}{c}", _nd(e["delta_f1"]), "Regime B menos A, media dos modelos"),
                    _macro(f"efeitoModelosIc{g}{c}", str(e["n_modelos_ic"])),
                    _macro(f"efeitoModelosIcExcluiZero{g}{c}", str(e["n_modelos_ic_exclui_zero"])),
                ]
        cheias = [e for e in celulas.values()]
        erro_a = _media([e["erro_a"] for e in cheias])
        erro_b = _media([e["erro_b"] for e in cheias])
        macros += [
            _macro(f"efeitoErroA{g}Media", _nd(erro_a)),
            _macro(f"efeitoErroB{g}Media", _nd(erro_b)),
            _macro(f"efeitoRazao{g}Media", _nd(_media([e["razao"] for e in cheias]), 2),
                   "media das razoes das celulas"),
            _macro(f"efeitoRazaoDasMedias{g}", _nd(None if not erro_a or erro_b is None else erro_b / erro_a, 2),
                   "erro medio do Regime B sobre o do A"),
            _macro(f"efeitoDeltaFUm{g}Media", _nd(_media([e["delta_f1"] for e in cheias]))),
        ]
    return macros


def tabela20_efeito(dados: dict) -> str | None:
    """Tabela 20: erro de ordenação por regime, razão e Delta F1 (ANL-06)."""
    efeito = _efeito(dados)
    if efeito is None:
        return None
    linhas = []

    def linha(rotulo, e):
        ic = f"{e['n_modelos_ic_exclui_zero']}/{e['n_modelos_ic']}" if e["n_modelos_ic"] else "n/d"
        return (
            f"{rotulo} & {_nd(e['erro_a'])} & {_nd(e['erro_b'])} & {_nd(e['razao'], 2)} & "
            f"{_nd(e['delta_f1'])} & {ic} \\\\"
        )

    for grupo, _g in GRUPOS_EFEITO:
        celulas = efeito[grupo]
        for teste in ORDEM_CORPORA:
            if teste in celulas:
                prefixo = "" if grupo == "diagonal" else "União $\\to$ "
                linhas.append(linha(prefixo + _rotulo(teste, ROTULO_CORPUS), celulas[teste]))
        cheias = list(celulas.values())
        if cheias:
            erro_a = _media([e["erro_a"] for e in cheias])
            erro_b = _media([e["erro_b"] for e in cheias])
            linhas.append(
                rf"\textbf{{Média}} & {_nd(erro_a)} & {_nd(erro_b)} & "
                f"{_nd(_media([e['razao'] for e in cheias]), 2)} & "
                f"{_nd(_media([e['delta_f1'] for e in cheias]))} &  \\\\"
            )
        if grupo == "diagonal":
            linhas.append(r"\midrule")
    nivel = (dados.get("offline") or {}).get("parametros", {}).get("bootstrap_nivel")
    nivel_texto = "" if nivel is None else rf" de {_num(nivel * 100, 0)}\%"
    return _cabecalho("results/matrix.json + results/loco.json + results/offline.json") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Erro de ordenação (1 menos AUROC) por regime, razão B sobre A e diferença de F1
(B menos A), na diagonal e no LOCO, na semente 42. IC conta modelos com intervalo{nivel_texto}
\emph{{bootstrap}} que exclui zero.}}
\label{{tab:efeito}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Célula & Erro A & Erro B & Razão & $\Delta$ F1 & IC exclui zero \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte("matrix.json", "loco.json", "offline.json")}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# Auditoria de viés ampliada: esquema e características por classe (T32, ANL-05)
# ---------------------------------------------------------------------------

# As três características que o orientador cita além do topo da importância.
CARACTERISTICAS_EXTRAS_AUDITORIA = ("has_https", "has_www", "www_count")
CLASSES_AUDITORIA = (("benigno", "Benigno"), ("phishing", "Phishing"))
SUFIXO_MACRO_CARACTERISTICA = {
    "has_https": "HasHttps", "has_www": "HasWww", "www_count": "WwwCount",
}
# (chave em `esquema_por_classe`, rótulo da linha, sufixo da macro)
LINHAS_ESQUEMA = (
    ("explicito", "Com esquema", "Explicito"),
    ("http", r"\emph{http}", "Http"),
    ("https", r"\emph{https}", "Https"),
)


def caracteristicas_da_auditoria(
    importances: dict,
    corpora=ORDEM_CORPORA,
    n: int = TOP_CARACTERISTICAS,
    extras=CARACTERISTICAS_EXTRAS_AUDITORIA,
) -> list[str]:
    """Características da auditoria: o topo de cada dataset e as três extras (ANL-05).

    O topo é o da tabela de importâncias no Regime B (as `n` de menor posto
    médio de cada dataset, na ordem dos datasets e depois dos postos, sem
    repetir), seguido de `has_https`, `has_www` e `www_count` quando ainda não
    estão entre elas.
    """
    escolhidas: list[str] = []
    for corpus in corpora:
        for nome in top_caracteristicas(importances, REGIME_B, corpus, n):
            if nome not in escolhidas:
                escolhidas.append(nome)
    for nome in extras:
        if nome not in escolhidas:
            escolhidas.append(nome)
    return escolhidas


def _auditoria(dados: dict) -> dict | None:
    """`esquema_por_classe` e `medias_por_classe` do diagnóstico; `None` sem o esquema.

    `diagnostics.json` anterior ao ANL-05 não traz as duas chaves: a tabela e as
    macros não saem, e rodar `python -m src.diagnostics` as produz.
    """
    diagnostico = dados.get("diagnostics") or {}
    esquema = diagnostico.get("esquema_por_classe")
    if not esquema:
        return None
    return {"esquema": esquema, "medias": diagnostico.get("medias_por_classe") or {}}


def macros_auditoria(dados: dict) -> list[str]:
    """Presença de esquema por classe e média de `has_https`, `has_www` e `www_count` (ANL-05)."""
    auditoria = _auditoria(dados)
    if auditoria is None:
        return []
    macros = []
    for corpus in ORDEM_CORPORA:
        c = SUFIXO_MACRO_CORPUS[corpus]
        for classe, k in CLASSES_AUDITORIA:
            e = (auditoria["esquema"].get(corpus) or {}).get(classe)
            if e:
                for chave, _rotulo_linha, nome in LINHAS_ESQUEMA:
                    fracao = e.get(chave)
                    macros.append(
                        _macro(f"esquema{nome}{c}{k}",
                               _nd(None if fracao is None else fracao * 100, 2),
                               "pontos percentuais das URLs da classe")
                    )
            medias = ((auditoria["medias"].get(corpus) or {}).get(classe) or {}).get("medias")
            if medias:
                for nome_carac, sufixo in SUFIXO_MACRO_CARACTERISTICA.items():
                    macros.append(
                        _macro(f"auditoriaMedia{sufixo}{c}{k}", _nd(medias.get(nome_carac), 2))
                    )
    return macros


def tabela21_auditoria(dados: dict) -> str | None:
    """Tabela 21: esquema e características do topo da importância, por classe e dataset (ANL-05).

    `None` sem `esquema_por_classe` no diagnóstico. A parte das características
    só sai com `medias_por_classe` e com as importâncias da semente 42.
    """
    auditoria = _auditoria(dados)
    if auditoria is None:
        return None
    corpora = [c for c in ORDEM_CORPORA if c in auditoria["esquema"]]
    if not corpora:
        return None
    n_colunas = 1 + 2 * len(corpora)

    def valores(por_corpus, funcao):
        return " & ".join(
            funcao(por_corpus.get(corpus) or {}, classe)
            for corpus in corpora
            for classe, _k in CLASSES_AUDITORIA
        )

    linhas = [rf"\multicolumn{{{n_colunas}}}{{l}}{{\textbf{{Esquema explícito na URL (\%)}}}} \\"]
    for chave, rotulo, _nome in LINHAS_ESQUEMA:
        linhas.append(
            f"{rotulo} & "
            + valores(
                auditoria["esquema"],
                lambda c, classe, chave=chave: _nd(
                    None if (c.get(classe) or {}).get(chave) is None
                    else c[classe][chave] * 100, 1),
            )
            + " \\\\"
        )
    caracteristicas = []
    importances = (dados.get("importances_por_seed") or {}).get(SEED_CANONICA)
    if auditoria["medias"] and importances:
        caracteristicas = caracteristicas_da_auditoria(importances, corpora)
    if caracteristicas:
        linhas.append(r"\midrule")
        linhas.append(
            rf"\multicolumn{{{n_colunas}}}{{l}}{{\textbf{{Média das características}}}} \\"
        )
        for nome in caracteristicas:
            linhas.append(
                rf"\texttt{{{_escapar(nome)}}} & "
                + valores(
                    auditoria["medias"],
                    lambda c, classe, nome=nome: _nd(
                        ((c.get(classe) or {}).get("medias") or {}).get(nome), 2),
                )
                + " \\\\"
            )
    cabecalho_corpora = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{_rotulo(c, ROTULO_CORPUS)}}}" for c in corpora
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(corpora))
    )
    classes = " & ".join(["Benigno & Phishing"] * len(corpora))
    parte_caracteristicas = (
        " e média das características do topo da importância, mais \\texttt{has\\_https}, "
        "\\texttt{has\\_www} e \\texttt{www\\_count}"
        if caracteristicas else ""
    )
    fontes = ("diagnostics.json", "importances.json") if caracteristicas else ("diagnostics.json",)
    return _cabecalho("results/diagnostics.json :: esquema_por_classe + medias_por_classe") + rf"""
\begin{{table}}[htb]
\centering
\caption{{Presença de esquema na URL{parte_caracteristicas}, por classe e \emph{{dataset}}.}}
\label{{tab:auditoria-vies}}
\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{l{'r' * (2 * len(corpora))}}}
\toprule
 & {cabecalho_corpora} \\
{cmidrules}
 & {classes} \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}
{_fonte(*fontes)}
\end{{table}}
"""


def blocos_frente_a(dados: dict) -> list[tuple[str, list[str]]]:
    """Blocos de macros da frente A que dependem de dados novos; só os com conteúdo."""
    blocos = [
        ("Deduplicacao por dataset e regime (DEDUP-04): splits.json :: dedup",
         macros_dedup(dados)),
        ("Linhas descartadas por motivo (PROV-04): corpora/manifest.json",
         macros_descartes(dados)),
        ("Secao privada da PSL contra so ICANN (PSL-04, REEX-13): diagnostics.json",
         macros_psl(dados)),
        ("Dominios com as duas classes e prevalencia por particao (DEDUP-05, REEX-08): "
         "diagnostics.json", macros_prevalencia(dados)),
        ("Sensibilidade sem esquema (SENS-02, REEX-05): sensibilidade_sem_esquema/ e "
         "rodada principal", macros_sensibilidade(dados)),
        ("Repeticao da semente 42 (REP-01, REEX-12): repeticao_seed_42/ e rodada principal",
         macros_repeticao(dados)),
        ("HPO avaliado fora da diagonal (HPO-02, REEX-11): hpo_fora_diagonal.json",
         macros_hpo_fora(dados)),
        ("Filtragem da sobreposicao de eTLD+1 (FILT-01, REEX-06): offline.json :: filtragem",
         macros_filtragem(dados)),
        ("Bootstrap pareado do Delta entre regimes (BOOT-01, EST-02): offline.json",
         macros_bootstrap_pareado(dados)),
        ("Decomposicao da variacao da AUROC fora da diagonal (ANL-01): matrix.json, "
         "por regime, semente e media", macros_decomposicao(dados)),
        ("Uniao do LOCO contra as fontes isoladas (ANL-02): matrix.json e loco.json, "
         "semente 42 e media das sementes", macros_uniao(dados)),
        ("Composicao da uniao do LOCO por dataset de origem (ANL-02): splits.json",
         macros_composicao(dados)),
        ("Predicoes degeneradas fora da diagonal e no LOCO (ANL-04): matrix.json e loco.json",
         macros_degeneradas(dados)),
        ("Erro de ordenacao, razao e Delta F1 (ANL-06): matrix.json, loco.json e offline.json",
         macros_efeito(dados)),
        ("Auditoria de vies ampliada: esquema por classe e caracteristicas (ANL-05): "
         "diagnostics.json", macros_auditoria(dados)),
    ]
    return [(titulo, macros) for titulo, macros in blocos if macros]


# Caracteres de URL que o LaTeX interpreta. Sem escapar, `_` e `^` só valem em
# modo matemático (erro), `%` comenta o resto da linha (some texto em silêncio) e
# `#` é argumento de macro. Como a URL vai dentro de `\texttt{}` na prosa e não
# num `\url{}`, o escape é por nossa conta.
_ESCAPE_TEX = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _url_tex(url: str) -> str:
    """URL literal, segura para ir dentro de `\\texttt{}`.

    🚨 O `%` é o caso perigoso e é comum em URL (codificação percentual): não
    escapado, ele comenta o resto da linha do `.tex` e o texto seguinte
    desaparece do PDF **sem erro de compilação**. É a mesma família de falha
    silenciosa da §13.4-21 (o `\\ref` órfão que imprime `??`).

    `\\allowbreak` depois de `/` e `.` deixa a URL quebrar de linha em vez de
    estourar a coluna: sem isso um endereço longo dá `Overfull \\hbox`, e o
    portão do DoD exige zero.
    """
    saida = "".join(_ESCAPE_TEX.get(c, c) for c in url)
    for sep in ("/", ".", "-", "?", "="):
        saida = saida.replace(sep, sep + r"\allowbreak{}")
    # O `[.]` da neutralização (`_neutralizar_url`) não pode ser aberto ao meio:
    # a quebra, quando houver, vem depois do `]`.
    return saida.replace(r"[.\allowbreak{}]", r"[.]\allowbreak{}")


_ESQUEMA_URL = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://")
_FIM_DO_HOST = re.compile(r"[/?#]")
_ESQUEMA_NEUTRO = {"http": "hxxp", "https": "hxxps"}


def _neutralizar_url(url: str, rotulo: int) -> str:
    """URL de phishing neutralizada, para não virar endereço clicável no PDF (ABNT-09).

    Com rótulo diferente de 1 a URL sai intacta. Com rótulo 1, `http` vira `hxxp`
    e `https` vira `hxxps` (sem distinção de caixa no casamento), e cada ponto do
    host vira `[.]`; o caminho e a consulta ficam como estão. O host vai até o
    primeiro `/`, `?` ou `#`, com ou sem esquema, e um IP literal também tem os
    pontos trocados. O escape para TeX continua por conta do `_url_tex`.
    """
    if rotulo != 1:
        return url
    prefixo, resto = "", url
    casou = _ESQUEMA_URL.match(url)
    if casou:
        esquema = _ESQUEMA_NEUTRO.get(casou.group(1).lower(), casou.group(1))
        prefixo, resto = f"{esquema}://", url[casou.end():]
    fim = _FIM_DO_HOST.search(resto)
    posicao = fim.start() if fim else len(resto)
    return prefixo + resto[:posicao].replace(".", "[.]") + resto[posicao:]


def numeros_prosa(dados: dict) -> str:
    """`numeros.tex`: um `\\newcommand` por valor que a prosa pode citar.

    Conjunto deliberadamente pequeno: só o que a §13.8 e o D1 já afirmam, que
    é o que Resultados, Discussão e Ameaças vão precisar dizer no meio de uma
    frase. Percentuais saem **sem** o sinal de `%`: quem escreve põe `\\%`.
    """
    diag = dados["diagnostics"]
    seeds = sorted(dados["matriz_por_seed"])

    # --- Vazamento nos splits oficiais (C1) --------------------------------
    vaz = diag["vazamento_split_oficial"]
    por_dominio = sorted(
        ((v["contaminacao_dominios_pct"], c) for c, v in vaz.items())
    )
    menor, segundo = por_dominio[0], por_dominio[1]
    maior = por_dominio[-1]
    linhas_max = max(v["contaminacao_linhas_pct"] for v in vaz.values())

    # O exemplo de URL da Introdução vem do ISCX-URL2016 e a escolha é
    # EDITORIAL, não medida: é o corpus em que o domínio mais vazado
    # (`pastehtml.com`) é um serviço de hospedagem abusado por campanhas, e não
    # um domínio malicioso. O leitor entende num relance que um modelo que
    # aprende esse domínio acerta o teste sem ter aprendido nada sobre phishing,
    # que é literalmente a frase da Introdução. Nos outros cinco o domínio mais
    # vazado é `wikipedia.org` ou `blogspot.com`, do lado benigno, e o exemplo
    # não sustenta a mesma leitura.
    _EX_CORPUS = "iscx2016"
    _ex = vaz[_EX_CORPUS]["exemplo"]
    if _ex is None:
        raise ReportError(
            f"{_EX_CORPUS}: diagnostics.json não traz `exemplo` de vazamento. "
            "Rode `python -m src.diagnostics`: o campo entrou na T5.9 e um "
            "diagnostics.json anterior a 08/08/2026 não o tem."
        )

    # URLs literalmente idênticas entre treino e teste do corte de referência
    # (LIT-03 e LIT-04): campo medido pelo `src.diagnostics` sobre a coluna `url`
    # bruta. Um diagnostics.json sem o campo para o `make tabelas`, em vez de
    # omitir a macro em silêncio.
    for _corpus, _v in vaz.items():
        if "urls_identicas_treino_teste" not in _v:
            raise ReportError(
                f"{_corpus}: diagnostics.json não traz `urls_identicas_treino_teste`. "
                "Rode `python -m src.diagnostics`."
            )
    macros_urls_identicas = [
        _macro(
            f"vazamentoUrlsIdenticas{SUFIXO_MACRO_CORPUS[c]}",
            _milhar(vaz[c]["urls_identicas_treino_teste"]),
            "URLs distintas, string exata, no treino e no teste do corte de "
            "referencia, antes da dedup",
        )
        for c in ORDEM_TODOS_CORPORA
        if c in vaz
    ]

    # --- Matriz cruzada (C2) ----------------------------------------------
    auroc = _celulas_por_seed(dados["matriz_por_seed"], "etld1", "auroc")
    mcc = _celulas_por_seed(dados["matriz_por_seed"], "etld1", "mcc")

    # --- Divergência AUROC x AUC-PR (T5.9) --------------------------------
    # A linha de AUC-PR das 16 células saiu do artigo com a T2 (D-C1), e a §2.1
    # o roteiro de revisão manda o destino ser UMA frase na §Resultados:
    # que a AUC-PR acompanha o AUROC, e qual é a maior divergência entre as
    # duas. Estas macros são esse destino; sem elas o corte da T2 apagaria a
    # AUC-PR do artigo inteiro em silêncio.
    aucpr = _celulas_por_seed(dados["matriz_por_seed"], "etld1", "auc_pr")
    divergencia = {
        k: aucpr[SEED_CANONICA][k] - v
        for k, v in auroc[SEED_CANONICA].items()
        if aucpr[SEED_CANONICA].get(k) is not None and v is not None
    }
    celula_div = max(divergencia, key=lambda k: abs(divergencia[k]))
    # A DIREÇÃO é derivada, não escrita: se uma rodada futura inverter o sinal,
    # a prosa passa a dizer a outra palavra sozinha em vez de virar afirmação
    # falsa que ninguém remediu. É a doutrina da §13.4-26; o `hpoMenorDeltaModelo`
    # a segue pelo nome do modelo, e o sinal dele vem de `hpoNumPerdem`.
    direcao_div = "acima" if divergencia[celula_div] > 0 else "abaixo"

    def parte(celulas, diagonal: bool):
        return [v for (tr, te), v in celulas.items() if (tr == te) is diagonal]

    fora_canonica = {
        k: v for k, v in auroc[SEED_CANONICA].items() if k[0] != k[1]
    }
    por_linha = {
        tr: _media([v for (t, _te), v in fora_canonica.items() if t == tr])
        for tr in ORDEM_CORPORA
    }
    melhor_linha = max(por_linha, key=por_linha.get)
    ordem_linha = sorted(por_linha, key=por_linha.get)
    pior_linha, segundo_pior_linha = ordem_linha[0], ordem_linha[1]
    # Contagem abaixo do acaso vai como FAIXA entre as seeds, nunca como
    # absoluto: é o caso que o PC5 nomeia explicitamente.
    abaixo = [
        sum(1 for (tr, te), v in auroc[s].items() if tr != te and v < 0.5)
        for s in seeds
    ]

    # --- LOCO (T6) ---------------------------------------------------------
    # Faixa sobre os 4 corpora da seed canônica, e NÃO sobre as 3 seeds: é o
    # que a coluna "Ganho" da Tabela 6 imprime, e prosa que citasse um mínimo
    # que não aparece em nenhuma linha da tabela seria lida como divergência.
    # A variação entre seeds já vai na própria tabela, por corpus.
    col_canonica = medias_da_coluna(
        dados["matriz_por_seed"][SEED_CANONICA], "etld1", "auroc"
    )
    loco_canonica = medias_loco(
        dados["loco_por_seed"][SEED_CANONICA], "etld1", "auroc"
    )
    ganhos = [
        loco_canonica[c] - col_canonica[c][1]
        for c in ORDEM_CORPORA
        if c in loco_canonica and col_canonica.get(c, (None, None))[1] is not None
    ]
    res_loco = diag["sobreposicao_loco"]["etld1"]["resumo"]
    res_par = diag["sobreposicao_entre_corpora"]["etld1"]["resumo"]

    # --- Ablação (T4) ------------------------------------------------------
    hpo = dados["hpo"]["modelos"]
    deltas = {
        k: v["auroc_teste_ajustado"] - v["auroc_teste_seco"] for k, v in hpo.items()
    }
    # O menor delta de teste: "maior perda" quando algum delta é <= 0 e "menor
    # ganho" quando todos são positivos. Quem decide qual dos dois é a contagem
    # `hpoNumPerdem` (modelos com delta <= 0).
    menor_delta_hpo = min(deltas, key=deltas.get)
    n_perdem = sum(1 for d in deltas.values() if d <= 0)

    # --- Viés de construção (Cap. 6, seção 6.10) ---------------------------
    # As duas pontas da bimodalidade que a tab:vies mostra: o maior gap entre os
    # limpos e o menor gap máximo entre os contaminados. Derivadas do veredito
    # gravado, e não de um limiar daqui, para a prosa nunca discordar da tabela.
    vies = diag["vies_construcao"]
    gaps_limpos = [v["gap_max_p"] for v in vies.values() if v["veredito"] != "contaminado"]
    gaps_contaminados = [
        v["gap_max_p"] for v in vies.values() if v["veredito"] == "contaminado"
    ]

    # --- Calibração (Cap. 6, seção 6.7) -------------------------------------
    # Mesma convenção do aurocDiagonal: média dos 6 modelos por célula, Regime B
    # (etld1), semente canônica; depois média das células da diagonal e de fora.
    brier = media_por_celula(dados["matriz_por_seed"][SEED_CANONICA], "etld1", "brier")
    ece = media_por_celula(dados["matriz_por_seed"][SEED_CANONICA], "etld1", "ece")
    ece_modelo = media_por_modelo(
        dados["matriz_por_seed"][SEED_CANONICA], "etld1", "ece"
    )
    ece_fora_modelo = {m: v for (m, na_diag), v in ece_modelo.items() if not na_diag}
    ece_melhor = min(ece_fora_modelo, key=ece_fora_modelo.get)
    ece_pior = max(ece_fora_modelo, key=ece_fora_modelo.get)

    # --- Cobertura das importâncias (Cap. 6, seção 6.6) ---------------------
    # O total é o número de ajustes da matriz (regime, dataset de treino,
    # modelo), contado em matrix.json, e não 6 x 8 digitado: se um regime ou um
    # modelo mudar, as duas macros mudam juntas.
    ajustes_total = {
        (r["regime"], r["train_corpus"], r["model"])
        for r in dados["matriz_por_seed"][SEED_CANONICA]
    }

    blocos = [
        ("Vazamento de dominio nos splits oficiais (C1) -- diagnostics.json", [
            _macro("vazamentoDominiosMax", _pct(maior[0]),
                   f"{maior[1]}, o pior dos {len(vaz)}"),
            _macro("vazamentoDominiosMin", _pct(menor[0]), f"{menor[1]}"),
            _macro("vazamentoDominiosSegundoMenor", _pct(segundo[0]),
                   f"{segundo[1]}: piso dos outros {len(vaz) - 1}"),
            _macro("vazamentoLinhasMax", _pct(linhas_max), "por linha, nao por dominio"),
            _macro("vazamentoCorpusPior", ROTULO_CORPUS.get(maior[1], maior[1])),
            _macro("vazamentoCorpusMelhor", ROTULO_CORPUS.get(menor[1], menor[1])),
            *macros_urls_identicas,
            # --- O exemplo concreto (roteiro de revisão) ---------------------
            # O artigo não mostrava UMA URL: um trabalho sobre características
            # léxicas de URL em que o leitor nunca vê um endereço obriga-o a
            # imaginar o objeto de estudo o tempo inteiro. Estas macros
            # são o destino, e existem para que o par NÃO seja digitado à mão
            # (PC3) — ele é derivado por
            # `diagnostics.escolher_exemplo_vazamento`, que exige as duas URLs
            # DISTINTAS. O par que o roteiro propunha a mão tinha a URL do lado
            # "teste" nas DUAS partições do ISCX, o que ilustraria duplicação
            # exata em vez do vazamento de domínio que o C1 mede.
            _macro("exemploDominio", _ex["dominio"]),
            _macro("exemploUrlTreino",
                   _url_tex(_neutralizar_url(_ex["url_treino"], _ex["label_treino"]))),
            _macro("exemploUrlTeste",
                   _url_tex(_neutralizar_url(_ex["url_teste"], _ex["label_teste"]))),
            _macro("exemploLinhasTreino", f"{_ex['linhas_treino']}"),
            _macro("exemploLinhasTeste", f"{_ex['linhas_teste']}"),
            _macro("exemploCorpus", ROTULO_CORPUS.get(_EX_CORPUS, _EX_CORPUS)),
        ]),
        ("Matriz cruzada (C2) -- etld1, seed canonica, media dos 6 modelos", [
            _macro("aurocDiagonal", _num(_media(parte(auroc[SEED_CANONICA], True)))),
            _macro("aurocForaDiagonal", _num(_media(parte(auroc[SEED_CANONICA], False)))),
            _macro("mccDiagonal", _num(_media(parte(mcc[SEED_CANONICA], True)))),
            _macro("mccForaDiagonal", _num(_media(parte(mcc[SEED_CANONICA], False)))),
            _macro("aurocForaMin", _num(min(fora_canonica.values())), "pior celula"),
            _macro("aurocForaMax", _num(max(fora_canonica.values())), "melhor celula"),
            _macro("aurocLinhaMelhor", _num(por_linha[melhor_linha]),
                   "media da linha de treino que melhor transfere"),
            _macro("aurocLinhaPior", _num(por_linha[pior_linha])),
            # O SEGUNDO pior por linha existe porque a assimetria do D1 não
            # está em o ISCX ser o último — está em ele estar isolado. A
            # §13.8 mede a margem (0,30 por linha contra 0,07 por coluna), e
            # sem esta macro a T5.5c teria de digitá-la (PC3). Ver §13.4-26.
            _macro("aurocLinhaSegundoPior", _num(por_linha[segundo_pior_linha])),
            _macro("corpusLinhaMelhor", ROTULO_CORPUS.get(melhor_linha, melhor_linha)),
            _macro("corpusLinhaPior", ROTULO_CORPUS.get(pior_linha, pior_linha)),
            _macro("celulasAbaixoDoAcasoMin", str(min(abaixo)),
                   f"de {len(fora_canonica)} celulas fora da diagonal"),
            _macro("celulasAbaixoDoAcasoMax", str(max(abaixo)),
                   f"faixa entre as {len(seeds)} seeds -- PC5"),
        ]),
        ("Divergencia AUROC x AUC-PR -- o destino da linha que saiu com a T2", [
            _macro("aucprDiagonal", _num(_media(parte(aucpr[SEED_CANONICA], True)))),
            _macro("aucprForaDiagonal",
                   _num(_media(parte(aucpr[SEED_CANONICA], False)))),
            _macro("divergenciaAucprMax", _num(abs(divergencia[celula_div])),
                   "|AUC-PR - AUROC| na celula mais divergente das 16"),
            _macro("divergenciaAucprTreino",
                   ROTULO_CORPUS.get(celula_div[0], celula_div[0])),
            _macro("divergenciaAucprTeste",
                   ROTULO_CORPUS.get(celula_div[1], celula_div[1])),
            _macro("divergenciaAucprDirecao", direcao_div,
                   "DERIVADO: onde a AUC-PR fica em relacao ao AUROC"),
        ]),
        ("Leave-one-corpus-out (T6) -- loco.json e diagnostics.json", [
            _macro("locoAuroc", _num(_media(list(loco_canonica.values())))),
            _macro("locoGanhoMin", _num(min(ganhos))),
            _macro("locoGanhoMax", _num(max(ganhos))),
            _macro("locoSobreposicaoMedia", _pct(res_loco["media_pct"])),
            _macro("locoSobreposicaoLinhasMedia", _pct(res_loco["linhas_media_pct"])),
            _macro("locoSobreposicaoLinhasMax", _pct(res_loco["linhas_max_pct"])),
        ]),
        ("Sobreposicao residual par a par entre corpora (T5.6)", [
            _macro("sobreposicaoParMedia", _pct(res_par["fora_diagonal_media_pct"])),
            _macro("sobreposicaoParMax", _pct(res_par["fora_diagonal_max_pct"])),
            _macro("sobreposicaoParLinhasMedia",
                   _pct(res_par["fora_diagonal_linhas_media_pct"])),
            _macro("sobreposicaoParLinhasMax",
                   _pct(res_par["fora_diagonal_linhas_max_pct"])),
        ]),
        # O modelo de menor delta de teste é DERIVADO, e por isso o nome da
        # macro não o cita: `\hpoPerdaMlp` viraria mentira silenciosa se outra
        # rodada elegesse outro modelo. Quem escreve usa `\hpoMenorDeltaModelo` junto
        # e confere o sinal em `\hpoNumPerdem`: 0 quer dizer que ninguém piora
        # e que as macros descrevem o MENOR GANHO.
        ("Ablacao de hiperparametros (T4) -- hpo.json", [
            _macro("hpoTentativas", str(dados["hpo"]["n_trials"]), "por modelo"),
            _macro("hpoGanhoMax", _num(max(deltas.values()), 5), "o maior ganho no teste"),
            _macro("hpoNumPerdem", str(n_perdem),
                   "DERIVADO: modelos com delta de teste <= 0"),
            _macro("hpoMenorDeltaModelo", _rotulo(menor_delta_hpo, ROTULO_MODELO),
                   "DERIVADO: menor delta de teste"),
            _macro("hpoMenorDeltaValor", _num(abs(deltas[menor_delta_hpo]), 5),
                   "valor ABSOLUTO do menor delta: maior perda se hpoNumPerdem > 0, "
                   "senao menor ganho"),
            _macro("hpoMenorDeltaCusto",
                   _num(hpo[menor_delta_hpo]["fit_seconds_ajustado"]
                        / hpo[menor_delta_hpo]["fit_seconds_seco"], 1), "vezes o tempo"),
        ]),
        ("Friedman e Nemenyi (E4, T8 e F3): matrix.json, etld1, AUROC", _macros_friedman(dados)),
        ("Vies de construcao (Cap. 6): diagnostics.json :: vies_construcao", [
            _macro("viesGapLimpoMax", _pct(max(gaps_limpos)),
                   "maior gap entre os limpos, pontos percentuais"),
            _macro("viesGapContaminadoMin", _pct(min(gaps_contaminados)),
                   "menor gap maximo entre os contaminados"),
        ]),
        ("Calibracao (Cap. 6): matrix.json, etld1, seed canonica, media dos 6 modelos", [
            _macro("brierDiagonal", _num(_media(parte(brier, True)))),
            _macro("brierForaDiagonal", _num(_media(parte(brier, False)))),
            _macro("eceDiagonal", _num(_media(parte(ece, True)))),
            _macro("eceForaDiagonal", _num(_media(parte(ece, False)))),
            _macro("eceForaMelhorModelo", _rotulo(ece_melhor, ROTULO_MODELO),
                   "DERIVADO: menor ECE medio fora da diagonal"),
            _macro("eceForaPiorModelo", _rotulo(ece_pior, ROTULO_MODELO),
                   "DERIVADO: maior ECE medio fora da diagonal"),
        ]),
        ("Cobertura das importancias (Cap. 6): importances.json e matrix.json", [
            _macro("importanciasAjustes", str(len(dados["importances"]["ajustes"])),
                   "ajustes com importancia por caracteristica"),
            _macro("importanciasAjustesTotal", str(len(ajustes_total)),
                   "ajustes da matriz, regime x dataset x modelo"),
        ]),
        ("Precisao, revocacao e taxa de falsos positivos (RUN-01): matrix.json e "
         "loco.json, seed canonica, media dos 6 modelos", macros_limiar(dados)),
        *blocos_frente_a(dados),
    ]

    corpo = "\n\n".join(
        f"% -- {titulo}\n" + "\n".join(macros) for titulo, macros in blocos
    )
    return _cabecalho(
        "results/diagnostics.json + matrix.json + seed_*/ + loco.json + hpo.json + "
        "splits.json + corpora/manifest.json"
    ) + f"""%
% Carregado no preambulo do artigo.tex. Cada comando abaixo e um numero MEDIDO:
% digitar qualquer um deles a mao no .tex viola o PC3 e a R5. Percentuais vem
% sem o sinal de porcentagem -- quem escreve poe a barra-porcento depois da
% macro.
%
% Macro nova se acrescenta em src/report.py::numeros_prosa, com teste.

{corpo}
"""


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------

# Nome de arquivo é CONTRATO com o `artigo.tex`. Hífen, nunca
# sublinhado: `_` dentro de `\texttt` quebra a compilação.
TABELAS = {
    "tabela1-vazamento.tex": tabela1_vazamento,
    "tabela5-vies.tex": tabela5_vies,
    # A T2 (matriz 4x4 com AUROC/AUC-PR) saiu do artigo na T5.9 por decisão
    # D-C1: ela e o mapa de calor publicavam a mesma matriz 4x4, e juntas
    # custavam ~1,05 página para mostrar 16 células. Fica a figura, que
    # comunica a inversão abaixo do acaso num relance.
    #
    # 🔁 `tabela2_matriz` NÃO foi apagada, de propósito: a câmera-ready de
    # 13/09 é a candidata mais forte a querê-la de volta, porque é a única
    # publicação da AUC-PR (o roteiro de revisão a lista como
    # prioridade ALTA de retorno). Reverter é descomentar a linha abaixo e
    # devolver o `\inputtabela` ao artigo.tex.
    #   "tabela2-matriz-etld1.tex": tabela2_matriz,
    "tabela3-diagonal.tex": tabela3_diagonal,
    "tabela4-ablacao.tex": tabela4_ablacao,
    "tabela6-loco.tex": tabela6_loco,
    # RUN-01: precisão, revocação e TFP por célula, nos dois regimes.
    "tabela10-limiar.tex": tabela10_limiar,
    # Rodadas adicionais (T25): a função devolve `None` quando a rodada não
    # existe, e o `gerar` não escreve o arquivo.
    "tabela11-sensibilidade.tex": tabela11_sensibilidade,
    "tabela12-repeticao.tex": tabela12_repeticao,
    "tabela13-hpo-fora.tex": tabela13_hpo_fora,
    # Análises offline (T26): vazio sem `offline.json`.
    "tabela14-filtragem.tex": tabela14_filtragem,
    "tabela15-bootstrap-pareado.tex": tabela15_bootstrap_pareado,
    # Análises da F04 (T30): sem os dados, a função devolve `None`.
    "tabela16-decomposicao.tex": tabela16_decomposicao,
    "tabela17-uniao.tex": tabela17_uniao,
    "tabela18-composicao.tex": tabela18_composicao,
    "tabela19-degeneradas.tex": tabela19_degeneradas,
    "tabela20-efeito.tex": tabela20_efeito,
    # ANL-05 (T32): só sai com `esquema_por_classe` no `diagnostics.json`.
    "tabela21-auditoria.tex": tabela21_auditoria,
    "tabela7-importancias.tex": tabela7_importancias,
    # E4 (EST-01): Friedman e Nemenyi. O diagrama de diferença crítica é TikZ
    # e por isso mora aqui, e não em FIGURAS: nasce no `make tabelas`.
    "tabela8-friedman.tex": tabela8_friedman,
    "figura3-diferenca-critica.tex": figura3_diferenca_critica,
    # Cap. 6 (T5): calibração por modelo, seção 6.7.
    "tabela9-calibracao.tex": tabela9_calibracao,
    # Spec 8, Apêndice B: configuração fixa (etapa 1) e melhor configuração do
    # HPO (etapa 2). O espaço de busca não sai daqui: o `hpo.json` não o grava.
    "apendiceB1-configuracao.tex": apendiceB1_configuracao,
    "apendiceB2-hpo.tex": apendiceB2_hpo,
    # Spec 8, Apêndice C: oito tabelas da matriz (regime x métrica) e duas do
    # LOCO (regime), com a semente 42 e a faixa das três sementes.
    **TABELAS_APENDICE_C,
    # Spec 8, Apêndice D: as execuções e as versões das bibliotecas.
    "apendiceD1-execucao.tex": apendiceD1_execucao,
    "apendiceD2-versoes.tex": apendiceD2_versoes,
    # Não é tabela: é o arquivo de `\newcommand` que o preâmbulo carrega para
    # a PROSA poder citar número sem digitá-lo (PC3). Mora no mesmo dicionário
    # porque tem o mesmo ciclo de vida — mesma fonte, mesma regeneração, mesmo
    # teste de proveniência — e separá-lo criaria um segundo caminho de
    # escrita para alguém esquecer de chamar.
    "numeros.tex": numeros_prosa,
}


def gerar(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    tables_dir: Path = DEFAULT_TABLES_DIR,
) -> list[Path]:
    """Gera todas as tabelas. Devolve os caminhos escritos, em ordem fixa.

    Tabela que depende de rodada adicional ausente devolve `None` e não é
    escrita: rodada ausente não é erro.
    """
    dados = carregar(results_dir)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    escritos = []
    for nome, funcao in TABELAS.items():
        conteudo = funcao(dados)
        if conteudo is None:
            continue
        destino = tables_dir / nome
        destino.write_text(conteudo, encoding="utf-8")
        escritos.append(destino)
    return escritos


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "T5.2 -- gera as tabelas LaTeX do artigo a partir de results/. "
            "Nenhum numero e digitado a mao (R5)."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--sem-figuras", action="store_true",
        help="gera so as tabelas (util para iterar sem esperar o matplotlib)",
    )
    args = parser.parse_args()

    try:
        escritos = gerar(args.results_dir, args.tables_dir)
        if not args.sem_figuras:
            escritos += gerar_figuras(args.results_dir, args.figures_dir)
    except ReportError as exc:
        print(f"ERRO: {exc}")
        return 1

    for caminho in escritos:
        print(f"gravado: {caminho.resolve().relative_to(TCC_ROOT).as_posix()}")
    print()
    print(f"{len(escritos)} arquivo(s). Compilar: make pdf, na raiz do repositorio")
    return 0


__all__ = [
    "ReportError",
    "carregar",
    "conferir_commits",
    "media_por_celula",
    "medias_da_coluna",
    "medias_loco",
    "media_por_modelo",
    "posto_medio",
    "top_caracteristicas",
    "tabela1_vazamento",
    "tabela5_vies",
    "tabela2_matriz",
    "tabela3_diagonal",
    "tabela4_ablacao",
    "tabela6_loco",
    "tabela10_limiar",
    "macros_limiar",
    "macros_dedup",
    "macros_descartes",
    "macros_psl",
    "macros_prevalencia",
    "macros_sensibilidade",
    "macros_repeticao",
    "macros_hpo_fora",
    "tabela11_sensibilidade",
    "tabela12_repeticao",
    "tabela13_hpo_fora",
    "macros_filtragem",
    "macros_bootstrap_pareado",
    "tabela14_filtragem",
    "tabela15_bootstrap_pareado",
    "macros_decomposicao",
    "macros_uniao",
    "macros_composicao",
    "macros_degeneradas",
    "macros_efeito",
    "tabela16_decomposicao",
    "tabela17_uniao",
    "tabela18_composicao",
    "tabela19_degeneradas",
    "tabela20_efeito",
    "macros_auditoria",
    "caracteristicas_da_auditoria",
    "tabela21_auditoria",
    "blocos_frente_a",
    "tabela7_importancias",
    "tabela9_calibracao",
    "apendiceB1_configuracao",
    "apendiceB2_hpo",
    "apendiceC_matriz",
    "apendiceC_loco",
    "TABELAS_APENDICE_C",
    "apendiceD1_execucao",
    "apendiceD2_versoes",
    "friedman_por_seed",
    "tabela8_friedman",
    "figura3_diferenca_critica",
    "numeros_prosa",
    "TABELAS",
    "gerar",
]


# ---------------------------------------------------------------------------
# Figuras (F1 e F2)
# ---------------------------------------------------------------------------
#
# Saem em PDF, não em PNG: o artigo é vetorial e uma figura rasterizada num
# artigo de 8-12 páginas aparece borrada na impressão. `matplotlib` grava PDF
# nativamente, sem passo extra.
#
# ⚠️ `matplotlib.use("Agg")` antes de importar `pyplot`: sem isso, num Windows
# sem display o backend interativo tenta abrir janela e o script trava sem
# mensagem — o tipo de falha que o Lucas não teria como diagnosticar. Aqui
# quem gera é o Julio, mas o script tem de ser rodável em qualquer máquina.

DEFAULT_FIGURES_DIR = TCC_ROOT / "latex" / "figuras"


def _plt():
    """Importa o `pyplot` já com backend não-interativo, e o seaborn junto."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.85)
    return plt, sns


def _virgula(valor, casas: int = 1) -> str:
    """Número com vírgula decimal para os textos das figuras (ABNT-06, FIG-01).

    O matplotlib formata com ponto. Toda anotação, rótulo de eixo e barra de
    cor passa por aqui ou por `_formatador_virgula`, para o PDF seguir o resto
    do texto. Valor ausente (célula sem dado) vira texto vazio.
    """
    if valor is None or valor != valor:  # None ou NaN
        return ""
    texto = f"{valor:.{casas}f}"
    if texto.startswith("-") and not texto.strip("-0."):
        texto = texto[1:]
    return texto.replace(".", ",")


def _formatador_virgula(casas: int = 1):
    """`FuncFormatter` que imprime os valores do eixo com vírgula decimal."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda valor, _posicao: _virgula(valor, casas))


def figura2_mapa_de_calor(dados: dict, destino: Path) -> Path:
    """F2 — mapa de calor da matriz cruzada (P1: comunica o achado central).

    Escala divergente centrada em **0,5**, e não no meio dos dados: 0,5 é o
    acaso, e a leitura que importa é "esta célula está acima ou abaixo de
    chutar". Centrar na média dos dados pintaria de cor neutra um valor que é
    o achado (§13.4-8: modelos treinados no ISCX chegam a AUROC 0,19-0,46 em
    outros corpora, isto é, a relação aprendida **inverte**).
    """
    plt, sns = _plt()
    celulas = media_por_celula(
        dados["matriz_por_seed"][SEED_CANONICA], "etld1", "auroc"
    )
    matriz = [
        [celulas.get((tr, te)) for te in ORDEM_CORPORA] for tr in ORDEM_CORPORA
    ]
    rotulos = [ROTULO_CORPUS.get(c, c) for c in ORDEM_CORPORA]

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    # Anotação em texto pronto, com vírgula (FIG-01): `fmt=".3f"` do seaborn
    # imprimiria ponto decimal.
    anotacoes = [[_virgula(v, 3) for v in linha] for linha in matriz]
    sns.heatmap(
        matriz, annot=anotacoes, fmt="", cmap="RdYlGn", center=0.5,
        vmin=0.0, vmax=1.0, linewidths=0.5, linecolor="white",
        xticklabels=rotulos, yticklabels=rotulos, ax=ax,
        cbar_kws={
            "label": "AUROC (média dos 6 modelos)",
            "format": _formatador_virgula(1),
        },
    )
    # D-H (§12): os dois rótulos de eixo são a razão de a T5.9 regenerar as
    # figuras SEM `--sem-figuras` uma vez. Trocar só as tabelas deixaria a
    # única imagem do artigo dizendo "Corpus" ao lado de sete "Dataset".
    ax.set_xlabel("Dataset de teste")
    ax.set_ylabel("Dataset de treino")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return destino


def figura1_curvas(dados: dict, results_dir: Path, destino: Path) -> Path:
    """F1 — curvas ROC e PR, diagonal contra fora-diagonal (P2).

    ⚠️ A curva PR gravada é o **envelope monótono**, não a AUC-PR das tabelas
    (§7.4). A área sob este envelope é otimista e integrar a figura dá outro
    número; a legenda no `artigo.tex` declara isso. Não é defeito: é a
    convenção clássica de precisão interpolada, e foi escolhida porque
    interpolar linearmente atravessa o degrau da curva PR (§11, 05/08/2026).
    """
    plt, sns = _plt()
    curvas = _ler_json(results_dir / "curves.json")
    grade = curvas["grade"]

    series = {"diagonal": {"roc": [], "pr": []}, "fora": {"roc": [], "pr": []}}
    for chave, pontos in curvas["celulas"].items():
        regime, treino, teste, _modelo = chave.split("|")
        if regime != "etld1":
            continue
        grupo = "diagonal" if treino == teste else "fora"
        series[grupo]["roc"].append(pontos["roc_tpr"])
        series[grupo]["pr"].append(pontos["pr_precision"])

    def media(curvas_do_grupo):
        return [
            sum(c[i] for c in curvas_do_grupo) / len(curvas_do_grupo)
            for i in range(len(grade))
        ]

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    cores = {"diagonal": "#1b7837", "fora": "#b2182b"}
    nomes = {"diagonal": "Diagonal (mesmo dataset)", "fora": "Fora da diagonal"}
    for grupo in ("diagonal", "fora"):
        if not series[grupo]["roc"]:
            continue
        ax_roc.plot(grade, media(series[grupo]["roc"]),
                    color=cores[grupo], label=nomes[grupo], linewidth=1.8)
        ax_pr.plot(grade, media(series[grupo]["pr"]),
                   color=cores[grupo], label=nomes[grupo], linewidth=1.8)

    ax_roc.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1, label="Acaso")
    ax_roc.set_xlabel("Taxa de falsos positivos")
    ax_roc.set_ylabel("Taxa de verdadeiros positivos")
    ax_roc.set_title("ROC")
    ax_pr.set_xlabel("Revocação")
    ax_pr.set_ylabel("Precisão (envelope monótono)")
    # Hífen simples: o matplotlib não converte `--` em traço, e o PDF mostraria
    # dois hífens no título.
    ax_pr.set_title("Precisão-revocação")
    for eixo in (ax_roc, ax_pr):
        eixo.set_xlim(0, 1)
        eixo.set_ylim(0, 1.02)
        # Vírgula decimal nos dois eixos (FIG-01).
        eixo.xaxis.set_major_formatter(_formatador_virgula(1))
        eixo.yaxis.set_major_formatter(_formatador_virgula(1))
        eixo.legend(loc="lower right", frameon=True, fontsize=7)

    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return destino


# Registro das figuras que o artigo usa. É consultado pelo `gerar_figuras()`
# abaixo, e essa consulta é o que dá dente à remoção: até a T5.9 este
# dicionário não era lido por código nenhum, então tirar uma entrada dele era
# declaração sem efeito — a mesma classe de defeito que a §11 de 05/08 nomeou
# ("hash de contrato não é hash de resultado"), e o `figura1-curvas.pdf`
# continuaria sendo gerado e re-carimbado (§13.4-22) depois de sair do artigo.
FIGURAS = {
    "figura2-mapa-calor.pdf": "F2 -- mapa de calor da matriz cruzada",
    # A F1 (curvas ROC/PR) saiu do artigo na T5.9 por decisão D-C3: custava
    # 1 página MEDIDA (§13.4-23) para ilustrar com menos densidade o mesmo
    # fato que o mapa de calor e a T3 já mostram.
    #
    # Religada no Cap. 6 (T6, 01/10/2026, CAP6-18): a monografia não tem o
    # teto de página do artigo, e a seção 6.3 usa a figura como `fig:curvas`.
    "figura1-curvas.pdf": "F1: curvas ROC e PR",
}


def gerar_figuras(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Gera as figuras **declaradas em `FIGURAS`**, na ordem do dicionário.

    O `if` por nome existe em vez de um laço direto porque as duas funções não
    têm a mesma assinatura: a F1 lê `results/curves.json` e precisa do
    `results_dir`, a F2 não. Uniformizar as assinaturas só para caber num laço
    trocaria clareza por simetria.
    """
    dados = carregar(results_dir)
    figures_dir = Path(figures_dir)
    escritos = []
    if "figura2-mapa-calor.pdf" in FIGURAS:
        escritos.append(
            figura2_mapa_de_calor(dados, figures_dir / "figura2-mapa-calor.pdf")
        )
    if "figura1-curvas.pdf" in FIGURAS:
        escritos.append(
            figura1_curvas(dados, Path(results_dir), figures_dir / "figura1-curvas.pdf")
        )
    return escritos


if __name__ == "__main__":
    raise SystemExit(main())
