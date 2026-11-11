"""E4: teste de Friedman com pós-teste de Nemenyi entre os seis modelos.

Por que este módulo existe
--------------------------
O EST-01 do PRD pede a comparação estatística dos seis modelos **sem novo
treino**, só a partir do `results/matrix.json`. O `report.py` declara que não
calcula métrica nenhuma ("aqui só entra aritmética que um revisor confere de
cabeça"), e postos, qui-quadrado e amplitude estudentizada não são essa
aritmética. Por isso a conta mora aqui, com teste próprio, e o `report.py` só
formata o que este módulo devolve.

O protocolo NÃO é escolha deste módulo
--------------------------------------
Ele foi prometido no Capítulo 4 (`04-metodologia/08-estatistica.tex`) e é
implementado exatamente como está escrito lá:

- os blocos são as **16 células** da matriz cruzada no **Regime B** (`etld1`);
- a métrica é a **AUROC**;
- o nível de significância é **0,05**;
- os postos são atribuídos em cada bloco separadamente, posto 1 para o melhor
  modelo, com posto médio em caso de empate (Demšar, 2006, p. 11);
- dois modelos diferem quando os postos médios diferem ao menos pela
  diferença crítica (CD) de Nemenyi.

A semente que decide é a canônica (42), pela convenção do mesmo capítulo; as
sementes 7 e 13 entram só como faixa, como em todas as outras tabelas.

O Cap. 4 também registra a ressalva: as quatro células de uma linha da matriz
compartilham o dataset de treino e não são blocos independentes. Este módulo
não corrige isso (não há correção no Demšar para blocos dependentes); a
ressalva é do texto.

Duas estatísticas, uma decisão
------------------------------
O Cap. 4 promete o teste de Friedman, e o `chi2` é o que decide. O F de Iman e
Davenport sai junto porque o Demšar o recomenda como menos conservador
(p. 11); se as duas discordassem, o texto teria de dizer, e o
`test_estatistica.py` trava que hoje concordam.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

# Protocolo do Cap. 4, Seção sec:met-estatistica. Mudar qualquer um destes três
# valores exige um AD novo no .specs/STATE.md e a reescrita daquela seção.
REGIME = "etld1"
METRICA = "auroc"
ALFA = 0.05


def blocos_da_matriz(
    registros: list, regime: str = REGIME, metrica: str = METRICA
) -> tuple[list, list, np.ndarray]:
    """Monta a matriz N x k (blocos x modelos) a partir dos registros do `matrix.json`.

    Devolve `(blocos, modelos, valores)`, com blocos e modelos em ordem
    alfabética para que duas execuções produzam a mesma matriz.

    Falha alto se um bloco não tiver os mesmos modelos que os outros ou se
    alguma métrica for `None`: o Friedman exige o quadro completo, e completar
    com zero ou descartar o bloco em silêncio mudaria o resultado sem aviso.
    """
    celulas: dict = {}
    for r in registros:
        if r["regime"] != regime:
            continue
        chave = (r["train_corpus"], r["test_corpus"])
        valor = r["metrics"].get(metrica)
        if valor is None:
            raise ValueError(
                f"{metrica} indefinida em {chave} para {r['model']}: o teste de "
                f"Friedman exige todos os modelos em todos os blocos"
            )
        if r["model"] in celulas.setdefault(chave, {}):
            raise ValueError(f"modelo {r['model']} repetido no bloco {chave}")
        celulas[chave][r["model"]] = valor

    if not celulas:
        raise ValueError(f"nenhum registro no regime {regime!r}")
    blocos = sorted(celulas)
    modelos = sorted(celulas[blocos[0]])
    for b in blocos:
        if sorted(celulas[b]) != modelos:
            raise ValueError(
                f"o bloco {b} tem os modelos {sorted(celulas[b])}, e não {modelos}"
            )
    valores = np.array([[celulas[b][m] for m in modelos] for b in blocos], dtype=float)
    return blocos, modelos, valores


def postos(valores: np.ndarray) -> np.ndarray:
    """Posto de cada modelo em cada bloco: 1 para o maior valor, médio no empate.

    A métrica é "quanto maior, melhor" (AUROC), por isso o sinal trocado.
    """
    return np.vstack([stats.rankdata(-linha, method="average") for linha in valores])


def friedman(valores: np.ndarray) -> dict:
    """Estatística de Friedman e o F de Iman e Davenport (Demšar, 2006, p. 11).

    Usa a fórmula do Demšar, sem a correção de empates que o
    `scipy.stats.friedmanchisquare` aplica: é a fórmula que o texto cita. Sem
    empate as duas coincidem, e o teste confere isso.
    """
    n, k = valores.shape
    if n < 2 or k < 2:
        raise ValueError(f"Friedman exige ao menos 2 blocos e 2 modelos, e não {n}x{k}")
    medios = postos(valores).mean(axis=0)
    qui2 = 12 * n / (k * (k + 1)) * (np.sum(medios**2) - k * (k + 1) ** 2 / 4)
    gl = k - 1
    denominador = n * (k - 1) - qui2
    # Concordância perfeita entre os blocos zera o denominador: o F é infinito e
    # o p é zero, e não uma divisão por zero.
    ff = np.inf if denominador <= 0 else (n - 1) * qui2 / denominador
    gl_ff = (k - 1, (k - 1) * (n - 1))
    return {
        "n_blocos": n,
        "k_modelos": k,
        "postos_medios": medios,
        "qui2": float(qui2),
        "gl": gl,
        "p": float(stats.chi2.sf(qui2, gl)),
        "ff": float(ff),
        "gl_ff": gl_ff,
        "p_ff": 0.0 if np.isinf(ff) else float(stats.f.sf(ff, *gl_ff)),
    }


def valor_critico_nemenyi(k: int, alfa: float = ALFA) -> float:
    """`q_alfa` da Tabela 5(a) do Demšar: amplitude estudentizada dividida por raiz de 2.

    Calculado em vez de copiado da tabela, para valer para qualquer `k`; o
    teste confere a linha de `alfa = 0,05` e a de `0,10` contra a tabela.
    """
    return float(stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2))


def diferenca_critica(k: int, n: int, alfa: float = ALFA) -> float:
    """CD de Nemenyi: `q_alfa * sqrt(k(k+1) / 6N)` (Demšar, 2006, p. 11)."""
    return valor_critico_nemenyi(k, alfa) * float(np.sqrt(k * (k + 1) / (6 * n)))


def pares_significativos(modelos: list, medios, cd: float) -> list[tuple[str, str]]:
    """Pares cujos postos médios diferem ao menos pela CD, `(melhor, pior)`.

    A comparação usa `>=`, que é o "ao menos" do texto. Ordem fixa: pelo posto
    do melhor e depois pelo do pior, para o `.tex` regenerar byte a byte.
    """
    pares = []
    for i, j in combinations(range(len(modelos)), 2):
        if abs(medios[i] - medios[j]) >= cd:
            melhor, pior = (i, j) if medios[i] < medios[j] else (j, i)
            pares.append((melhor, pior))
    pares.sort(key=lambda p: (medios[p[0]], medios[p[1]], modelos[p[0]], modelos[p[1]]))
    return [(modelos[a], modelos[b]) for a, b in pares]


def grupos_sem_diferenca(modelos: list, medios, cd: float) -> list[tuple[str, ...]]:
    """Grupos maximais de modelos consecutivos no posto que não diferem entre si.

    São as barras horizontais do diagrama de diferença crítica: cada uma liga
    modelos cuja amplitude de postos fica abaixo da CD. Um grupo contido em
    outro não é desenhado. Modelo isolado (sem par abaixo da CD) não gera barra.
    """
    ordem = sorted(range(len(modelos)), key=lambda i: (medios[i], modelos[i]))
    grupos = []
    for a in range(len(ordem)):
        b = a
        while b + 1 < len(ordem) and medios[ordem[b + 1]] - medios[ordem[a]] < cd:
            b += 1
        if b > a and not (grupos and grupos[-1][1] >= b):
            grupos.append((a, b))
    return [tuple(modelos[ordem[i]] for i in range(a, b + 1)) for a, b in grupos]


def comparar(
    registros: list,
    regime: str = REGIME,
    metrica: str = METRICA,
    alfa: float = ALFA,
) -> dict:
    """Friedman e Nemenyi sobre os registros de uma semente do `matrix.json`."""
    blocos, modelos, valores = blocos_da_matriz(registros, regime, metrica)
    resultado = friedman(valores)
    medios = resultado["postos_medios"]
    cd = diferenca_critica(len(modelos), len(blocos), alfa)
    rejeita = resultado["p"] < alfa
    # O pós-teste só se aplica com a hipótese nula rejeitada (Demšar, 2006,
    # p. 11). Sem rejeição, nenhum par é declarado diferente, mesmo que a
    # distância de postos passe da CD.
    pares = pares_significativos(modelos, medios, cd) if rejeita else []
    resultado.update({
        "regime": regime,
        "metrica": metrica,
        "alfa": alfa,
        "blocos": blocos,
        "modelos": modelos,
        "postos_medios": {m: float(v) for m, v in zip(modelos, medios)},
        "rejeita_h0": rejeita,
        "q_alfa": valor_critico_nemenyi(len(modelos), alfa),
        "cd": cd,
        "pares_significativos": pares,
        "grupos": grupos_sem_diferenca(modelos, medios, cd),
    })
    return resultado


__all__ = [
    "REGIME",
    "METRICA",
    "ALFA",
    "blocos_da_matriz",
    "postos",
    "friedman",
    "valor_critico_nemenyi",
    "diferenca_critica",
    "pares_significativos",
    "grupos_sem_diferenca",
    "comparar",
]
