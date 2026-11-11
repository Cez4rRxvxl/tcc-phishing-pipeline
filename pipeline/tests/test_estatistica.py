"""E4: o `estatistica.py` calcula o Friedman e o Nemenyi que o Cap. 6 publica.

Três famílias de teste:

1. **Oráculo externo.** O exemplo resolvido do Demšar (2006, Tabela 6: quatro
   variantes do C4.5 em 14 datasets, AUC) tem postos médios, estatísticas e CD
   publicados. Reproduzi-los prova a fórmula sem depender dos nossos dados.
2. **O protocolo do Cap. 4.** 16 blocos do Regime B, AUROC, alfa 0,05: se
   alguém trocar a constante, o texto da Seção sec:met-estatistica passa a
   mentir, e estes testes caem.
3. **Os dados reais.** O resultado sobre o `matrix.json` fica travado, junto
   com a concordância entre o qui-quadrado e o F de Iman e Davenport.

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import estatistica

REPO_ROOT = Path(__file__).resolve().parent.parent
CAP4_ESTATISTICA = (
    REPO_ROOT.parent / "latex" / "capitulos" / "04-metodologia" / "08-estatistica.tex"
)

# Demšar (2006), Tabela 6. Colunas: C4.5, C4.5+m, C4.5+cf, C4.5+m+cf.
DEMSAR_TABELA6 = np.array([
    [0.763, 0.768, 0.771, 0.798],
    [0.599, 0.591, 0.590, 0.569],
    [0.954, 0.971, 0.968, 0.967],
    [0.628, 0.661, 0.654, 0.657],
    [0.882, 0.888, 0.886, 0.898],
    [0.936, 0.931, 0.916, 0.931],
    [0.661, 0.668, 0.609, 0.685],
    [0.583, 0.583, 0.563, 0.625],
    [0.775, 0.838, 0.866, 0.875],
    [1.000, 1.000, 1.000, 1.000],
    [0.940, 0.962, 0.965, 0.962],
    [0.619, 0.666, 0.614, 0.669],
    [0.972, 0.981, 0.975, 0.975],
    [0.957, 0.978, 0.946, 0.970],
])
DEMSAR_MODELOS = ["c45", "c45_m", "c45_cf", "c45_m_cf"]
# Os postos entre parênteses na mesma tabela. A AUC impressa está arredondada a
# 3 casas: no mushroom (linha 13), 0,975 e 0,975 receberam os postos 2 e 3, e
# um empate na AUC impressa daria 2,5 e 2,5. O Friedman é testado sobre estes
# postos; o `postos()`, sobre as outras 13 linhas.
DEMSAR_POSTOS = np.array([
    [4, 3, 2, 1], [1, 2, 3, 4], [4, 1, 2, 3], [4, 1, 3, 2], [4, 2, 3, 1],
    [1, 2.5, 4, 2.5], [3, 2, 4, 1], [2.5, 2.5, 4, 1], [4, 3, 2, 1],
    [2.5, 2.5, 2.5, 2.5], [4, 2.5, 1, 2.5], [3, 2, 4, 1], [4, 1, 2, 3],
    [3, 1, 4, 2],
])
LINHA_MUSHROOM = 12
# Postos viram valores com o sinal trocado: o maior valor tem o posto 1.
DEMSAR_VALORES_SEM_ARREDONDAMENTO = -DEMSAR_POSTOS


def _registros(valores: np.ndarray, modelos: list, regime: str = "etld1") -> list:
    """Converte uma matriz blocos x modelos no formato do `matrix.json`."""
    return [
        {"regime": regime, "train_corpus": f"t{i:02d}", "test_corpus": "x",
         "model": m, "metrics": {"auroc": float(v)}}
        for i, linha in enumerate(valores)
        for m, v in zip(modelos, linha)
    ]


# ---------------------------------------------------------------------------
# 1. O exemplo resolvido do Demšar
# ---------------------------------------------------------------------------


def test_postos_casam_com_os_publicados_inclusive_nos_empates():
    """Empates em iris, lung cancer e primary tumor recebem posto médio."""
    calculados = estatistica.postos(DEMSAR_TABELA6)
    outras = [i for i in range(len(DEMSAR_POSTOS)) if i != LINHA_MUSHROOM]
    assert calculados[outras].tolist() == DEMSAR_POSTOS[outras].tolist()


def test_reproduz_os_postos_medios_da_tabela6_do_demsar():
    medios = estatistica.postos(DEMSAR_VALORES_SEM_ARREDONDAMENTO).mean(axis=0)
    assert medios == pytest.approx([3.143, 2.000, 2.893, 1.964], abs=5e-4)


def test_reproduz_o_qui2_e_o_ff_do_demsar():
    res = estatistica.friedman(DEMSAR_VALORES_SEM_ARREDONDAMENTO)
    assert res["qui2"] == pytest.approx(9.28, abs=5e-3)
    assert res["ff"] == pytest.approx(3.69, abs=5e-3)
    assert res["gl_ff"] == (3, 39)
    # "The critical value of F(3, 39) for alfa = 0.05 is 2.85, so we reject."
    assert stats.f.ppf(0.95, 3, 39) == pytest.approx(2.85, abs=5e-3)
    assert res["p_ff"] < 0.05


@pytest.mark.parametrize("alfa, linha", [
    (0.05, [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]),
    (0.10, [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920]),
])
def test_valor_critico_casa_com_a_tabela5a_do_demsar(alfa, linha):
    """O `q_alfa` é calculado e não copiado; a tabela de 2 a 10 classificadores confere."""
    for k, esperado in zip(range(2, 11), linha):
        assert estatistica.valor_critico_nemenyi(k, alfa) == pytest.approx(esperado, abs=1e-3)


def test_reproduz_a_cd_e_a_conclusao_do_demsar():
    """Com alfa 0,05 nenhum par difere (CD 1,25); com 0,10, o C4.5 puro perde para dois."""
    assert estatistica.diferenca_critica(4, 14, 0.05) == pytest.approx(1.25, abs=5e-3)
    assert estatistica.diferenca_critica(4, 14, 0.10) == pytest.approx(1.12, abs=5e-3)

    res05 = estatistica.comparar(_registros(DEMSAR_VALORES_SEM_ARREDONDAMENTO, DEMSAR_MODELOS), alfa=0.05)
    assert res05["pares_significativos"] == []

    res10 = estatistica.comparar(_registros(DEMSAR_VALORES_SEM_ARREDONDAMENTO, DEMSAR_MODELOS), alfa=0.10)
    assert sorted(res10["pares_significativos"]) == [("c45_m", "c45"), ("c45_m_cf", "c45")]


def test_sem_empate_a_formula_coincide_com_a_do_scipy():
    """O scipy corrige empates e a fórmula do Demšar não; sem empate, são a mesma conta."""
    rng = np.random.default_rng(0)
    valores = rng.random((16, 6))
    res = estatistica.friedman(valores)
    ref = stats.friedmanchisquare(*valores.T)
    assert res["qui2"] == pytest.approx(ref.statistic)
    assert res["p"] == pytest.approx(ref.pvalue)


# ---------------------------------------------------------------------------
# 2. Casos de borda
# ---------------------------------------------------------------------------


def test_posto_1_e_o_maior_valor():
    assert estatistica.postos(np.array([[0.9, 0.5, 0.7]])).tolist() == [[1.0, 3.0, 2.0]]


def test_concordancia_perfeita_nao_divide_por_zero():
    valores = np.tile([0.9, 0.8, 0.7], (5, 1))
    res = estatistica.friedman(valores)
    assert np.isinf(res["ff"]) and res["p_ff"] == 0.0


def test_sem_rejeicao_nenhum_par_e_declarado_diferente():
    """O pós-teste só vale com H0 rejeitada, mesmo que a distância passe da CD."""
    valores = np.array([[0.9, 0.8, 0.7], [0.7, 0.8, 0.9]] * 2)
    res = estatistica.comparar(_registros(valores, ["a", "b", "c"]))
    assert not res["rejeita_h0"]
    assert res["pares_significativos"] == []


def test_metrica_indefinida_falha_em_vez_de_virar_zero():
    regs = _registros(np.array([[0.9, 0.8], [0.7, 0.6]]), ["a", "b"])
    regs[0]["metrics"]["auroc"] = None
    with pytest.raises(ValueError, match="indefinida"):
        estatistica.blocos_da_matriz(regs)


def test_bloco_incompleto_falha():
    regs = _registros(np.array([[0.9, 0.8], [0.7, 0.6]]), ["a", "b"])
    with pytest.raises(ValueError, match="modelos"):
        estatistica.blocos_da_matriz(regs[:-1])


def test_ignora_o_outro_regime():
    regs = _registros(np.array([[0.9, 0.8], [0.7, 0.6]]), ["a", "b"])
    regs += _registros(np.array([[0.1, 0.2]]), ["a", "b"], regime="oficial")
    blocos, _, valores = estatistica.blocos_da_matriz(regs)
    assert len(blocos) == 2 and valores.min() >= 0.6


def test_grupos_sao_maximais_e_sem_modelo_isolado():
    medios = [1.0, 1.5, 2.0, 4.0, 6.0]
    grupos = estatistica.grupos_sem_diferenca(list("abcde"), medios, cd=1.1)
    assert grupos == [("a", "b", "c")]


# ---------------------------------------------------------------------------
# 3. O protocolo do Cap. 4 e os dados reais
# ---------------------------------------------------------------------------


def test_protocolo_e_o_prometido_no_cap4():
    """Regime B, AUROC e 0,05, como a Seção sec:met-estatistica escreve."""
    texto = CAP4_ESTATISTICA.read_text(encoding="utf-8")
    assert "16 células da matriz cruzada no Regime B" in texto
    assert "a métrica é a AUROC" in texto
    assert "nível de significância é 0,05" in texto
    assert (estatistica.REGIME, estatistica.METRICA, estatistica.ALFA) == (
        "etld1", "auroc", 0.05
    )


@pytest.fixture(scope="module")
def matriz():
    caminho = REPO_ROOT / "results" / "matrix.json"
    if not caminho.exists():
        pytest.skip("results/matrix.json ausente")
    return json.loads(caminho.read_text(encoding="utf-8"))


def test_os_blocos_sao_as_16_celulas_e_os_6_modelos(matriz):
    blocos, modelos, valores = estatistica.blocos_da_matriz(matriz)
    assert valores.shape == (16, 6)
    assert len({tr for tr, _ in blocos}) == 4 and len({te for _, te in blocos}) == 4
    assert len(modelos) == 6


def test_resultado_da_semente_canonica_esta_travado(matriz):
    """Valores de 01/10/2026. Se mudarem, o Cap. 6 muda junto, e alguém tem de saber."""
    res = estatistica.comparar(matriz)
    # Expectativa da rodada consolidada (AD-055, que substitui o AD-010). Valores
    # da rodada antiga: qui2 = 22,0, H0 rejeitada, postos 2,4375 (LightGBM) e
    # 4,6875 (regressão logística) e quatro pares significativos. Agora o teste
    # de Friedman não rejeita H0 na semente canônica (Iman-Davenport a 0,0507,
    # logo acima de 0,05) e nenhum par supera a diferença crítica.
    assert res["qui2"] == pytest.approx(10.75, abs=1e-9)
    assert not res["rejeita_h0"]
    assert res["p_ff"] > estatistica.ALFA, "Friedman e Iman-Davenport discordam"
    assert res["p_ff"] == pytest.approx(0.0507, abs=5e-5)
    assert res["cd"] == pytest.approx(1.885, abs=5e-4)
    assert res["postos_medios"]["lightgbm"] == pytest.approx(2.75)
    assert res["postos_medios"]["logistic_regression"] == pytest.approx(4.625)
    assert sorted(res["pares_significativos"]) == []
