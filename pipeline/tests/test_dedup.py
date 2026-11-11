"""T8: deduplicação pelo `url_norm` por regime (spec DEDUP, critérios 1 a 6).

Quadro sintético, uma linha por caso da spec. As posições esperadas estão
escritas à mão a partir das regras da spec, não derivadas da implementação.

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

import numpy as np
import pandas as pd
import pytest

import dedup
from data.url_canonicalization import canonicalize_url
from dedup import DedupError, deduplicar

# pos  url_norm            rótulo  split   caso
#  0   a.com/1             0       train   primeira de uma duplicata no treino
#  1   a.com/1             0       train   duplicata no treino
#  2   b.com/1             1       test    primeira de uma duplicata no teste
#  3   b.com/1             1       test    duplicata no teste
#  4   c.com/1             0       train   URL repetida entre treino e teste,
#  5   c.com/1             0       test    com o mesmo rótulo
#  6   d.com/1             0       train   conflito dentro do treino
#  7   d.com/1             1       train
#  8   e.com/1             0       train   conflito entre treino e teste
#  9   e.com/1             1       test
# 10   f.com/x (www.)      1       train   variante com www.
# 11   f.com/x (sem www.)  1       train   variante sem www.
# 12   None                0       train   sem url_norm
# 13   None                0       train   sem url_norm
_LINHAS = [
    ("http://a.com/1", 0, "train"),
    ("http://a.com/1", 0, "train"),
    ("http://b.com/1", 1, "test"),
    ("http://b.com/1", 1, "test"),
    ("http://c.com/1", 0, "train"),
    ("http://c.com/1", 0, "test"),
    ("http://d.com/1", 0, "train"),
    ("http://d.com/1", 1, "train"),
    ("http://e.com/1", 0, "train"),
    ("http://e.com/1", 1, "test"),
    ("http://www.f.com/x", 1, "train"),
    ("http://f.com/x", 1, "train"),
    ("", 0, "train"),
    ("", 0, "train"),
]


@pytest.fixture
def quadro():
    norm = [canonicalize_url(u)["url_norm"] if u else None for u, _, _ in _LINHAS]
    return pd.DataFrame(
        {
            "url_norm": norm,
            "label": [r for _, r, _ in _LINHAS],
            "split_oficial": [s for _, _, s in _LINHAS],
        }
    )


def test_variantes_www_tem_o_mesmo_url_norm(quadro):
    # Premissa do quadro: as duas variantes colapsam na mesma chave.
    assert quadro["url_norm"][10] == quadro["url_norm"][11]
    assert pd.isna(quadro["url_norm"][12])


class TestRegimeB:
    def test_dedup_no_dataset_inteiro_mantem_primeira_ocorrencia(self, quadro):
        # DEDUP 1: primeira ocorrência por url_norm no dataset inteiro (c.com
        # fica só a de treino); DEDUP 4: d.com e e.com saem por inteiro;
        # DEDUP 1 + www.: só a posição 10 de f.com; sem url_norm ficam.
        r = deduplicar(quadro, "etld1")
        assert r.posicoes.tolist() == [0, 2, 4, 10, 12, 13]

    def test_contagens(self, quadro):
        # DEDUP 5.
        c = deduplicar(quadro, "etld1").contagens
        assert c == {
            "n_antes": 14,
            "n_conflito_linhas": 4,
            "n_conflito_urls": 2,
            "n_duplicatas_removidas": 4,  # 1, 3, 5, 11
            "n_final": 6,
            "n_sem_url_norm": 2,
        }

    def test_sem_url_norm_nao_participa_da_dedup(self, quadro):
        # DEDUP (premissa): as duas linhas sem chave não se deduplicam entre si.
        r = deduplicar(quadro, "etld1")
        assert {12, 13} <= set(r.posicoes.tolist())


class TestRegimeA:
    def test_dedup_dentro_do_treino_e_dentro_do_teste(self, quadro):
        # DEDUP 2: duplicata de treino (1) e de teste (3) saem; a URL repetida
        # entre treino e teste (4 e 5) fica nas duas partições (DEDUP 3);
        # DEDUP 4: d.com e e.com saem por inteiro; 11 é duplicata de 10.
        r = deduplicar(quadro, "oficial")
        assert r.posicoes.tolist() == [0, 2, 4, 5, 10, 12, 13]

    def test_contagens_incluem_repetidas_entre_treino_e_teste(self, quadro):
        # DEDUP 3 e 5.
        c = deduplicar(quadro, "oficial").contagens
        assert c == {
            "n_antes": 14,
            "n_conflito_linhas": 4,
            "n_conflito_urls": 2,
            "n_duplicatas_removidas": 3,  # 1, 3, 11
            "n_final": 7,
            "n_sem_url_norm": 2,
            "n_repetidas_treino_teste": 1,  # c.com/1
        }

    def test_regime_b_nao_tem_a_contagem_de_repetidas(self, quadro):
        assert "n_repetidas_treino_teste" not in deduplicar(quadro, "etld1").contagens


class TestDeterminismo:
    @pytest.mark.parametrize("regime", ["oficial", "etld1"])
    def test_duas_chamadas_dao_as_mesmas_posicoes(self, quadro, regime):
        # DEDUP 6.
        a = deduplicar(quadro, regime)
        b = deduplicar(quadro.copy(), regime)
        assert a.posicoes.tolist() == b.posicoes.tolist()
        assert a.contagens == b.contagens

    @pytest.mark.parametrize("regime", ["oficial", "etld1"])
    def test_posicoes_ordenadas_e_unicas(self, quadro, regime):
        p = deduplicar(quadro, regime).posicoes
        assert p.tolist() == sorted(set(p.tolist()))

    def test_primeira_ocorrencia_independe_do_indice_do_quadro(self, quadro):
        # As posições são posicionais: um índice embaralhado não muda nada.
        embaralhado = quadro.copy()
        embaralhado.index = np.arange(len(quadro))[::-1] + 100
        assert (
            deduplicar(embaralhado, "etld1").posicoes.tolist()
            == deduplicar(quadro, "etld1").posicoes.tolist()
        )


class TestEntradaInvalida:
    def test_regime_desconhecido(self, quadro):
        with pytest.raises(DedupError, match="regime"):
            deduplicar(quadro, "xyz")

    def test_coluna_ausente(self, quadro):
        with pytest.raises(DedupError, match="url_norm"):
            deduplicar(quadro.drop(columns=["url_norm"]), "etld1")

    def test_regime_a_sem_split_oficial(self, quadro):
        with pytest.raises(DedupError, match="split_oficial"):
            deduplicar(quadro.drop(columns=["split_oficial"]), "oficial")


def test_modulo_exporta_a_api():
    assert set(dedup.__all__) >= {"deduplicar", "ResultadoDedup"}
