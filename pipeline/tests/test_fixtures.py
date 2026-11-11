"""Confere os fixtures congelados antes da reexecucao consolidada (T1, RUN-04, ANL-01).

Os arquivos de `tests/fixtures/` foram gerados por script a partir de
`latex/tabelas/numeros.tex` e `results/matrix.json` da semente 42 antiga; aqui so se
confere que o conteudo congelado tem a forma que a spec descreve.
"""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _macros():
    texto = (FIXTURES / "macros_antes.txt").read_text(encoding="utf-8")
    return [linha for linha in texto.splitlines() if linha]


def _auroc():
    return json.loads(
        (FIXTURES / "auroc_fora_diagonal_seed42_antiga.json").read_text(encoding="utf-8")
    )


def test_macros_antes_tem_91_nomes_unicos():
    # T1 Done when: 91 macros no primeiro arquivo (RUN-04).
    nomes = _macros()
    assert len(nomes) == 91
    assert len(set(nomes)) == 91


def test_auroc_fora_diagonal_tem_72_registros_por_regime():
    # T1 Done when: 72 registros por regime (4 treinos x 3 testes x 6 modelos).
    registros = _auroc()
    por_regime = {}
    for r in registros:
        por_regime[r["regime"]] = por_regime.get(r["regime"], 0) + 1
    assert por_regime == {"etld1": 72, "oficial": 72}


def test_auroc_fora_diagonal_traz_treino_teste_e_modelo_sem_diagonal():
    # T1 What: cada registro tem treino, teste e modelo; nenhum e da diagonal.
    registros = _auroc()
    for r in registros:
        assert {"regime", "train_corpus", "test_corpus", "model", "auroc"} <= set(r)
        assert r["train_corpus"] != r["test_corpus"]
        assert 0.0 <= r["auroc"] <= 1.0
    chaves = {(r["regime"], r["train_corpus"], r["test_corpus"], r["model"]) for r in registros}
    assert len(chaves) == 144
