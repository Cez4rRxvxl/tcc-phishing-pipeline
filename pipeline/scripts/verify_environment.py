"""
Verificacao de ambiente — Tarefa T1.1 (SDD.md secao 9).

Prova que as bibliotecas do projeto (nucleo, os 6 modelos do artigo e HPO)
importam corretamente no ambiente ativo, sem depender de "pip disse OK".

NAO TREINA NENHUM MODELO. Nao chama .fit() em nenhum estimador nem em dados
reais — apenas instancia classes vazias/com hiperparametros minimos para
provar que import + construtor funcionam (restricao operacional da SDD.md
secao 0: nenhum treinamento nesta maquina).

Uso:
    python scripts/verify_environment.py

Saida: 0 se tudo OK, 1 se qualquer verificacao falhar.
"""
import sys

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print("-" * 60)

results = []


def check(name, fn):
    try:
        fn()
        print(f"[OK]   {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        results.append((name, False, str(e)))


# --- Interpretador --------------------------------------------------------

# Acrescentado em 06/08/2026. Ate aqui as 18 checagens eram todas de
# biblioteca: a versao do Python era *impressa* na linha 19 e nunca
# *conferida*. Foi por isso que a rodada 2 saiu num 3.14.2 enquanto a
# documentacao mandava 3.14.3, com "18/18 OK" na tela. Naquele caso o
# desvio era inofensivo -- os wheels de numpy/scipy/sklearn/catboost tem
# etiqueta `cp314`, que nao distingue patch, logo o binario que faz a
# conta era o mesmo. O que o portao nao podia continuar deixando passar e
# a troca de versao *menor*: num 3.13 ou 3.15 os wheels seriam outros
# binarios e o numero poderia mudar de verdade, com a mesma tela verde.
# Confere-se (3, 14) e nao o patch, porque e a granularidade que os wheels
# de fato distinguem -- exigir patch exato daria falha onde nao ha risco.
PYTHON_MINOR_EXIGIDO = (3, 14)


_EXIGIDO_STR = f"{PYTHON_MINOR_EXIGIDO[0]}.{PYTHON_MINOR_EXIGIDO[1]}"


def _python():
    atual = sys.version_info[:2]
    assert atual == PYTHON_MINOR_EXIGIDO, (
        f"este projeto exige Python {_EXIGIDO_STR}.x, e o interpretador"
        f" ativo e {atual[0]}.{atual[1]}."
        f" Os wheels pinados na requirements.lock sao cp"
        f"{PYTHON_MINOR_EXIGIDO[0]}{PYTHON_MINOR_EXIGIDO[1]}:"
        f" noutra versao menor o pip instalaria binarios diferentes."
        f" Crie o venv com: py -{_EXIGIDO_STR} -m venv venv"
    )


check(f"python {_EXIGIDO_STR}.x", _python)


# --- Nucleo ---------------------------------------------------------------

def _numpy():
    import numpy as np
    assert np.array([1, 2, 3]).sum() == 6


def _pandas():
    import pandas as pd
    assert len(pd.DataFrame({"a": [1, 2]})) == 2


def _pyarrow():
    import pyarrow as pa
    assert len(pa.array([1, 2, 3])) == 3


def _scipy():
    import scipy
    from scipy import stats  # noqa: F401


def _joblib():
    import joblib  # noqa: F401


def _tldextract():
    import tldextract
    ext = tldextract.extract("http://sub.example.com/path")
    assert ext.domain == "example"


def _pyyaml():
    import yaml
    assert yaml.safe_load("a: 1")["a"] == 1


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401


check("numpy", _numpy)
check("pandas", _pandas)
check("pyarrow", _pyarrow)
check("scipy", _scipy)
check("joblib", _joblib)
check("tldextract", _tldextract)
check("pyyaml", _pyyaml)
check("matplotlib", _matplotlib)

# --- Os 6 modelos do artigo (D4) ------------------------------------------
# 3 vem do scikit-learn, 3 sao bibliotecas de GBM dedicadas.

def _sklearn_logreg():
    from sklearn.linear_model import LogisticRegression
    LogisticRegression(max_iter=100)


def _sklearn_rf():
    from sklearn.ensemble import RandomForestClassifier
    RandomForestClassifier(n_estimators=10)


def _sklearn_mlp():
    from sklearn.neural_network import MLPClassifier
    MLPClassifier(max_iter=100)


def _sklearn_metrics():
    from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: F401
    from sklearn.model_selection import GroupShuffleSplit  # noqa: F401


check("sklearn.LogisticRegression", _sklearn_logreg)
check("sklearn.RandomForestClassifier", _sklearn_rf)
check("sklearn.MLPClassifier", _sklearn_mlp)
check("sklearn.metrics/model_selection", _sklearn_metrics)


def _xgboost():
    import xgboost as xgb
    xgb.XGBClassifier(n_estimators=10)


def _lightgbm():
    import lightgbm as lgb
    lgb.LGBMClassifier(n_estimators=10)


def _catboost():
    import catboost as cb
    cb.CatBoostClassifier(iterations=10, verbose=False)


check("xgboost.XGBClassifier", _xgboost)
check("lightgbm.LGBMClassifier", _lightgbm)
check("catboost.CatBoostClassifier", _catboost)

# --- HPO (Etapa 2 do D5 — nao executado aqui, so importado) ---------------

def _optuna():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    assert study is not None


check("optuna", _optuna)

# --- Dev --------------------------------------------------------------

def _pytest():
    import pytest  # noqa: F401


def _ruff():
    import subprocess
    out = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True, text=True, check=True,
    )
    assert "ruff" in out.stdout.lower()


check("pytest", _pytest)
check("ruff", _ruff)

print("-" * 60)
n_fail = sum(1 for _, ok, _ in results if not ok)
n_total = len(results)
print(f"Total: {n_total - n_fail}/{n_total} OK")
if n_fail:
    print("FALHAS:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("TODAS AS BIBLIOTECAS IMPORTAM CORRETAMENTE.")
    sys.exit(0)
