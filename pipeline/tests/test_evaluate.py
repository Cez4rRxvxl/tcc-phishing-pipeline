"""T4.2 — DoD: a matriz de confusão de B1 tem de dar 0,7914, nunca 2,85 nem nan.

Este arquivo é o teste de regressão do bug que invalidou o repositório legado
(docs/diagnostico.md §3.1, B1). A matriz exigida é

    tn = 338389, fp = 3433, fn = 5457, tp = 18165

e o valor correto do MCC é **0,7914**. O pipeline legado publicou **2,8497**
por overflow de `int64`, e três outros modelos deram `nan` — foi comparando
`2.85 > nan` que ele elegeu um campeão (B2).

Além do valor, os testes abaixo travam as três defesas do módulo:
aritmética exata, conferência contra o sklearn e validação de faixa.
"""

import math

import numpy as np
import pytest

import evaluate
from evaluate import MetricsError

# A matriz de confusão exata que o pipeline legado avaliou (B1).
B1_TN, B1_FP, B1_FN, B1_TP = 338389, 3433, 5457, 18165
B1_MCC_CORRETO = 0.7913687783833526
B1_MCC_PUBLICADO_ERRADO = 2.849736445840131


@pytest.fixture(scope="module")
def b1_arrays():
    """Reconstrói os vetores y_true/y_pred que produzem a matriz de B1.

    365.444 linhas — barato de materializar, e permite conferir a rota
    `matthews_corrcoef` do sklearn contra a rota por contagens.
    """
    y_true = np.array([0] * (B1_TN + B1_FP) + [1] * (B1_FN + B1_TP), dtype=np.int64)
    y_pred = np.array(
        [0] * B1_TN + [1] * B1_FP + [0] * B1_FN + [1] * B1_TP, dtype=np.int64
    )
    return y_true, y_pred


# ===========================================================================
# O DoD da T4.2
# ===========================================================================
class TestB1RegressaoMCC:
    def test_matriz_de_b1_da_07914(self):
        """O critério objetivo de conclusão da tarefa."""
        valor = evaluate.mcc_from_confusion(B1_TN, B1_FP, B1_FN, B1_TP)
        assert valor == pytest.approx(0.7914, abs=5e-5)
        assert valor == pytest.approx(B1_MCC_CORRETO, rel=1e-12)

    def test_nao_e_o_valor_publicado_errado(self):
        """2,8497 é impossível (MCC ∈ [-1,1]) e foi o que o legado publicou."""
        valor = evaluate.mcc_from_confusion(B1_TN, B1_FP, B1_FN, B1_TP)
        assert valor != pytest.approx(B1_MCC_PUBLICADO_ERRADO, rel=1e-6)
        assert -1.0 <= valor <= 1.0

    def test_nao_e_nan(self):
        """Nos outros 3 modelos o overflow caiu em faixa negativa e
        `np.sqrt` devolveu nan — o que permitiu `2.85 > nan` (B2)."""
        valor = evaluate.mcc_from_confusion(B1_TN, B1_FP, B1_FN, B1_TP)
        assert not math.isnan(valor)
        assert math.isfinite(valor)

    def test_concorda_com_sklearn(self, b1_arrays):
        """SDD §4 D6 exige `sklearn.metrics.matthews_corrcoef`. As duas rotas
        do módulo são independentes e têm de concordar."""
        y_true, y_pred = b1_arrays
        por_contagem = evaluate.mcc_from_confusion(B1_TN, B1_FP, B1_FN, B1_TP)
        por_sklearn = evaluate.mcc(y_true, y_pred)
        assert por_sklearn == pytest.approx(por_contagem, rel=1e-12)
        assert por_sklearn == pytest.approx(0.7914, abs=5e-5)

    def test_confusion_counts_reproduz_a_matriz(self, b1_arrays):
        y_true, y_pred = b1_arrays
        assert evaluate.confusion_counts(y_true, y_pred) == (B1_TN, B1_FP, B1_FN, B1_TP)


# ===========================================================================
# A causa raiz: tipo, não fórmula
# ===========================================================================
class TestCausaRaizDoOverflow:
    def test_o_produto_marginal_estoura_int64(self):
        """Documenta a aritmética: o produto dos 4 fatores vale ~6,0e19 contra
        um teto de int64 de 9,22e18. Este teste não valida o nosso código —
        registra a evidência que justifica ele existir."""
        produto = (
            (B1_TP + B1_FP) * (B1_TP + B1_FN) * (B1_TN + B1_FP) * (B1_TN + B1_FN)
        )
        assert produto == 59_964_496_224_571_849_872
        assert produto > np.iinfo(np.int64).max

    def test_a_implementacao_legada_realmente_dava_285(self):
        """Reproduz o bug exato de `07c_multiobjective_eval.py:50`.

        Se algum dia o numpy passar a levantar exceção em vez de dar a volta,
        este teste falha — e é um bom motivo para ler o cabeçalho do módulo.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            tn, fp, fn, tp = (np.int64(x) for x in (B1_TN, B1_FP, B1_FN, B1_TP))
            numerador = tp * tn - fp * fn
            denominador = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            legado = float(numerador / denominador)
        assert legado == pytest.approx(B1_MCC_PUBLICADO_ERRADO, rel=1e-9)
        assert legado > 1.0  # impossível para um MCC

    def test_numpy_int64_na_entrada_nao_contamina(self):
        """Passar `numpy.int64` (o que `confusion_matrix().ravel()` devolve)
        tem de dar o mesmo resultado: `int(...)` converte para precisão
        arbitrária antes de qualquer multiplicação."""
        como_np = evaluate.mcc_from_confusion(
            np.int64(B1_TN), np.int64(B1_FP), np.int64(B1_FN), np.int64(B1_TP)
        )
        como_py = evaluate.mcc_from_confusion(B1_TN, B1_FP, B1_FN, B1_TP)
        assert como_np == como_py

    def test_dtypes_estreitos_tambem_nao_contaminam(self):
        """uint16/int32 na entrada: mesma resposta. Relevante porque as
        features do projeto são uint16 (B13) e é fácil um dtype estreito
        vazar para uma contagem."""
        valores = [
            evaluate.mcc_from_confusion(*(dt(x) for x in (300, 5, 7, 200)))
            for dt in (int, np.int32, np.int64, np.uint16, np.uint32)
        ]
        assert len(set(valores)) == 1

    def test_corpus_gigante_nao_estoura(self):
        """Escala em que o int64 morre com folga: 10 milhões por célula, cujo
        produto marginal é ~1,6e29."""
        valor = evaluate.mcc_from_confusion(
            10_000_000, 1_000_000, 1_000_000, 10_000_000
        )
        assert -1.0 <= valor <= 1.0
        assert valor == pytest.approx(0.8181818181, rel=1e-9)

    def test_escala_extrema_permanece_na_faixa(self):
        valor = evaluate.mcc_from_confusion(10**12, 10**11, 10**11, 10**12)
        assert -1.0 <= valor <= 1.0


# ===========================================================================
# Propriedades matemáticas do MCC
# ===========================================================================
class TestPropriedadesDoMCC:
    def test_classificador_perfeito_da_1(self):
        assert evaluate.mcc_from_confusion(100, 0, 0, 100) == pytest.approx(1.0)

    def test_classificador_invertido_da_menos_1(self):
        assert evaluate.mcc_from_confusion(0, 100, 100, 0) == pytest.approx(-1.0)

    def test_previu_tudo_positivo_da_zero(self):
        """Denominador zero (tn+fp com fp=tudo, tn=0... aqui: nenhuma predição
        negativa). O sklearn devolve 0,0; nós também. NUNCA nan — foi o nan que
        permitiu B2."""
        valor = evaluate.mcc_from_confusion(0, 0, 0, 500)
        assert valor == 0.0
        assert not math.isnan(valor)

    def test_previu_tudo_negativo_da_zero(self):
        valor = evaluate.mcc_from_confusion(500, 0, 0, 0)
        assert valor == 0.0

    def test_concorda_com_sklearn_em_casos_aleatorios(self):
        """Verificação cruzada em 200 matrizes sorteadas, incluindo casos
        degenerados."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = int(rng.integers(10, 400))
            y_true = rng.integers(0, 2, size=n)
            y_pred = rng.integers(0, 2, size=n)
            tn, fp, fn, tp = evaluate.confusion_counts(y_true, y_pred)
            nosso = evaluate.mcc_from_confusion(tn, fp, fn, tp)
            deles = evaluate.mcc(y_true, y_pred)
            assert nosso == pytest.approx(deles, abs=1e-12)

    def test_contagem_negativa_e_rejeitada(self):
        with pytest.raises(MetricsError, match="negativa"):
            evaluate.mcc_from_confusion(-1, 0, 0, 10)


# ===========================================================================
# Validação de faixa — a defesa que faltava em B1
# ===========================================================================
class TestValidacaoDeFaixa:
    def _metricas_validas(self):
        return {
            "auroc": 0.95, "auc_pr": 0.90,
            "precision": 0.84, "recall": 0.77, "f1": 0.80,
            "fpr": 0.01, "mcc": 0.79, "brier": 0.03, "ece": 0.02,
            "threshold": 0.5,
            "tp": 18165, "fp": 3433, "tn": 338389, "fn": 5457,
        }

    def test_metricas_validas_passam(self):
        evaluate.validate_metrics(self._metricas_validas())

    def test_mcc_de_285_e_barrado(self):
        """O teste que teria impedido o artigo legado de existir."""
        m = self._metricas_validas()
        m["mcc"] = B1_MCC_PUBLICADO_ERRADO
        with pytest.raises(MetricsError, match="fora de"):
            evaluate.validate_metrics(m)

    def test_mcc_nan_e_barrado(self):
        m = self._metricas_validas()
        m["mcc"] = float("nan")
        with pytest.raises(MetricsError, match="finita"):
            evaluate.validate_metrics(m)

    @pytest.mark.parametrize(
        "campo,valor",
        [
            ("auroc", 1.5), ("auroc", -0.1),
            ("auc_pr", 2.0),
            ("precision", 1.01), ("recall", -0.01), ("f1", 3.0),
            ("fpr", 1.2), ("brier", -1.0), ("ece", 1.5),
            ("threshold", 1.5), ("mcc", -2.0),
        ],
    )
    def test_cada_metrica_fora_de_faixa_e_barrada(self, campo, valor):
        m = self._metricas_validas()
        m[campo] = valor
        with pytest.raises(MetricsError):
            evaluate.validate_metrics(m)

    def test_metrica_ausente_e_barrada(self):
        m = self._metricas_validas()
        del m["mcc"]
        with pytest.raises(MetricsError, match="ausentes"):
            evaluate.validate_metrics(m)

    def test_none_e_aceito_em_metrica_indefinida(self):
        """AUROC indefinido (teste com uma classe só) vem como null, não como
        0 — reportar 0 seria inventar um número."""
        m = self._metricas_validas()
        m["auroc"] = None
        m["auc_pr"] = None
        evaluate.validate_metrics(m)

    def test_contagem_negativa_e_barrada(self):
        m = self._metricas_validas()
        m["fp"] = -1
        with pytest.raises(MetricsError, match="contagem"):
            evaluate.validate_metrics(m)

    def test_contagem_nao_inteira_e_barrada(self):
        m = self._metricas_validas()
        m["tp"] = 18165.0
        with pytest.raises(MetricsError, match="contagem"):
            evaluate.validate_metrics(m)

    def test_matriz_vazia_e_barrada(self):
        m = self._metricas_validas()
        m.update({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        with pytest.raises(MetricsError, match="vazia"):
            evaluate.validate_metrics(m)

    def test_booleano_nao_passa_por_numero(self):
        m = self._metricas_validas()
        m["mcc"] = True
        with pytest.raises(MetricsError, match="numérica"):
            evaluate.validate_metrics(m)


# ===========================================================================
# evaluate_predictions — o bloco `metrics` do contrato §7.3
# ===========================================================================
class TestEvaluatePredictions:
    @pytest.fixture
    def dados(self):
        rng = np.random.default_rng(7)
        n = 2000
        y_true = rng.integers(0, 2, size=n)
        # Score correlacionado com o rótulo, com sobreposição.
        y_score = np.clip(0.3 * y_true + rng.normal(0.35, 0.18, size=n), 0, 1)
        return y_true, y_score

    def test_esquema_exato_de_7_3(self, dados):
        y_true, y_score = dados
        metrics, _ = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        assert tuple(metrics) == evaluate.METRIC_KEYS

    def test_todas_as_metricas_na_faixa(self, dados):
        y_true, y_score = dados
        metrics, _ = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        evaluate.validate_metrics(metrics)

    def test_contagens_somam_o_total(self, dados):
        y_true, y_score = dados
        metrics, _ = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(y_true)

    def test_mcc_bate_com_sklearn(self, dados):
        y_true, y_score = dados
        metrics, _ = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        esperado = evaluate.mcc(y_true, (y_score >= 0.5).astype(int))
        assert metrics["mcc"] == pytest.approx(esperado, abs=1e-12)

    def test_limiar_usa_maior_ou_igual(self):
        """Convenção: prevê phishing quando p >= limiar. Tem de casar com o
        `precision_recall_curve`, senão o limiar escolhido na validação não
        significa a mesma coisa no teste."""
        y_true = np.array([0, 1])
        y_score = np.array([0.5, 0.5])
        metrics, _ = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        assert metrics["tp"] == 1 and metrics["fp"] == 1
        assert metrics["tn"] == 0 and metrics["fn"] == 0

    def test_classificador_perfeito(self):
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.array([0.01] * 50 + [0.99] * 50)
        metrics, warnings = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["auc_pr"] == pytest.approx(1.0)
        assert metrics["mcc"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["fpr"] == pytest.approx(0.0)
        assert warnings == []

    def test_previsao_tudo_phishing_expoe_precision_baixa(self):
        """B6: o "campeão" legado previa tudo como phishing e tinha Precision
        0,0689, mas liderava o ranking porque ele era ordenado por Recall puro.
        Aqui o registro tem de mostrar o estrago, não esconder."""
        y_true = np.array([0] * 950 + [1] * 50)
        y_score = np.full(1000, 0.99)
        metrics, _ = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(0.05)
        assert metrics["fpr"] == pytest.approx(1.0)
        assert metrics["mcc"] == 0.0  # sem nenhuma predição negativa
        assert metrics["auroc"] == pytest.approx(0.5)

    def test_teste_com_uma_classe_avisa_e_devolve_none(self):
        y_true = np.zeros(100, dtype=int)
        y_score = np.full(100, 0.2)
        metrics, warnings = evaluate.evaluate_predictions(y_true, y_score, threshold=0.5)
        assert metrics["auroc"] is None
        assert metrics["auc_pr"] is None
        assert any("única classe" in w for w in warnings)
        assert metrics["mcc"] == 0.0

    def test_score_fora_de_faixa_e_rejeitado(self):
        """Margem de decisão (decision_function) em vez de probabilidade
        invalidaria Brier e ECE — é parente de B8."""
        y_true = np.array([0, 1, 0, 1])
        with pytest.raises(MetricsError, match=r"fora de \[0, 1\]"):
            evaluate.evaluate_predictions(y_true, np.array([-2.0, 3.1, 0.2, 0.9]), 0.5)

    def test_nan_no_score_e_rejeitado(self):
        y_true = np.array([0, 1, 0, 1])
        with pytest.raises(MetricsError, match="nan"):
            evaluate.evaluate_predictions(y_true, np.array([0.1, float("nan"), 0.2, 0.9]), 0.5)

    def test_rotulo_fora_de_0_1_e_rejeitado(self):
        """Corpora vêm com -1/1 e 2/1; `corpora.py` normaliza. Se um vetor
        cru vazar até aqui, tem de doer."""
        y_true = np.array([-1, 1, -1, 1])
        with pytest.raises(MetricsError, match="rótulos fora"):
            evaluate.evaluate_predictions(y_true, np.array([0.1, 0.9, 0.2, 0.8]), 0.5)

    def test_conjunto_vazio_e_rejeitado(self):
        with pytest.raises(MetricsError, match="vazio"):
            evaluate.evaluate_predictions(np.array([]), np.array([]), 0.5)

    def test_formatos_diferentes_sao_rejeitados(self):
        with pytest.raises(MetricsError, match="formatos diferentes"):
            evaluate.evaluate_predictions(np.array([0, 1]), np.array([0.5]), 0.5)

    def test_limiar_invalido_e_rejeitado(self):
        y_true = np.array([0, 1])
        y_score = np.array([0.2, 0.8])
        for limiar in (-0.1, 1.5):
            with pytest.raises(MetricsError, match="limiar"):
                evaluate.evaluate_predictions(y_true, y_score, limiar)

    def test_determinismo(self, dados):
        """RNF1: duas chamadas idênticas dão métricas idênticas."""
        y_true, y_score = dados
        a, _ = evaluate.evaluate_predictions(y_true, y_score, 0.4)
        b, _ = evaluate.evaluate_predictions(y_true, y_score, 0.4)
        assert a == b


# ===========================================================================
# Calibração
# ===========================================================================
class TestCalibracao:
    def test_ece_de_modelo_perfeitamente_calibrado_e_baixo(self):
        """Gera p e sorteia o rótulo com probabilidade p: por construção o
        modelo está calibrado, então o ECE tem de ser próximo de 0."""
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, size=200_000)
        y = (rng.uniform(0, 1, size=200_000) < p).astype(int)
        ece = evaluate.expected_calibration_error(y, p, n_bins=15)
        assert ece < 0.01

    def test_ece_de_modelo_descalibrado_e_alto(self):
        """Modelo que grita 0,99 para tudo, mas acerta metade."""
        y = np.array([0] * 500 + [1] * 500)
        p = np.full(1000, 0.99)
        ece = evaluate.expected_calibration_error(y, p, n_bins=15)
        assert ece == pytest.approx(0.49, abs=0.01)

    def test_ece_na_faixa_0_1(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(20, 500))
            y = rng.integers(0, 2, size=n)
            p = rng.uniform(0, 1, size=n)
            assert 0.0 <= evaluate.expected_calibration_error(y, p) <= 1.0

    def test_ece_bins_extremos(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        # Bin 0 e bin final: p=0.0 e p=1.0 não podem cair fora do range.
        assert evaluate.expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)

    def test_ece_bin_unico_e_media_global(self):
        y = np.array([0, 0, 0, 1])
        p = np.full(4, 0.6)
        assert evaluate.expected_calibration_error(y, p, n_bins=1) == pytest.approx(0.35)

    def test_brier_de_b8_seria_visivel(self):
        """B8: XGBoost com Brier 0,5257 e ECE 0,69 a 96% de acurácia. O ponto
        é que Brier e ECE ruins convivem com acurácia alta — por isso os dois
        entram no relatório (D6), sem recalibração por cima."""
        y = np.array([0] * 960 + [1] * 40)
        p = np.concatenate([np.full(960, 0.72), np.full(40, 0.72)])
        metrics, _ = evaluate.evaluate_predictions(y, p, threshold=0.9)
        assert metrics["brier"] > 0.4
        assert metrics["ece"] > 0.6
        # ... e ainda assim quase tudo classificado corretamente como benigno:
        assert (metrics["tn"] + metrics["tp"]) / 1000 == pytest.approx(0.96)


# ===========================================================================
# Seleção de limiar (SDD §4 D6)
# ===========================================================================
class TestSelecaoDeLimiar:
    @pytest.fixture
    def val(self):
        rng = np.random.default_rng(3)
        n = 3000
        y = rng.integers(0, 2, size=n)
        p = np.clip(0.35 * y + rng.normal(0.3, 0.15, size=n), 0, 1)
        return y, p

    def test_limiar_na_faixa(self, val):
        y, p = val
        for criterio in evaluate.THRESHOLD_CRITERIA:
            limiar = evaluate.select_threshold(y, p, criterion=criterio)
            assert 0.0 <= limiar <= 1.0

    def test_limiar_f1_e_pelo_menos_tao_bom_quanto_05(self, val):
        y, p = val
        limiar = evaluate.select_threshold(y, p, criterion="f1")
        m_sel, _ = evaluate.evaluate_predictions(y, p, limiar)
        m_meio, _ = evaluate.evaluate_predictions(y, p, 0.5)
        assert m_sel["f1"] >= m_meio["f1"]

    def test_determinismo(self, val):
        y, p = val
        a = evaluate.select_threshold(y, p, "f1")
        b = evaluate.select_threshold(y, p, "f1")
        assert a == b

    def test_empate_resolve_para_o_menor_limiar(self):
        """Empate de F1 é resolvido a favor do recall: numa detecção de
        phishing, deixar passar é pior que alarmar."""
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        limiar = evaluate.select_threshold(y, p, "f1")
        assert limiar == pytest.approx(0.8)  # não 0.9

    def test_youden_separa_perfeitamente(self):
        y = np.array([0] * 50 + [1] * 50)
        p = np.array([0.1] * 50 + [0.9] * 50)
        limiar = evaluate.select_threshold(y, p, "youden")
        m, _ = evaluate.evaluate_predictions(y, p, limiar)
        assert m["mcc"] == pytest.approx(1.0)

    def test_validacao_de_uma_classe_cai_para_05(self):
        y = np.zeros(50, dtype=int)
        p = np.linspace(0, 1, 50)
        assert evaluate.select_threshold(y, p) == 0.5

    def test_criterio_desconhecido_e_rejeitado(self, val):
        y, p = val
        with pytest.raises(MetricsError, match="critério"):
            evaluate.select_threshold(y, p, "acuracia")


# ===========================================================================
# Registro de resultado (RF7, RNF4)
# ===========================================================================
class TestRegistroDeResultado:
    @pytest.fixture
    def metrics(self):
        rng = np.random.default_rng(11)
        y = rng.integers(0, 2, size=500)
        p = np.clip(0.3 * y + rng.normal(0.35, 0.2, size=500), 0, 1)
        m, _ = evaluate.evaluate_predictions(y, p, 0.5)
        return m

    def test_esquema_do_registro(self, metrics):
        rec = evaluate.make_result_record(
            run_id="abc1234-20260805T101500Z",
            regime="etld1",
            train_corpus="grambeddings",
            test_corpus="phishstorm",
            model="xgboost",
            metrics=metrics,
            fit_seconds=12.5,
        )
        for campo in ("run_id", "regime", "train_corpus", "test_corpus", "model",
                      "metrics", "fit_seconds"):
            assert campo in rec
        assert tuple(rec["metrics"]) == evaluate.METRIC_KEYS
        assert rec["diagonal"] is False
        assert rec["n_test"] == 500

    def test_diagonal_marcada(self, metrics):
        rec = evaluate.make_result_record(
            run_id="x", regime="etld1", train_corpus="iscx2016",
            test_corpus="iscx2016", model="mlp", metrics=metrics, fit_seconds=1.0,
        )
        assert rec["diagonal"] is True

    def test_avisos_ficam_fora_do_bloco_metrics(self, metrics):
        """RF7 — esquema estável: `metrics` tem exatamente as 14 chaves de
        §7.3; aviso e proveniência ficam no nível do registro."""
        rec = evaluate.make_result_record(
            run_id="x", regime="oficial", train_corpus="a", test_corpus="b",
            model="catboost", metrics=metrics, fit_seconds=1.0,
            warnings=["algum aviso"],
        )
        assert rec["warnings"] == ["algum aviso"]
        assert "warnings" not in rec["metrics"]

    def test_registro_valida_as_metricas(self, metrics):
        ruim = dict(metrics)
        ruim["mcc"] = B1_MCC_PUBLICADO_ERRADO
        with pytest.raises(MetricsError):
            evaluate.make_result_record(
                run_id="x", regime="etld1", train_corpus="a", test_corpus="b",
                model="xgboost", metrics=ruim, fit_seconds=1.0,
            )

    def test_regime_invalido_e_rejeitado(self, metrics):
        with pytest.raises(MetricsError, match="regime"):
            evaluate.make_result_record(
                run_id="x", regime="aleatorio", train_corpus="a", test_corpus="b",
                model="xgboost", metrics=metrics, fit_seconds=1.0,
            )

    def test_run_id_carimba_commit_e_timestamp(self):
        """RNF4: todo artefato carimbado com commit hash e timestamp."""
        run_id = evaluate.make_run_id()
        assert "-" in run_id
        assert run_id.endswith("Z")
        commit, _, ts = run_id.rpartition("-")
        assert commit
        assert len(ts) == len("20260805T101500Z")

    def test_run_id_nao_usa_utcnow_depreciado(self):
        """Pendência 6 de §13.4: `datetime.utcnow()` é depreciado no 3.14."""
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error", DeprecationWarning)
            evaluate.make_run_id()

    def test_json_serializavel(self, metrics):
        import json

        rec = evaluate.make_result_record(
            run_id="x", regime="etld1", train_corpus="a", test_corpus="b",
            model="lightgbm", metrics=metrics, fit_seconds=1.0,
        )
        recarregado = json.loads(json.dumps(rec))
        assert recarregado["metrics"]["mcc"] == rec["metrics"]["mcc"]

    def test_dataframe_achatado_para_csv(self, metrics):
        recs = [
            evaluate.make_result_record(
                run_id="x", regime="etld1", train_corpus="a", test_corpus=t,
                model="xgboost", metrics=metrics, fit_seconds=1.0,
            )
            for t in ("a", "b")
        ]
        df = evaluate.records_to_dataframe(recs)
        assert len(df) == 2
        for k in evaluate.METRIC_KEYS:
            assert k in df.columns
        assert "metrics" not in df.columns


# ---------------------------------------------------------------------------
# Curvas em grade fixa (T4.6) — e o defeito da primeira versão
# ---------------------------------------------------------------------------
class TestCurvasEmGradeFixa:
    """A curva PR usa envelope monótono, não interpolação linear.

    A primeira versão da `curve_points` fazia
    `np.interp(grade, recall[argsort(recall)], precisao[...])`. Como o recall
    vem repetido em massa (medido: 3.952 pontos repetidos em 4.001 a 3% de
    prevalência), o `np.interp` atravessava o degrau e inventava trecho que
    nenhum limiar produz. Para um classificador **perfeito** a curva gravada
    descia em reta de precisão 1,0 até 0,5, com área 0,7525 em vez de 1,0 —
    e esse é justamente o regime do projeto (a diagonal do ISCX dá 0,99999
    com quatro modelos, e ensemble de árvore empata score em massa).

    Estes testes travam a correção pelo comportamento, não pela implementação.
    """

    @staticmethod
    def _envelope_referencia(y, s, grade):
        """`max{p : r >= g}` por laço explícito — implementação independente.

        Mesma doutrina da defesa 2 do B1: o valor rápido é conferido contra
        uma rota lenta e obviamente correta, nunca contra si mesmo.
        """
        from sklearn.metrics import precision_recall_curve

        p, r, _ = precision_recall_curve(y, s)
        return [max(p[r >= g]) for g in grade]

    def test_classificador_perfeito_tem_precisao_um_em_toda_a_grade(self):
        y = np.array([0] * 500 + [1] * 500)
        s = np.array([0.1] * 500 + [0.9] * 500)
        pontos = evaluate.curve_points(y, s)
        assert pontos["pr_precision"] == [1.0] * len(pontos["grade"])
        assert pontos["roc_tpr"] == [1.0] * len(pontos["grade"])

    def test_score_com_muitos_empates_nao_afunda_a_precisao(self):
        """Ensemble de árvore produz poucos valores distintos de score."""
        rng = np.random.default_rng(1)
        y = np.array([0] * 2000 + [1] * 2000)
        s = np.where(
            y == 1,
            rng.choice([0.98, 0.99, 1.0], 4000),
            rng.choice([0.0, 0.01, 0.02], 4000),
        )
        pontos = evaluate.curve_points(y, s)
        assert min(pontos["pr_precision"]) == 1.0

    @pytest.mark.parametrize("prev,sep", [(0.30, 1.2), (0.03, 2.0), (0.50, 0.3)])
    def test_bate_com_a_implementacao_de_referencia(self, prev, sep):
        rng = np.random.default_rng(7)
        y = (rng.random(3000) < prev).astype(int)
        s = 1.0 / (1.0 + np.exp(-rng.normal(sep * y, 1.0)))
        pontos = evaluate.curve_points(y, s)
        esperado = self._envelope_referencia(y, s, pontos["grade"])
        for obtido, ref in zip(pontos["pr_precision"], esperado):
            assert obtido == pytest.approx(ref, abs=1e-6)

    def test_precisao_e_nao_crescente_em_recall(self):
        """Propriedade do envelope: se cai e volta a subir, não é envelope."""
        rng = np.random.default_rng(3)
        y = (rng.random(4000) < 0.2).astype(int)
        s = np.round(1.0 / (1.0 + np.exp(-rng.normal(1.5 * y, 1.0))), 2)
        pr = evaluate.curve_points(y, s)["pr_precision"]
        assert all(b <= a + 1e-9 for a, b in zip(pr, pr[1:]))

    def test_roc_continua_interpolada_linearmente(self):
        """Na ROC a interpolação é legítima: o ponto intermediário é atingível
        randomizando entre os dois classificadores. A área na grade tem de
        bater com o `roc_auc_score`."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(11)
        y = (rng.random(5000) < 0.4).astype(int)
        s = 1.0 / (1.0 + np.exp(-rng.normal(1.2 * y, 1.0)))
        pontos = evaluate.curve_points(y, s)
        area = np.trapezoid(pontos["roc_tpr"], pontos["grade"])
        assert area == pytest.approx(roc_auc_score(y, s), abs=5e-3)

    @pytest.mark.parametrize("rotulo", [0, 1])
    def test_classe_unica_devolve_none_nunca_zero(self, rotulo):
        """Decisão de 04/08/2026: métrica indefinida nunca vira 0."""
        rng = np.random.default_rng(5)
        y = np.full(50, rotulo, dtype=int)
        assert evaluate.curve_points(y, rng.random(50)) is None

    def test_grade_tem_101_pontos_e_e_a_mesma_em_toda_celula(self):
        rng = np.random.default_rng(13)
        pontos = [
            evaluate.curve_points(
                (rng.random(500) < 0.3).astype(int), rng.random(500)
            )
            for _ in range(3)
        ]
        assert all(len(p["grade"]) == 101 for p in pontos)
        assert all(p["grade"] == pontos[0]["grade"] for p in pontos)

    def test_grade_e_a_mesma_constante_do_modulo(self):
        pontos = evaluate.curve_points(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
        )
        assert pontos["grade"] == [float(v) for v in evaluate.GRADE_CURVA]

    def test_area_do_envelope_nao_e_a_auc_pr_reportada(self):
        """Trava a incoerência que a legenda da F1 tem de declarar (§7.4).

        A área sob o envelope é otimista; a AUC-PR das tabelas é a *average
        precision*. São grandezas diferentes, e o teste existe para que
        ninguém "conserte" a divergência trocando uma pela outra.
        """
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(17)
        y = (rng.random(8000) < 0.05).astype(int)
        s = np.round(1.0 / (1.0 + np.exp(-rng.normal(1.5 * y, 1.0))), 2)
        pontos = evaluate.curve_points(y, s)
        area_envelope = np.trapezoid(pontos["pr_precision"], pontos["grade"])
        assert area_envelope > average_precision_score(y, s)


# ---------------------------------------------------------------------------
# IC por bootstrap (05/08/2026)
# ---------------------------------------------------------------------------
class TestBootstrapCI:
    """A defesa 2 do B1 aplicada a uma fórmula nova.

    `_auroc_e_ap_de_grupos` reimplementa AUROC e average precision para poder
    reamostrar em O(n_grupos) — chamar `roc_auc_score` 500 vezes por célula
    reordenaria 160 mil scores e custaria ~13 min só no Grambeddings. Toda vez
    que este projeto reimplementou uma métrica, a regra é conferir contra a rota
    do sklearn (foi o que faltou no B1). Aqui a conferência é sob `sample_weight`
    **arbitrário**, não só sob os pesos que o bootstrap por acaso sorteia.
    """

    @staticmethod
    def _preparar(y, s):
        """Reproduz a redução a grupos que a `bootstrap_ci` faz por dentro."""
        ordem = np.argsort(-s, kind="stable")
        ys = y[ordem].astype(float)
        ss = s[ordem]
        grupo_de = np.cumsum(np.r_[True, ss[1:] != ss[:-1]]) - 1
        return ordem, ys, grupo_de, int(grupo_de[-1]) + 1

    @pytest.mark.parametrize(
        "nome,n,prev,arredondar",
        [
            ("continuo", 3000, 0.30, None),   # um grupo por linha
            ("empatado", 3000, 0.05, 1),      # dezenas de grupos
            ("binario", 2000, 0.50, 0),       # dois grupos
        ],
    )
    def test_formula_bate_com_sklearn_sob_peso_arbitrario(
        self, nome, n, prev, arredondar
    ):
        from sklearn.metrics import average_precision_score, roc_auc_score

        rng = np.random.default_rng(19)
        y = (rng.random(n) < prev).astype(int)
        s = 1.0 / (1.0 + np.exp(-rng.normal(1.5 * y, 1.0)))
        if arredondar is not None:
            s = np.round(s, arredondar)
        ordem, ys, grupo_de, n_grupos = self._preparar(y, s)

        for k in range(20):
            # Metade dos casos com peso inteiro cheio, metade com muitos zeros —
            # peso zero é o que o bootstrap produz de fato (numa reamostra com
            # reposição, ~37% das linhas não saem sorteadas).
            if k % 2:
                pesos = rng.integers(0, 5, size=n).astype(float)
            else:
                pesos = (rng.random(n) < 0.4).astype(float) * rng.integers(
                    1, 4, size=n
                )
            if pesos[y == 1].sum() == 0 or pesos[y == 0].sum() == 0:
                continue
            au, ap = evaluate._auroc_e_ap_ponderados(ys, grupo_de, n_grupos, pesos)
            na_ordem_original = np.empty(n)
            na_ordem_original[ordem] = pesos
            assert not np.isnan(au) and not np.isnan(ap)
            assert au == pytest.approx(
                roc_auc_score(y, s, sample_weight=na_ordem_original), abs=1e-12
            )
            assert ap == pytest.approx(
                average_precision_score(y, s, sample_weight=na_ordem_original),
                abs=1e-12,
            )

    def test_ponto_cai_dentro_do_intervalo(self):
        """A trava do bug que a primeira versão tinha.

        Ela concatenava `(pos, neg)` e sorteava depois, sem deslocar os índices
        do bloco negativo — os sorteios da classe negativa caíam no bloco dos
        positivos. O IC saía deslocado (ponto 0,8028 com IC [0,5470; 0,5526]) e
        nada mais no código acusava.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        rng = np.random.default_rng(23)
        for prev in (0.03, 0.30, 0.50):
            y = (rng.random(4000) < prev).astype(int)
            s = 1.0 / (1.0 + np.exp(-rng.normal(1.5 * y, 1.0)))
            ic = evaluate.bootstrap_ci(y, s, n_resamples=200, seed=42)
            assert ic["auroc_lo"] <= roc_auc_score(y, s) <= ic["auroc_hi"]
            assert ic["auc_pr_lo"] <= average_precision_score(y, s) <= ic["auc_pr_hi"]

    def test_intervalo_estreita_quando_a_amostra_cresce(self):
        """É o achado que o IC existe para comunicar: o teste do ISCX tem ~4 mil
        linhas e o do Grambeddings ~160 mil, e a precisão da estimativa não é a
        mesma nos dois."""
        rng = np.random.default_rng(29)
        larguras = []
        for n in (2000, 20000):
            y = (rng.random(n) < 0.4).astype(int)
            s = 1.0 / (1.0 + np.exp(-rng.normal(1.2 * y, 1.0)))
            ic = evaluate.bootstrap_ci(y, s, n_resamples=200, seed=42)
            larguras.append(ic["auroc_hi"] - ic["auroc_lo"])
        assert larguras[1] < larguras[0] / 2

    def test_determinismo_e_invariancia_a_ordem(self):
        """RNF1 pelo caminho que importa: a retomada executa em outra ordem."""
        rng = np.random.default_rng(31)
        y = (rng.random(2000) < 0.3).astype(int)
        s = rng.random(2000)
        a = evaluate.bootstrap_ci(y, s, n_resamples=100, seed=42)
        assert a == evaluate.bootstrap_ci(y, s, n_resamples=100, seed=42)
        perm = rng.permutation(2000)
        assert a == evaluate.bootstrap_ci(y[perm], s[perm], n_resamples=100, seed=42)

    def test_seed_diferente_da_intervalo_diferente(self):
        rng = np.random.default_rng(37)
        y = (rng.random(2000) < 0.3).astype(int)
        s = 1.0 / (1.0 + np.exp(-rng.normal(y, 1.0)))
        a = evaluate.bootstrap_ci(y, s, n_resamples=100, seed=1)
        b = evaluate.bootstrap_ci(y, s, n_resamples=100, seed=2)
        assert a != b

    @pytest.mark.parametrize("rotulo", [0, 1])
    def test_classe_unica_devolve_none_nunca_zero(self, rotulo):
        rng = np.random.default_rng(41)
        y = np.full(60, rotulo, dtype=int)
        assert evaluate.bootstrap_ci(y, rng.random(60)) is None

    def test_classificador_perfeito_tem_intervalo_degenerado_em_um(self):
        y = np.array([0] * 300 + [1] * 300)
        s = np.array([0.2] * 300 + [0.8] * 300)
        ic = evaluate.bootstrap_ci(y, s, n_resamples=100, seed=42)
        assert ic["auroc_lo"] == ic["auroc_hi"] == 1.0
        assert ic["auc_pr_lo"] == ic["auc_pr_hi"] == 1.0

    @pytest.mark.parametrize(
        "kwargs", [{"n_resamples": 1}, {"nivel": 0.0}, {"nivel": 1.0}]
    )
    def test_parametro_invalido_levanta(self, kwargs):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.4, 0.6, 0.9])
        with pytest.raises(MetricsError):
            evaluate.bootstrap_ci(y, s, **kwargs)

    def test_nivel_maior_da_intervalo_mais_largo(self):
        rng = np.random.default_rng(43)
        y = (rng.random(3000) < 0.3).astype(int)
        s = 1.0 / (1.0 + np.exp(-rng.normal(1.2 * y, 1.0)))
        estreito = evaluate.bootstrap_ci(y, s, n_resamples=400, seed=42, nivel=0.80)
        largo = evaluate.bootstrap_ci(y, s, n_resamples=400, seed=42, nivel=0.99)
        assert largo["auroc_hi"] - largo["auroc_lo"] > (
            estreito["auroc_hi"] - estreito["auroc_lo"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
