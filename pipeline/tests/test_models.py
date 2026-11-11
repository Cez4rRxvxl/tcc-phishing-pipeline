"""T4.1 — DoD: os 6 modelos instanciam; GPU detectada ou fallback silencioso.

⛔ Nenhum teste aqui chama `.fit()` em dados reais (SDD §0). Verificam construção,
leitura de config e escolha de dispositivo; os testes do MLP agrupado (T10)
ajustam redes pequenas em dados sintéticos.

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

import pytest

import models
from models import GpuInfo, ModelConfigError

REPO_ROOT = models.REPO_ROOT


@pytest.fixture(scope="module")
def config():
    return models.load_experiment_config()


@pytest.fixture
def sem_gpu():
    return GpuInfo(available=False, reason="fixture de teste")


@pytest.fixture
def com_gpu():
    return GpuInfo(
        available=True,
        n_devices=1,
        names=("NVIDIA GeForce RTX 4060",),
        detected_by="fixture de teste",
    )


# ---------------------------------------------------------------------------
# DoD — os 6 instanciam
# ---------------------------------------------------------------------------
class TestSeisModelosInstanciam:
    def test_todos_os_seis_da_ordem_canonica(self, config, sem_gpu):
        built = models.build_all_models(config, gpu=sem_gpu)
        assert list(built) == list(models.MODEL_ORDER)
        assert len(built) == 6

    @pytest.mark.parametrize("name", models.MODEL_ORDER)
    def test_cada_modelo_isolado(self, config, name, sem_gpu):
        est = models.build_model(name, config, gpu=sem_gpu)
        assert est is not None
        assert hasattr(est, "fit")
        assert hasattr(est, "predict_proba"), (
            f"{name} não expõe predict_proba — AUROC, AUC-PR, Brier e ECE "
            f"(SDD §4 D6) todos dependem de probabilidade, não de rótulo."
        )

    def test_nenhum_modelo_vem_ajustado(self, config, sem_gpu):
        """Instanciar não pode treinar (SDD §0)."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        for name, est in models.build_all_models(config, gpu=sem_gpu).items():
            target = est.named_steps["clf"] if hasattr(est, "named_steps") else est
            if name in ("logistic_regression", "random_forest", "mlp"):
                with pytest.raises(NotFittedError):
                    check_is_fitted(target)

    def test_familias_cobrem_as_cinco_de_d4(self, config):
        familias = {spec["familia"] for spec in config["modelos"].values()}
        assert familias == {"linear", "bagging", "boosting", "rede_neural"}


# ---------------------------------------------------------------------------
# RF5 — hiperparâmetros vêm do YAML, não do código
# ---------------------------------------------------------------------------
class TestHiperparametrosVemDoConfig:
    def test_params_do_yaml_chegam_ao_estimador(self, config, sem_gpu):
        est = models.build_model("random_forest", config, gpu=sem_gpu)
        assert est.n_estimators == config["modelos"]["random_forest"]["params"]["n_estimators"]
        assert est.min_samples_leaf == 2

    def test_alterar_o_config_altera_o_modelo(self, config, sem_gpu):
        """Prova que o valor não está duplicado em Python (RF5)."""
        import copy

        alterado = copy.deepcopy(config)
        alterado["modelos"]["xgboost"]["params"]["max_depth"] = 3
        est = models.build_model("xgboost", alterado, gpu=sem_gpu)
        assert est.max_depth == 3

    def test_seed_do_config_propaga_para_todos(self, config, sem_gpu):
        seed = int(config["seed"])
        for name, est in models.build_all_models(config, gpu=sem_gpu).items():
            target = est.named_steps["clf"] if hasattr(est, "named_steps") else est
            if name == "catboost":
                assert target.get_params()["random_seed"] == seed
            else:
                assert target.get_params()["random_state"] == seed

    def test_mlp_hidden_layers_convertido_para_tupla(self, config, sem_gpu):
        """YAML devolve lista; o sklearn exige tupla."""
        est = models.build_model("mlp", config, gpu=sem_gpu)
        assert isinstance(est.named_steps["clf"].hidden_layer_sizes, tuple)

    def test_nenhum_modelo_usa_class_weight(self, config, sem_gpu):
        """Decisão declarada em experiment.yml: reponderar classe distorce a
        probabilidade prevista, e D6 exige Brier/ECE sem recalibração — foi
        assim que B8 passou (Brier 0,5257 a 96% de acurácia)."""
        for name, est in models.build_all_models(config, gpu=sem_gpu).items():
            target = est.named_steps["clf"] if hasattr(est, "named_steps") else est
            params = target.get_params()
            assert params.get("class_weight") is None, f"{name} usa class_weight"
            assert params.get("scale_pos_weight") in (None, 1, 1.0), f"{name} pondera classe"
            assert params.get("auto_class_weights") is None, f"{name} pondera classe"
            assert params.get("is_unbalance") in (None, False), f"{name} pondera classe"


# ---------------------------------------------------------------------------
# Padronização declarada no config
# ---------------------------------------------------------------------------
class TestEscala:
    def test_linear_e_mlp_sao_padronizados(self, config, sem_gpu):
        for name in ("logistic_regression", "mlp"):
            est = models.build_model(name, config, gpu=sem_gpu)
            assert hasattr(est, "named_steps"), f"{name} deveria vir num Pipeline"
            assert "escala" in est.named_steps

    def test_arvores_nao_sao_padronizadas(self, config, sem_gpu):
        """Árvore é invariante a transformação monotônica — padronizar só
        gastaria memória e mudaria o dtype de uint16 para float64."""
        for name in ("random_forest", "xgboost", "lightgbm", "catboost"):
            est = models.build_model(name, config, gpu=sem_gpu)
            assert not hasattr(est, "named_steps"), f"{name} não precisa de escala"

    def test_pipeline_expoe_predict_proba(self, config, sem_gpu):
        est = models.build_model("logistic_regression", config, gpu=sem_gpu)
        assert hasattr(est, "predict_proba")


class TestParalelismo:
    def test_n_jobs_do_config_chega_aos_gbm(self, config, sem_gpu):
        for name in ("xgboost", "lightgbm"):
            est = models.build_model(name, config, gpu=sem_gpu)
            assert est.get_params()["n_jobs"] == config["cpu"]["n_jobs"]

    def test_logistic_regression_nao_recebe_n_jobs(self, config, sem_gpu):
        """Desde o scikit-learn 1.8 o parâmetro não tem efeito, emite
        FutureWarning e sai na 1.10 — passá-lo quebraria a execução do Lucas
        numa wheel mais nova por um parâmetro que nunca fez nada."""
        assert models._n_jobs_kwarg("logistic_regression", config) == {}

    def test_construir_os_seis_nao_emite_futurewarning(self, config, sem_gpu):
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error", FutureWarning)
            models.build_all_models(config, gpu=sem_gpu)


# ---------------------------------------------------------------------------
# RF10 / RH4 — GPU automática com fallback silencioso
# ---------------------------------------------------------------------------
class TestDeteccaoDeGpu:
    def test_detect_gpu_nunca_lanca(self):
        """O requisito é fallback SILENCIOSO: numa máquina sem GPU (esta),
        detectar não pode ser erro."""
        info = models.detect_gpu()
        assert isinstance(info, GpuInfo)
        assert isinstance(info.available, bool)
        assert info.describe()  # sempre legível para o log (RH3)

    def test_esta_maquina_nao_tem_gpu(self):
        """SDD §13.3: esta máquina tem 12 núcleos e nenhuma GPU. Se este teste
        falhar na máquina do Lucas, é o esperado — ele tem GPU NVIDIA."""
        info = models.detect_gpu()
        if info.available:
            pytest.skip(f"máquina com GPU ({info.describe()}) — provavelmente a do Lucas")
        assert info.n_devices == 0
        assert info.reason

    def test_deteccao_em_cache(self):
        """Detectar 48 vezes no laço da matriz seria desperdício."""
        assert models.detect_gpu() is models.detect_gpu()

    def test_force_cpu_por_ambiente(self, monkeypatch):
        models.detect_gpu.cache_clear()
        monkeypatch.setenv("PHISHING_FORCE_CPU", "1")
        try:
            info = models.detect_gpu()
            assert info.available is False
            assert "PHISHING_FORCE_CPU" in info.reason
        finally:
            models.detect_gpu.cache_clear()

    def test_params_de_cpu_sao_explicitos(self, config, sem_gpu):
        assert models.gpu_params("xgboost", config, gpu=sem_gpu) == {"device": "cpu"}
        assert models.gpu_params("catboost", config, gpu=sem_gpu) == {"task_type": "CPU"}

    def test_params_de_gpu_quando_ha_gpu(self, config, com_gpu):
        assert models.gpu_params("xgboost", config, gpu=com_gpu) == {"device": "cuda"}
        assert models.gpu_params("catboost", config, gpu=com_gpu) == {
            "task_type": "GPU",
            "devices": "0",
        }

    def test_lightgbm_fica_em_cpu_mesmo_com_gpu(self, config, com_gpu):
        """A wheel oficial do LightGBM no PyPI é compilada sem o tree learner
        de GPU; passar device="gpu" nela levanta "GPU Tree Learner was not
        enabled in this build" e quebraria a execução do Lucas."""
        assert models.gpu_params("lightgbm", config, gpu=com_gpu) == {}
        assert "lightgbm" not in config["gpu"]["habilitado_para"]

    def test_sklearn_ignora_gpu(self, config, com_gpu):
        for name in ("logistic_regression", "random_forest", "mlp"):
            assert models.gpu_params(name, config, gpu=com_gpu) == {}

    def test_modelos_constroem_com_gpu_simulada(self, config, com_gpu):
        """Construir com device="cuda" não exige GPU presente — o XGBoost só
        cobra o dispositivo no `.fit()`. Garante que a máquina do Lucas não vai
        encontrar um erro de construção."""
        est = models.build_model("xgboost", config, gpu=com_gpu)
        assert est.get_params()["device"] == "cuda"
        est_cb = models.build_model("catboost", config, gpu=com_gpu)
        assert est_cb.get_params()["task_type"] == "GPU"

    def test_gpu_auto_false_desliga_tudo(self, config, com_gpu):
        import copy

        cfg = copy.deepcopy(config)
        cfg["gpu"]["auto"] = False
        assert models.gpu_params("xgboost", cfg, gpu=com_gpu) == {"device": "cpu"}


# ---------------------------------------------------------------------------
# Validação da config (RH1 — falhar alto e claro)
# ---------------------------------------------------------------------------
class TestValidacaoDeConfig:
    def test_config_do_repo_carrega(self, config):
        assert config["seed"] == 42
        assert set(config["corpora"]["treino_avaliacao"]) == {
            "grambeddings", "kaggle03", "phishstorm", "iscx2016",
        }
        assert set(config["corpora"]["demonstracao_vies"]) == {"pdrcnn", "sid321axn"}
        assert config["regimes"] == ["oficial", "etld1"]

    def test_corpora_de_vies_nunca_treinam(self, config):
        """R6 do SDD: nenhum corpus com viés de construção medido entra no
        treino. PDRCNN (gap 94,9 p) e sid321axn (59,9 p) só aparecem em
        Ameaças à Validade."""
        treino = set(config["corpora"]["treino_avaliacao"])
        vies = set(config["corpora"]["demonstracao_vies"])
        assert treino & vies == set()
        assert "dataset" not in treino and "urldata" not in treino

    def test_arquivo_inexistente_falha_claro(self, tmp_path):
        with pytest.raises(ModelConfigError, match="não encontrada"):
            models.load_experiment_config(tmp_path / "nao_existe.yml")

    def test_modelo_faltando_falha_claro(self, tmp_path):
        import yaml

        cfg = models.load_experiment_config()
        del cfg["modelos"]["catboost"]
        path = tmp_path / "incompleto.yml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ModelConfigError, match="faltam modelos"):
            models.load_experiment_config(path)

    def test_modelo_extra_falha_claro(self, tmp_path):
        """Adicionar modelo é mudar decisão travada (D4) — tem de doer."""
        import yaml

        cfg = models.load_experiment_config()
        cfg["modelos"]["extratrees"] = {"familia": "bagging", "params": {}}
        path = tmp_path / "extra.yml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ModelConfigError, match="não previstos"):
            models.load_experiment_config(path)

    def test_modelo_desconhecido_em_build(self, config):
        with pytest.raises(ModelConfigError, match="desconhecido"):
            models.build_model("extratrees", config)

    def test_chave_obrigatoria_ausente(self, tmp_path):
        import yaml

        path = tmp_path / "sem_seed.yml"
        path.write_text(yaml.safe_dump({"modelos": {}}), encoding="utf-8")
        with pytest.raises(ModelConfigError, match="seed"):
            models.load_experiment_config(path)


# ---------------------------------------------------------------------------
# Overrides do smoke
# ---------------------------------------------------------------------------
class TestSmokeOverrides:
    def test_overrides_reduzem_orcamento(self, config, sem_gpu):
        smoke = models.apply_smoke_overrides(config)
        est = models.build_model("random_forest", smoke, gpu=sem_gpu)
        assert est.n_estimators == 40

    def test_overrides_nao_contaminam_a_config_real(self, config):
        """Deep copy: uma execução de verdade não pode herdar n_estimators=40."""
        antes = config["modelos"]["random_forest"]["params"]["n_estimators"]
        models.apply_smoke_overrides(config)
        assert config["modelos"]["random_forest"]["params"]["n_estimators"] == antes
        assert antes == 300

    def test_smoke_marcado_na_proveniencia(self, config):
        smoke = models.apply_smoke_overrides(config)
        prov = models.model_provenance("xgboost", smoke)
        assert prov["smoke"] is True
        assert models.model_provenance("xgboost", config)["smoke"] is False

    def test_override_de_modelo_inexistente_falha(self, config):
        import copy

        cfg = copy.deepcopy(config)
        cfg["smoke"]["overrides"]["extratrees"] = {"n_estimators": 1}
        with pytest.raises(ModelConfigError, match="inexistente"):
            models.apply_smoke_overrides(cfg)

    def test_todos_os_seis_tem_override(self, config):
        """Se um modelo ficar sem override, o smoke estoura os 60 s por causa
        dele e ninguém entende por quê."""
        assert set(config["smoke"]["overrides"]) == set(models.MODEL_ORDER)


# ---------------------------------------------------------------------------
# Proveniência (RNF4)
# ---------------------------------------------------------------------------
class TestProveniencia:
    def test_registro_completo(self, config, sem_gpu):
        prov = models.model_provenance("lightgbm", config, gpu=sem_gpu)
        assert prov["model"] == "lightgbm"
        assert prov["familia"] == "boosting"
        assert prov["seed"] == 42
        assert prov["gpu_detected"] is False
        assert prov["params"]["n_estimators"] == 400
        assert prov["escala"] is False

    def test_determinismo_do_lightgbm_declarado(self, config):
        """RNF1: sem `deterministic` + `force_row_wise`, a redução paralela do
        LightGBM pode variar a ordem das somas em float entre execuções."""
        params = config["modelos"]["lightgbm"]["params"]
        assert params["deterministic"] is True
        assert params["force_row_wise"] is True

    def test_catboost_nao_escreve_no_repo(self, config):
        """A limpeza do legado removeu `catboost_info/`; não deixar voltar."""
        assert config["modelos"]["catboost"]["params"]["allow_writing_files"] is False


# ---------------------------------------------------------------------------
# positive_class_proba — guarda contra indexar a coluna errada
# ---------------------------------------------------------------------------
class TestPositiveClassProba:
    def test_ordem_normal(self):
        import numpy as np

        class Fake:
            classes_ = [0, 1]

            def predict_proba(self, X):
                return np.array([[0.9, 0.1], [0.2, 0.8]])

        assert models.positive_class_proba(Fake(), None).tolist() == [0.1, 0.8]

    def test_ordem_invertida(self):
        """Se `classes_` vier [1, 0], indexar [:, 1] pegaria a classe errada
        calado — e a AUROC sairia invertida (≈ 1 - AUROC real)."""
        import numpy as np

        class Fake:
            classes_ = [1, 0]

            def predict_proba(self, X):
                return np.array([[0.1, 0.9], [0.8, 0.2]])

        assert models.positive_class_proba(Fake(), None).tolist() == [0.1, 0.8]

    def test_predicao_do_random_forest_e_bit_a_bit_reproduzivel(self):
        """RNF1 — regressão do único ponto de não-determinismo encontrado.

        `RandomForestClassifier.predict_proba` acumula as árvores em paralelo;
        soma em float não é associativa, e a ordem de término das threads
        mudava o último bit (2,22e-16 medido). Isso desfazia empates de score e
        alterava AUROC/AUC-PR: 30 dos 192 registros do smoke mudavam entre duas
        execuções.

        Este é o único teste da suíte que ajusta um modelo. É deliberado e
        respeita SDD §0: 500 linhas sintéticas, 20 árvores, ~0,1 s — não é
        treinamento, é a prova de que o determinismo está travado.

        Dois detalhes do cenário são **obrigatórios** para o teste não ser
        vácuo, e foram descobertos medindo:

          - **Folhas impuras** (`min_samples_leaf=2` + rótulo com ruído). Com
            rótulo perfeitamente separável, toda árvore vota exatamente 0,0 ou
            1,0, a soma em float é exata, a ordem das threads não muda nada — e
            o teste passaria mesmo com o defeito presente. A primeira versão
            deste teste era vácua por isso.
          - **`n_jobs=-1`**: é a redução paralela que introduz a variação.
        """
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        X = rng.integers(0, 100, size=(500, 12)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] + rng.integers(0, 40, 500) > 120).astype(int)

        rf = RandomForestClassifier(
            n_estimators=20, min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X, y)

        repeticoes = [models.positive_class_proba(rf, X) for _ in range(6)]
        for i, p in enumerate(repeticoes[1:], start=2):
            assert np.array_equal(repeticoes[0], p), (
                f"predição do RandomForest divergiu na repetição {i} "
                f"(maxdif={np.abs(repeticoes[0] - p).max():.3e}) — "
                f"_predicao_serial deixou de ser aplicada, RNF1 quebrado"
            )

    def test_o_cenario_do_teste_acima_realmente_expoe_o_defeito(self):
        """Prova que o teste anterior não é vácuo.

        Roda o MESMO cenário chamando `predict_proba` direto (sem a
        serialização) e exige que a divergência apareça. Se algum dia o
        scikit-learn passar a reduzir de forma determinística, este teste falha
        primeiro — e aí `_predicao_serial` pode ser reavaliada em vez de
        carregada por superstição.
        """
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        X = rng.integers(0, 100, size=(500, 12)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] + rng.integers(0, 40, 500) > 120).astype(int)

        rf = RandomForestClassifier(
            n_estimators=20, min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X, y)

        cru = [rf.predict_proba(X)[:, 1] for _ in range(6)]
        maxdif = max(float(np.abs(cru[0] - p).max()) for p in cru[1:])
        assert maxdif > 0.0, (
            "o cenário deixou de expor a não-determinação da redução paralela; "
            "reveja se `_predicao_serial` ainda é necessária (RNF1)"
        )
        assert maxdif < 1e-12, (
            f"divergência de {maxdif:.3e} é grande demais para ser ruído de "
            f"ordem de soma — suspeite de floresta diferente, não de ULP"
        )

    def test_n_jobs_e_restaurado_apos_a_predicao(self):
        """A serialização é só da predição: o `fit` seguinte tem de continuar
        podendo usar todos os núcleos."""
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
        X = np.arange(40, dtype=np.float32).reshape(20, 2)
        rf.fit(X, np.array([0, 1] * 10))
        models.positive_class_proba(rf, X)
        assert rf.n_jobs == -1

    def test_treino_degenerado_sem_positivos(self):
        import numpy as np

        class Fake:
            classes_ = [0]

            def predict_proba(self, X):
                return np.array([[1.0], [1.0]])

        out = models.positive_class_proba(Fake(), None)
        assert out.tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# T10: MLP com validação interna agrupada (spec MLP, critérios 1 a 6)
#
# Os ajustes abaixo usam dados sintéticos pequenos (3.600 linhas, 41 ou 10
# colunas) e redes de uma camada pequena. Nenhum treina num corpus real.
# ---------------------------------------------------------------------------
def _dados_agrupados(n_grupos=1200, por_grupo=3, n_cols=10, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    grupos = np.repeat(np.arange(n_grupos).astype(str), por_grupo)
    n = len(grupos)
    X = rng.normal(size=(n, n_cols))
    y = ((X[:, 0] + 0.8 * rng.normal(size=n)) > 0).astype(int)
    return X, y, grupos


# Parâmetros reduzidos só para os testes de comportamento; os parâmetros do
# `experiment.yml` são conferidos à parte (TestMLPParametrosDoYml).
_KW_MLP = dict(
    hidden_layer_sizes=(32,),
    max_iter=200,
    n_iter_no_change=3,
    validation_fraction=0.1,
    random_state=42,
    early_stopping=True,
)


class TestMLPValidacaoAgrupada:
    def test_validacao_nao_compartilha_grupo_e_tem_cerca_de_10_por_cento(
        self, monkeypatch
    ):
        # MLP 1 e 2. A validação é capturada na chamada que o sklearn faz a
        # cada época (`_update_no_improvement_count` recebe X_val); a coluna 0
        # de X carrega o índice da linha, o que permite recuperar os grupos.
        import numpy as np
        from sklearn.neural_network import MLPClassifier

        X, y, grupos = _dados_agrupados()
        X[:, 0] = np.arange(len(y))
        vistos = []
        original = MLPClassifier._update_no_improvement_count

        def espiao(self, early_stopping, X_, y_, sample_weight):
            vistos.append(np.asarray(X_)[:, 0].astype(int))
            return original(self, early_stopping, X_, y_, sample_weight)

        monkeypatch.setattr(MLPClassifier, "_update_no_improvement_count", espiao)
        modelo = models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos)

        val_ids = vistos[0]
        fit_ids = np.setdiff1d(np.arange(len(y)), val_ids)
        assert not set(grupos[val_ids]) & set(grupos[fit_ids])
        assert 0.08 <= len(val_ids) / len(y) <= 0.12
        # O registro da fração obtida bate com o corte de fato usado.
        assert modelo.fracao_validacao_obtida_ == pytest.approx(len(val_ids) / len(y))
        assert modelo.n_grupos_validacao_ == len(set(grupos[val_ids]))

    @pytest.mark.parametrize("semente", [1, 42])
    def test_grupo_dominante_segue_agrupado_e_registra_a_fracao_real(
        self, monkeypatch, semente
    ):
        # Caso de borda do MLP-01: um eTLD+1 com mais de 10% das linhas do
        # treino impede a fração pedida. O MLP segue pelo sorteio agrupado
        # (sem erro e sem cair na validação por linha) e registra em
        # `fracao_validacao_obtida_` a fração de fato obtida, que sai da
        # faixa de 8% a 12%. O dominante tem 300 de 800 linhas (37,5%).
        import numpy as np
        from sklearn.neural_network import MLPClassifier

        rng = np.random.default_rng(0)
        grupos = np.array(
            ["dominante"] * 300 + [f"g{i}" for i in range(100) for _ in range(5)]
        )
        n = len(grupos)
        X = rng.normal(size=(n, 4))
        X[:, 0] = np.arange(n)
        y = rng.integers(0, 2, n)
        vistos = []
        original = MLPClassifier._update_no_improvement_count

        def espiao(self, early_stopping, X_, y_, sample_weight):
            vistos.append(np.asarray(X_)[:, 0].astype(int))
            return original(self, early_stopping, X_, y_, sample_weight)

        monkeypatch.setattr(MLPClassifier, "_update_no_improvement_count", espiao)
        kw = dict(_KW_MLP, hidden_layer_sizes=(4,), max_iter=5, random_state=semente)
        modelo = models.MLPValidacaoAgrupada(**kw).fit(X, y, groups=grupos)

        val_ids = vistos[0]
        fit_ids = np.setdiff1d(np.arange(n), val_ids)
        assert not set(grupos[val_ids]) & set(grupos[fit_ids])
        fracao = modelo.fracao_validacao_obtida_
        assert fracao == pytest.approx(len(val_ids) / n)
        assert not 0.08 <= fracao <= 0.12
        assert modelo.n_grupos_validacao_ == len(set(grupos[val_ids]))

    def test_semantica_de_parada_e_restauracao_igual_ao_laco_de_referencia(self):
        # MLP 4 (MLP-03). Reproduz à mão, com `partial_fit` época a época, a
        # regra do early stopping do sklearn: o escore é a acurácia na
        # validação; conta-se uma época sem melhora quando
        # `escore < melhor + tol`; para quando a contagem passa de
        # `n_iter_no_change`; ficam os pesos da melhor época.
        #
        # O `partial_fit` recria o gerador aleatório a cada chamada e não
        # acumula a permutação das linhas, então a referência embaralha por
        # conta própria (`shuffle=False`), com o mesmo fluxo de sorteio: depois
        # dos sorteios da inicialização dos pesos, uma permutação acumulada
        # por época.
        import numpy as np
        from sklearn.neural_network import MLPClassifier
        from sklearn.utils import shuffle

        import splits

        X, y, grupos = _dados_agrupados()
        # Corte fixado: o mesmo que o regime B usa para recortar validação.
        fit_idx, val_idx = splits._grupo_estratificado_once(grupos, y, 0.1, 42)

        modelo = models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos)

        Xf, yf, Xv, yv = X[fit_idx], y[fit_idx], X[val_idx], y[val_idx]
        rs = np.random.RandomState(42)
        camadas = [X.shape[1], 32, 1]
        for entrada, saida in zip(camadas[:-1], camadas[1:]):
            rs.uniform(-1, 1, (entrada, saida))
            rs.uniform(-1, 1, saida)
        ref = MLPClassifier(**{**_KW_MLP, "early_stopping": False, "shuffle": False})
        melhor, contagem, escores = -np.inf, 0, []
        ordem = np.arange(len(yf))
        for epoca in range(_KW_MLP["max_iter"]):
            ordem = shuffle(ordem, random_state=rs)
            if epoca == 0:
                ref.partial_fit(Xf[ordem], yf[ordem], classes=[0, 1])
            else:
                ref.partial_fit(Xf[ordem], yf[ordem])
            escore = ref.score(Xv, yv)
            escores.append(escore)
            if escore < melhor + ref.tol:
                contagem += 1
            else:
                contagem = 0
            if escore > melhor:
                melhor = escore
                melhores_pesos = [c.copy() for c in ref.coefs_]
                melhores_vieses = [i.copy() for i in ref.intercepts_]
            if contagem > ref.n_iter_no_change:
                break

        # O caso exercita a restauração: a melhor época não é a última.
        assert int(np.argmax(escores)) < len(escores) - 1
        assert modelo.n_iter_ == len(escores)
        assert modelo.validation_scores_ == escores
        assert modelo.best_validation_score_ == melhor
        for obtido, esperado in zip(modelo.coefs_, melhores_pesos):
            assert np.array_equal(obtido, esperado)
        for obtido, esperado in zip(modelo.intercepts_, melhores_vieses):
            assert np.array_equal(obtido, esperado)

    def test_mesma_semente_da_escores_identicos(self):
        # MLP 5.
        import numpy as np

        X, y, grupos = _dados_agrupados()
        a = models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos)
        b = models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos)
        assert a.validation_scores_ == b.validation_scores_
        assert np.array_equal(a.predict_proba(X), b.predict_proba(X))

    def test_sem_grupos_falha_alto(self):
        # Achado 3 do lote F: antes, `fit` sem grupos caia no MLP nativo (validacao
        # por linha) mantendo a proveniencia `agrupada_etld1`. O Regime A usa o
        # MLPClassifier nativo (validacao_agrupada=False), nunca esta classe.
        X, y, _ = _dados_agrupados()
        with pytest.raises(ValueError, match="groups"):
            models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y)

    def test_mlp_nativo_do_regime_a_segue_por_linha_e_ignora_grupos(self, sem_gpu):
        # MLP 3 (Regime A): sem a opcao ligada, `ajustar` usa o nativo, com ou
        # sem grupos, e a validacao e a nativa (por linha, estratificada).
        import numpy as np
        from sklearn.neural_network import MLPClassifier

        X, y, grupos = _dados_agrupados()
        config = _config_pequena(models.load_experiment_config())
        nativo = models.build_model("mlp", config, gpu=sem_gpu, validacao_agrupada=False)
        assert type(models._final_estimator(nativo)) is MLPClassifier
        models.ajustar(nativo, "mlp", X, y, grupos=None)
        outro = models.build_model("mlp", config, gpu=sem_gpu, validacao_agrupada=False)
        models.ajustar(outro, "mlp", X, y, grupos=grupos)
        a, b = models._final_estimator(nativo), models._final_estimator(outro)
        assert a.validation_scores_ == b.validation_scores_
        for ca, cb in zip(a.coefs_, b.coefs_):
            assert np.array_equal(ca, cb)

    def test_funcao_substituta_nao_chamada_levanta_erro(self, monkeypatch):
        # Design (tabela de erros): se o `train_test_split` interno não for
        # chamado exatamente uma vez, o `fit` levanta `RuntimeError`.
        from sklearn.neural_network import MLPClassifier

        X, y, grupos = _dados_agrupados(n_grupos=100)
        monkeypatch.setattr(MLPClassifier, "fit", lambda self, X_, y_: self)
        with pytest.raises(RuntimeError, match="0 vez"):
            models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos)

    def test_funcao_substituta_chamada_duas_vezes_levanta_erro(self, monkeypatch):
        from sklearn.neural_network import MLPClassifier
        from sklearn.neural_network import _multilayer_perceptron as mod

        X, y, grupos = _dados_agrupados(n_grupos=100)

        def fit_falso(self, X_, y_):
            mod.train_test_split(X_, y_)
            mod.train_test_split(X_, y_)
            return self

        monkeypatch.setattr(MLPClassifier, "fit", fit_falso)
        with pytest.raises(RuntimeError, match="2 vez"):
            models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos)

    def test_grupos_com_tamanho_diferente_de_y_levantam_erro(self):
        X, y, grupos = _dados_agrupados(n_grupos=100)
        with pytest.raises(ValueError, match="groups"):
            models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos[:-1])

    def test_o_train_test_split_do_sklearn_e_restaurado_depois_do_fit(self):
        from sklearn.neural_network import _multilayer_perceptron as mod

        original = mod.train_test_split
        X, y, grupos = _dados_agrupados(n_grupos=100)
        models.MLPValidacaoAgrupada(**_KW_MLP).fit(X, y, groups=grupos)
        assert mod.train_test_split is original


class TestBuildModelEAjustar:
    def test_sem_a_opcao_devolve_o_mlp_nativo(self, config, sem_gpu):
        from sklearn.neural_network import MLPClassifier

        clf = models.build_model("mlp", config, gpu=sem_gpu).named_steps["clf"]
        assert type(clf) is MLPClassifier

    def test_com_a_opcao_devolve_o_mlp_agrupado_com_os_mesmos_params(
        self, config, sem_gpu
    ):
        nativo = models.build_model("mlp", config, gpu=sem_gpu).named_steps["clf"]
        agrupado = models.build_model(
            "mlp", config, gpu=sem_gpu, validacao_agrupada=True
        ).named_steps["clf"]
        assert type(agrupado) is models.MLPValidacaoAgrupada
        assert agrupado.get_params() == nativo.get_params()

    @pytest.mark.parametrize("nome", [n for n in models.MODEL_ORDER if n != "mlp"])
    def test_a_opcao_nao_muda_os_outros_modelos(self, config, sem_gpu, nome):
        a = models.build_model(nome, config, gpu=sem_gpu)
        b = models.build_model(nome, config, gpu=sem_gpu, validacao_agrupada=True)
        assert type(a) is type(b)
        assert not isinstance(
            models._final_estimator(b), models.MLPValidacaoAgrupada
        )

    def test_ajustar_roteia_os_grupos_pelo_pipeline(self, config, sem_gpu):
        X, y, grupos = _dados_agrupados(n_grupos=300)
        pequeno = _config_pequena(config)
        modelo = models.build_model("mlp", pequeno, gpu=sem_gpu, validacao_agrupada=True)
        out = models.ajustar(modelo, "mlp", X, y, grupos=grupos)
        assert out is modelo
        clf = modelo.named_steps["clf"]
        assert 0.08 <= clf.fracao_validacao_obtida_ <= 0.12
        assert clf.n_grupos_validacao_ > 0

    def test_ajustar_sem_grupos_com_validacao_agrupada_levanta(self, config, sem_gpu):
        # Achado 3 do lote F: este teste esperava o fallback silencioso para a
        # validacao nativa (sem `fracao_validacao_obtida_`); agora e erro.
        X, y, _ = _dados_agrupados(n_grupos=300)
        modelo = models.build_model(
            "mlp", _config_pequena(config), gpu=sem_gpu, validacao_agrupada=True
        )
        with pytest.raises(ValueError, match="groups"):
            models.ajustar(modelo, "mlp", X, y)

    def test_ajustar_ignora_grupos_no_mlp_nativo_e_nos_demais(self, config, sem_gpu):
        X, y, grupos = _dados_agrupados(n_grupos=300)
        nativo = models.build_model("mlp", _config_pequena(config), gpu=sem_gpu)
        assert models.ajustar(nativo, "mlp", X, y, grupos=grupos) is nativo
        assert hasattr(nativo.named_steps["clf"], "coefs_")
        rl = models.build_model("logistic_regression", config, gpu=sem_gpu)
        assert models.ajustar(rl, "logistic_regression", X, y, grupos=grupos) is rl
        assert hasattr(rl.named_steps["clf"], "coef_")


def _config_pequena(config):
    """Cópia da config com o MLP reduzido, só para ajustes rápidos nos testes."""
    import copy

    pequena = copy.deepcopy(config)
    pequena["modelos"]["mlp"]["params"].update(
        hidden_layer_sizes=[8], max_iter=15, n_iter_no_change=3
    )
    return pequena


class TestMLPParametrosDoYml:
    def test_os_parametros_vem_do_yml_e_o_arquivo_nao_muda(self, config, sem_gpu):
        # MLP 6. `tol` não está no `experiment.yml`: vale o default do sklearn
        # (spec-precision gap, registrado no relatório da tarefa).
        import hashlib

        from sklearn.neural_network import MLPClassifier

        caminho = models.DEFAULT_CONFIG_PATH
        antes = hashlib.sha256(caminho.read_bytes()).hexdigest()
        params = config["modelos"]["mlp"]["params"]
        clf = models.build_model(
            "mlp", config, gpu=sem_gpu, validacao_agrupada=True
        ).named_steps["clf"]
        assert clf.validation_fraction == params["validation_fraction"] == 0.1
        assert clf.n_iter_no_change == params["n_iter_no_change"] == 10
        assert clf.max_iter == params["max_iter"] == 200
        assert clf.early_stopping is True
        assert clf.tol == MLPClassifier().tol
        assert hashlib.sha256(caminho.read_bytes()).hexdigest() == antes


class TestProvenienciaValidacaoInterna:
    def test_mlp_agrupado_e_nativo(self, config, sem_gpu):
        # MLP 1 e 3: a proveniência diz qual validação foi usada.
        agrupado = models.model_provenance(
            "mlp", config, gpu=sem_gpu, validacao_agrupada=True
        )
        assert agrupado["validacao_interna"] == "agrupada_etld1"
        assert models.model_provenance("mlp", config, gpu=sem_gpu)[
            "validacao_interna"
        ] == "linha"

    def test_modelos_sem_validacao_interna_gravam_none(self, config, sem_gpu):
        prov = models.model_provenance("random_forest", config, gpu=sem_gpu)
        assert prov["validacao_interna"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
