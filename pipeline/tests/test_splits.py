"""T2.4 — DoD: teste automatizado prova sobreposição de domínio = 0 (docs/requisitos.md RF4).

Cobre os dois regimes de SDD §4 D3:
  - Regime B ("etld1"): sobreposição de domínio entre partições tem de ser 0.
  - Regime A ("oficial"): o split da fonte é preservado tal como distribuído
    (e o vazamento dele **não** é corrigido — é o que a Tabela T3 mede).

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

import numpy as np
import pandas as pd
import pytest

import splits
from splits import SplitError, SplitIndices

# ---------------------------------------------------------------------------
# Fixtures — corpus sintético com domínios repetidos entre linhas
# ---------------------------------------------------------------------------


def _synthetic_corpus(n_domains: int = 120, rows_per_domain: int = 4) -> pd.DataFrame:
    """Corpus sintético já "canonizado" (tem `etld_plus_one`).

    Cada domínio aparece em várias linhas — é essa repetição que faz o split
    ingênuo vazar e que o particionador agrupado tem de respeitar.
    """
    rows = []
    for d in range(n_domains):
        domain = f"dominio{d:04d}.com"
        for r in range(rows_per_domain):
            rows.append(
                {
                    "label": d % 2,
                    "url": f"http://www.{domain}/pagina{r}",
                    "corpus": "sintetico",
                    "split_oficial": "train" if d % 5 else "test",
                    "etld_plus_one": domain,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def corpus():
    return _synthetic_corpus()


@pytest.fixture(scope="module")
def real_sample():
    """Amostra pequena de um corpus real, canonizada de verdade.

    Pula quando o parquet não está presente, para que a suíte fique verde num
    clone parcial. Os `data.parquet` **são** versionados desde 04/08/2026 (a P5
    foi revertida — ver SDD §11), então na prática este skip não dispara; ele
    existe só para não amarrar a suíte a ~96 MB de dados.
    """
    from pathlib import Path

    import features

    parquet = Path(__file__).parent.parent / "corpora" / "iscx2016" / "data.parquet"
    if not parquet.exists():
        pytest.skip(f"corpus real ausente: {parquet} (não é versionado)")
    df = pd.read_parquet(parquet, columns=["label", "url", "corpus", "split_oficial"])
    df = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)
    return features.build_feature_frame(df)


# ---------------------------------------------------------------------------
# DoD da T2.4 — sobreposição de domínio = 0 no Regime B
# ---------------------------------------------------------------------------
class TestNoDomainOverlap:
    """O critério objetivo de conclusão da T2.4."""

    def test_zero_overlap_between_all_partitions(self, corpus):
        split = splits.split_etld1(corpus, seed=42)
        overlap = splits.domain_overlap(corpus, split)
        assert overlap == {"train_val": 0, "train_test": 0, "val_test": 0}

    def test_assert_helper_passes_on_valid_split(self, corpus):
        split = splits.split_etld1(corpus, seed=42)
        splits.assert_no_domain_overlap(corpus, split)  # não deve lançar

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
    def test_zero_overlap_for_many_seeds(self, corpus, seed):
        """A garantia é estrutural, não sorte de uma seed específica."""
        split = splits.split_etld1(corpus, seed=seed)
        assert splits.domain_overlap(corpus, split) == {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        }

    def test_zero_overlap_with_skewed_domain_sizes(self):
        """Domínios de tamanhos muito diferentes (um domínio dominante).

        Caso real: `google.com` aparece milhares de vezes num corpus de
        benignos, enquanto domínios de phishing aparecem uma vez cada. O
        agrupamento tem de continuar íntegro.
        """
        rows = [{"label": 0, "url": f"http://google.com/{i}", "etld_plus_one": "google.com"}
                for i in range(500)]
        rows += [{"label": 1, "url": f"http://phish{i}.tk/login", "etld_plus_one": f"phish{i}.tk"}
                 for i in range(200)]
        df = pd.DataFrame(rows)
        split = splits.split_etld1(df, corpus="skew", seed=42)
        assert splits.domain_overlap(df, split) == {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        }

    def test_assert_helper_detects_injected_leak(self, corpus):
        """Prova que a checagem não é vacuamente verdadeira.

        Constrói à mão um split contaminado (uma linha do treino movida para o
        teste sem levar o domínio) e exige que `assert_no_domain_overlap`
        acuse. Sem este teste, a checagem poderia estar sempre passando por
        estar comparando a coisa errada.
        """
        n = len(corpus)
        bad = SplitIndices(
            corpus="sintetico",
            regime="etld1",
            train=np.arange(0, n - 2),
            val=np.array([n - 2]),
            # a última linha compartilha domínio com a penúltima, que está no train
            test=np.array([n - 1]),
        )
        with pytest.raises(SplitError, match="vazamento de domínio"):
            splits.assert_no_domain_overlap(corpus, bad)


# ---------------------------------------------------------------------------
# Integridade da partição
# ---------------------------------------------------------------------------
class TestPartitionIntegrity:
    def test_covers_every_row_exactly_once(self, corpus):
        split = splits.split_etld1(corpus, seed=42)
        union = np.concatenate([split.train, split.val, split.test])
        assert len(union) == len(corpus)
        assert len(np.unique(union)) == len(corpus)
        assert set(union) == set(range(len(corpus)))

    def test_proportions_are_approximately_respected(self, corpus):
        split = splits.split_etld1(corpus, test_size=0.20, val_size=0.10, seed=42)
        n = len(corpus)
        # Tolerância larga: o particionador distribui domínios, não linhas — a
        # fração de linhas só se aproxima da pedida. (Nos 4 corpora reais o
        # `StratifiedGroupKFold` acerta 70/10/20 na casa do centésimo; o
        # sintético é pequeno e desequilibrado de propósito.)
        assert 0.10 < len(split.test) / n < 0.32
        assert 0.03 < len(split.val) / n < 0.20
        assert len(split.train) > len(split.test)

    def test_all_partitions_contain_both_classes(self, corpus):
        """Sem as duas classes, AUROC e MCC ficam indefinidos no teste."""
        split = splits.split_etld1(corpus, seed=42)
        labels = corpus["label"].to_numpy()
        for idx in (split.train, split.val, split.test):
            assert set(labels[idx]) == {0, 1}

    def test_val_size_zero_gives_empty_val(self, corpus):
        split = splits.split_etld1(corpus, val_size=0.0, seed=42)
        assert len(split.val) == 0
        assert len(split.train) + len(split.test) == len(corpus)


# ---------------------------------------------------------------------------
# Determinismo (RNF1)
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_gives_identical_split(self, corpus):
        a = splits.split_etld1(corpus, seed=42)
        b = splits.split_etld1(corpus, seed=42)
        assert np.array_equal(a.train, b.train)
        assert np.array_equal(a.val, b.val)
        assert np.array_equal(a.test, b.test)

    def test_different_seed_gives_different_split(self, corpus):
        a = splits.split_etld1(corpus, seed=42)
        b = splits.split_etld1(corpus, seed=99)
        assert not np.array_equal(a.test, b.test)

    def test_row_order_does_not_change_domain_assignment(self, corpus):
        """Embaralhar as linhas não pode mudar em que partição cada domínio cai.

        O particionador opera sobre os grupos únicos obtidos por
        `np.unique` (ordem canônica), então a atribuição domínio→partição é
        invariante à ordem das linhas. Este teste trava essa propriedade: sem
        ela, reordenar o parquet mudaria silenciosamente todos os resultados.
        """
        shuffled = corpus.sample(frac=1.0, random_state=7).reset_index(drop=True)

        a = splits.split_etld1(corpus, seed=42)
        b = splits.split_etld1(shuffled, seed=42)

        domains_a = set(corpus["etld_plus_one"].to_numpy()[a.test])
        domains_b = set(shuffled["etld_plus_one"].to_numpy()[b.test])
        assert domains_a == domains_b

    def test_official_regime_val_is_deterministic(self, corpus):
        a = splits.split_official(corpus, seed=42)
        b = splits.split_official(corpus, seed=42)
        assert np.array_equal(a.val, b.val)


# ---------------------------------------------------------------------------
# Regime A — split oficial preservado
# ---------------------------------------------------------------------------
class TestOfficialRegime:
    def test_test_partition_is_exactly_the_official_test(self, corpus):
        split = splits.split_official(corpus, val_size=0.1, seed=42)
        official_test = np.flatnonzero(corpus["split_oficial"].to_numpy() == "test")
        assert np.array_equal(split.test, official_test)

    def test_val_comes_only_from_official_train(self, corpus):
        split = splits.split_official(corpus, val_size=0.1, seed=42)
        assert (corpus["split_oficial"].to_numpy()[split.val] == "train").all()

    def test_val_is_stratified_by_label(self):
        """Corpus desbalanceado 97/3 (padrão do PDRCNN): a validação precisa
        conter as duas classes, senão a seleção de limiar (D6) é sem sentido."""
        rows = [{"label": 1, "url": f"http://p{i}.tk", "etld_plus_one": f"p{i}.tk",
                 "split_oficial": "train"} for i in range(970)]
        rows += [{"label": 0, "url": f"http://b{i}.com", "etld_plus_one": f"b{i}.com",
                  "split_oficial": "train"} for i in range(30)]
        rows += [{"label": 1, "url": "http://t.tk", "etld_plus_one": "t.tk",
                  "split_oficial": "test"},
                 {"label": 0, "url": "http://t.com", "etld_plus_one": "t.com",
                  "split_oficial": "test"}]
        df = pd.DataFrame(rows)
        split = splits.split_official(df, corpus="desbalanceado", val_size=0.1, seed=42)
        val_labels = df["label"].to_numpy()[split.val]
        assert set(val_labels) == {0, 1}

    def test_official_regime_removes_no_row(self, corpus):
        """O Regime A não filtra nada — reproduz a fonte, vazamento incluído."""
        split = splits.split_official(corpus, val_size=0.1, seed=42)
        assert split.n_total == len(corpus)

    def test_leaky_official_split_reproduces_overlap(self):
        """Corpus onde o split oficial compartilha domínio de propósito:
        o Regime A tem de reportar a sobreposição, e o B tem de zerá-la."""
        rows = []
        for d in range(50):
            domain = f"comum{d}.com"
            rows.append({"label": d % 2, "url": f"http://{domain}/a",
                         "etld_plus_one": domain, "split_oficial": "train"})
            rows.append({"label": d % 2, "url": f"http://{domain}/b",
                         "etld_plus_one": domain, "split_oficial": "test"})
        df = pd.DataFrame(rows)

        a = splits.split_official(df, corpus="vazando", val_size=0.0, seed=42)
        assert splits.domain_overlap(df, a)["train_test"] == 50  # 100% de vazamento

        b = splits.split_etld1(df, corpus="vazando", seed=42)
        assert splits.domain_overlap(df, b)["train_test"] == 0

    def test_missing_split_column_raises(self):
        df = pd.DataFrame({"label": [0, 1], "url": ["a.com", "b.com"],
                           "etld_plus_one": ["a.com", "b.com"]})
        with pytest.raises(SplitError, match="split_oficial"):
            splits.split_official(df)

    def test_unknown_split_value_raises(self, corpus):
        bad = corpus.copy()
        bad.loc[0, "split_oficial"] = "validation"
        with pytest.raises(SplitError, match="valores inesperados"):
            splits.split_official(bad)


# ---------------------------------------------------------------------------
# Chave de agrupamento — linhas sem eTLD+1
# ---------------------------------------------------------------------------
class TestGroupKeys:
    def test_uses_etld_when_present(self, corpus):
        keys = splits.group_keys(corpus)
        assert keys.iloc[0] == "dominio0000.com"

    def test_rows_without_etld_get_url_based_group(self):
        """Cada URL inválida distinta vira seu próprio grupo — nunca um único
        grupo "sem domínio" indivisível.

        `""` e `None` colapsam no mesmo grupo, e isso é o comportamento
        correto: são linhas sem URL alguma, indistinguíveis entre si, e o lado
        seguro do agrupamento é mantê-las juntas (nunca separadas). Na prática
        elas não chegam aqui — `corpora.py` já as descarta em
        `dropped_empty_url`.
        """
        df = pd.DataFrame({
            "label": [0, 1, 0, 1],
            "url": ["", "lixo [[[", None, "outro lixo ]]]"],
            "etld_plus_one": [None, None, None, None],
        })
        keys = splits.group_keys(df)
        assert all(str(k).startswith(splits.NO_ETLD1_PREFIX) for k in keys)
        assert keys.iloc[0] == keys.iloc[2]  # "" e None: mesma linha degenerada
        assert len({keys.iloc[0], keys.iloc[1], keys.iloc[3]}) == 3  # lixos distintos separados

    def test_identical_invalid_urls_share_a_group(self):
        """Duas ocorrências da MESMA URL inválida não podem se separar —
        seria vazamento literal (a mesma linha em treino e em teste)."""
        df = pd.DataFrame({
            "label": [0, 1],
            "url": ["lixo [[[", "lixo [[["],
            "etld_plus_one": [None, None],
        })
        keys = splits.group_keys(df)
        assert keys.iloc[0] == keys.iloc[1]

    def test_empty_string_etld_treated_as_missing(self):
        df = pd.DataFrame({"label": [0], "url": ["x"], "etld_plus_one": [""]})
        keys = splits.group_keys(df)
        assert str(keys.iloc[0]).startswith(splits.NO_ETLD1_PREFIX)

    def test_split_survives_rows_without_etld(self):
        """Um corpus com linhas inválidas ainda particiona sem vazar."""
        rows = [{"label": i % 2, "url": f"http://d{i}.com/x", "etld_plus_one": f"d{i}.com"}
                for i in range(60)]
        rows += [{"label": 1, "url": f"lixo{i} [[[", "etld_plus_one": None} for i in range(10)]
        df = pd.DataFrame(rows)
        split = splits.split_etld1(df, corpus="com_invalidas", seed=42)
        assert splits.domain_overlap(df, split) == {
            "train_val": 0, "train_test": 0, "val_test": 0,
        }
        assert split.n_total == len(df)

    def test_missing_etld_column_raises_with_actionable_message(self):
        df = pd.DataFrame({"label": [0], "url": ["a.com"]})
        with pytest.raises(SplitError, match="build_feature_frame"):
            splits.group_keys(df)


# ---------------------------------------------------------------------------
# Fachada, validação de argumentos e relatório
# ---------------------------------------------------------------------------
class TestFacadeAndGuards:
    def test_make_split_dispatches(self, corpus):
        assert splits.make_split(corpus, "etld1", seed=42).regime == "etld1"
        assert splits.make_split(corpus, "oficial", seed=42).regime == "oficial"

    def test_unknown_regime_raises(self, corpus):
        with pytest.raises(SplitError, match="regime desconhecido"):
            splits.make_split(corpus, "aleatorio")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"test_size": 0.0},
            {"test_size": 1.0},
            {"val_size": -0.1},
            {"test_size": 0.6, "val_size": 0.5},
        ],
    )
    def test_invalid_proportions_raise(self, corpus, kwargs):
        with pytest.raises(SplitError):
            splits.split_etld1(corpus, **kwargs)

    def test_too_few_domains_raises(self):
        df = pd.DataFrame({
            "label": [0, 1, 0, 1],
            "url": ["http://a.com/1", "http://a.com/2", "http://b.com/1", "http://b.com/2"],
            "etld_plus_one": ["a.com", "a.com", "b.com", "b.com"],
        })
        with pytest.raises(SplitError, match="domínio"):
            splits.split_etld1(df, corpus="minusculo")

    def test_empty_corpus_raises(self):
        df = pd.DataFrame({"label": [], "url": [], "etld_plus_one": []})
        with pytest.raises(SplitError):
            splits.split_etld1(df, corpus="vazio")

    def test_mixed_corpora_raises(self, corpus):
        mixed = corpus.copy()
        mixed.loc[0, "corpus"] = "outro"
        with pytest.raises(SplitError, match="mistura"):
            splits.split_etld1(mixed)

    def test_frames_returns_disjoint_slices(self, corpus):
        split = splits.split_etld1(corpus, seed=42)
        frames = split.frames(corpus)
        assert sum(len(f) for f in frames.values()) == len(corpus)
        assert set(frames["train"]["etld_plus_one"]) & set(frames["test"]["etld_plus_one"]) == set()

    def test_split_summary_shape(self, corpus):
        split = splits.split_etld1(corpus, seed=42)
        summary = splits.split_summary(corpus, split)
        assert summary["regime"] == "etld1"
        assert summary["domain_overlap"]["train_test"] == 0
        assert summary["n_domains_total"] == corpus["etld_plus_one"].nunique()
        for part in ("train", "val", "test"):
            assert summary["partitions"][part]["n_rows"] > 0
            assert 0 <= summary["partitions"][part]["phishing_pct"] <= 100

    def test_cross_corpus_overlap_measured_not_filtered(self):
        """A fora-diagonal pode compartilhar domínio entre corpora distintos;
        a função mede isso (para Ameaças à Validade) sem alterar nada."""
        train_df = pd.DataFrame({
            "url": ["http://a.com/1", "http://b.com/1"],
            "etld_plus_one": ["a.com", "b.com"],
        })
        test_df = pd.DataFrame({
            "url": ["http://b.com/2", "http://c.com/1", "http://d.com/1"],
            "etld_plus_one": ["b.com", "c.com", "d.com"],
        })
        stats = splits.cross_corpus_domain_overlap(
            train_df, [0, 1], test_df, [0, 1, 2]
        )
        assert stats["shared_domains"] == 1  # b.com
        assert stats["test_domains"] == 3
        assert stats["rows_with_shared_domain_pct"] == pytest.approx(33.33, abs=0.01)


# ---------------------------------------------------------------------------
# T4.10 — sobreposição da UNIÃO (leave-one-corpus-out, SDD §13.4-17)
# ---------------------------------------------------------------------------
class TestUniaoDeTreinos:
    """`union_domain_overlap`: o que a matriz par a par não conseguia medir."""

    @staticmethod
    def _quadro(dominios):
        return pd.DataFrame({
            "url": [f"http://{d}/1" for d in dominios],
            "etld_plus_one": list(dominios),
        })

    def test_um_unico_quadro_reproduz_a_funcao_par_a_par(self):
        """Com um elemento, tem de dar o MESMO que `cross_corpus_domain_overlap`.

        Não é redundância: a T2 sai de uma função e a T6 da outra, e se as duas
        contassem 'domínio' de formas diferentes o artigo compararia números
        incomparáveis — foi esse tipo de divergência que produziu a correção da
        §3.3 registrada na §11 de 04/08/2026.
        """
        treino = self._quadro(["a.com", "b.com"])
        teste = self._quadro(["b.com", "c.com", "d.com"])

        par = splits.cross_corpus_domain_overlap(treino, [0, 1], teste, [0, 1, 2])
        uniao = splits.union_domain_overlap([(treino, [0, 1])], teste, [0, 1, 2])

        for chave in par:
            assert uniao[chave] == par[chave], f"divergiu em '{chave}'"
        assert uniao["n_train_corpora"] == 1

    def test_a_uniao_pega_o_que_nenhum_par_sozinho_pegaria(self):
        """O ponto inteiro da T4.10: a união é maior que qualquer parcela.

        Cada corpus de treino compartilha UM domínio distinto com o teste.
        Par a par, cada um mede 1 de 3 (33,33%); a união mede 2 de 3 (66,67%).
        Somar as parcelas daria 100%, que está errado — é por isso que o SDD
        só tinha limites derivados antes desta função.
        """
        a = self._quadro(["x.com", "ruido1.com"])
        b = self._quadro(["y.com", "ruido2.com"])
        teste = self._quadro(["x.com", "y.com", "z.com"])

        pa = splits.cross_corpus_domain_overlap(a, [0, 1], teste, [0, 1, 2])
        pb = splits.cross_corpus_domain_overlap(b, [0, 1], teste, [0, 1, 2])
        uniao = splits.union_domain_overlap(
            [(a, [0, 1]), (b, [0, 1])], teste, [0, 1, 2]
        )

        assert pa["shared_domains"] == 1
        assert pb["shared_domains"] == 1
        assert uniao["shared_domains"] == 2
        assert uniao["shared_domain_pct"] == pytest.approx(66.67, abs=0.01)
        # Entre o maior isolado e a soma — os dois limites que a §13.4-17 tinha.
        assert pa["shared_domains"] < uniao["shared_domains"] < (
            pa["shared_domains"] + pb["shared_domains"] + 1
        )
        assert uniao["n_train_corpora"] == 2

    def test_dominio_repetido_entre_treinos_nao_conta_duas_vezes(self):
        """Os conjuntos se intersectam, e é por isso que somar superestima."""
        a = self._quadro(["comum.com", "so_a.com"])
        b = self._quadro(["comum.com", "so_b.com"])
        teste = self._quadro(["comum.com", "outro.com"])

        uniao = splits.union_domain_overlap(
            [(a, [0, 1]), (b, [0, 1])], teste, [0, 1]
        )
        assert uniao["shared_domains"] == 1          # 'comum.com', uma vez só
        assert uniao["train_domains"] == 3           # comum, so_a, so_b
        assert uniao["shared_domain_pct"] == pytest.approx(50.0)

    def test_lista_vazia_falha_alto(self):
        teste = self._quadro(["a.com"])
        with pytest.raises(splits.SplitError):
            splits.union_domain_overlap([], teste, [0])


# ---------------------------------------------------------------------------
# Corpus real — o teste que importa de verdade (roda se o parquet existir)
# ---------------------------------------------------------------------------
class TestRealCorpus:
    """O teste que importa de verdade: dados reais, canonização real."""

    def test_etld1_regime_has_zero_overlap_on_real_data(self, real_sample):
        split = splits.split_etld1(real_sample, seed=42)
        assert splits.domain_overlap(real_sample, split) == {
            "train_val": 0, "train_test": 0, "val_test": 0,
        }

    def test_official_regime_leaks_on_real_data(self, real_sample):
        """SDD §3.3 mediu 79,6% de contaminação no ISCXURL2016. Numa amostra
        de 3.000 linhas o número exato varia, mas o vazamento tem de existir —
        se aparecer 0, o Regime A parou de reproduzir o split da fonte."""
        split = splits.split_official(real_sample, val_size=0.1, seed=42)
        overlap = splits.domain_overlap(real_sample, split)
        assert overlap["train_test"] > 0

    def test_both_regimes_preserve_all_rows(self, real_sample):
        for regime in splits.REGIMES:
            split = splits.make_split(real_sample, regime, seed=42)
            assert split.n_total == len(real_sample)


# ---------------------------------------------------------------------------
# RF4 revisto — o split é agrupado E estratificado (05/08/2026)
# ---------------------------------------------------------------------------
class TestEstratificacaoPorGrupo:
    """O ganho que motivou trocar `GroupShuffleSplit` por `StratifiedGroupKFold`.

    O sorteio cego não olhava o rótulo, e no ISCX real produziu validação com
    31,14% de phishing contra 56,41% no teste — 25,27 p.p. Como o limiar sai da
    validação (D6) e é transportado para as 4 células de teste da linha, o desvio
    contaminava as métricas com limiar de 24 dos 192 registros.

    Medido nos 4 corpora reais depois da troca: amplitude de 0,00 a 0,01 p.p., e
    partições de exatamente 70/10/20% (o ISCX vinha com 26,29% de teste).

    A garantia de sobreposição zero **não** muda com a estratificação — quem
    cobre isso é `TestNoDomainOverlap`, que roda sobre o mesmo particionador.
    """

    @staticmethod
    def _prevalencias(df, split):
        y = df["label"].to_numpy()
        return {
            p: 100.0 * y[getattr(split, p)].mean()
            for p in ("train", "val", "test")
        }

    def test_prevalencia_fica_proxima_nas_tres_particoes(self, corpus):
        split = splits.split_etld1(corpus, test_size=0.2, val_size=0.1, seed=42)
        prev = self._prevalencias(corpus, split)
        amplitude = max(prev.values()) - min(prev.values())
        # Domínio é indivisível, então o balanço é aproximado, nunca exato — mas
        # o sorteio cego chegava a 25 p.p. e este limite o barraria.
        assert amplitude < 10.0, f"prevalencia desequilibrada: {prev}"

    def test_e_melhor_que_o_sorteio_cego(self, corpus):
        """Comparação direta contra o comportamento anterior ao RF4 revisto.

        `_group_shuffle_once` é mantido no módulo justamente como referência do
        que se deixou para trás (e para o caso de classe única). Se algum dia a
        estratificação parar de ajudar, este teste avisa.
        """
        import numpy as np

        grupos = splits.group_keys(corpus).to_numpy()
        y = corpus["label"].to_numpy()

        trainval, teste = splits._group_shuffle_once(grupos, 0.2, 42)
        sub_tr, sub_val = splits._group_shuffle_once(
            grupos[trainval], 0.1 / 0.8, 42
        )
        cego = {
            "train": trainval[sub_tr], "val": trainval[sub_val], "test": teste,
        }
        prev_cego = {p: 100.0 * y[i].mean() for p, i in cego.items()}
        amp_cego = max(prev_cego.values()) - min(prev_cego.values())

        split = splits.split_etld1(corpus, test_size=0.2, val_size=0.1, seed=42)
        prev_novo = self._prevalencias(corpus, split)
        amp_novo = max(prev_novo.values()) - min(prev_novo.values())

        assert amp_novo <= amp_cego, (
            f"o sorteio estratificado ficou pior que o cego: "
            f"{amp_novo:.2f} vs {amp_cego:.2f} p.p."
        )
        assert not np.array_equal(np.sort(split.test), np.sort(teste)), (
            "as duas rotas deram a mesma partição — a estratificação não rodou"
        )

    def test_corpus_de_classe_unica_cai_no_sorteio_cego_sem_levantar(self, corpus):
        """Estratificar não faz sentido com uma classe só: o sorteio agrupado
        simples é o comportamento correto, não uma exceção a corrigir. A
        sobreposição zero continua valendo."""
        uma_classe = corpus.copy()
        uma_classe["label"] = 1
        split = splits.split_etld1(uma_classe, test_size=0.2, val_size=0.1, seed=42)
        assert splits.domain_overlap(uma_classe, split) == {
            "train_val": 0, "train_test": 0, "val_test": 0,
        }

    @pytest.mark.parametrize("test_size", [0.30, 0.40, 0.45])
    def test_fracao_nao_aproximavel_por_kfold_levanta(self, corpus, test_size):
        """Sendo K-fold, a fração é quantizada em 1/k.

        0,30 → o k mais próximo é 3, que dá 0,3333: desvio de 0,033. Em vez de
        devolver em silêncio uma partição de tamanho diferente do pedido, levanta.
        Os valores do projeto (0,20 e 0,125) são recíprocos exatos.
        """
        with pytest.raises(splits.SplitError, match="aproximável|aproximavel"):
            splits.split_etld1(
                corpus, test_size=test_size, val_size=0.10, seed=42
            )

    @pytest.mark.parametrize("test_size", [0.10, 0.125, 0.20, 0.25, 0.50])
    def test_fracao_reciproca_exata_sai_no_tamanho_pedido(self, corpus, test_size):
        split = splits.split_etld1(corpus, test_size=test_size, val_size=0.10, seed=42)
        obtido = len(split.test) / len(corpus)
        # O sintético é pequeno e tem domínios de tamanho desigual; nos corpora
        # reais isto sai exato (medido: 20,00% no ISCX).
        assert abs(obtido - test_size) < 0.12, (
            f"pediu {test_size}, saiu {obtido:.4f}"
        )


class TestCanariaDeParticao:
    """`splits.fingerprint_particao` — pega troca de algoritmo de partição.

    A `experiment.yml` declara `test_size` e `val_size`, mas **não** o
    particionador. Sem esta canária, trocar `GroupShuffleSplit` por
    `StratifiedGroupKFold` deixaria `matrix.config_fingerprint` idêntico e a
    retomada (RH2) aceitaria registros da rodada anterior como válidos.
    """

    def test_e_deterministica(self):
        assert splits.fingerprint_particao() == splits.fingerprint_particao()

    def test_muda_ao_trocar_o_particionador(self, monkeypatch):
        antes = splits.fingerprint_particao()
        monkeypatch.setattr(
            splits,
            "_grupo_estratificado_once",
            lambda grupos, rotulos, frac, seed: splits._group_shuffle_once(
                grupos, frac, seed
            ),
        )
        assert splits.fingerprint_particao() != antes

    def test_o_corpus_sintetico_tem_prevalencia_desigual(self):
        """Se todo domínio tivesse a mesma prevalência, estratificar e sortear
        cego dariam o mesmo resultado e a canária não distinguiria os dois."""
        import numpy as np

        # Reproduz o corpus interno da canária.
        n_dominios, por_dominio = 60, 5
        prevalencias = {
            (d * 7) % (por_dominio + 1) for d in range(n_dominios)
        }
        assert len(prevalencias) > 1
        assert len(prevalencias) == len(np.unique(list(prevalencias)))

    def test_mudou_em_relacao_ao_valor_anterior(self):
        """T9: a canária ganhou duplicatas e conflito e passa por
        `particionar_corpus`, então o valor muda. 8e7a6010e1fb580d é o valor
        de antes da deduplicação."""
        assert splits.fingerprint_particao() != "8e7a6010e1fb580d"

    def test_muda_se_a_dedup_deixar_de_remover_duplicatas(self, monkeypatch):
        import dedup

        antes = splits.fingerprint_particao()
        original = dedup.deduplicar

        def sem_dedup(quadro, regime, **kw):
            r = original(quadro, regime, **kw)
            return dedup.ResultadoDedup(
                posicoes=np.arange(len(quadro)), contagens=r.contagens
            )

        monkeypatch.setattr(dedup, "deduplicar", sem_dedup)
        assert splits.fingerprint_particao() != antes


# ---------------------------------------------------------------------------
# T9: fachada particionar_corpus (spec DEDUP 2, 7 e caso de borda de classes)
# ---------------------------------------------------------------------------


def _corpus_com_duplicatas() -> tuple[pd.DataFrame, dict]:
    """Corpus sintético com `url_norm` e linhas anexadas que a dedup remove.

    Devolve o quadro e as posições (índice do quadro original) com papel
    conhecido, escritas à mão a partir das regras da spec.
    """
    base = _synthetic_corpus()
    base["url_norm"] = base["url"]
    n = len(base)
    extras = []
    # n: duplicata exata do treino (linha 0 é de domínio 0, split "test"; por
    # isso copia-se a linha 4, domínio 1, "train").
    extras.append(dict(base.iloc[4]))
    # n+1: URL do treino repetida no teste com o mesmo rótulo (Regime A fica).
    rep = dict(base.iloc[4])
    rep["split_oficial"] = "test"
    extras.append(rep)
    # n+2: conflito com a linha 8 (rótulo oposto, mesma URL).
    conf = dict(base.iloc[8])
    conf["label"] = 1 - conf["label"]
    extras.append(conf)
    quadro = pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
    papeis = {
        "duplicata_treino": n,
        "repetida_no_teste": n + 1,
        "conflito_novo": n + 2,
        "conflito_original": 8,
        "n": n,
    }
    return quadro, papeis


class TestParticionarCorpus:
    @pytest.mark.parametrize("regime", ["oficial", "etld1"])
    def test_posicoes_indexam_o_quadro_original_e_cobrem_as_mantidas(self, regime):
        import dedup

        quadro, _ = _corpus_com_duplicatas()
        split, contagens = splits.particionar_corpus(
            quadro, regime, corpus="sintetico", seed=42, test_size=0.2, val_size=0.1
        )
        mantidas = dedup.deduplicar(quadro, regime).posicoes
        uniao = np.concatenate([split.train, split.val, split.test])
        # Disjuntas e cobrindo exatamente as posições mantidas pela dedup.
        assert len(uniao) == len(np.unique(uniao))
        assert sorted(uniao.tolist()) == mantidas.tolist()
        assert uniao.max() < len(quadro)
        assert contagens["n_final"] == len(uniao)

    def test_regime_b_remove_duplicata_e_conflito_conhecidos(self):
        quadro, p = _corpus_com_duplicatas()
        split, c = splits.particionar_corpus(
            quadro, "etld1", corpus="sintetico", seed=42
        )
        uniao = set(np.concatenate([split.train, split.val, split.test]).tolist())
        # DEDUP 1 e 4: as duas duplicatas anexadas e as duas linhas do conflito.
        assert not uniao & {
            p["duplicata_treino"],
            p["repetida_no_teste"],
            p["conflito_novo"],
            p["conflito_original"],
        }
        assert c["n_conflito_linhas"] == 2
        assert c["n_duplicatas_removidas"] == 2

    def test_regime_b_nenhum_etld1_em_duas_particoes(self):
        quadro, _ = _corpus_com_duplicatas()
        split, _ = splits.particionar_corpus(
            quadro, "etld1", corpus="sintetico", seed=42
        )
        # `domain_overlap` indexa o quadro original com as posições devolvidas.
        assert splits.domain_overlap(quadro, split) == {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        }

    def test_regime_a_validacao_sai_do_treino_deduplicado(self):
        # DEDUP 7.
        quadro, p = _corpus_com_duplicatas()
        split, c = splits.particionar_corpus(
            quadro, "oficial", corpus="sintetico", seed=42
        )
        treino_e_val = np.concatenate([split.train, split.val])
        # Treino e validação vêm só do treino oficial, sem url_norm repetido.
        assert set(quadro["split_oficial"].to_numpy()[treino_e_val]) == {"train"}
        assert quadro["url_norm"].to_numpy()[treino_e_val].tolist() == list(
            dict.fromkeys(quadro["url_norm"].to_numpy()[treino_e_val].tolist())
        )
        # A duplicata do treino não entra em nenhuma partição.
        assert p["duplicata_treino"] not in set(
            np.concatenate([split.train, split.val, split.test]).tolist()
        )
        # DEDUP 3: a URL do treino repetida no teste, com o mesmo rótulo, fica.
        assert p["repetida_no_teste"] in set(split.test.tolist())
        assert c["n_repetidas_treino_teste"] == 1
        # O teste é exatamente o teste oficial deduplicado.
        assert set(quadro["split_oficial"].to_numpy()[split.test]) == {"test"}

    @pytest.mark.parametrize("regime", ["oficial", "etld1"])
    def test_particao_sem_uma_das_classes_levanta_split_error(self, regime):
        quadro, _ = _corpus_com_duplicatas()
        if regime == "oficial":
            # O teste oficial passa a ter só a classe 0.
            quadro.loc[quadro["split_oficial"] == "test", "label"] = 0
            particao = "test"
        else:
            # Um único domínio com as duas classes e o resto só de classe 0:
            # alguma partição fica sem a classe 1.
            quadro["label"] = 0
            quadro.loc[quadro["etld_plus_one"] == "dominio0001.com", "label"] = 1
            particao = None
        with pytest.raises(SplitError) as exc:
            splits.particionar_corpus(quadro, regime, corpus="sintetico", seed=42)
        msg = str(exc.value)
        assert "sintetico" in msg and f"'{regime}'" in msg
        if particao:
            assert f"'{particao}'" in msg
        else:
            assert any(f"'{n}'" in msg for n in ("train", "val", "test"))

    def test_erro_de_regime_desconhecido(self):
        quadro, _ = _corpus_com_duplicatas()
        with pytest.raises(Exception, match="regime"):
            splits.particionar_corpus(quadro, "xyz", corpus="sintetico", seed=42)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
