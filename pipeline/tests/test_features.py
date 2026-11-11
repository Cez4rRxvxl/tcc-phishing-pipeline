"""Testes de regressão do extrator de features léxicas (T3.1/T3.2/T3.3).

Cobre especificamente:
  - B4 (`src/features/lexical_features.py`): `urlparse()` sem scheme
    devolvia `netloc=''`, zerando todas as features de domínio. Corrigido
    garantindo scheme antes do parse.
  - B3 (`src/pipelines/05_build_feature_matrix.py`): features extraídas da
    URL crua em vez da URL canonizada. Corrigido em
    `src/features/__init__.py::build_feature_frame`, que canoniza antes de
    extrair.
  - T3.2: nenhuma das 41 features satura o dtype auditado.

Convenção de import: `pyproject.toml` define `[tool.pytest.ini_options]
pythonpath = ["src"]`, então os módulos de `src/` são importáveis
diretamente por nome de topo (`features`, `data`, ...), igual ao resto do
código do projeto.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import features
from features.lexical_features import (
    LexicalFeatureExtractor,
    extract_lexical_features,
    get_feature_names,
)

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def extractor():
    return LexicalFeatureExtractor()


# ---------------------------------------------------------------------------
# B4 — scheme ausente não pode mais zerar features de domínio
# ---------------------------------------------------------------------------
class TestSchemeHandling:
    """Regressão direta de B4 (docs/diagnostico.md §3.1 e §3.5)."""

    def test_no_scheme_with_path(self):
        """`google.com/algo` (sem scheme) -> domain_length deve ser 10, não 0."""
        feats = extract_lexical_features("google.com/algo")
        assert feats["domain_length"] == 10
        assert feats["has_ip"] == 0

    def test_no_scheme_bare_domain(self):
        """`google.com` sozinho (domínio nu) -> domain_length == 10."""
        feats = extract_lexical_features("google.com")
        assert feats["domain_length"] == 10

    def test_with_scheme_features_coherent(self):
        """URL completa com scheme: todas as features relacionadas devem bater."""
        feats = extract_lexical_features("https://www.google.com/path?a=1")
        assert feats["has_https"] == 1
        assert feats["has_www"] == 1
        assert feats["domain_length"] == len("www.google.com")  # 14
        assert feats["has_query"] == 1
        assert feats["subdomain_count"] == 1  # "www" antes de google.com
        assert feats["path_length"] == len("/path")

    def test_ip_address(self):
        """`http://192.168.1.1/x` -> has_ip deve ser 1."""
        feats = extract_lexical_features("http://192.168.1.1/x")
        assert feats["has_ip"] == 1
        assert feats["domain_length"] == len("192.168.1.1")

    def test_port(self):
        """`http://exemplo.com:8080/` -> has_port deve ser 1."""
        feats = extract_lexical_features("http://exemplo.com:8080/")
        assert feats["has_port"] == 1
        assert feats["has_ip"] == 0

    def test_default_port_not_flagged(self):
        """Porta padrão explícita (443 para https) não deve marcar has_port."""
        feats = extract_lexical_features("https://exemplo.com:443/")
        assert feats["has_port"] == 0

    def test_subdomains(self):
        """`http://a.b.c.exemplo.com/` -> subdomain_count deve ser coerente (3)."""
        feats = extract_lexical_features("http://a.b.c.exemplo.com/")
        assert feats["subdomain_count"] == 3
        assert feats["domain_length"] == len("a.b.c.exemplo.com")

    def test_has_https_no_false_positive_on_substring(self):
        """
        Bug adicional encontrado além de B4: o código antigo fazia
        `url.lower().startswith('https')`, que dava falso positivo para
        hosts sem scheme cujo texto começa com a substring "https" (ex.:
        `httpsecure.com`). A correção usa o scheme explícito real.
        """
        feats = extract_lexical_features("httpsecure.com/login")
        assert feats["has_https"] == 0

    def test_ftp_scheme_is_not_https(self):
        feats = extract_lexical_features("ftp://files.exemplo.com/a")
        assert feats["has_https"] == 0

    def test_injected_scheme_does_not_leak_into_whole_string_features(self):
        """
        O `http://` injetado internamente para viabilizar o parse não pode
        vazar para features de texto corrido (url_length, contagens de
        caracteres, entropia) — senão toda URL sem scheme ganharia +7
        caracteres artificiais e um viés sistemático novo.
        """
        feats = extract_lexical_features("google.com")
        assert feats["url_length"] == len("google.com")  # 10, não 17
        assert feats["slash_count"] == 0
        assert feats["dot_count"] == 1


# ---------------------------------------------------------------------------
# Entradas degeneradas: nunca lançar exceção (RF3 / DoD)
# ---------------------------------------------------------------------------
class TestDegenerateInputs:
    def test_empty_string(self):
        feats = extract_lexical_features("")
        assert feats["domain_length"] == 0
        assert feats["url_length"] == 0

    def test_none(self):
        feats = extract_lexical_features(None)
        assert feats["domain_length"] == 0

    def test_whitespace_only(self):
        feats = extract_lexical_features("   ")
        assert feats["domain_length"] == 0
        assert feats["url_length"] == 0

    def test_malformed_url_no_exception(self):
        for bad in [
            "not a url at all !!! [[[",
            "http://",
            "://foo",
            "[[[:::]]]",
            "http://[::::::::]",
            "a" * 5000,  # muito comprido, sem estrutura de URL
            "%%%%%%%%%%",
            12345,  # tipo errado (não é string)
            [],
            {},
        ]:
            feats = extract_lexical_features(bad)
            assert isinstance(feats, dict)
            assert feats["domain_length"] >= 0

    def test_batch_never_raises(self):
        from features.lexical_features import extract_features_batch

        urls = ["google.com/algo", "", None, "http://192.168.1.1:8080/x",
                "not a url [[[", "https://www.exemplo.com.br/produto?id=1"]
        df = extract_features_batch(urls)
        assert len(df) == len(urls)
        assert df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Prova de regressão do bug histórico: domain_length média ~4,48 (B4)
# ---------------------------------------------------------------------------
class TestDomainLengthRegression:
    def test_domain_length_mean_not_near_zero_for_schemeless_domains(self):
        """
        Simula o padrão do corpus legado (`dataset.csv`): benignos são
        domínios nus, sem scheme, sem path (evidência da SDD §3.2). Antes
        da correção de B4, `domain_length` para essas linhas era 0 e a
        média do corpus caía para ~4,48. Depois da correção, a média deve
        refletir o comprimento real dos domínios.
        """
        bare_domains = [
            "google.com", "facebook.com", "amazon.com", "wikipedia.org",
            "montaguemillennium.com", "museumneka.com", "nationonenews.com",
            "example.co.uk", "a.io", "banco-do-brasil-seguro.com.br",
        ]
        expected_mean = np.mean([len(d) for d in bare_domains])

        lengths = [extract_lexical_features(u)["domain_length"] for u in bare_domains]
        observed_mean = np.mean(lengths)

        assert observed_mean == pytest.approx(expected_mean)
        assert observed_mean > 10  # bem longe do ~4,48 do bug
        assert all(length > 0 for length in lengths)


# ---------------------------------------------------------------------------
# T3.2 — nenhuma feature satura o dtype auditado
# ---------------------------------------------------------------------------
class TestDtypeAudit:
    def test_dtype_map_covers_all_41_features(self):
        assert set(features.DTYPE_MAP.keys()) == set(get_feature_names())
        assert len(features.DTYPE_MAP) == 41

    def test_promoted_features_are_uint16(self):
        """
        domain_length e special_char_count eram uint8 no mapa legado e
        estavam a um passo de saturar (máx. observado 246 e 239,
        respectivamente, segundo a auditoria da tarefa). Ambas devem ter
        sido promovidas.
        """
        promoted = [
            "domain_length", "special_char_count", "dot_count", "slash_count",
            "hyphen_count", "underscore_count", "percent_count", "at_count",
            "question_count", "equal_count", "ampersand_count", "hash_count",
            "subdomain_count", "path_tokens", "query_params", "www_count",
        ]
        for feat in promoted:
            assert features.DTYPE_MAP[feat] == "uint16", (
                f"{feat} deveria ter sido promovida para uint16"
            )

    def test_boolean_features_stay_uint8(self):
        for feat in features.FEATURE_GROUPS["boolean"]:
            assert features.DTYPE_MAP[feat] == "uint8"

    def test_no_feature_saturates_on_adversarial_url(self):
        """
        Constrói uma URL adversarial que ultrapassaria uint8 (limite 255)
        em várias contagens de caractere — o tipo de URL de ofuscação que
        motivou a auditoria de dtype (percent_count observado = 600 e
        special_char_count observado = 784 na amostra real testada,
        ambos > 255). `apply_dtypes` com o mapa atual não deve lançar
        `OverflowError`, e nenhum valor pode ser truncado/quebrado pelo
        cast.
        """
        adversarial_url = "http://exemplo.com/" + ("%20" * 300) + ("." * 300)
        feats = extract_lexical_features(adversarial_url)

        assert feats["percent_count"] > 255
        assert feats["dot_count"] > 255

        X = pd.DataFrame([feats], columns=get_feature_names())
        X_typed = features.apply_dtypes(X)  # não deve lançar

        assert X_typed.loc[0, "percent_count"] == feats["percent_count"]
        assert X_typed.loc[0, "dot_count"] == feats["dot_count"]
        for col in X_typed.columns:
            limit = features._UINT_LIMITS.get(str(X_typed[col].dtype))
            if limit is not None:
                assert X_typed[col].max() <= limit, f"{col} saturou {X_typed[col].dtype}"

    def test_apply_dtypes_raises_instead_of_silently_truncating(self):
        """
        Prova que a rede de segurança funciona: se alguém tentar forçar uma
        feature de volta para um dtype pequeno demais para o valor
        observado, `apply_dtypes` deve levantar `OverflowError` em vez de
        deixar o `.astype()` truncar silenciosamente (o mecanismo exato de
        B4/B10: um bug fica invisível porque o valor errado ainda "parece"
        um número válido).
        """
        X = pd.DataFrame({"special_char_count": [300]})
        with pytest.raises(OverflowError):
            features.apply_dtypes(X, dtype_map={"special_char_count": "uint8"}, strict=True)


# ---------------------------------------------------------------------------
# Contrato de 41 features (B10) — não pode haver drift silencioso
# ---------------------------------------------------------------------------
class TestFeatureContract:
    def test_exactly_41_features(self):
        assert len(get_feature_names()) == 41

    def test_matches_features_json_contract(self):
        """
        `config/features.json` é o contrato de quais 41 features existem
        (corrige B10: README alegava "104+ features") e de seus dtypes
        (corrige B13: overflow silencioso de uint8).
        Este teste barra qualquer adição/remoção silenciosa de feature.

        Nota: o arquivo vivia em `artifacts/models/` no pipeline legado e foi
        movido para `config/` na limpeza do legado — `artifacts/` passou a ser
        diretório de saída gerada, não versionada.
        """
        features_json_path = PROJECT_ROOT / "config" / "features.json"
        assert features_json_path.exists(), (
            f"contrato de features ausente em {features_json_path}"
        )

        metadata = json.loads(features_json_path.read_text(encoding="utf-8"))
        assert set(metadata["feature_names"]) == set(get_feature_names())


# ---------------------------------------------------------------------------
# B3 — build_feature_frame precisa canonizar antes de extrair
# ---------------------------------------------------------------------------
class TestBuildFeatureFrame:
    def test_preserves_original_columns_and_row_order(self):
        df = pd.DataFrame({
            "label": [0, 1, 0],
            "url": ["google.com/algo", "http://malicioso.tk/login.php", ""],
            "corpus": ["c1", "c1", "c1"],
        })
        out = features.build_feature_frame(df)

        assert list(out["label"]) == [0, 1, 0]
        assert list(out["corpus"]) == ["c1", "c1", "c1"]
        assert len(out) == 3

    def test_canonicalization_runs_before_extraction(self):
        """
        Corrige B3: `etld_plus_one` (produzido pela canonização) deve
        existir e `domain_length` não pode ser 0 para uma URL sem scheme
        válida — provando que a extração não voltou a operar sobre a URL
        crua sem qualquer canonização.
        """
        df = pd.DataFrame({"url": ["google.com/algo"]})
        out = features.build_feature_frame(df)

        assert out.loc[0, "etld_plus_one"] == "google.com"
        assert out.loc[0, "invalid_url"] == False  # noqa: E712
        assert out.loc[0, "domain_length"] == 10

    def test_invalid_url_flagged_without_exception(self):
        df = pd.DataFrame({"url": ["", None, "not a url [[[ !!!"]})
        out = features.build_feature_frame(df)

        assert out["invalid_url"].all()
        assert out["domain_length"].tolist() == [0, 0, 0]

    def test_www_not_zeroed_by_canonicalization(self):
        """
        Decisão de design documentada em `features/__init__.py`: `url_norm`
        remove `www.`, mas `has_www`/`www_count` têm que refletir a URL
        original, não `url_norm`. Este teste existe para travar essa
        decisão — se alguém trocar a fonte de extração para `url_norm`,
        este teste quebra.
        """
        df = pd.DataFrame({"url": ["https://www.exemplo.com/"]})
        out = features.build_feature_frame(df)

        assert "www." not in out.loc[0, "url_norm"]  # canonização removeu www
        assert out.loc[0, "has_www"] == 1  # mas a feature não foi afetada
        assert out.loc[0, "www_count"] >= 1

    def test_missing_url_column_raises_keyerror(self):
        df = pd.DataFrame({"not_url": ["google.com"]})
        with pytest.raises(KeyError):
            features.build_feature_frame(df)

    def test_dtypes_applied_in_output(self):
        df = pd.DataFrame({"url": ["google.com/algo"] * 5})
        out = features.build_feature_frame(df)
        assert str(out["domain_length"].dtype) == "uint16"
        assert str(out["has_ip"].dtype) == "uint8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestT34DefeitosDoExtrator:
    """T3.4 — os 3 defeitos remanescentes do extrator, corrigidos em 05/08/2026.

    Cada um foi reproduzido e medido em amostra de 109.965 linhas reais dos 4
    corpora de treino antes da correção. São
    testes de valor, não de tipo: o DoD da T3.4 pede que cada correção seja
    coberta, e o risco aqui é justamente alguém "simplificar" a lógica de volta
    para contagem de pontos ou `in url`.
    """

    # ------------------------------------------------------------------ (a)
    def test_has_www_nao_dispara_fora_do_host(self):
        """`'www.' in url` dava falso positivo em 0,8% das linhas reais."""
        df = features.extract_features(
            [
                "https://site.com/www.algo",       # www no path, não no host
                "https://site.com/?u=www.x.com",   # www na query
                "https://www.site.com/x",          # www de verdade
            ]
        )
        assert df.loc[0, "has_www"] == 0
        assert df.loc[1, "has_www"] == 0
        assert df.loc[2, "has_www"] == 1

    def test_www_count_continua_contando_a_url_inteira(self):
        """A correção do (a) não pode contaminar `www_count`, que é outra
        feature e conta ocorrências em qualquer posição — ver a decisão de
        04/08/2026 na §11 sobre canonização e `has_www`/`www_count`."""
        df = features.extract_features(["https://site.com/www.algo"])
        assert df.loc[0, "has_www"] == 0
        assert df.loc[0, "www_count"] >= 1

    # ------------------------------------------------------------------ (b)
    def test_domain_length_ignora_a_porta(self):
        df = features.extract_features(
            ["https://site.com:8080/x", "https://site.com/x"]
        )
        assert df.loc[0, "domain_length"] == df.loc[1, "domain_length"] == 8

    def test_domain_length_ignora_userinfo(self):
        """`user:pw@host` — o userinfo não é domínio. Cai junto com a porta
        porque ambos saem do mesmo helper."""
        df = features.extract_features(["http://user:pw@site.com/x"])
        assert df.loc[0, "domain_length"] == 8

    # ------------------------------------------------------------------ (c)
    def test_subdomain_count_nao_conta_sufixo_composto(self):
        """Era o defeito de maior alcance: 10,0% das linhas reais, com taxa
        variando 2,6x entre corpora — confundidor direto da C2."""
        df = features.extract_features(
            [
                "https://bbc.co.uk/x",          # antes: 1 — errado, é o eTLD+1
                "https://a.b.bbc.co.uk/x",      # antes: 3 — errado, são 2
                "https://site.com.br/x",        # antes: 1 — errado
                "https://a.b.c.site.com.br/x",  # antes: 4 — errado, são 3
                "https://site.com/x",           # sempre esteve certo
                "https://www.site.com/x",       # sempre esteve certo
            ]
        )
        assert list(df["subdomain_count"]) == [0, 2, 0, 3, 0, 1]

    def test_subdomain_count_de_ip_e_zero(self):
        """IP não tem subdomínio. A contagem de pontos dava 2 para IPv4."""
        df = features.extract_features(["http://192.168.1.1/p", "http://200.1.2.3/x"])
        assert list(df["subdomain_count"]) == [0, 0]

    def test_subdomain_count_usa_a_mesma_definicao_do_split(self):
        """Coerência interna: `diagnostics.py` e `splits.py` medem domínio com
        `tldextract`. Se a feature contasse de outro jeito, a Tabela 1 e o
        Regime B falariam de "domínio" com definições diferentes — que é o
        defeito que a §11 (entrada de 04/08/2026 sobre a T2.3) descreve."""
        import tldextract

        for host in ("a.b.bbc.co.uk", "www.site.com", "a.b.c.site.com.br"):
            esperado = len(
                [r for r in tldextract.extract(host).subdomain.split(".") if r]
            )
            obtido = features.extract_features([f"https://{host}/x"]).loc[
                0, "subdomain_count"
            ]
            assert obtido == esperado, host

    # ---------------------------------------------------------------- borda
    def test_host_malformado_nao_derruba_o_extrator(self):
        """O helper usa fatiamento de string em vez de `ParseResult.hostname`
        justamente porque no Python 3.14 a propriedade levanta `ValueError` em
        host entre colchetes malformado, e os corpora têm linhas assim."""
        urls = ["http://[::1]:8080/p", "http://[malformado/p", "http://:8080/p", ""]
        df = features.extract_features(urls)  # não pode levantar
        assert len(df) == len(urls)


class TestPortaIpEEntropiaDoHost:
    """T6 (FEAT-01, FEAT-02, FEAT-03): critérios FEAT 1 a 7 e 9 da spec, com as
    URLs exatas da spec, direto no extrator (sem canonização)."""

    def test_feat1_credenciais_nao_sao_porta(self):
        # FEAT AC1
        feats = extract_lexical_features("http://user:pass@example.com/")
        assert feats["has_port"] == 0

    def test_feat2_porta_nao_padrao(self):
        # FEAT AC2
        feats = extract_lexical_features("http://example.com:8080/")
        assert feats["has_port"] == 1

    def test_feat3_ipv6_sem_porta(self):
        # FEAT AC3
        feats = extract_lexical_features("http://[2001:db8::1]/a")
        assert feats["has_ip"] == 1
        assert feats["has_port"] == 0

    def test_feat4_ipv6_com_porta(self):
        # FEAT AC4
        feats = extract_lexical_features("http://[2001:db8::1]:8080/a")
        assert feats["has_ip"] == 1
        assert feats["has_port"] == 1

    def test_feat5_ipv4_com_userinfo_e_porta(self):
        # FEAT AC5 (emenda do Design: porta 8080 marca has_port; a 80 nao)
        feats = extract_lexical_features("http://user@192.168.0.1:8080/")
        assert feats["has_ip"] == 1
        assert feats["has_port"] == 1

    def test_feat6_entropia_do_dominio_ignora_userinfo_e_porta(self):
        # FEAT AC6
        com = extract_lexical_features("http://user:pw@example.com:8080/")
        sem = extract_lexical_features("http://example.com/")
        assert com["domain_entropy"] == sem["domain_entropy"]
        assert com["domain_entropy"] > 0

    @pytest.mark.parametrize(
        "url",
        ["http://[2001:db8::1/a", "http://[::1/", "http://user@[2001:db8::1:8080/"],
    )
    def test_feat7_colchete_sem_fechamento_nao_levanta(self, url):
        # FEAT AC7: as tres caracteristicas saem sem excecao.
        feats = extract_lexical_features(url)
        for nome in ("has_ip", "has_port", "domain_entropy"):
            assert nome in feats
        assert feats["has_ip"] == 0

    @pytest.mark.parametrize(
        "url", ["http://user@192.168.0.1:80/", "https://example.com:443/"]
    )
    def test_feat9_porta_padrao_nao_marca_has_port(self, url):
        # FEAT AC9
        assert extract_lexical_features(url)["has_port"] == 0


class TestOpcaoStripScheme:
    """T7 (SENS-01, FEAT-04, PSL-03): opção `strip_scheme` e canária ampliada."""

    PAR = ("https://a.com/x", "a.com/x")
    # Características calculadas sobre `parsed` (domínio, caminho, consulta) e a
    # do esquema: a opção não pode alterá-las (SENS-01, critério 2).
    DE_PARSED = [
        "has_https", "domain_length", "path_length", "query_length",
        "fragment_length", "path_tokens", "query_params", "subdomain_count",
        "has_ip", "has_port", "has_www", "has_double_slash", "has_query",
        "has_fragment", "has_suspicious_tld", "has_url_shortener",
        "has_suspicious_extension", "domain_entropy",
    ]

    @staticmethod
    def _linha(url, strip):
        return features.extract_features([url], strip_scheme=strip).iloc[0]

    def test_texto_inteiro_igual_com_e_sem_esquema(self):
        # SENS-01 AC1. A lista de caracteristicas de texto inteiro sai do
        # extrator: sao as que mudam com o esquema quando a opcao esta desligada.
        com_off = self._linha(self.PAR[0], False)
        sem_off = self._linha(self.PAR[1], False)
        texto_inteiro = [
            n for n in features.FEATURE_NAMES
            if com_off[n] != sem_off[n] and n != "has_https"
        ]
        assert "url_length" in texto_inteiro  # a lista nao esta vazia
        com_on = self._linha(self.PAR[0], True)
        sem_on = self._linha(self.PAR[1], True)
        for nome in texto_inteiro:
            assert com_on[nome] == sem_on[nome], nome
        # e nenhuma outra, alem de has_https, difere
        for nome in features.FEATURE_NAMES:
            if nome != "has_https":
                assert com_on[nome] == sem_on[nome], nome

    def test_esquema_e_parsed_iguais_aos_da_opcao_desligada(self):
        # SENS-01 AC2, numa URL com porta, consulta e fragmento.
        url = "https://www.a.com:8080/p/q.exe?x=1&y=2#f"
        on = self._linha(url, True)
        off = self._linha(url, False)
        for nome in self.DE_PARSED:
            assert on[nome] == off[nome], nome
        assert on["has_https"] == 1

    def test_opcao_desligada_e_o_padrao(self):
        a = features.extract_features(["https://a.com/x"])
        b = features.extract_features(["https://a.com/x"], strip_scheme=False)
        assert a.equals(b)

    def test_build_feature_frame_repassa_a_opcao(self):
        df = pd.DataFrame({"url": ["https://a.com/x"]})
        on = features.build_feature_frame(df, strip_scheme=True)
        assert on.loc[0, "url_length"] == len("a.com/x")
        off = features.build_feature_frame(df)
        assert off.loc[0, "url_length"] == len("https://a.com/x")

    def test_fingerprint_difere_com_a_opcao(self):
        assert features.fingerprint_valores(True) != features.fingerprint_valores(False)
        assert features.fingerprint_valores() == features.fingerprint_valores(False)

    def test_fingerprint_muda_com_o_sha256_da_psl(self, monkeypatch):
        # PSL-03 AC7: o sha256 da PSL entra no fingerprint.
        import psl

        antes = features.fingerprint_valores()
        monkeypatch.setattr(psl, "sha256", lambda: "0" * 64)
        assert features.fingerprint_valores() != antes

    def test_canario_inclui_as_tres_urls_do_design(self):
        # FEAT-04: a canaria cobre IPv6 com porta, userinfo com IP e github.io.
        for url in (
            "http://[2001:db8::1]:8080/a",
            "http://user@192.168.0.1:8080/",
            "https://user.github.io/x",
        ):
            assert url in features.URLS_CANARIO

    def test_canario_muda_se_has_port_voltar_a_marcar_credenciais(self, monkeypatch):
        # FEAT-04: um cache anterior a correcao seria recusado.
        from features.lexical_features import LexicalFeatureExtractor

        antes = features.fingerprint_valores()
        monkeypatch.setattr(
            LexicalFeatureExtractor,
            "_has_porta_nao_padrao",
            classmethod(lambda cls, netloc: ":" in netloc),
        )
        assert features.fingerprint_valores() != antes


class TestExtratorPelaPSLVersionada:
    """T5 (PSL-02, PSL-03): subdominio e TLD vem da PSL versionada, sem rede."""

    def test_subdomain_count_respeita_a_secao_privada_e_o_sufixo_composto(self):
        # PSL-02: user.github.io e eTLD+1 (0 subdominios); a.b.bbc.co.uk tem 2.
        df = features.extract_features(
            ["https://user.github.io/x", "https://a.b.bbc.co.uk/"]
        )
        assert list(df["subdomain_count"]) == [0, 2]

    def test_extracao_com_socket_bloqueado(self, monkeypatch):
        # PSL-03 AC5: sem rede ao extrair caracteristicas.
        import socket

        import psl

        psl.extrator.cache_clear()

        def bloqueado(*args, **kwargs):
            raise OSError("rede bloqueada no teste")

        monkeypatch.setattr(socket, "socket", bloqueado)
        df = features.extract_features(
            ["https://user.github.io/x", "https://a.b.bbc.co.uk/"]
        )
        assert list(df["subdomain_count"]) == [0, 2]
        psl.extrator.cache_clear()


class TestCanariaDaCanonizacao:
    """Achado 4 do lote F: o `url_norm` (chave da dedup, AD-057) e o eTLD+1
    (chave do Regime B) entram no `fingerprint_valores`, que alimenta o contrato
    do cache e o `config_fingerprint` do livro-caixa. Mudar a canonizacao tem de
    invalidar os dois."""

    def test_canario_exercita_www_query_tracking_esquema_e_psl(self):
        quadro = features.canonicalize_urls(features.URLS_CANARIO_CANONIZACAO)
        norm = list(quadro["url_norm"])
        etld = list(quadro["etld_plus_one"])
        # www removido, query ordenada e sem tracking, fragmento fora.
        assert "https://exemplo.com/a?a=1&b=2" in norm
        # porta padrao fora e esquema acrescentado.
        assert "http://site.com/x" in norm
        assert "http://site.com/sem-esquema?y=2&z=1" in norm
        # PSL: sufixo composto e dominio privado.
        assert "bbc.co.uk" in etld and "user.github.io" in etld
        assert all(isinstance(v, str) for v in norm)

    def test_muda_quando_a_normalizacao_da_query_muda(self, monkeypatch):
        from data.url_canonicalization import URLCanonicalizer

        antes = features.fingerprint_valores()
        monkeypatch.setattr(
            URLCanonicalizer, "_normalize_query", lambda self, query: query or None
        )
        assert features.fingerprint_valores() != antes

    def test_muda_quando_o_url_norm_muda(self, monkeypatch):
        antes = features.fingerprint_valores()
        original = features.canonicalize_url

        def outro_url_norm(url):
            saida = dict(original(url))
            if isinstance(saida.get("url_norm"), str):
                saida["url_norm"] = saida["url_norm"] + "/"
            return saida

        monkeypatch.setattr(features, "canonicalize_url", outro_url_norm)
        assert features.fingerprint_valores() != antes

    def test_muda_quando_o_etld_plus_one_muda(self, monkeypatch):
        antes = features.fingerprint_valores()
        original = features.canonicalize_url

        def outro_etld(url):
            saida = dict(original(url))
            saida["etld_plus_one"] = "fixo.com"
            return saida

        monkeypatch.setattr(features, "canonicalize_url", outro_etld)
        assert features.fingerprint_valores() != antes


class TestCanariaDeValores:
    """`features.fingerprint_valores` — a defesa contra rodada misturada.

    O problema que ela resolve: nem `matrix.config_fingerprint` nem o contrato
    de features (nomes + dtypes) mudavam ao corrigir a T3.4. Os **valores**
    mudaram em 10% das linhas reais e as duas impressões digitais continuaram
    idênticas, o que faria a retomada (RH2) aceitar o livro-caixa da rodada 1 e
    produzir uma tabela com metade dos números de cada rodada, sem rastro.

    A canária extrai um conjunto fixo de URLs e hasheia os **valores**. Estes
    testes travam as duas propriedades que a tornam útil: muda quando a lógica
    muda, e não muda quando só a prosa muda. Sem eles nada impede que alguém
    remova a canária numa "simplificação" — e ela é a peça que sustenta a
    separação entre as duas rodadas.
    """

    def test_e_deterministica(self):
        assert features.fingerprint_valores() == features.fingerprint_valores()

    def test_muda_quando_a_logica_de_extracao_muda(self, monkeypatch):
        from features.lexical_features import LexicalFeatureExtractor

        antes = features.fingerprint_valores()
        original = LexicalFeatureExtractor._extract_count_features

        def com_defeito(self, url, parsed):
            saida = original(self, url, parsed)
            saida["dot_count"] = saida["dot_count"] + 1
            return saida

        monkeypatch.setattr(
            LexicalFeatureExtractor, "_extract_count_features", com_defeito
        )
        assert features.fingerprint_valores() != antes

    def test_muda_se_o_subdomain_count_voltar_a_contar_pontos(self, monkeypatch):
        """O defeito concreto da T3.4-c, que a canária existe para pegar."""
        from features.lexical_features import LexicalFeatureExtractor

        antes = features.fingerprint_valores()
        original = LexicalFeatureExtractor._extract_count_features

        def ingenuo(self, url, parsed):
            saida = original(self, url, parsed)
            if parsed.netloc:
                saida["subdomain_count"] = max(0, len(parsed.netloc.split(".")) - 2)
            return saida

        monkeypatch.setattr(
            LexicalFeatureExtractor, "_extract_count_features", ingenuo
        )
        assert features.fingerprint_valores() != antes

    def test_nao_muda_quando_so_a_docstring_muda(self, monkeypatch):
        """Era o argumento contra hashear o fonte: comentário não é resultado."""
        from features.lexical_features import LexicalFeatureExtractor

        antes = features.fingerprint_valores()
        monkeypatch.setattr(
            LexicalFeatureExtractor._extract_count_features,
            "__doc__",
            "outra explicacao, mesmo comportamento",
        )
        assert features.fingerprint_valores() == antes

    def test_as_urls_canario_exercitam_as_features_que_a_T34_mexeu(self):
        """Cobertura mínima: a canária precisa variar nas features corrigidas.

        Uma feature constante nas 10 URLs é invisível para a canária — um bug
        que a zerasse passaria batido. As três da T3.4 **têm** de variar.
        """
        quadro = features.extract_features(features.URLS_CANARIO)
        for coluna in ("subdomain_count", "domain_length", "has_www", "has_port"):
            assert quadro[coluna].nunique() > 1, (
                f"'{coluna}' e constante nas URLs canario: a canaria nao "
                f"detectaria um defeito nela"
            )

    def test_cobertura_declarada_das_features_constantes(self):
        """Documenta o limite conhecido em vez de deixá-lo implícito.

        Estas três features não variam nas 13 URLs canário (eram quatro nas 10
        de 05/08/2026: `path_tokens` saiu da lista ao entrarem as três URLs da
        T7, FEAT-04). A canária **não** as protege, e é melhor ter isso escrito
        num teste que falha ao mudar do que numa nota que ninguém lê. Se este
        teste falhar porque a lista encolheu, ótimo: apague o nome dela daqui.
        """
        quadro = features.extract_features(features.URLS_CANARIO)
        constantes = {
            c for c in features.FEATURE_NAMES if quadro[c].nunique() == 1
        }
        assert constantes == {
            "underscore_count",
            "has_suspicious_extension",
            "has_double_slash",
        }
