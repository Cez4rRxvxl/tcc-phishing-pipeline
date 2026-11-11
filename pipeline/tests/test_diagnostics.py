"""T2.3 — DoD: reproduzir as tabelas de SDD §3.2/§3.3, gravar
`results/diagnostics.json` e emitir a matriz 4×4 de sobreposição entre corpora
chamando `splits.cross_corpus_domain_overlap()`.

Três coisas que estes testes existem para travar:

1. As **definições operacionais** de `scheme` e `path`. São elas que produzem
   os números da Tabela 5; mudá-las sem querer muda uma tabela do artigo.
2. A **diagonal zero** da matriz entre corpora no regime eTLD+1. É a prova
   mais barata de que a fiação do split está certa.
3. O **artefato entregue** (`results/diagnostics.json`) não pode estar
   parcial nem vazio — é dele que a T5.3 e a T5.6 leem (R5).

Convenção de import: `pyproject.toml` define `pythonpath = ["src"]`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import diagnostics
import splits
from diagnostics import DiagnosticsError

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Definições operacionais dos atributos de viés (SDD §3.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,esperado",
    [
        ("http://exemplo.com/a", True),
        ("https://exemplo.com", True),
        ("ftp://exemplo.com/x", True),
        ("HTTP://EXEMPLO.COM", True),
        ("exemplo.com/a", False),
        ("//exemplo.com/a", False),
        ("", False),
        (None, False),
        (12345, False),
    ],
)
def test_tem_scheme(url, esperado):
    assert diagnostics.tem_scheme(url) is esperado


def test_tem_scheme_nao_casa_substring_e_regressao_do_b14():
    """B14: `has_https` era `url.startswith('https')`, teste de substring.

    `httpsecure.com` não tem scheme algum. Se `tem_scheme` casasse a
    substring, o gap de scheme — o atributo de viés mais forte que medimos —
    passaria a contar hosts que só começam com as letras certas.
    """
    assert diagnostics.tem_scheme("httpsecure.com/login") is False
    assert diagnostics.tem_scheme("ftpserver.example.org") is False


@pytest.mark.parametrize(
    "url,esperado",
    [
        ("http://exemplo.com/pagina", True),
        ("exemplo.com/pagina", True),          # sem scheme, mas tem path
        ("http://exemplo.com/", False),        # barra final solta não é path
        ("http://exemplo.com", False),
        ("exemplo.com", False),
        ("http://exemplo.com/?q=1", True),
        ("http://exemplo.com:8080/x", True),
        ("http://exemplo.com?q=1", False),     # query sem path
        ("", False),
        (None, False),
    ],
)
def test_tem_path(url, esperado):
    assert diagnostics.tem_path(url) is esperado


def test_tem_path_nao_estoura_em_host_binario():
    """`urlparse` levanta ValueError em host malformado no Python 3.14.

    O corpus `sid321axn` tem 18 linhas de bytes corrompidos (SDD §3.2). Um
    diagnóstico não pode morrer por causa do lixo que ele existe para
    descrever — por isso `tem_path` é textual, sem `urlparse`.
    """
    lixo = "\udcedh\udcad9{\udcedmOfD\udcc0\x9e3[::1]\x125"
    assert diagnostics.tem_path(lixo) in (True, False)  # o que importa: não levanta
    assert diagnostics.tem_scheme(lixo) is False


# ---------------------------------------------------------------------------
# Fixtures sintéticas
# ---------------------------------------------------------------------------


def _corpus_sintetico(
    nome: str = "sintetico",
    n_dominios: int = 20,
    linhas_por_dominio: int = 2,
    scheme_no_phishing: bool = True,
    scheme_no_benigno: bool = True,
    sufixo_dominio: str = "",
    primeiro_dominio: int = 0,
) -> pd.DataFrame:
    """Corpus já canonizado (tem `etld_plus_one`), no contrato de SDD §7.3.

    `scheme_no_*` controla o viés de construção: com
    `scheme_no_benigno=False` o corpus fica com gap de scheme de 100 p, que é
    a forma extrema do que o `dataset.csv` legado exibia (~98 p).

    `primeiro_dominio` desloca a numeração dos domínios, para montar dois
    corpora que se sobrepõem *parcialmente* — é o cenário da fora-diagonal.

    O `split_oficial` é atribuído por **linha**, não por domínio: cada domínio
    aparece no treino e no teste oficiais. Isso é de propósito, e é o que
    torna a fixture realista — o vazamento de domínio no split da fonte é
    justamente o achado da §3.3. Atribuir o split por domínio produziria uma
    partição já disjunta, na qual nem o Regime A vazaria, e os testes da
    matriz passariam sem medir nada.
    """
    linhas = []
    for d in range(primeiro_dominio, primeiro_dominio + n_dominios):
        dominio = f"dominio{d:03d}{sufixo_dominio}.com"
        for r in range(linhas_por_dominio):
            label = d % 2
            com_scheme = scheme_no_phishing if label == 1 else scheme_no_benigno
            prefixo = "http://" if com_scheme else ""
            linhas.append(
                {
                    "label": label,
                    "url": f"{prefixo}{dominio}/pagina{r}",
                    "corpus": nome,
                    "split_oficial": "train" if r % 2 == 0 else "test",
                    "etld_plus_one": dominio,
                    # Colunas que `canonizar_colunas` anexa; `url_norm` é a chave
                    # da deduplicação de `splits.particionar_corpus` (DEDUP-05).
                    "url_norm": f"http://{dominio}/pagina{r}",
                    "host": dominio,
                }
            )
    return pd.DataFrame(linhas)


@pytest.fixture
def corpus():
    return _corpus_sintetico()


# ---------------------------------------------------------------------------
# medir_vies — Tabela 5
# ---------------------------------------------------------------------------


def test_medir_vies_detecta_gap_de_scheme_extremo():
    """Benignos sem scheme e phishing com scheme ⇒ gap 100 p ⇒ contaminado."""
    frame = _corpus_sintetico(scheme_no_benigno=False)
    v = diagnostics.medir_vies(frame)

    assert v["scheme"]["phishing_pct"] == 100.0
    assert v["scheme"]["benigno_pct"] == 0.0
    assert v["scheme"]["gap_p"] == 100.0
    assert v["gap_max_p"] == 100.0
    assert v["veredito"] == "contaminado"


def test_medir_vies_corpus_sem_vies_e_limpo():
    frame = _corpus_sintetico()
    v = diagnostics.medir_vies(frame)

    assert v["scheme"]["gap_p"] == 0.0
    assert v["path"]["gap_p"] == 0.0
    assert v["veredito"] == "limpo"


def test_medir_vies_reporta_contagens_e_balanco(corpus):
    v = diagnostics.medir_vies(corpus)

    assert v["n_total"] == len(corpus)
    assert v["n_treino"] + v["n_teste"] == v["n_total"]
    assert v["balanco"]["benigno"] + v["balanco"]["phishing"] == v["n_total"]
    assert v["balanco"]["benigno_pct"] == pytest.approx(50.0)


def test_medir_vies_mede_sobre_o_treino_nao_sobre_o_corpus_inteiro():
    """O gap declarado na §3.2 é o do split de treino.

    Este corpus é construído para que treino e corpus inteiro discordem: só o
    lado benigno do TESTE recebe scheme. Se a medição usasse o corpus todo, o
    gap cairia.
    """
    frame = _corpus_sintetico(scheme_no_benigno=False)
    e_teste = frame["split_oficial"] == "test"
    e_benigno = frame["label"] == 0
    frame.loc[e_teste & e_benigno, "url"] = "http://" + frame.loc[
        e_teste & e_benigno, "url"
    ]

    v = diagnostics.medir_vies(frame)
    assert v["scheme"]["gap_p"] == 100.0                 # medido no treino
    assert v["scheme"]["gap_corpus_inteiro_p"] < 100.0   # diluído no corpus todo


def test_medir_vies_falha_alto_sem_coluna_obrigatoria(corpus):
    with pytest.raises(DiagnosticsError, match="split_oficial"):
        diagnostics.medir_vies(corpus.drop(columns=["split_oficial"]))


def test_medir_vies_falha_alto_sem_split_de_treino(corpus):
    so_teste = corpus.assign(split_oficial="test")
    with pytest.raises(DiagnosticsError, match="sem split de treino"):
        diagnostics.medir_vies(so_teste)


# ---------------------------------------------------------------------------
# medir_vazamento_split_oficial — Tabela 1
# ---------------------------------------------------------------------------


def test_medir_vazamento_conta_dominios_e_linhas():
    """Contagens exatas num caso montado à mão.

    treino: a.com, b.com, c.com   ·   teste: b.com, c.com (2×), d.com
    ⇒ 2 de 3 domínios do teste vazam (66,67%); 3 de 4 linhas (75,0%).
    """
    frame = pd.DataFrame(
        {
            "label": [0, 1, 0, 1, 0, 1, 0],
            "url": [f"http://x{i}" for i in range(7)],
            "corpus": ["c"] * 7,
            "split_oficial": ["train"] * 3 + ["test"] * 4,
            "etld_plus_one": ["a.com", "b.com", "c.com", "b.com", "c.com", "c.com", "d.com"],
        }
    )
    v = diagnostics.medir_vazamento_split_oficial(frame)

    assert v["dominios_treino"] == 3
    assert v["dominios_teste"] == 3
    assert v["dominios_sobrepostos"] == 2
    assert v["contaminacao_dominios_pct"] == pytest.approx(66.67, abs=0.01)
    assert v["linhas_teste_com_dominio_valido"] == 4
    assert v["linhas_teste_contaminadas"] == 3
    assert v["contaminacao_linhas_pct"] == pytest.approx(75.0)


def test_medir_vazamento_exclui_linhas_sem_etld1_e_as_contabiliza():
    """Linha sem eTLD+1 não é domínio — não entra na contagem, mas aparece.

    Se entrasse, um "domínio" vazio compartilhado entre treino e teste
    inflaria a sobreposição em +1 em todos os corpora. Foi exatamente esse
    detalhe (mais o heurístico ingênuo de dois rótulos) que fazia a auditoria
    original da §3.3 dar números diferentes destes — ver SDD §11.
    """
    frame = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "url": ["http://a.com/1", "lixo", "http://a.com/2", "lixo2"],
            "corpus": ["c"] * 4,
            "split_oficial": ["train", "train", "test", "test"],
            "etld_plus_one": ["a.com", None, "a.com", ""],
        }
    )
    v = diagnostics.medir_vazamento_split_oficial(frame)

    assert v["dominios_treino"] == 1
    assert v["dominios_teste"] == 1
    assert v["dominios_sobrepostos"] == 1
    assert v["contaminacao_dominios_pct"] == 100.0
    assert v["linhas_sem_etld1"] == {"treino": 1, "teste": 1}


def test_medir_vazamento_com_dtype_string_do_pandas():
    """`carregar_corpus_canonizado` entrega `etld_plus_one` como dtype "string".

    Com esse dtype, `serie != ""` propaga pd.NA e um `&` ingênuo com array
    booleano quebraria. O caminho tem de funcionar nos dois dtypes.
    """
    frame = pd.DataFrame(
        {
            "label": pd.array([0, 1], dtype="int8"),
            "url": pd.array(["http://a.com/1", "http://a.com/2"], dtype="string"),
            "corpus": pd.array(["c", "c"], dtype="string"),
            "split_oficial": pd.array(["train", "test"], dtype="string"),
            "etld_plus_one": pd.array(["a.com", pd.NA], dtype="string"),
        }
    )
    v = diagnostics.medir_vazamento_split_oficial(frame)
    assert v["dominios_treino"] == 1
    assert v["dominios_teste"] == 0
    assert v["linhas_sem_etld1"]["teste"] == 1


def _frame_urls(treino: list, teste: list) -> pd.DataFrame:
    n = len(treino) + len(teste)
    return pd.DataFrame(
        {
            "label": [1] * n,
            "url": pd.array(treino + teste, dtype="string"),
            "corpus": ["c"] * n,
            "split_oficial": ["train"] * len(treino) + ["test"] * len(teste),
            "etld_plus_one": ["a.com"] * n,
        }
    )


def test_urls_identicas_conta_distintas_em_string_exata():
    """LIT-01 e LIT-02: duplicata interna conta uma vez; caixa diferente não casa."""
    frame = _frame_urls(
        ["http://a.com/1", "http://a.com/1", "http://b.com/2", "http://c.com/x"],
        ["http://a.com/1", "http://b.com/2", "http://C.com/x"],
    )
    v = diagnostics.medir_vazamento_split_oficial(frame)
    assert v["urls_identicas_treino_teste"] == 2


def test_urls_identicas_sem_intersecao_da_zero_e_o_campo_existe():
    frame = _frame_urls(["http://a.com/1"], ["http://b.com/2"])
    v = diagnostics.medir_vazamento_split_oficial(frame)
    assert "urls_identicas_treino_teste" in v
    assert v["urls_identicas_treino_teste"] == 0


def test_urls_identicas_ignora_url_nula_nos_dois_lados():
    frame = _frame_urls([pd.NA, "http://a.com/1"], [pd.NA, "http://b.com/2"])
    v = diagnostics.medir_vazamento_split_oficial(frame)
    assert v["urls_identicas_treino_teste"] == 0


def test_urls_identicas_nao_normaliza_barra_final():
    frame = _frame_urls(["http://a.com/p/"], ["http://a.com/p"])
    v = diagnostics.medir_vazamento_split_oficial(frame)
    assert v["urls_identicas_treino_teste"] == 0


def test_medir_vazamento_falha_alto_sem_canonizacao(corpus):
    with pytest.raises(DiagnosticsError, match="etld_plus_one"):
        diagnostics.medir_vazamento_split_oficial(corpus.drop(columns=["etld_plus_one"]))


# ---------------------------------------------------------------------------
# Matriz de sobreposição ENTRE corpora — o DoD estendido da T2.3
# ---------------------------------------------------------------------------


@pytest.fixture
def frames_matriz():
    """Quatro corpora sintéticos: 'a' e 'b' se sobrepõem em parte; 'c' e 'd' não.

    A sobreposição entre 'a' e 'b' é **parcial** (domínios 008–039 dos 040 de
    cada um) e não total, de propósito: com conjuntos de domínio idênticos o
    particionador — determinístico e com a mesma seed — escolheria exatamente
    as mesmas partições nos dois corpora, e o teste de 'b' seria o complemento
    do treino de 'a'. A sobreposição mediria zero, sem que nada estivesse
    errado.

    A faixa compartilhada foi ampliada de 020–039 para 008–039 em 05/08/2026,
    ao trocar o `GroupShuffleSplit` pelo `StratifiedGroupKFold` (RF4 revisto).
    Com 20 domínios compartilhados e 8 no teste de cada corpus, a direção
    `b -> a` passou a medir 0,0: os domínios que 'a' põe no teste caíam todos
    na validação de 'b'. Não era defeito da medição — `a -> b` continuava
    detectando —, era a mesma armadilha que este docstring já descrevia,
    reaparecendo na outra direção. Ampliar a faixa remove a sensibilidade em
    vez de afrouxar a asserção, que é o que **não** se deve fazer aqui: o
    teste existe para provar que o resíduo da §13.4-7 é detectável.
    """
    return {
        "a": _corpus_sintetico("a", n_dominios=40),
        "b": _corpus_sintetico("b", n_dominios=40, primeiro_dominio=8),
        "c": _corpus_sintetico("c", n_dominios=40, sufixo_dominio="-c"),  # disjunto
        "d": _corpus_sintetico("d", n_dominios=40, sufixo_dominio="-d"),  # disjunto
    }


def _matriz(frames, regime):
    return diagnostics.matriz_sobreposicao_entre_corpora(
        frames, regime=regime, seed=42, test_size=0.2, val_size=0.1, val_size_oficial=0.1
    )


def test_matriz_diagonal_e_zero_no_regime_etld1(frames_matriz):
    """O ponto do Regime B: dentro de um corpus, treino e teste não se tocam."""
    matriz = _matriz(frames_matriz, "etld1")
    for nome in frames_matriz:
        assert matriz[nome][nome]["shared_domain_pct"] == 0.0
        assert matriz[nome][nome]["diagonal"] is True


def test_matriz_detecta_sobreposicao_fora_da_diagonal(frames_matriz):
    """'a' e 'b' têm os mesmos domínios; 'c' e 'd' são disjuntos dos dois.

    É este resíduo que a §13.4-7 manda medir e declarar, nunca filtrar.
    """
    matriz = _matriz(frames_matriz, "etld1")

    assert matriz["a"]["b"]["shared_domain_pct"] > 0.0
    assert matriz["b"]["a"]["shared_domain_pct"] > 0.0
    assert matriz["a"]["c"]["shared_domain_pct"] == 0.0
    assert matriz["c"]["d"]["shared_domain_pct"] == 0.0
    assert matriz["a"]["b"]["diagonal"] is False


def test_matriz_usa_cross_corpus_domain_overlap_do_splits(frames_matriz, monkeypatch):
    """Chamador de `splits.cross_corpus_domain_overlap()` — literalmente o DoD.

    A função existia sem chamador nenhum (SDD §13.4-7). Este teste falha se
    alguém reimplementar a medição aqui em vez de usar a de `splits.py`, o que
    faria a Tabela de ameaças divergir do split de fato executado.
    """
    import splits

    chamadas = []
    original = splits.cross_corpus_domain_overlap

    def espiao(*args, **kwargs):
        chamadas.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(splits, "cross_corpus_domain_overlap", espiao)
    _matriz(frames_matriz, "etld1")

    assert len(chamadas) == len(frames_matriz) ** 2  # matriz cheia, 4x4 = 16


def test_matriz_regime_oficial_tem_diagonal_nao_nula(frames_matriz):
    """No Regime A a diagonal vaza — é o que a Tabela 3 do artigo mede."""
    matriz = _matriz(frames_matriz, "oficial")
    assert matriz["a"]["a"]["shared_domain_pct"] > 0.0


def test_matriz_rejeita_regime_desconhecido(frames_matriz):
    with pytest.raises(DiagnosticsError, match="regime desconhecido"):
        _matriz(frames_matriz, "inventado")


def test_resumo_da_matriz_separa_diagonal_de_fora_diagonal(frames_matriz):
    matriz = _matriz(frames_matriz, "etld1")
    resumo = diagnostics._resumir_matriz(matriz)

    assert resumo["diagonal_max_pct"] == 0.0
    assert resumo["fora_diagonal_max_pct"] > 0.0
    assert (
        resumo["fora_diagonal_min_pct"]
        <= resumo["fora_diagonal_media_pct"]
        <= resumo["fora_diagonal_max_pct"]
    )


# ---------------------------------------------------------------------------
# T4.10 — contaminação do leave-one-corpus-out (SDD §13.4-17)
# ---------------------------------------------------------------------------


def _loco(frames, regime, indices=None):
    return diagnostics.sobreposicao_loco(
        frames, regime=regime, seed=42, test_size=0.2, val_size=0.1,
        val_size_oficial=0.1, indices=indices,
    )


def test_loco_cobre_todos_os_corpora_e_nenhum_treina_em_si_mesmo(frames_matriz):
    """O corpus retido não pode entrar na própria união de treino.

    Se entrasse, a T6 mediria treino contendo o teste — que é exatamente o
    número inflado de dentro de casa que o artigo critica (D3-b). No pipeline
    isso é conferido por `matrix.check_loco_records`; aqui, na medição.
    """
    por_retido = _loco(frames_matriz, "etld1")

    assert set(por_retido) == set(frames_matriz)
    for retido, celula in por_retido.items():
        assert retido not in celula["train_corpora"]
        assert len(celula["train_corpora"]) == len(frames_matriz) - 1
        assert celula["n_train_corpora"] == len(frames_matriz) - 1


@pytest.fixture
def frames_loco_entrelacado():
    """Quatro corpora em que a UNIÃO mede mais que qualquer par isolado.

    A `frames_matriz` não serve para isto, e a razão é estrutural: lá só 'a' e
    'b' se sobrepõem, então a união sem 'a' contamina exatamente o tanto que
    'b' já contaminava sozinho. União = maior par, e o teste passaria sem
    exercitar nada — a mesma armadilha que a §13.5 registra ao ampliar a faixa
    daquela fixture em 05/08/2026.

    Aqui 'd' tem 40 domínios, 'a' cobre os **pares** e 'b' os **ímpares**: cada
    um alcança metade do teste de 'd' e nenhum alcança a outra metade. Medido
    com seed 42: união = 5 domínios contra 3 do maior par isolado. 'c' é
    disjunto, para que exista um caso de contaminação nula na mesma tabela.
    """
    def entrelacado(nome, dominios):
        quadro = pd.concat(
            [_corpus_sintetico(nome, n_dominios=1, primeiro_dominio=d) for d in dominios],
            ignore_index=True,
        )
        # Fixture ajustada na frente A (T19): `label = d % 2` deixava 'a' (só
        # pares) com uma única classe e 'b' (só ímpares) com a outra, e o
        # particionamento com deduplicação (DEDUP, caso de borda da spec) recusa
        # partição sem uma das classes. Os rótulos passam a alternar de domínio
        # em domínio; os domínios de cada corpus não mudam, e é deles que
        # depende a asserção. As contagens "5 contra 3" do docstring são da
        # fixture anterior; a asserção continua sendo união maior que o maior par.
        quadro["label"] = (quadro.index // 2) % 2
        return quadro

    return {
        "a": entrelacado("a", range(0, 40, 2)),
        "b": entrelacado("b", range(1, 40, 2)),
        "c": _corpus_sintetico("c", n_dominios=40, sufixo_dominio="-c"),
        "d": _corpus_sintetico("d", n_dominios=40),
    }


def test_loco_nunca_fica_abaixo_do_maior_par_a_par(frames_matriz):
    """Limite inferior garantido: a união contém cada treino isolado.

    É um dos dois limites derivados que a §13.4-17 carregava na falta da
    medição direta. Falha se a união deixar de somar todos os corpora — por
    exemplo, se alguém passar só o primeiro da lista.
    """
    matriz = _matriz(frames_matriz, "etld1")
    por_retido = _loco(frames_matriz, "etld1")

    for retido, celula in por_retido.items():
        maior_par = max(
            matriz[treino][retido]["shared_domains"]
            for treino in celula["train_corpora"]
        )
        assert celula["shared_domains"] >= maior_par, (
            f"união do LOCO sem '{retido}' mediu menos que o maior par a par"
        )


def test_loco_supera_o_maior_par_quando_os_treinos_alcancam_partes_distintas(
    frames_loco_entrelacado
):
    """A razão de ser da T4.10: a união mede o que nenhum par isolado mede.

    Sem este teste, `sobreposicao_loco` poderia devolver o máximo par a par e
    nada acusaria — e o SDD continuaria com o limite derivado no lugar do
    número, que é exatamente a pendência §13.4-17.
    """
    matriz = _matriz(frames_loco_entrelacado, "etld1")
    por_retido = _loco(frames_loco_entrelacado, "etld1")

    celula = por_retido["d"]
    maior_par = max(
        matriz[treino]["d"]["shared_domains"] for treino in celula["train_corpora"]
    )
    assert celula["shared_domains"] > maior_par, (
        f"união sem 'd' mediu {celula['shared_domains']}, o maior par mediu "
        f"{maior_par} — a fixture deixou de exercitar o efeito da união"
    )
    # E continua abaixo da soma das parcelas: os conjuntos se intersectam, e é
    # por isso que somar par a par superestimava (§13.4-17).
    soma_das_parcelas = sum(
        matriz[treino]["d"]["shared_domains"] for treino in celula["train_corpora"]
    )
    assert celula["shared_domains"] <= soma_das_parcelas


def test_loco_usa_union_domain_overlap_do_splits(frames_matriz, monkeypatch):
    """Chamador de `splits.union_domain_overlap()` — o DoD da T4.10.

    Mesma doutrina do teste equivalente da matriz: se alguém reimplementar a
    contagem aqui, a T6 e a T2 passam a contar 'domínio' de formas diferentes.
    """
    import splits

    chamadas = []
    original = splits.union_domain_overlap

    def espiao(*args, **kwargs):
        chamadas.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(splits, "union_domain_overlap", espiao)
    _loco(frames_matriz, "etld1")

    assert len(chamadas) == len(frames_matriz)


def test_loco_e_matriz_saem_das_mesmas_particoes(frames_matriz):
    """As duas medições têm de compartilhar partição, senão não se comparam.

    A §13.4-17 pede a união do LOCO **ao lado** da média da matriz. Se cada
    uma sorteasse a sua partição, os dois números seriam de experimentos
    diferentes e a comparação não significaria nada.
    """
    indices = diagnostics._particionar_para_diagnostico(
        frames_matriz, "etld1", 42, 0.2, 0.1, 0.1
    )
    compartilhado = _loco(frames_matriz, "etld1", indices=indices)
    proprio = _loco(frames_matriz, "etld1")

    for retido in frames_matriz:
        assert compartilhado[retido] == proprio[retido]
        # E o teste medido é o mesmo que a matriz usa naquele corpus.
        assert compartilhado[retido]["test_rows"] == len(indices[retido].test)


def test_resumo_do_loco_aponta_o_pior_corpus_retido(frames_matriz):
    por_retido = _loco(frames_matriz, "etld1")
    resumo = diagnostics._resumir_loco(por_retido)

    assert resumo["min_pct"] <= resumo["media_pct"] <= resumo["max_pct"]
    pior = resumo["corpus_retido_mais_contaminado"]
    assert por_retido[pior]["shared_domain_pct"] == resumo["max_pct"]


def test_loco_rejeita_corpus_unico():
    with pytest.raises(DiagnosticsError, match="ao menos 2 corpora"):
        _loco({"a": _corpus_sintetico("a", n_dominios=40)}, "etld1")


# ---------------------------------------------------------------------------
# Ponta a ponta, sobre corpora sintéticos em disco
# ---------------------------------------------------------------------------


@pytest.fixture
def corpora_dir_falso(tmp_path):
    """Seis parquet minúsculos com os nomes reais dos corpora.

    Permite exercitar `executar()` inteiro — inclusive a canonização de
    verdade e a matriz 4×4 — sem depender dos ~96 MB de parquet real e sem
    gastar os ~6 min da execução completa.
    """
    for nome in diagnostics.TODOS_OS_CORPORA:
        frame = _corpus_sintetico(nome, n_dominios=24, sufixo_dominio=f"-{nome[:4]}")
        destino = tmp_path / nome
        destino.mkdir(parents=True)
        frame[["label", "url", "corpus", "split_oficial"]].to_parquet(
            destino / "data.parquet", index=False, engine="pyarrow"
        )
    return tmp_path


def test_carregar_corpus_canonizado_anexa_etld1(corpora_dir_falso):
    frame = diagnostics.carregar_corpus_canonizado("iscx2016", corpora_dir=corpora_dir_falso)

    # Expectativa trocada na T19 da frente A (DEDUP-05, design componente 12): a
    # carga passa a anexar `url_norm` (chave da deduplicação de
    # `splits.particionar_corpus`) e `host` (entrada do REEX-13).
    assert list(frame.columns) == [
        "label", "url", "corpus", "split_oficial", "etld_plus_one", "url_norm", "host",
    ]
    assert frame["etld_plus_one"].notna().all()
    assert (frame["etld_plus_one"] != "").all()
    # As 41 features NÃO são extraídas: nada aqui depende delas.
    assert "url_length" not in frame.columns


def test_carregar_corpus_falha_alto_com_corpus_ausente(tmp_path):
    with pytest.raises(DiagnosticsError, match="não encontrado"):
        diagnostics.carregar_corpus_canonizado("iscx2016", corpora_dir=tmp_path)


def test_executar_produz_o_contrato_completo(corpora_dir_falso):
    d = diagnostics.executar(corpora_dir=corpora_dir_falso)

    assert d["parcial"] is False
    # Expectativa ampliada na T19 da frente A (PSL-04, DEDUP-05, ANL-05): as
    # cinco chaves finais são os diagnósticos novos; as demais não mudam.
    assert set(d) == {
        "run_id", "gerado_em", "parcial", "config_fingerprint", "parametros",
        "definicoes", "vies_construcao", "vazamento_split_oficial",
        "sobreposicao_entre_corpora", "sobreposicao_loco",
        "psl_privada_contra_icann", "dominios_duas_classes", "desvio_prevalencia",
        "esquema_por_classe", "medias_por_classe",
    }
    assert set(d["vies_construcao"]) == set(diagnostics.TODOS_OS_CORPORA)
    assert set(d["vazamento_split_oficial"]) == set(diagnostics.TODOS_OS_CORPORA)
    # `config_fingerprint` entrou em 05/08/2026 (§13.4-18): a
    # `sobreposicao_entre_corpora` é medida SOBRE as partições do Regime B, então
    # este arquivo envelhece junto com o particionador — e é a fonte das T1/T5.
    # Sem o carimbo, um diagnóstico de outra rodada conviveria com o
    # `matrix.json` da atual sem nada acusar. Pode ser `None` se as bibliotecas
    # de treino não estiverem instaláveis, mas a chave tem de existir.
    assert "config_fingerprint" in d
    # Os dois regimes de SDD §4 D3, cada um com matriz 4×4 e resumo.
    assert set(d["sobreposicao_entre_corpora"]) == {"oficial", "etld1"}
    for bloco in d["sobreposicao_entre_corpora"].values():
        assert len(bloco["matriz"]) == 4
        assert all(len(linha) == 4 for linha in bloco["matriz"].values())
        assert "resumo" in bloco
    # T4.10: a contaminação da união, nos mesmos dois regimes. Um corpus
    # retido por regime, cada um treinando nos outros três.
    assert set(d["sobreposicao_loco"]) == {"oficial", "etld1"}
    for bloco in d["sobreposicao_loco"].values():
        assert len(bloco["por_corpus_retido"]) == 4
        assert "resumo" in bloco
        for retido, celula in bloco["por_corpus_retido"].items():
            assert retido not in celula["train_corpora"]
            assert celula["n_train_corpora"] == 3


def test_executar_ordena_as_tabelas_como_o_artigo(corpora_dir_falso):
    """Viés crescente (limpos primeiro) e vazamento decrescente (pior primeiro)."""
    d = diagnostics.executar(corpora_dir=corpora_dir_falso)

    gaps = [v["gap_max_p"] for v in d["vies_construcao"].values()]
    assert gaps == sorted(gaps)

    contaminacoes = [
        v["contaminacao_dominios_pct"] for v in d["vazamento_split_oficial"].values()
    ]
    assert contaminacoes == sorted(contaminacoes, reverse=True)


def test_executar_marca_parcial_com_subconjunto_de_corpora(corpora_dir_falso):
    """Rodar com `--corpora` reduzido é legítimo, mas o arquivo sai marcado.

    Sem a marca, um JSON incompleto alimentaria a T5.3 e a T5.6 sem ninguém
    notar — que é a forma exata de violar a R5.
    """
    d = diagnostics.executar(corpora_dir=corpora_dir_falso, corpora=("iscx2016",))

    assert d["parcial"] is True
    assert d["sobreposicao_entre_corpora"] == {}


def test_gravar_produz_json_legivel(corpora_dir_falso, tmp_path):
    d = diagnostics.executar(corpora_dir=corpora_dir_falso, corpora=("iscx2016",))
    destino = diagnostics.gravar(d, out_path=tmp_path / "sub" / "diagnostics.json")

    assert destino.exists()
    with destino.open(encoding="utf-8") as f:
        relido = json.load(f)
    assert relido["vies_construcao"]["iscx2016"] == d["vies_construcao"]["iscx2016"]


# ---------------------------------------------------------------------------
# T19: diagnósticos da frente A (PSL privada, domínios com as duas classes,
# desvio de prevalência, esquema e médias por classe), todos sem treino
# ---------------------------------------------------------------------------


def test_psl_privada_conta_urls_grupos_icann_e_grupos_resultantes():
    """REEX-13 (PSL-04): `a.github.io` e `b.github.io` eram um grupo ICANN só."""
    r = diagnostics.psl_privada_contra_icann(["a.github.io", "b.github.io", "x.com"])

    assert r == {
        "n_urls": 3,
        "n_urls_mudam": 2,
        "n_grupos_icann_afetados": 1,   # github.io
        "n_grupos_resultantes": 2,      # a.github.io e b.github.io
    }


def test_psl_privada_conta_cada_url_e_ignora_host_vazio_e_ip():
    hosts = ["a.github.io"] * 3 + ["a.example.com", "", "192.168.0.1", None]
    r = diagnostics.psl_privada_contra_icann(hosts)

    assert r["n_urls"] == 7
    assert r["n_urls_mudam"] == 3
    assert r["n_grupos_icann_afetados"] == 1
    assert r["n_grupos_resultantes"] == 1


def test_dominios_com_as_duas_classes_batem_com_a_conta_a_mao():
    frame = pd.DataFrame(
        {
            #            d1 d1 d2 d2 d3 d3 d3 (sem)
            "etld_plus_one": ["d1.com"] * 2 + ["d2.com"] * 2 + ["d3.com"] * 3 + [""],
            "label": [0, 1, 0, 0, 1, 1, 0, 1],
        }
    )
    r = diagnostics.dominios_duas_classes(frame)

    # d1 (0,1) e d3 (1,1,0) têm as duas classes; d2 só uma; a linha sem eTLD+1
    # fica de fora.
    assert r == {"n_dominios": 3, "n_dominios_duas_classes": 2, "n_linhas_duas_classes": 5}


def test_desvio_de_prevalencia_por_particao_bate_com_a_conta_a_mao():
    # A linha 10 (phishing) não é elegível: não entra na referência.
    frame = pd.DataFrame({"label": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]})
    split = splits.SplitIndices(
        corpus="ds", regime="etld1",
        train=np.arange(0, 6), val=np.arange(6, 8), test=np.arange(8, 10),
    )
    r = diagnostics.desvio_prevalencia(frame, split)

    # Dataset deduplicado: 3 phishing em 10 linhas = 0,3.
    assert r["n_dataset"] == 10
    assert r["prevalencia_dataset"] == pytest.approx(0.3)
    assert r["train"]["prevalencia"] == pytest.approx(2 / 6)
    assert r["train"]["desvio"] == pytest.approx(2 / 6 - 0.3)
    assert r["val"]["prevalencia"] == pytest.approx(0.5)
    assert r["val"]["desvio"] == pytest.approx(0.2)
    assert r["test"]["prevalencia"] == pytest.approx(0.0)
    assert r["test"]["desvio"] == pytest.approx(-0.3)
    assert [r[p]["n"] for p in ("train", "val", "test")] == [6, 2, 2]


def test_esquema_por_classe_separa_http_https_e_nenhum():
    frame = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "url": [
                "http://a.com", "https://b.com", "c.com", "https://d.com",
                "http://x.com", "ftp://y.com", "z.com/a", "httpsecure.com",
            ],
        }
    )
    r = diagnostics.esquema_por_classe(frame)

    assert r["benigno"] == {"n": 4, "explicito": 0.75, "http": 0.25, "https": 0.5, "nenhum": 0.25}
    # ftp conta como esquema explícito; `httpsecure.com` não (B14).
    assert r["phishing"] == {"n": 4, "explicito": 0.5, "http": 0.25, "https": 0.0, "nenhum": 0.5}


def test_medias_por_classe_cobrem_as_41_caracteristicas_e_batem_com_a_conta_a_mao():
    import features

    frame = pd.DataFrame(
        {
            "label": [0, 0, 1, 1, 1],
            "url": [
                "http://a.com", "http://abcd.com",
                "https://x.com/login", "http://y.com/a/b", "z.com",
            ],
        }
    )
    r = diagnostics.medias_por_classe(frame)

    assert set(r) == {"benigno", "phishing"}
    assert r["benigno"]["n"] == 2 and r["phishing"]["n"] == 3
    assert list(r["benigno"]["medias"]) == list(features.FEATURE_NAMES)
    assert len(r["benigno"]["medias"]) == 41
    # À mão: comprimentos 12 e 15 (benigno); 19, 16 e 5 (phishing).
    assert r["benigno"]["medias"]["url_length"] == pytest.approx(13.5)
    assert r["phishing"]["medias"]["url_length"] == pytest.approx((19 + 16 + 5) / 3)

    # Referência independente: média do quadro de características inteiro.
    X = features.extract_features(frame["url"].tolist()).astype(float)
    for nome, rotulo in (("benigno", 0), ("phishing", 1)):
        esperado = X[frame["label"].to_numpy() == rotulo].mean()
        for f in features.FEATURE_NAMES:
            assert r[nome]["medias"][f] == pytest.approx(esperado[f])


def test_medias_por_classe_nao_dependem_do_tamanho_do_bloco():
    frame = pd.DataFrame(
        {"label": [0, 1, 0, 1, 1], "url": ["a.com", "http://b.com/x", "c.org/y", "d.net", "e.io"]}
    )
    grande = diagnostics.medias_por_classe(frame, chunk=1000)
    pequeno = diagnostics.medias_por_classe(frame, chunk=2)
    for classe in ("benigno", "phishing"):
        assert grande[classe]["n"] == pequeno[classe]["n"]
        assert grande[classe]["medias"] == pytest.approx(pequeno[classe]["medias"])


@pytest.mark.parametrize(
    "funcao",
    [
        diagnostics.psl_privada_contra_icann,
        diagnostics.dominios_duas_classes,
        diagnostics.desvio_prevalencia,
        diagnostics.esquema_por_classe,
        diagnostics.medias_por_classe,
    ],
)
def test_diagnosticos_novos_nao_usam_models_nem_fit(funcao):
    """Nada de treino nos diagnósticos novos (T19, REEX-13 e ANL-05 sem treinar)."""
    nomes = set(funcao.__code__.co_names)
    for const in funcao.__code__.co_consts:
        if hasattr(const, "co_names"):
            nomes |= set(const.co_names)
    assert "models" not in nomes
    assert not {"fit", "fit_transform", "partial_fit", "ajustar"} & nomes


def test_particoes_do_diagnostico_sao_as_da_fachada_com_deduplicacao():
    """O diagnóstico e a matriz veem as mesmas partições (design componente 12)."""
    frame = _corpus_sintetico("a", n_dominios=40)
    # Repete as quatro primeiras linhas: duplicatas pelo `url_norm`.
    frame = pd.concat([frame, frame.iloc[:4]], ignore_index=True)

    obtido = diagnostics._particionar_para_diagnostico(
        {"a": frame}, "etld1", seed=42, test_size=0.2, val_size=0.1, val_size_oficial=0.1
    )["a"]
    esperado, contagens = splits.particionar_corpus(
        frame, "etld1", corpus="a", seed=42, test_size=0.2, val_size=0.1
    )

    assert contagens["n_duplicatas_removidas"] == 4
    assert np.array_equal(obtido.train, esperado.train)
    assert np.array_equal(obtido.val, esperado.val)
    assert np.array_equal(obtido.test, esperado.test)
    assert obtido.n_total == len(frame) - 4


def test_executar_grava_os_diagnosticos_novos_por_dataset(corpora_dir_falso):
    d = diagnostics.executar(corpora_dir=corpora_dir_falso)

    todos = set(diagnostics.TODOS_OS_CORPORA)
    matriz = {"grambeddings", "kaggle03", "phishstorm", "iscx2016"}
    assert set(d["psl_privada_contra_icann"]) == todos
    assert set(d["dominios_duas_classes"]) == todos
    assert set(d["esquema_por_classe"]) == todos
    assert set(d["medias_por_classe"]) == matriz
    assert set(d["desvio_prevalencia"]) == {"oficial", "etld1"}
    for por_corpus in d["desvio_prevalencia"].values():
        assert set(por_corpus) == matriz
        assert set(por_corpus["iscx2016"]) == {"n_dataset", "prevalencia_dataset",
                                               "train", "val", "test"}
    assert set(d["dominios_duas_classes"]["iscx2016"]) == {"bruto", "deduplicado_etld1"}
    assert set(d["psl_privada_contra_icann"]["iscx2016"]) == {
        "n_urls", "n_urls_mudam", "n_grupos_icann_afetados", "n_grupos_resultantes",
    }


def test_executar_sem_caracteristicas_pula_so_as_medias(corpora_dir_falso):
    d = diagnostics.executar(corpora_dir=corpora_dir_falso, caracteristicas=False)

    assert d["medias_por_classe"] == {}
    assert set(d["esquema_por_classe"]) == set(diagnostics.TODOS_OS_CORPORA)


# ---------------------------------------------------------------------------
# Regressão sobre corpus real e sobre o artefato entregue
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iscx_real():
    """Corpus real menor (19.965 URLs, ~3 s). Pula se o parquet não estiver lá."""
    if not (REPO_ROOT / "corpora" / "iscx2016" / "data.parquet").exists():
        pytest.skip("corpora/iscx2016/data.parquet ausente")
    return diagnostics.carregar_corpus_canonizado("iscx2016")


def test_regressao_iscx2016_vies(iscx_real):
    """Ancora a Tabela 5 no ISCXURL2016 real.

    Valores medidos em 04/08/2026 com a canonização e as definições deste
    módulo. Se mudarem, uma tabela do artigo mudou — atualize o SDD §3.2 e
    registre em §11 antes de mexer aqui.
    """
    v = diagnostics.medir_vies(iscx_real)

    assert v["n_treino"] == 15972
    assert v["n_total"] == 19965
    assert v["scheme"]["gap_p"] == pytest.approx(0.20, abs=0.01)
    assert v["path"]["gap_p"] == pytest.approx(3.38, abs=0.01)
    assert v["veredito"] == "limpo"


def test_regressao_iscx2016_vazamento(iscx_real):
    """Ancora a Tabela 1: o achado central do artigo (SDD §3.3).

    Expectativa trocada na T4 da frente A (PSL-02, PSL-03: eTLD+1 pela PSL
    versionada, com a seção de domínios privados). Os valores antigos (2877,
    1275, 1001, 78,51% e 92,31%) são os da PSL só com a seção ICANN, e foram
    reproduzidos com `include_psl_private_domains=False`; os atuais são os da
    PSL com a seção privada.
    """
    v = diagnostics.medir_vazamento_split_oficial(iscx_real)

    assert v["dominios_treino"] == 2972
    assert v["dominios_teste"] == 1299
    assert v["dominios_sobrepostos"] == 1016
    assert v["contaminacao_dominios_pct"] == pytest.approx(78.21, abs=0.01)
    assert v["contaminacao_linhas_pct"] == pytest.approx(92.06, abs=0.01)


def test_artefato_entregue_esta_completo_e_coerente():
    """`results/diagnostics.json` é versionado e é o que a T5.3/T5.6 leem (R5).

    Este teste não recalcula nada — ele guarda o artefato contra ficar
    parcial, vazio ou incoerente no repositório.
    """
    caminho = REPO_ROOT / "results" / "diagnostics.json"
    if not caminho.exists():
        pytest.skip("results/diagnostics.json ausente — rode `python -m src.diagnostics`")

    with caminho.open(encoding="utf-8") as f:
        d = json.load(f)

    assert d["parcial"] is False, "diagnóstico parcial não pode alimentar o artigo"
    assert set(d["vies_construcao"]) == set(diagnostics.TODOS_OS_CORPORA)
    assert set(d["sobreposicao_entre_corpora"]) == {"oficial", "etld1"}

    etld1 = d["sobreposicao_entre_corpora"]["etld1"]
    assert etld1["resumo"]["diagonal_max_pct"] == 0.0
    assert etld1["resumo"]["fora_diagonal_max_pct"] > 0.0

    # T4.10 (§13.4-17): a contaminação da união, que a legenda da T6 declara.
    # Sem esta chave o artigo voltaria a ter só os limites derivados.
    assert set(d["sobreposicao_loco"]) == {"oficial", "etld1"}
    loco = d["sobreposicao_loco"]["etld1"]
    assert set(loco["por_corpus_retido"]) == set(d["parametros"]["corpora_matriz"])
    for retido, celula in loco["por_corpus_retido"].items():
        assert retido not in celula["train_corpora"], (
            f"'{retido}' entrou no próprio treino da união — seria vazamento "
            f"do corpus retido, o único defeito que invalida a T6"
        )
        assert celula["n_train_corpora"] == 3

    # Os quatro corpora de treino/avaliação continuam limpos e os dois de
    # demonstração de viés continuam contaminados (SDD §4 D2, R6).
    for nome in ("grambeddings", "kaggle03", "phishstorm", "iscx2016"):
        assert d["vies_construcao"][nome]["veredito"] == "limpo"
    for nome in ("pdrcnn", "sid321axn"):
        assert d["vies_construcao"][nome]["veredito"] == "contaminado"
