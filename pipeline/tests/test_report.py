"""T5.2 — o `report.py` é a ponte entre `results/` e o PDF, e o que ele erra
chega impresso no artigo.

Três famílias de teste, por ordem de importância:

1. **A convenção de agregação está travada.** A célula da matriz é a média dos
   6 modelos, e isso não foi escolhido: é o que reproduz os números que a
   §13.8 publicou. Um teste guarda esse valor, para que ninguém troque a
   agregação por mediana ou por "melhor modelo" e o artigo passe a dizer outra
   coisa sem ninguém notar.
2. **Formatação numérica.** Dois defeitos reais foram cometidos e corrigidos
   aqui, e os dois só apareceram no PDF renderizado, não na saída do módulo.
3. **O `.tex` compila.** Um comando não definido derruba a compilação inteira,
   e o `artigo.tex` não carrega `amsmath`.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# O artigo.tex saiu de `paper/` e virou insumo em `docs/artigo/` do TCC II;
# as tabelas geradas vivem em `latex/tabelas/`.
ARTIGO_TEX = REPO_ROOT.parent / "docs" / "artigo" / "artigo.tex"
TABELAS_DIR = REPO_ROOT.parent / "latex" / "tabelas"
sys.path.insert(0, str(REPO_ROOT / "src"))

import report  # noqa: E402


@pytest.fixture(scope="module")
def dados():
    if not (REPO_ROOT / "results" / "matrix.json").exists():
        pytest.skip("results/matrix.json ausente — rode scripts/run_all.py")
    return report.carregar()


# ---------------------------------------------------------------------------
# 1. A convenção de agregação
# ---------------------------------------------------------------------------


def test_media_por_celula_reproduz_os_numeros_publicados(dados):
    """Canária da convenção: média dos 6 modelos, seed 42, regime eTLD+1.

    A §13.8 publicou "diagonal 0,9741 -> fora-diagonal 0,6856" **antes** deste
    módulo existir. Foi reproduzindo esses dois valores que a convenção de
    agregação foi identificada, em vez de escolhida (§11, 06/08/2026). Se
    alguém trocar a média por mediana, por "melhor modelo" ou por média
    ponderada, este teste cai — e tem de cair, porque o artigo passaria a
    afirmar outro número com a mesma frase.

    O mesmo exercício expôs que o par de MCC da §13.8 (0,864 / 0,304) **não**
    reproduzia sob nenhuma convenção; ele foi corrigido para 0,863 / 0,303.
    """
    celulas = report.media_por_celula(
        dados["matriz_por_seed"][report.SEED_CANONICA], "etld1", "auroc"
    )
    diagonal = [v for (tr, te), v in celulas.items() if tr == te]
    fora = [v for (tr, te), v in celulas.items() if tr != te]

    # Expectativa da rodada consolidada (AD-055, que substitui o AD-010).
    # Valores da rodada antiga do artigo: 0,9741 e 0,6856.
    assert sum(diagonal) / len(diagonal) == pytest.approx(0.9790, abs=5e-5)
    assert sum(fora) / len(fora) == pytest.approx(0.6768, abs=5e-5)


def test_metrica_indefinida_e_excluida_da_media_e_nao_vira_zero():
    """`None` fora da média, nunca 0 — a doutrina do `evaluate.py`.

    Zero num AUROC indefinido cria um número falso que atravessa a tabela até
    o PDF. Foi comparar valor inválido que produziu os bugs B2 e B7.
    """
    registros = [
        {"regime": "etld1", "train_corpus": "a", "test_corpus": "b",
         "metrics": {"auroc": 0.8}},
        {"regime": "etld1", "train_corpus": "a", "test_corpus": "b",
         "metrics": {"auroc": None}},
    ]
    celulas = report.media_por_celula(registros, "etld1", "auroc")
    assert celulas[("a", "b")] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 2. Formatação — os dois defeitos que o PDF revelou
# ---------------------------------------------------------------------------


def test_milhar_nao_corrompe_a_virgula_decimal():
    """Regressão: `.replace(',', '.')` na linha inteira estragava o decimal.

    A primeira versão formatava o separador de milhar substituindo vírgula por
    ponto na **linha** da tabela, o que atingia também a vírgula decimal que o
    `_num()` acabara de produzir: `78,51` saía `78.51` no PDF.
    """
    assert report._milhar(113848) == "113.848"
    assert report._num(78.51, 2) == "78,51"
    linha = f"{report._milhar(2877)} & {report._num(78.51, 2)}"
    assert linha == "2.877 & 78,51"


def test_ponto_sempre_dentro_da_propria_faixa():
    """Regressão: ponto em 4 casas e faixa em 3 punha o ponto fora do intervalo.

    Medido no primeiro PDF: `0,0028 [0,003--0,004]` e `-0,0071
    [-0,007--0,006]`. Com o mesmo número de casas nos dois, não acontece.
    """
    valores = [0.0028, 0.0031, 0.0044]
    ponto = report._num(valores[0], 4)
    faixa = report._faixa(valores, 4, ";~")
    lo = faixa.split(";~")[0].lstrip("[")
    assert lo <= ponto or float(lo.replace(",", ".")) <= valores[0]
    assert ponto == "0,0028"
    assert faixa == "[0,0028;~0,0044]"


def test_faixa_com_negativo_nao_usa_travessao():
    """`[-0,011--0,010]` é ilegível: o travessão colide com o menos."""
    assert report._faixa([-0.011, -0.010], 4, ";~") == "[-0,0110;~-0,0100]"
    # Coluna sempre positiva mantém o travessão, que é a convenção de intervalo.
    assert report._faixa([0.9, 0.95], 3) == "[0,900--0,950]"


def test_faixa_vazia_com_uma_seed_so():
    """Sem as seeds extras a tabela sai sem faixa, e não quebra."""
    assert report._faixa([0.9], 4) == ""
    assert report._faixa([], 4) == ""


# ---------------------------------------------------------------------------
# 3. O `.tex` gerado compila
# ---------------------------------------------------------------------------


def test_gera_exatamente_os_nomes_que_o_artigo_espera(dados, tmp_path):
    """Nome de arquivo é contrato com o `artigo.tex`.

    O `\\input` é protegido por `\\IfFileExists`: um nome errado não quebra a
    compilação, ele troca a tabela por um aviso — falha silenciosa exatamente
    onde ela custa mais.
    """
    escritos = report.gerar(REPO_ROOT / "results", tmp_path)
    nomes = {p.name for p in escritos}
    # A `tabela2-matriz-etld1.tex` saiu deste conjunto na T5.9 (decisão D-C1):
    # a T2 e o mapa de calor publicavam a mesma matriz 4x4 e juntas custavam
    # ~1,05 página. A função `tabela2_matriz` continua no módulo, de propósito
    # (o roteiro de revisão: prioridade ALTA de retorno na câmera-ready
    # de 13/09, porque é a única publicação da AUC-PR). Se ela voltar, esta
    # lista volta com ela — é este teste que trava o contrato de nomes.
    assert nomes == {
        "tabela1-vazamento.tex",
        "tabela5-vies.tex",
        "tabela3-diagonal.tex",
        "tabela4-ablacao.tex",
        "tabela6-loco.tex",
        # RUN-01 (frente A, T23): precisão, revocação e TFP por célula.
        "tabela10-limiar.tex",
        "tabela7-importancias.tex",
        # E4 (EST-01): Friedman e Nemenyi, tabela e diagrama de diferença crítica.
        "tabela8-friedman.tex",
        "figura3-diferenca-critica.tex",
        # Cap. 6 (T5): calibração por modelo.
        "tabela9-calibracao.tex",
        # Spec 8, Apêndice B: configuração fixa e melhor configuração do HPO.
        "apendiceB1-configuracao.tex",
        "apendiceB2-hpo.tex",
        # Spec 8, Apêndice C: matriz (regime x métrica) e LOCO (regime).
        *{
            f"apendiceC-matriz-{r}-{m}.tex"
            for r in ("a", "b")
            for m in ("auroc", "aucpr", "f1", "mcc")
        },
        "apendiceC-loco-a.tex",
        "apendiceC-loco-b.tex",
        # Spec 8, Apêndice D: execuções e versões das bibliotecas.
        "apendiceD1-execucao.tex",
        "apendiceD2-versoes.tex",
        "numeros.tex",
        # Tabelas condicionais: só saem quando a rodada correspondente existe em
        # `results/`. Com a rodada consolidada (AD-055, que substitui o AD-010)
        # todas existem. A ausência delas é coberta, com dados sintéticos, em
        # `test_report_rodadas.py`.
        "tabela11-sensibilidade.tex",  # sensibilidade sem esquema (sementes 42, 7, 13)
        "tabela12-repeticao.tex",  # repetição da semente 42 (RUN-01)
        "tabela13-hpo-fora.tex",  # HPO avaliado fora da diagonal
        "tabela14-filtragem.tex",  # filtragem do offline.json
        "tabela15-bootstrap-pareado.tex",  # bootstrap pareado do offline.json
        # Análises da F04 (T30, ANL-01, ANL-02, ANL-04 e ANL-06): saem da
        # matriz, do LOCO, do splits.json e do offline.json.
        "tabela16-decomposicao.tex",
        "tabela17-uniao.tex",
        "tabela18-composicao.tex",
        "tabela19-degeneradas.tex",
        "tabela20-efeito.tex",
        # Tabela condicional (ANL-05): gerada quando o diagnostics.json traz
        # esquema_por_classe e medias_por_classe. O caso sem essas chaves segue
        # coberto por test_report_rodadas.
        "tabela21-auditoria.tex",
    }
    # Só as chamadas de verdade: o cabeçalho do artigo.tex tem comentários que
    # mencionam `paper/tables/*.tex` em prosa, e eles não são pedidos de tabela.
    esperados_pelo_artigo = set()
    for linha in ARTIGO_TEX.read_text(
        encoding="utf-8"
    ).splitlines():
        if linha.lstrip().startswith("%") or "{tables/" not in linha:
            continue
        esperados_pelo_artigo.add(linha.split("{tables/")[1].split("}")[0])
    assert esperados_pelo_artigo <= nomes, (
        f"o artigo.tex pede tabelas que o report.py não gera: "
        f"{esperados_pelo_artigo - nomes}"
    )


def test_nenhuma_tabela_usa_comando_de_pacote_nao_carregado(dados, tmp_path):
    """Regressão: `\\text{}` é do `amsmath`, que o `artigo.tex` não carrega.

    Custou 12 erros de compilação numa legenda. `\\mathrm` é LaTeX puro e faz
    o mesmo trabalho aqui.
    """
    proibidos = ("\\text{", "\\intertext", "\\numprint", "\\SI{", "\\toprule*")
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        conteudo = caminho.read_text(encoding="utf-8")
        for comando in proibidos:
            assert comando not in conteudo, (
                f"{caminho.name} usa '{comando}', que o artigo.tex não carrega"
            )


def test_toda_linha_de_tabela_termina_com_quebra_dupla(dados, tmp_path):
    """Regressão: um `\\` solto no lugar de `\\\\` derruba a tabela inteira.

    Aconteceu ao escrever o módulo por heredoc de shell, que consumiu uma das
    barras. O sintoma no LaTeX é `Misplaced \\noalign` a partir do `\\midrule`
    — mensagem que não aponta para a linha culpada.
    """
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        for n, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), 1):
            s = linha.rstrip()
            if "&" not in s or s.lstrip().startswith("%"):
                continue
            assert s.endswith("\\\\"), (
                f"{caminho.name}:{n} tem célula mas não termina em '\\\\': {s[-40:]!r}"
            )


def test_resizebox_abre_e_fecha_nas_tabelas_largas(dados, tmp_path):
    """Chave desbalanceada em `\\resizebox` engole o resto do documento.

    ⚠️ Este teste **descobre** quais tabelas usam `\\resizebox` em vez de
    nomeá-las, e a razão é a T5.9: ele nomeava a T2 e a T3, e as duas deixaram
    de casar no mesmo dia — a T2 saiu do artigo (D-C1) e a T3 encolheu de 24
    linhas para 4 (D-C3), ficando estreita o bastante para dispensar o
    `\\resizebox`. Um teste que nomeia arquivos quebra por *mudança de escopo*
    e não por defeito, e quem o consertar tende a afrouxá-lo. Varrendo tudo, a
    invariante que importa — chave balanceada onde houver `\\resizebox` — segue
    travada em qualquer tabela, inclusive nas que ainda não existem.
    """
    com_resizebox = []
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        # Linha de comentário fora: um `\resizebox` citado num `%` não é um
        # `\resizebox`. Sem isto o teste acusa a T3, cuja legenda explica por
        # que ela deixou de usar um.
        conteudo = "\n".join(
            linha for linha in caminho.read_text(encoding="utf-8").splitlines()
            if not linha.lstrip().startswith("%")
        )
        n = conteudo.count("\\resizebox")
        assert n <= 1, f"{caminho.name}: {n} \\resizebox na mesma tabela"
        if n:
            com_resizebox.append(caminho.name)
            assert "\\end{tabular}}" in conteudo, (
                f"{caminho.name}: resizebox aberto e não fechado"
            )
    assert com_resizebox, (
        "nenhuma tabela usa \\resizebox — o teste deixou de medir alguma coisa"
    )


def test_cada_tabela_declara_a_origem_dos_numeros(dados, tmp_path):
    """R5: todo número do artigo sai de `results/`, e o `.tex` diz de onde."""
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        cabecalho = caminho.read_text(encoding="utf-8")[:400]
        assert "NAO EDITE A MAO" in cabecalho
        # Quase tudo sai de `results/`. A tabela de versões do Apêndice D sai do
        # `requirements.lock`, que mora na raiz do pipeline, e o cabeçalho o diz.
        assert (
            "% Fonte: results/" in cabecalho
            or (caminho.name == "apendiceD2-versoes.tex"
                and "% Fonte: requirements.lock" in cabecalho)
        ), caminho.name


def test_a_seed_canonica_e_obrigatoria(tmp_path):
    """O resultado do artigo é a seed 42 (§7.4). Sem ela, nada é gerado."""
    vazio = tmp_path / "results_vazio"
    vazio.mkdir()
    with pytest.raises(report.ReportError):
        report.carregar(vazio)


# ---------------------------------------------------------------------------
# 4. O achado que a T3 tornou visível
# ---------------------------------------------------------------------------


def test_a_queda_da_diagonal_nao_e_medivel_em_dois_dos_quatro_corpora(dados):
    """Sustenta a leitura que a T3 por modelo permitiu (§11, 06/08/2026).

    Medido na seed 42: em `kaggle03` e `iscx2016`, |ΔAUROC| fica **dentro** da
    largura do IC por bootstrap nos **6** modelos — a queda não é distinguível
    de ruído de amostragem. Em `grambeddings` e `phishstorm` fica fora nos 6.
    A separação é limpa por corpus, e é mais afiada do que a média de 0,0093
    que o texto citava antes.

    Se este teste cair, a afirmação do artigo mudou e a §13.8 precisa ser
    reescrita — não é para ser "consertado" ajustando o limiar.

    ➕ **A T5.9 mudou o status desta garantia: ela passou a ser PUBLICADA.**
    Até 08/08/2026 o `6/6` e o `0/6` só existiam aqui — o artigo pedia ao
    leitor que os derivasse comparando 24 pares de |Δ| contra 24 larguras de
    IC. A T3 encolhida imprime a contagem numa coluna, e a segunda metade
    deste teste confere que o que está impresso é o que está medido. Sem essa
    segunda metade o corte teria movido a garantia para uma tabela que ninguém
    verifica.

    ➖ **A frente A2 (TAB-02) tirou a contagem da tabela**: o critério passou a
    ser o IC pareado. A medição acima fica, como lastro da leitura; a segunda
    metade confere que a coluna e o trecho do subtítulo realmente saíram.
    """
    registros = dados["matriz_por_seed"][report.SEED_CANONICA]
    por_chave = {
        (r["train_corpus"], r["model"], r["regime"]): r
        for r in registros
        if r["diagonal"]
    }
    dentro_do_ruido = {}
    for corpus in report.ORDEM_CORPORA:
        n = 0
        for modelo in report.ROTULO_MODELO:
            of = por_chave.get((corpus, modelo, "oficial"))
            et = por_chave.get((corpus, modelo, "etld1"))
            if not of or not et or not et.get("ci"):
                continue
            delta = abs(et["metrics"]["auroc"] - of["metrics"]["auroc"])
            largura = et["ci"]["auroc_hi"] - et["ci"]["auroc_lo"]
            n += delta <= largura
        dentro_do_ruido[corpus] = n

    assert dentro_do_ruido["kaggle03"] == 6
    assert dentro_do_ruido["iscx2016"] == 6
    # Expectativa da rodada consolidada (AD-055, que substitui o AD-010): era 0/6
    # nos dois. Agora só a regressão logística (a de IC mais largo) cai dentro do
    # ruído nesses dois corpora; os outros cinco modelos seguem fora.
    assert dentro_do_ruido["grambeddings"] == 1
    assert dentro_do_ruido["phishstorm"] == 1

    # A2 (TAB-02): o critério |Δ| <= IC saiu da T3, substituído pelo IC pareado
    # da `tab:bootstrap-pareado`. A tabela já NÃO imprime a contagem medida acima
    # nem a marca "sim"/"não" por modelo; ela traz só a saturação do Regime A no
    # subtítulo do bloco (cabeçalho de 8 colunas, com as de média das sementes).
    tex = report.tabela3_diagonal(dados)
    linhas = tex.splitlines()
    for corpus in dentro_do_ruido:
        rotulo = report.ROTULO_CORPUS[corpus]
        inicio = f"\\multicolumn{{8}}{{l}}{{\\textbf{{{rotulo}}}:"
        i = next((k for k, ln in enumerate(linhas) if ln.startswith(inicio)), None)
        assert i is not None, f"a T3 não traz bloco para {rotulo}"
        assert "na largura do IC" not in linhas[i], linhas[i]
        bloco = linhas[i + 1:i + 7]
        assert len(bloco) == 6 and all(" & " in ln for ln in bloco), rotulo
        marcas = {ln.rsplit(" & ", 1)[1].removesuffix(" \\\\") for ln in bloco}
        assert not marcas & {"sim", "não"}, (
            f"as linhas de {rotulo} ainda trazem a marca do critério antigo"
        )
    # E a média por dataset continua PROIBIDA: no ISCX ela dá -0,0027 e leria
    # como "queda desprezível" o que é teto de métrica (§13.4-20).
    assert "-0,0027" not in tex, (
        "a média por dataset voltou à T3 — é exatamente o número que a "
        "§13.4-20 proíbe"
    )


# ---------------------------------------------------------------------------
# 5. T5.5a — as três tabelas que faltavam e o `numeros.tex`
# ---------------------------------------------------------------------------


def test_as_tabelas_regeneram_byte_a_byte(dados, tmp_path):
    """RNF1 sobre o artefato, não sobre a métrica.

    As 4 tabelas antigas já regeneravam idênticas e ninguém provava; as 3 novas
    trouxeram ordenação por valor (o posto da T7), que é onde empate vira não
    determinismo. O LightGBM grava contagem de divisões e dezenas de
    características empatam em zero — se a ordenação não fosse estável e o
    desempate não seguisse a ordem fixa de `config/features.json`, duas
    execuções trocariam duas linhas da T7 e o `git diff` acusaria mudança onde
    não houve.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    report.gerar(REPO_ROOT / "results", a)
    report.gerar(REPO_ROOT / "results", b)
    for caminho in sorted(a.iterdir()):
        assert caminho.read_bytes() == (b / caminho.name).read_bytes(), (
            f"{caminho.name} mudou entre duas execuções sobre os mesmos dados"
        )


def test_o_posto_medio_e_imune_a_escala(dados):
    """A T7 agrega por posto porque os 5 modelos gravam em escalas diferentes.

    Prova direta: multiplicar as importâncias de um ajuste por mil não pode
    mudar nada. Uma média dos valores brutos mudaria — e é exatamente o que
    acontece hoje entre modelos, já que o LightGBM grava contagem de divisões
    (soma ~24.800) e o Random Forest soma 1.
    """
    imp = dados["importances"]
    antes = report.posto_medio(imp, "etld1", "iscx2016")

    chave = "etld1|iscx2016|random_forest"
    inflado = {
        "features": imp["features"],
        "ajustes": {
            k: ({f: v * 1000 for f, v in vals.items()} if k == chave else vals)
            for k, vals in imp["ajustes"].items()
        },
    }
    assert report.posto_medio(inflado, "etld1", "iscx2016") == antes


def test_as_escalas_das_importancias_sao_mesmo_incompativeis(dados):
    """Canária do fato que motivou a decisão do posto (§11, 07/08/2026).

    Se um dia o pipeline passar a normalizar as importâncias antes de gravar,
    este teste cai — e aí a decisão do posto pode ser revista com fundamento,
    em vez de sobreviver como folclore.
    """
    ajustes = dados["importances"]["ajustes"]
    somas = {
        modelo: sum(ajustes[f"etld1|grambeddings|{modelo}"].values())
        for modelo in ("random_forest", "xgboost", "catboost", "lightgbm")
    }
    assert somas["random_forest"] == pytest.approx(1.0, abs=1e-6)
    assert somas["xgboost"] == pytest.approx(1.0, abs=1e-6)
    assert somas["catboost"] == pytest.approx(100.0, abs=1e-4)
    assert somas["lightgbm"] > 1000, "LightGBM deixou de gravar contagem bruta"

    # A regressão logística grava coeficiente COM SINAL — é por isso que o
    # posto usa |valor|, e não o valor.
    logistica = ajustes["etld1|grambeddings|logistic_regression"]
    assert min(logistica.values()) < 0


def test_a_t7_nao_afirma_que_o_iscx_nao_compartilha_nenhuma(dados, tmp_path):
    """PC5, no caso concreto que quase entrou no artigo.

    Na seed 42 o ISCX não compartilha **nenhuma** das cinco primeiras
    características com os outros corpora. É a frase mais forte da T7 e ela é
    **falsa em dois terços das execuções**: na seed 7 ele compartilha uma e na
    seed 13, duas. A célula tem de sair com a faixa ao lado do ponto.
    """
    por_seed = dados["importances_por_seed"]
    assert len(por_seed) >= 2, "sem as seeds extras este teste não mede nada"

    contagens = []
    for seed, imp in por_seed.items():
        tops = {
            c: set(report.top_caracteristicas(
                imp, "etld1", c, report.TOP_CARACTERISTICAS))
            for c in report.ORDEM_CORPORA
        }
        outros = set().union(*(v for c, v in tops.items() if c != "iscx2016"))
        contagens.append(len(tops["iscx2016"] & outros))

    assert min(contagens) != max(contagens), (
        "as seeds passaram a concordar; reveja se a faixa ainda é necessária"
    )
    texto = (tmp_path / "t7.tex")
    texto.write_text(report.tabela7_importancias(dados), encoding="utf-8")
    rodape = [l for l in texto.read_text(encoding="utf-8").splitlines()
              if l.startswith("Compart.")]
    assert rodape and "[" in rodape[0], (
        "a linha 'Compart.' saiu sem faixa entre seeds — vira absoluto (PC5)"
    )


def test_a_t6_reproduz_a_leitura_publicada_do_loco(dados):
    """Canária da convenção da T6, na linha da canária da T2.

    A §13.8 publicou "0,686 (1->1) -> 0,887 (união), contra 0,974 na diagonal"
    antes de a tabela existir. Foi reproduzindo esses três valores que a
    convenção foi identificada: média dos 6 modelos, por corpus de **teste**,
    regime eTLD+1. Se alguém trocar a base de comparação — por linha em vez de
    coluna, ou mediana em vez de média —, o ganho do LOCO muda e o artigo passa
    a afirmar outro número com a mesma frase.
    """
    registros = dados["matriz_por_seed"][report.SEED_CANONICA]
    coluna = report.medias_da_coluna(registros, "etld1", "auroc")
    loco = report.medias_loco(
        dados["loco_por_seed"][report.SEED_CANONICA], "etld1", "auroc"
    )
    corpora = report.ORDEM_CORPORA

    def media(vals):
        return sum(vals) / len(vals)

    # Expectativa da rodada consolidada (AD-055, que substitui o AD-010).
    # Valores da rodada antiga do artigo: 0,9741, 0,6856 e 0,8870.
    assert media([coluna[c][0] for c in corpora]) == pytest.approx(0.9790, abs=5e-5)
    assert media([coluna[c][1] for c in corpora]) == pytest.approx(0.6768, abs=5e-5)
    assert media([loco[c] for c in corpora]) == pytest.approx(0.8862, abs=5e-5)


def test_a_t6_declara_a_sobreposicao_residual_do_loco(dados):
    """§13.4-17: sem isso a T6 aparece ao lado da T2 carregando contaminação
    não declarada, que é a leitura alternativa mais fácil para o ganho.

    O D3 manda **medir e declarar, nunca filtrar** — a tabela tem de imprimir
    os números do `diagnostics.json`, não escondê-los na legenda em prosa.
    """
    tex = report.tabela6_loco(dados)
    resumo = dados["diagnostics"]["sobreposicao_loco"]["etld1"]
    for corpus, v in resumo["por_corpus_retido"].items():
        assert report._pct(v["shared_domain_pct"]) in tex, corpus
        assert report._pct(v["rows_with_shared_domain_pct"]) in tex, corpus
    assert report._pct(resumo["resumo"]["linhas_max_pct"]) in tex
    assert "nunca filtrada" in tex


def _legenda_tex(tex: str) -> str:
    """Conteúdo do primeiro `\\caption{...}` do arquivo."""
    i = tex.find("\\caption{")
    assert i >= 0, "sem \\caption"
    corpo, _ = _grupo_em(tex, i + len("\\caption"))
    return corpo


def test_a_legenda_da_ablacao_diz_que_a_configuracao_e_identica_entre_os_datasets(dados):
    """FIG-02 (ESC-04, emenda do Design): não é idêntica "para os seis modelos".

    A configuração fixa de `config/experiment.yml` tem hiperparâmetros próprios
    de cada modelo; o que ela mantém igual é o mesmo ajuste nos quatro
    *datasets*. A legenda dizia outra coisa.
    """
    legenda = _legenda_tex(report.tabela4_ablacao(dados))

    assert "idêntica entre os" in legenda
    assert "datasets" in legenda
    assert "seis modelos" not in legenda
    assert "para os seis" not in legenda


def test_a_t4_imprime_todos_os_modelos_com_o_delta_no_corpo(dados):
    """§7.4 e §11 de 06/08: o resultado do HPO é mostrado em todos os modelos.

    O D5 proíbe selecionar resultado: nenhum modelo vira "nota" e nenhum fica
    de fora. Cada modelo tem de ter uma linha no corpo da tabela com o delta de
    teste (ajustado menos seco), e a frase de perda da legenda é DERIVADA dos
    dados: só existe quando algum delta é <= 0. Na rodada consolidada nenhum
    modelo perde no teste (o menor delta é positivo), então a legenda não
    imprime frase de perda; o antigo achado do MLP (delta negativo, 16,5x o
    custo) era da rodada do artigo e deixou de valer.
    """
    tex = report.tabela4_ablacao(dados)
    linhas = tex.splitlines()
    modelos = dados["hpo"]["modelos"]
    assert modelos
    deltas = {}
    for nome, v in modelos.items():
        delta = v["auroc_teste_ajustado"] - v["auroc_teste_seco"]
        deltas[nome] = delta
        rotulo = report._rotulo(nome, report.ROTULO_MODELO)
        assert any(
            l.startswith(rotulo) and report._num(delta, 5) in l for l in linhas
        ), f"{nome}: sem linha no corpo com o delta de teste"

    perdas = {k: d for k, d in deltas.items() if d <= 0}
    if not perdas:
        assert "piora" not in tex
    else:
        pior = min(perdas, key=perdas.get)
        legenda = _legenda_tex(tex)
        assert report._rotulo(pior, report.ROTULO_MODELO) in legenda
        assert "piora" in legenda


def test_a_t4_frase_de_perda_cobre_os_tres_ramos(dados):
    """A frase de perda da legenda da T4 depende só da contagem de perdas.

    Zero perdas: nenhuma frase (e nenhuma "piora"). Uma perda: "o único que"
    com o rótulo dela. Duas: "um dos que" com o rótulo da MAIOR perda. E o
    trecho "após melhorar na validação" só sai quando o delta de validação do
    modelo de maior perda é positivo. A rodada consolidada só exercita o
    primeiro ramo; sem este teste os outros dois ficariam sem guarda.
    """
    nomes = [n for n in report.ROTULO_MODELO if n in dados["hpo"]["modelos"]]
    assert len(nomes) >= 2
    a, b = nomes[0], nomes[1]

    def com_deltas(teste: dict, val: dict | None = None) -> dict:
        hpo = copy.deepcopy(dados["hpo"])
        for nome, v in hpo["modelos"].items():
            v["auroc_teste_ajustado"] = v["auroc_teste_seco"] + teste.get(nome, 0.001)
            v["auroc_val_ajustado"] = v["auroc_val_seco"] + (val or {}).get(nome, 0.001)
        return dict(dados, hpo=hpo)

    rot = lambda n: report._rotulo(n, report.ROTULO_MODELO)  # noqa: E731

    sem = report.tabela4_ablacao(com_deltas({}))
    assert "piora" not in sem

    uma = _legenda_tex(report.tabela4_ablacao(com_deltas({a: -0.01})))
    assert "o único que" in uma and rot(a) in uma and "piora" in uma
    assert "após melhorar na validação" in uma

    duas = _legenda_tex(report.tabela4_ablacao(com_deltas({a: -0.01, b: -0.02})))
    assert "um dos que" in duas and "o único que" not in duas
    assert rot(b) in duas and "piora" in duas

    sem_val = _legenda_tex(
        report.tabela4_ablacao(com_deltas({a: -0.01}, val={a: -0.002}))
    )
    assert "piora" in sem_val and rot(a) in sem_val
    assert "após melhorar na validação" not in sem_val


def _macros_de(conteudo: str) -> dict:
    """`{nome: valor}` das linhas `\\newcommand{\\nome}{valor}` do numeros.tex."""
    macros = {}
    for linha in conteudo.splitlines():
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*(%.*)?$", linha)
        if m:
            macros[m.group(1)] = m.group(2)
    return macros


def test_macros_das_urls_identicas_saem_por_dataset_com_milhar(dados):
    """LIT-03: uma macro por dataset; 1386 sai `1.386` e 0 sai `0`."""
    diag = copy.deepcopy(dados["diagnostics"])
    valores = {"grambeddings": 1386, "kaggle03": 0, "phishstorm": 12,
               "iscx2016": 1386, "pdrcnn": 30, "sid321axn": 1158}
    for corpus, n in valores.items():
        diag["vazamento_split_oficial"][corpus]["urls_identicas_treino_teste"] = n
    macros = _macros_de(report.numeros_prosa(dict(dados, diagnostics=diag)))
    esperado = {
        "vazamentoUrlsIdenticasGrambeddings": "1.386",
        "vazamentoUrlsIdenticasKaggleTres": "0",
        "vazamentoUrlsIdenticasPhishstorm": "12",
        "vazamentoUrlsIdenticasIscx": "1.386",
        "vazamentoUrlsIdenticasPdrcnn": "30",
        "vazamentoUrlsIdenticasKaggleUm": "1.158",
    }
    for nome, valor in esperado.items():
        assert macros[nome] == valor, nome
    assert len([n for n in macros if n.startswith("vazamentoUrlsIdenticas")]) == 6


def test_dataset_sem_o_campo_das_urls_identicas_para_o_report(dados):
    """LIT-04: diagnostics.json antigo para com ReportError que nomeia o campo."""
    diag = copy.deepcopy(dados["diagnostics"])
    del diag["vazamento_split_oficial"]["iscx2016"]["urls_identicas_treino_teste"]
    with pytest.raises(report.ReportError) as exc:
        report.numeros_prosa(dict(dados, diagnostics=diag))
    assert "urls_identicas_treino_teste" in str(exc.value)
    assert "src.diagnostics" in str(exc.value)


def test_a_macro_do_iscx_e_a_da_dedup_nao_se_confundem(dados):
    """LIT-05: a medida literal (1.386) e a da dedup pelo url_norm (1.405) são grandezas
    distintas, e a macro antiga mantém o nome."""
    macros = _macros_de(report.numeros_prosa(dados))
    assert macros["vazamentoUrlsIdenticasIscx"] == "1.386"
    conteudo_dedup = "\n".join(report.macros_dedup(dados))
    assert r"\newcommand{\dedupRepetidasTreinoTesteIscxRegimeA}{1.405}" in conteudo_dedup


def test_as_macros_do_menor_delta_do_hpo_tem_nome_neutro_e_o_valor_de_antes(dados):
    """HPO-01 e HPO-02: `hpoMenorDelta{Modelo,Valor,Custo}` existem com os valores
    que `hpoPior{Modelo,Perda,Custo}` tinham (calculados aqui dos mesmos dados), e
    nenhum nome `hpoPior` é gerado."""
    macros = _macros_de(report.numeros_prosa(dados))
    hpo = dados["hpo"]["modelos"]
    deltas = {k: v["auroc_teste_ajustado"] - v["auroc_teste_seco"] for k, v in hpo.items()}
    menor = min(deltas, key=deltas.get)
    assert macros["hpoMenorDeltaModelo"] == report._rotulo(menor, report.ROTULO_MODELO)
    assert macros["hpoMenorDeltaValor"] == report._num(abs(deltas[menor]), 5)
    razao = hpo[menor]["fit_seconds_ajustado"] / hpo[menor]["fit_seconds_seco"]
    assert macros["hpoMenorDeltaCusto"] == report._num(razao, 1)
    assert not [nome for nome in macros if nome.startswith("hpoPior")]
    assert "hpoPior" not in report.numeros_prosa(dados)


def test_a_macro_hpo_menor_delta_diz_a_verdade_sobre_o_sinal(dados):
    """`\\hpoMenorDeltaModelo` designa o menor delta de teste. A contagem
    `\\hpoNumPerdem` (modelos com delta <= 0) é o
    que diz se "perda" é verdade: com 0, ninguém piora e a macro designa o menor
    GANHO; com mais de 0, o modelo da macro tem de ter delta <= 0.
    """
    macros = _macros_de(report.numeros_prosa(dados))
    deltas = {
        k: v["auroc_teste_ajustado"] - v["auroc_teste_seco"]
        for k, v in dados["hpo"]["modelos"].items()
    }
    n = int(macros["hpoNumPerdem"])
    assert n == sum(1 for d in deltas.values() if d <= 0)
    pior = min(deltas, key=deltas.get)
    assert macros["hpoMenorDeltaModelo"] == report._rotulo(pior, report.ROTULO_MODELO)
    if n == 0:
        assert all(d > 0 for d in deltas.values())
    else:
        assert deltas[pior] <= 0


def test_a_macro_hpo_num_perdem_conta_as_perdas_sinteticas(dados):
    """Com um modelo piorando no teste, a contagem é 1 e a macro o designa."""
    nomes = [n for n in report.ROTULO_MODELO if n in dados["hpo"]["modelos"]]
    hpo = copy.deepcopy(dados["hpo"])
    v = hpo["modelos"][nomes[0]]
    v["auroc_teste_ajustado"] = v["auroc_teste_seco"] - 0.01
    macros = _macros_de(report.numeros_prosa(dict(dados, hpo=hpo)))
    assert macros["hpoNumPerdem"] == "1"
    assert macros["hpoMenorDeltaModelo"] == report._rotulo(nomes[0], report.ROTULO_MODELO)


def test_o_delta_impresso_na_t4_bate_com_o_campo_do_pipeline(dados):
    """A T4 imprime (ajustado - seco) para a coluna e as células fecharem.

    O `hpo.json` também grava `delta_teste`, arredondado. Os dois têm de
    concordar: se divergirem, o pipeline mudou de definição de delta e a
    tabela ficaria dizendo uma coisa e o JSON outra.
    """
    for nome, v in dados["hpo"]["modelos"].items():
        calculado = v["auroc_teste_ajustado"] - v["auroc_teste_seco"]
        assert calculado == pytest.approx(v["delta_teste"], abs=1e-5), nome


def test_numeros_tex_so_define_comandos_e_nao_repete_nenhum(dados, tmp_path):
    """O `numeros.tex` entra no **preâmbulo**: qualquer linha que não seja
    `\\newcommand` vira texto solto antes do `\\begin{document}` e derruba a
    compilação. Nome duplicado também derruba — de propósito, e este teste
    pega antes do LaTeX.
    """
    conteudo = report.numeros_prosa(dados)
    nomes = []
    for n, linha in enumerate(conteudo.splitlines(), 1):
        limpa = linha.split("%")[0].strip() if not linha.startswith("%") else ""
        if not limpa:
            continue
        assert limpa.startswith("\\newcommand{\\"), f"linha {n}: {linha!r}"
        nome = limpa.split("{\\", 1)[1].split("}", 1)[0]
        # LaTeX só aceita letras em nome de comando: um dígito trunca o nome
        # e o resto vira texto impresso no PDF, em silêncio.
        assert nome.isalpha(), f"{nome!r} tem caractere que o LaTeX não aceita"
        nomes.append(nome)
    assert len(nomes) == len(set(nomes)), "macro definida duas vezes"
    assert len(nomes) >= 20, "o conjunto encolheu — a prosa da T5.5 depende dele"


def test_numeros_tex_concorda_com_as_tabelas(dados):
    """O risco do `numeros.tex` é a prosa afirmar um número e a tabela ao lado
    mostrar outro — pior que não ter mecanismo nenhum, porque parece checado.
    """
    conteudo = report.numeros_prosa(dados)
    macros = dict(
        linha.split("{\\", 1)[1].split("}{", 1)[0:1]
        + [linha.split("}{", 1)[1].rsplit("}", 1)[0]]
        for linha in (
            l.split("%")[0].strip()
            for l in conteudo.splitlines()
            if l.startswith("\\newcommand")
        )
    )
    celulas = report.media_por_celula(
        dados["matriz_por_seed"][report.SEED_CANONICA], "etld1", "auroc"
    )
    diagonal = [v for (tr, te), v in celulas.items() if tr == te]
    fora = [v for (tr, te), v in celulas.items() if tr != te]
    assert macros["aurocDiagonal"] == report._num(sum(diagonal) / len(diagonal))
    assert macros["aurocForaDiagonal"] == report._num(sum(fora) / len(fora))

    # A faixa de ganho do LOCO tem de ser a da MESMA seed que a Tabela 6
    # imprime: um mínimo que não aparece em nenhuma linha da tabela seria lido
    # como divergência entre texto e tabela.
    t6 = report.tabela6_loco(dados)
    assert macros["locoGanhoMin"] in t6
    assert macros["locoGanhoMax"] in t6


def test_a_assimetria_do_iscx_nao_depende_de_ordenacao_de_ganho(dados):
    """Trava a saída (b) da §13.4-26 (decisão do Julio, §11 de 07/08/2026).

    A §13.8 afirmava que "o maior ganho do LOCO é com o ISCX retido". Era
    falso nas 3 seeds — o maior é o PhishStorm, e a própria frase publicava
    o número dele como topo da faixa. A frase foi trocada por uma que a
    medição sustenta sozinha, e é ela que este teste guarda:

    1. o ISCX é o pior nas **duas** leituras da matriz, por linha e por
       coluna, em CADA semente;
    2. a margem por **linha** é maior que a por coluna em cada semente, e
       na MÉDIA das sementes é mais de 3 vezes maior: é essa desproporção
       que é a assimetria do D1, não a posição no ranking;
    3. sob a união de três fontes, a coluna do ISCX **recupera** (mais de
       0,15 de AUROC) em cada semente.

    Rodada consolidada (AD-063: a afirmação vale na média das sementes): a
    razão margem por linha / margem por coluna é 1,86 na semente canônica
    (42), 4,79 na 7 e 7,61 na 13, e 3,66 na média. A desproporção "mais de 3
    vezes" vale na média, não na semente 42 sozinha, e por isso o limiar de 3
    é aplicado à média das margens e não a cada semente. Não se acrescenta
    piso por semente: o teste guarda o que o texto afirma.

    Se qualquer um dos três cair, a Discussão precisa ser reescrita — de
    novo. Não é para ser "consertado" afrouxando o teste.
    """
    margens_linha, margens_coluna = [], []
    for semente, registros in dados["matriz_por_seed"].items():
        coluna = report.medias_da_coluna(registros, "etld1", "auroc")
        loco = report.medias_loco(
            dados["loco_por_seed"][semente], "etld1", "auroc"
        )
        por_linha = {}
        for treino in report.ORDEM_CORPORA:
            fora = [
                r["metrics"]["auroc"]
                for r in registros
                if r["regime"] == "etld1"
                and r["train_corpus"] == treino
                and r["test_corpus"] != treino
                and r["metrics"]["auroc"] is not None
            ]
            por_linha[treino] = sum(fora) / len(fora)

        ord_linha = sorted(por_linha, key=por_linha.get)
        ord_coluna = sorted(report.ORDEM_CORPORA, key=lambda c: coluna[c][1])
        assert ord_linha[0] == "iscx2016", f"semente {semente}"
        assert ord_coluna[0] == "iscx2016", f"semente {semente}"

        margem_linha = por_linha[ord_linha[1]] - por_linha[ord_linha[0]]
        margem_coluna = coluna[ord_coluna[1]][1] - coluna[ord_coluna[0]][1]
        assert margem_linha > margem_coluna, (
            f"semente {semente}: o ISCX deixou de estar mais isolado por "
            "linha que por coluna; a assimetria do D1 mudou"
        )
        margens_linha.append(margem_linha)
        margens_coluna.append(margem_coluna)

        assert loco["iscx2016"] > coluna["iscx2016"][1] + 0.15, (
            f"semente {semente}: o ISCX deixou de ser aprendível a partir "
            "dos outros"
        )

    media_linha = sum(margens_linha) / len(margens_linha)
    media_coluna = sum(margens_coluna) / len(margens_coluna)
    assert media_linha > 3 * media_coluna, (
        "na média das sementes a margem por linha do ISCX deixou de passar de "
        f"3 vezes a por coluna ({media_linha / media_coluna:.2f}); a "
        "assimetria do D1 mudou"
    )


# ---------------------------------------------------------------------------
# 6. T5.9 — o corte de página, e as garantias que ele criou
# ---------------------------------------------------------------------------


def _grupo_em(tex: str, i: int) -> tuple[str, int]:
    """Lê o grupo `{...}` equilibrado que começa em `tex[i] == '{'`.

    Devolve `(conteúdo, índice logo depois do '}')`.
    """
    assert tex[i] == "{", f"esperava '{{' em {i}, achei {tex[i]!r}"
    j, nivel = i + 1, 1
    while j < len(tex) and nivel:
        if tex[j] == "{":
            nivel += 1
        elif tex[j] == "}":
            nivel -= 1
        j += 1
    return tex[i + 1:j - 1], j


def _palavras(corpo: str) -> list[str]:
    """Palavras de um corpo de legenda, sem comando, chave nem matemática."""
    corpo = corpo.replace("~", " ")
    corpo = re.sub(r"\\[a-zA-Z]+\s*", " ", corpo)
    corpo = re.sub(r"[{}$\\]", " ", corpo)
    return [t for t in corpo.split() if any(c.isalnum() for c in t)]


def _palavras_de_legenda(tex: str) -> list[str]:
    """Palavras de um `\\caption{...}`, sem comando, chave nem matemática.

    A contagem não precisa casar dígito a dígito com a da §5 do
    o roteiro de revisão (que mediu 751 para as sete legendas, contra 741
    por este método) — precisa ser **a mesma convenção sempre**, para o teto
    de 40 ser verificável em vez de opinião.
    """
    i = tex.find("\\caption{")
    if i < 0:
        return []
    corpo, _ = _grupo_em(tex, i + len("\\caption"))
    return _palavras(corpo)


TETO_LEGENDA = 40


def test_nenhuma_legenda_passa_do_teto_de_quarenta_palavras(dados, tmp_path):
    """D-C2 (decisão do Julio, 07/08/2026): legenda tem teto de 40 palavras.

    Antes da T5.9 a média era 107 palavras por legenda e a maior tinha 133. O
    que estourava o teto não era o que a tabela mostra — era justificativa
    metodológica e instrução ao leitor sobre como interpretar, que a §5.1 tirou
    da legenda e o parágrafo de convenções da Metodologia recebeu, dito uma vez
    em vez de sete.

    Este teste existe porque legenda é o lugar do artigo onde texto cresce sem
    ninguém notar: ela não aparece na contagem de prosa por seção, não tem
    revisor natural, e cada linha dela custa página. Sem teto travado, a
    câmera-ready de 13/09 tem seis semanas para desfazer o corte.
    """
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        if caminho.name == "numeros.tex":
            continue  # não é tabela: é arquivo de \newcommand
        palavras = _palavras_de_legenda(caminho.read_text(encoding="utf-8"))
        assert palavras, f"{caminho.name} não tem \\caption"
        assert len(palavras) <= TETO_LEGENDA, (
            f"{caminho.name}: legenda com {len(palavras)} palavras, "
            f"teto {TETO_LEGENDA} (D-C2)"
        )


def test_legenda_de_figura_do_artigo_tambem_respeita_o_teto():
    """O teto de 40 palavras (D-C2) vale para legenda de FIGURA também.

    ⚠️ Achado ao executar a Fase 3 da T5.9, e é um buraco de cobertura, não um
    defeito novo: as sete legendas de tabela saem do `src/report.py` e o teste
    acima as vigia porque as gera. A legenda da F2 mora no `artigo.tex`, no
    argumento `#4` do `\\incluifigura` — nenhum teste a lia. Ela ficou em
    exatamente 40 palavras no corte da Fase 2, isto é, **no teto e sem margem**:
    uma palavra acrescentada na câmera-ready estouraria a D-C2 em silêncio.

    O teste vale para qualquer `\\incluifigura` do artigo, não só a F2 — é o que
    faz a garantia sobreviver ao retorno da F1 em 13/09. Um teste que nomeia o
    caso concreto quebra por mudança de escopo, e quem o conserta tende a
    afrouxá-lo: foi o que aconteceu com o
    `test_resizebox_abre_e_fecha_nas_tabelas_largas` no mesmo dia.
    """
    artigo = ARTIGO_TEX.read_text(encoding="utf-8")
    # Só as chamadas de verdade: a definição da macro e os comentários que
    # explicam como revertê-la também contêm a palavra `\incluifigura`.
    corpo = "\n".join(
        linha for linha in artigo.splitlines()
        if not linha.lstrip().startswith("%")
    )
    corpo = corpo[corpo.index("\\begin{document}"):]

    achadas = 0
    for m in re.finditer(r"\\incluifigura\s*(?=\{)", corpo):
        i = m.end()
        for _ in range(3):                      # #1 aviso, #2 caminho, #3 rótulo
            _, i = _grupo_em(corpo, i)
        legenda, _ = _grupo_em(corpo, i)        # #4 é a legenda que renderiza
        palavras = _palavras(legenda)
        achadas += 1
        assert len(palavras) <= TETO_LEGENDA, (
            f"legenda de figura no artigo.tex com {len(palavras)} palavras, "
            f"teto {TETO_LEGENDA} (D-C2): {' '.join(palavras[:12])}..."
        )

    assert achadas, (
        "nenhum \\incluifigura no corpo do artigo.tex — se a última figura saiu "
        "de propósito, este teste perdeu o objeto e deve sair junto"
    )


def test_o_par_de_urls_de_exemplo_esta_nos_dois_lados_do_corte(dados, tmp_path):
    """O exemplo de vazamento tem de ilustrar vazamento de DOMÍNIO.

    🚨 Esta é a garantia que o roteiro de revisão não tinha. O par proposto lá
    à mão era ``.../1eeytsv.html'' (treino) e ``.../ba1n3qguq.html'' (teste), e
    a medição em 08/08 mostrou que a segunda URL está **nas duas partições** do
    ISCX: publicá-la como "a do teste" ilustraria duplicação exata de URL, que é
    um defeito diferente — e mais forte — do que o vazamento de domínio que o C1
    mede. Um revisor que conferisse encontraria o exemplo principal do artigo
    demonstrando outra coisa.

    O teste trava as três condições que fazem o exemplo dizer o que o artigo
    afirma: mesmo domínio dos dois lados, URLs distintas, e contagens positivas
    dos dois lados.
    """
    diag = json.loads(
        (REPO_ROOT / "results" / "diagnostics.json").read_text(encoding="utf-8")
    )
    for corpus, v in diag["vazamento_split_oficial"].items():
        ex = v.get("exemplo")
        if ex is None:
            continue                       # domínio sem par distinto: honesto
        assert ex["url_treino"] != ex["url_teste"], (
            f"{corpus}: o par de exemplo repete a MESMA URL — isso é duplicação "
            f"exata, não vazamento de domínio ({ex['url_treino']})"
        )
        for lado in ("url_treino", "url_teste"):
            assert ex["dominio"] in ex[lado], (
                f"{corpus}: {lado} não pertence ao domínio {ex['dominio']}"
            )
        assert ex["linhas_treino"] > 0 and ex["linhas_teste"] > 0, (
            f"{corpus}: o domínio do exemplo não está nos dois lados do corte"
        )


def test_legenda_da_tabela_de_vazamento_usa_corte_de_referencia(dados):
    """TAB-01: a legenda traz o termo do AD-070 e nenhuma forma de "oficial".

    A linha de comentário `% Fonte:` cita a chave `vazamento_split_oficial` do JSON,
    que não muda: a checagem é sobre o texto da tabela, sem os comentários.
    """
    tex = report.tabela1_vazamento(dados)
    texto = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    assert (
        r"Vazamento de domínio no corte de referência dos seis \emph{datasets}" in texto
    )
    assert "oficia" not in texto


@pytest.mark.parametrize(
    "url, esperado",
    [
        ("http://a.b.com/x.html", "hxxp://a[.]b[.]com/x.html"),
        ("https://site.org/?q=a.b", "hxxps://site[.]org/?q=a.b"),
        ("HTTP://X.COM/", "hxxp://X[.]COM/"),
        ("www.x.com/p.php", "www[.]x[.]com/p.php"),
        ("http://10.0.0.1/a", "hxxp://10[.]0[.]0[.]1/a"),
    ],
)
def test_neutralizar_url_de_phishing(url, esperado):
    """URL-01 e URL-02: esquema vira hxxp(s) e os pontos do host viram `[.]`."""
    assert report._neutralizar_url(url, 1) == esperado


@pytest.mark.parametrize(
    "url",
    ["http://a.b.com/x.html", "https://site.org/?q=a.b", "www.x.com/p.php", "http://10.0.0.1/a"],
)
def test_neutralizar_url_de_rotulo_zero_sai_intacta(url):
    """URL-03."""
    assert report._neutralizar_url(url, 0) == url


def test_macro_do_exemplo_sai_neutralizada_e_com_pontos_de_quebra(dados):
    """URL-01, URL-02, URL-05: `hxxp:` e `[.]` na macro, com `\\allowbreak{}` do `_url_tex`."""
    diag = copy.deepcopy(dados["diagnostics"])
    ex = diag["vazamento_split_oficial"]["iscx2016"]["exemplo"]
    ex["url_treino"], ex["label_treino"] = "http://pastehtml.com/view/1dpt0t8.html", 1
    ex["url_teste"], ex["label_teste"] = "http://pastehtml.com/view/au6g36wgl.html", 1
    conteudo = report.numeros_prosa(dict(dados, diagnostics=diag))
    for nome in ("exemploUrlTreino", "exemploUrlTeste"):
        # `_macros_de` corta no primeiro `}`, que aqui é o do `\allowbreak{}`.
        linha = next(
            ln for ln in conteudo.splitlines() if ln.startswith(f"\\newcommand{{\\{nome}}}")
        )
        assert "hxxp:" in linha
        assert "[.]" in linha
        assert "http:" not in linha.replace("hxxp:", "")
        assert "\\allowbreak{}" in linha


def test_a_url_de_exemplo_e_escapada_para_o_latex(dados, tmp_path):
    """`%` numa URL comenta o resto da linha do `.tex`, em silêncio.

    Codificação percentual é comum em URL. Um `%` não escapado não derruba a
    compilação: ele apaga do PDF o resto da linha, sem aviso. Mesma família da
    §13.4-21 (o `\\ref` órfão que imprime `??`).
    """
    assert report._url_tex("a%20b") .startswith("a\\%")
    assert "\\_" in report._url_tex("x_y")
    assert "\\#" in report._url_tex("x#y")
    assert "\\textasciitilde{}" in report._url_tex("~u")

    numeros = next(
        c for c in report.gerar(REPO_ROOT / "results", tmp_path)
        if c.name == "numeros.tex"
    ).read_text(encoding="utf-8")
    for linha in numeros.splitlines():
        if "exemploUrl" not in linha:
            continue
        corpo = linha.split("}{", 1)[1].rsplit("}", 1)[0]
        # Todo `%` restante tem de estar escapado; `\%` é o único legítimo.
        assert "%" not in corpo.replace("\\%", ""), (
            f"URL de exemplo com `%` não escapado, que comeria a linha: {linha}"
        )


def test_o_nome_do_experimento_loco_e_o_mesmo_na_prosa_e_na_tabela():
    """`leave-one-corpus-out` é NOME DE EXPERIMENTO, não a palavra "corpus".

    🚨 Achado ao executar a Fase 3 da T5.9, e foi um defeito real que este teste
    existe para impedir. A troca de terminologia da D-H (`corpus` → `dataset`) foi
    aplicada por varredura no `artigo.tex` e renomeou as três ocorrências para
    ``leave-one-dataset-out'' — um termo que não existe na literatura. A legenda
    da T6 sai do `src/report.py`, que a varredura não tocou, e continuou dizendo
    ``Leave-one-corpus-out'': a prosa e a tabela que ela referencia passariam a
    chamar o mesmo experimento por nomes diferentes, na mesma página.

    A D-H é troca de EXIBIÇÃO do substantivo. Nome próprio de experimento fica,
    pela mesma razão que `fonte` e `benchmark` ficam (§12.2 do roteiro): não são
    rotação de sinônimo, são termos que o argumento usa. O acrônimo LOCO também
    depende dele.

    Nem a compilação nem o `pdftotext` pegam isto — as duas grafias são texto
    válido. Só a leitura pega, e revisão humana não escala para seis semanas de
    câmera-ready.
    """
    artigo = ARTIGO_TEX.read_text(encoding="utf-8")
    tabela = (TABELAS_DIR / "tabela6-loco.tex").read_text(
        encoding="utf-8"
    )

    assert "leave-one-dataset-out" not in artigo.lower(), (
        "a prosa inventou ``leave-one-dataset-out''. O experimento se chama "
        "leave-one-corpus-out (LOCO), e é assim que a legenda da T6 o nomeia"
    )
    assert "leave-one-corpus-out" in artigo.lower(), (
        "a prosa deixou de nomear o experimento LOCO"
    )
    assert "leave-one-corpus-out" in tabela.lower(), (
        "a legenda da T6 deixou de nomear o experimento LOCO — se o nome mudou "
        "de propósito, mude nos DOIS lugares e ajuste este teste"
    )


def test_gerar_figuras_respeita_o_dicionario_figuras(dados, tmp_path):
    """Tirar uma figura de `FIGURAS` tem de tirá-la da geração.

    ⚠️ Até a T5.9 o `FIGURAS` **não era lido por código nenhum** — o
    `gerar_figuras()` chamava as duas funções na mão. Remover a entrada da F1,
    que é o que a §2.2 do roteiro de corte manda fazer, teria sido declaração
    sem efeito: o `figura1-curvas.pdf` continuaria sendo gerado e re-carimbado
    (§13.4-22) depois de já ter saído do artigo. É a mesma classe de defeito
    que o `contrato_features_fingerprint()` teve em 05/08 — declarar uma coisa
    e o código fazer outra.
    """
    escritos = report.gerar_figuras(REPO_ROOT / "results", tmp_path)
    assert {p.name for p in escritos} == set(report.FIGURAS)
    # A F1 voltou a FIGURAS no Cap. 6 (T6, CAP6-18): a seção 6.3 a usa como
    # `fig:curvas`. Se ela sair de novo, sai também do capítulo.
    assert {"figura1-curvas.pdf", "figura2-mapa-calor.pdf"} <= set(report.FIGURAS)
    assert all(p.stat().st_size > 0 for p in escritos)


def test_virgula_para_as_figuras():
    """FIG-01: vírgula decimal, valor ausente vazio e sem zero negativo."""
    assert report._virgula(0.8456, 3) == "0,846"
    assert report._virgula(0.2, 1) == "0,2"
    assert report._virgula(1, 1) == "1,0"
    assert report._virgula(-0.00001, 3) == "0,000"
    assert report._virgula(None) == "" and report._virgula(float("nan")) == ""
    formatador = report._formatador_virgula(1)
    assert formatador(0.6, 0) == "0,6"


def _textos_da_figura(fig) -> list[str]:
    """Todo texto que a figura imprime: anotações, rótulos, títulos, ticks e legenda."""
    from matplotlib.text import Text

    fig.canvas.draw()  # os rótulos de tick só existem depois do desenho
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _figuras_geradas(monkeypatch, tmp_path) -> list:
    """Roda `gerar_figuras` e devolve as figuras do matplotlib que foram salvas."""
    from matplotlib.figure import Figure

    salvas = []
    original = Figure.savefig

    def espiar(self, *args, **kwargs):
        salvas.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", espiar)
    report.gerar_figuras(REPO_ROOT / "results", tmp_path)
    return salvas


def test_o_detector_de_ponto_decimal_distingue_ponto_de_virgula():
    """Sensor do teste abaixo: uma figura com o formato padrão do matplotlib é reprovada."""
    plt, _sns = report._plt()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.text(0.5, 0.5, "0.846")
    com_ponto = [t for t in _textos_da_figura(fig) if re.search(r"\d\.\d", t)]
    plt.close(fig)
    assert "0.846" in com_ponto and len(com_ponto) > 1  # a anotação e os ticks

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.xaxis.set_major_formatter(report._formatador_virgula(1))
    ax.yaxis.set_major_formatter(report._formatador_virgula(1))
    ax.text(0.5, 0.5, report._virgula(0.846, 3))
    assert not [t for t in _textos_da_figura(fig) if re.search(r"\d\.\d", t)]
    plt.close(fig)


def test_nenhuma_figura_de_figuras_imprime_numero_com_ponto_decimal(dados, tmp_path, monkeypatch):
    """FIG-01 (ABNT-06): anotações, eixos e barra de cor de TODA figura de `FIGURAS`.

    A lista vem de `FIGURAS`, e não de duas figuras fixas: uma figura nova
    registrada lá entra no teste sem que ninguém lembre de acrescentá-la. O
    texto de cada figura salva é lido depois de desenhada, então cobre também os
    rótulos de tick e os da barra de cor.
    """
    figuras = _figuras_geradas(monkeypatch, tmp_path)

    # Uma figura salva por entrada de FIGURAS.
    assert len(figuras) == len(report.FIGURAS)
    assert {p.name for p in tmp_path.iterdir()} == set(report.FIGURAS)
    for fig in figuras:
        textos = _textos_da_figura(fig)
        assert textos
        com_ponto = [t for t in textos if re.search(r"\d\.\d", t)]
        assert not com_ponto, f"número com ponto decimal numa figura: {com_ponto[:5]}"
    # O mapa de calor imprime as AUROC com três casas e vírgula.
    todos = [t for fig in figuras for t in _textos_da_figura(fig)]
    assert any(re.fullmatch(r"\d,\d{3}", t) for t in todos)
    # Ticks dos eixos das curvas e da barra de cor, com uma casa e vírgula.
    assert any(re.fullmatch(r"\d,\d", t) for t in todos)


def test_nenhuma_tabela_referencia_rotulo_que_o_artigo_nao_define(dados, tmp_path):
    """Um `\\ref{}` órfão não derruba a compilação: imprime `??` no PDF.

    Achado ao executar a T5.9: a legenda da T6 dizia ``a coluna correspondente
    da Tabela~\\ref{tab:matriz}'', e a T2 foi justamente a tabela cortada. O
    `\\label{tab:matriz}` deixou de existir, e o LaTeX teria impresso
    ``Tabela ??'' com um aviso no `.log` que a compilação classifica como
    *warning*, não erro. Nenhum teste pegava isso — só o portão do DoD, que é
    manual.
    """
    rotulos = set()
    usados = set()
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        conteudo = caminho.read_text(encoding="utf-8")
        rotulos |= set(re.findall(r"\\label\{([^}]+)\}", conteudo))
        usados |= set(re.findall(r"\\ref\{([^}]+)\}", conteudo))
    artigo = ARTIGO_TEX.read_text(encoding="utf-8")
    rotulos |= set(re.findall(r"\\label\{([^}]+)\}", artigo))

    assert usados <= rotulos, (
        f"tabela gerada referencia rótulo que ninguém define: {usados - rotulos}"
    )


def test_a_divergencia_aucpr_sobrevive_ao_corte_da_t2(dados):
    """A AUC-PR saiu das 16 células com a T2; tem de sobrar em macro (§2.1).

    A T2 era a **única** publicação da AUC-PR no artigo. Cortá-la sem destino
    apagaria do trabalho inteiro, em silêncio, uma das duas métricas primárias
    declaradas na Metodologia — e um revisor que leia "reportamos AUROC e
    AUC-PR" e não ache a segunda em lugar nenhum tem razão em reclamar.

    A direção da divergência é **derivada** (`\\divergenciaAucprDirecao`) e não
    escrita na prosa, pela cicatriz da §13.4-26: se uma rodada futura inverter
    o sinal, a frase passa a dizer a outra palavra sozinha em vez de virar uma
    afirmação falsa que sobrevive porque ninguém remediu.
    """
    conteudo = report.numeros_prosa(dados)
    macros = {
        m.group(1): m.group(2)
        for m in re.finditer(
            r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}", conteudo
        )
    }
    for nome in ("aucprDiagonal", "aucprForaDiagonal", "divergenciaAucprMax",
                 "divergenciaAucprTreino", "divergenciaAucprTeste",
                 "divergenciaAucprDirecao"):
        assert nome in macros, f"macro {nome} ausente do numeros.tex"

    celulas = report.media_por_celula(
        dados["matriz_por_seed"][report.SEED_CANONICA], "etld1", "auc_pr"
    )
    diagonal = [v for (tr, te), v in celulas.items() if tr == te]
    fora = [v for (tr, te), v in celulas.items() if tr != te]
    assert macros["aucprDiagonal"] == report._num(sum(diagonal) / len(diagonal))
    assert macros["aucprForaDiagonal"] == report._num(sum(fora) / len(fora))

    # A frase que a §Resultados vai escrever é "a AUC-PR acompanha o AUROC":
    # ela só se sustenta se as duas caírem no mesmo sentido da diagonal para
    # fora dela. Se um dia não caírem, a frase tem de deixar de ser escrita.
    auroc = report.media_por_celula(
        dados["matriz_por_seed"][report.SEED_CANONICA], "etld1", "auroc"
    )
    def medias(c):
        d = [v for (tr, te), v in c.items() if tr == te]
        f = [v for (tr, te), v in c.items() if tr != te]
        return sum(d) / len(d), sum(f) / len(f)
    d_pr, f_pr = medias(celulas)
    d_roc, f_roc = medias(auroc)
    assert d_pr > f_pr and d_roc > f_roc, "a AUC-PR deixou de acompanhar o AUROC"

    # A célula mais divergente e a direção são derivadas — confere as duas.
    div = {k: celulas[k] - auroc[k] for k in auroc if k in celulas}
    pior = max(div, key=lambda k: abs(div[k]))
    assert macros["divergenciaAucprMax"] == report._num(abs(div[pior]))
    assert macros["divergenciaAucprTreino"] == report.ROTULO_CORPUS[pior[0]]
    assert macros["divergenciaAucprTeste"] == report.ROTULO_CORPUS[pior[1]]
    assert macros["divergenciaAucprDirecao"] == (
        "acima" if div[pior] > 0 else "abaixo"
    )


# ---------------------------------------------------------------------------
# E4: Friedman e Nemenyi (T8 e F3) e a legenda da tab:vies
# ---------------------------------------------------------------------------


def test_legenda_da_tab_vies_sem_float_cru_nem_abreviatura(dados):
    """Pendência do Cap. 4: a legenda saía com "20.0 pontos" e o cabeçalho com
    "% phish.". O limiar é inteiro e tem vírgula decimal quando não for; o
    Kaggle-01 e o PDRCNN têm "classe positiva", e não phishing (Cap. 4)."""
    tex = report.tabela5_vies(dados)
    assert "20.0" not in tex and "phish." not in tex
    assert "20 pontos percentuais" in tex
    assert r"\% positiva" in tex


def test_diagrama_cd_nao_usa_o_operador_de_dois_hifens(dados):
    """O gate 5 lê `--` como travessão; o TikZ desenha com `to`."""
    corpo = "\n".join(
        linha for linha in report.figura3_diferenca_critica(dados).splitlines()
        if not linha.lstrip().startswith("%")
    )
    assert "--" not in corpo
    assert r"\begin{tikzpicture}" in corpo and r"\label{fig:diferenca-critica}" in corpo


def test_diagrama_cd_tem_um_rotulo_por_modelo_e_a_cd(dados):
    tex = report.figura3_diferenca_critica(dados)
    for rotulo in report.ROTULO_MODELO.values():
        assert f"{{{rotulo} (" in tex, f"{rotulo} fora do diagrama"
    canon = report.friedman_por_seed(dados)[report.SEED_CANONICA]
    assert f"CD = {report._num(canon['cd'], 3)}" in tex


def test_macros_do_friedman_concordam_com_a_tabela8(dados):
    """A prosa cita `\\friedmanQui` e `\\nemenyiCD` ao lado da T8: o mesmo número."""
    numeros = report.numeros_prosa(dados)
    tabela = report.tabela8_friedman(dados)

    def macro(nome):
        m = re.search(r"\\newcommand\{\\" + nome + r"\}\{([^}]*)\}", numeros)
        assert m, f"macro {nome} ausente"
        return m.group(1)

    assert rf"$\chi^2_F$ = {macro('friedmanQui')}," in tabela
    assert f"(CD) = {macro('nemenyiCD')}," in tabela
    for sufixo, chave in (("Lightgbm", "lightgbm"), ("Logistica", "logistic_regression")):
        assert f"{report.ROTULO_MODELO[chave]} & {macro('posto' + sufixo)} &" in tabela


def test_tabela8_lista_os_pares_nos_dois_sentidos(dados):
    """Cada par aparece na linha dos dois modelos, com a mesma contagem de sementes."""
    tabela = report.tabela8_friedman(dados)
    linhas = {
        l.split(" & ")[0]: l.split(" & ")[3] for l in tabela.splitlines()
        if l.count(" & ") == 3 and not l.startswith("Modelo")
    }
    # Expectativa da rodada consolidada (AD-055, que substitui o AD-010): a
    # contagem de sementes era 3 nos dois sentidos, agora é 2.
    assert "Reg. logística (2)" in linhas["LightGBM"]
    assert "LightGBM (2)" in linhas["Reg. logística"]
    assert linhas["MLP"].startswith("nenhum")


# ---------------------------------------------------------------------------
# Cap. 6: macros de viés, calibração e importâncias (T1)
# ---------------------------------------------------------------------------


def _macros_de(conteudo: str) -> dict:
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}", conteudo)
    }


def _results_json(nome: str):
    return json.loads((REPO_ROOT / "results" / nome).read_text(encoding="utf-8"))


def test_macros_do_vies_sao_as_pontas_da_bimodalidade(dados):
    """O maior gap entre os limpos e o menor entre os contaminados, do JSON."""
    macros = _macros_de(report.numeros_prosa(dados))
    vies = _results_json("diagnostics.json")["vies_construcao"]
    limpos = [v["gap_max_p"] for v in vies.values() if v["veredito"] == "limpo"]
    contaminados = [v["gap_max_p"] for v in vies.values() if v["veredito"] == "contaminado"]
    assert limpos and contaminados
    assert macros["viesGapLimpoMax"] == report._pct(max(limpos))
    assert macros["viesGapContaminadoMin"] == report._pct(min(contaminados))
    # A bimodalidade que a prosa afirma: os limpos inteiros abaixo dos contaminados.
    assert max(limpos) < min(contaminados)


def _calibracao_recalculada(metrica: str):
    """Média por célula (6 modelos) e depois por diagonal e fora, Regime B, seed 42."""
    celulas: dict = {}
    for r in _results_json("matrix.json"):
        if r["regime"] != "etld1" or r["metrics"][metrica] is None:
            continue
        celulas.setdefault((r["train_corpus"], r["test_corpus"]), []).append(
            r["metrics"][metrica]
        )
    medias = {k: sum(v) / len(v) for k, v in celulas.items()}
    diag = [v for (tr, te), v in medias.items() if tr == te]
    fora = [v for (tr, te), v in medias.items() if tr != te]
    return sum(diag) / len(diag), sum(fora) / len(fora)


def test_macros_de_calibracao_recalculadas_de_results(dados):
    macros = _macros_de(report.numeros_prosa(dados))
    for metrica, prefixo in (("brier", "brier"), ("ece", "ece")):
        diag, fora = _calibracao_recalculada(metrica)
        assert macros[f"{prefixo}Diagonal"] == report._num(diag)
        assert macros[f"{prefixo}ForaDiagonal"] == report._num(fora)


def test_melhor_e_pior_ece_fora_da_diagonal_sao_derivados(dados):
    """O nome do modelo sai do JSON, com o rótulo das outras tabelas."""
    macros = _macros_de(report.numeros_prosa(dados))
    por_modelo: dict = {}
    for r in _results_json("matrix.json"):
        if r["regime"] != "etld1" or r["train_corpus"] == r["test_corpus"]:
            continue
        por_modelo.setdefault(r["model"], []).append(r["metrics"]["ece"])
    assert all(len(v) == 12 for v in por_modelo.values())
    medias = {m: sum(v) / len(v) for m, v in por_modelo.items()}
    assert macros["eceForaMelhorModelo"] == report.ROTULO_MODELO[min(medias, key=medias.get)]
    assert macros["eceForaPiorModelo"] == report.ROTULO_MODELO[max(medias, key=medias.get)]


def test_cobertura_das_importancias_contada_nos_dados(dados):
    """O artigo dizia "40 dos 48": o MLP não expõe importância."""
    macros = _macros_de(report.numeros_prosa(dados))
    ajustes = _results_json("importances.json")["ajustes"]
    total = {
        (r["regime"], r["train_corpus"], r["model"])
        for r in _results_json("matrix.json")
    }
    assert macros["importanciasAjustes"] == str(len(ajustes))
    assert macros["importanciasAjustesTotal"] == str(len(total))
    # Os que faltam são exatamente os do modelo sem importância.
    com_importancia = {tuple(k.split("|")) for k in ajustes}
    assert {m for (_r, _c, m) in total - com_importancia} == {"mlp"}


# ---------------------------------------------------------------------------
# Cap. 6: fonte visível e vocabulário do Cap. 4 (T2, AD-039)
# ---------------------------------------------------------------------------


def _sem_comentarios(tex: str) -> str:
    return "\n".join(
        linha for linha in tex.splitlines() if not linha.lstrip().startswith("%")
    )


def test_todo_float_gerado_tem_fonte_visivel_dentro_dele(dados, tmp_path):
    """AD-039: a fonte vai impressa sob o float, e não só no cabeçalho `%`."""
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        if caminho.name == "numeros.tex":
            continue
        corpo = _sem_comentarios(caminho.read_text(encoding="utf-8"))
        m = re.search(r"\\begin\{(table|figure)\}(.*?)\\end\{\1\}", corpo, re.S)
        assert m, f"{caminho.name}: sem float"
        dentro = m.group(2)
        assert "\\fonte{Elaboração própria, a partir de \\texttt{pipeline/" in dentro, (
            f"{caminho.name}: float sem \\fonte visível"
        )
        # A fonte vem depois do conteúdo, como pede a NBR 14724.
        fim = max(dentro.rfind("\\end{tabular}"), dentro.rfind("\\end{tikzpicture}"))
        assert fim >= 0 and dentro.index("\\fonte{") > fim, caminho.name


def test_legendas_geradas_usam_o_vocabulario_do_cap4(dados, tmp_path):
    """CAP6-05: Regime A, Regime B e semente, como o Cap. 4 os define."""
    for caminho in report.gerar(REPO_ROOT / "results", tmp_path):
        if caminho.name == "numeros.tex":
            continue
        corpo = _sem_comentarios(caminho.read_text(encoding="utf-8"))
        legenda = " ".join(_palavras_de_legenda(corpo)).lower()
        assert "regime etld+1" not in corpo.lower(), caminho.name
        for proibida in ("seed", "seeds", "repetições", "oficial"):
            assert proibida not in legenda.split(), (
                f"{caminho.name}: legenda diz {proibida!r}"
            )


# ---------------------------------------------------------------------------
# Cap. 6: tab:diagonal por modelo (T3)
# ---------------------------------------------------------------------------


def _linhas_de_modelo(tex: str) -> list[str]:
    rotulos = tuple(v + " &" for v in report.ROTULO_MODELO.values())
    return [ln for ln in tex.splitlines() if ln.startswith(rotulos)]


def test_a_t3_tem_uma_linha_por_modelo_e_dataset(dados):
    tex = report.tabela3_diagonal(dados)
    linhas = _linhas_de_modelo(tex)
    assert len(linhas) == 24
    # ANL-03 (T31) e A2 (TAB-02): 6 colunas mais as duas de média das sementes (A e B)
    # = 7 separadores; a coluna do critério antigo saiu.
    assert all(ln.count(" & ") == 7 for ln in linhas)
    # ANL-03 (T31): com as colunas de média das sementes a tabela passa 115 pt da
    # largura do texto (medido na compilação) e é reduzida por `\resizebox`; sem
    # elas (uma semente só) continua sem redução, como antes.
    assert "\\resizebox{\\textwidth}" in _sem_comentarios(tex)
    assert "\\label{tab:diagonal}" in tex


def test_a_t3_recalcula_uma_linha_de_results(dados):
    """Grambeddings x Reg. logística, do JSON: AUROC A e B, IC e os dois Δ."""
    def reg(base, regime):
        regs = json.loads((base / "matrix.json").read_text(encoding="utf-8"))
        return next(
            r for r in regs
            if r["regime"] == regime and r["diagonal"]
            and r["train_corpus"] == "grambeddings" and r["model"] == "logistic_regression"
        )

    base = REPO_ROOT / "results"
    a, b = reg(base, "oficial"), reg(base, "etld1")
    deltas = []
    for sub in ("", "seed_07", "seed_13"):
        sa, sb = reg(base / sub if sub else base, "oficial"), reg(base / sub if sub else base, "etld1")
        deltas.append(sb["metrics"]["auroc"] - sa["metrics"]["auroc"])
    esperado = (
        f"Reg. logística & {report._num(a['metrics']['auroc'])} & "
        f"{report._num(b['metrics']['auroc'])} & "
        f"{{\\scriptsize ({report._num(b['ci']['auroc_lo'])};\\,"
        f"{report._num(b['ci']['auroc_hi'])})}} & "
        f"{report._num(deltas[0])} {{\\scriptsize {report._faixa(deltas, 4, ';' + chr(92) + ',')}}} & "
    )
    linhas = _linhas_de_modelo(report.tabela3_diagonal(dados))
    assert linhas[0].startswith(esperado), linhas[0]


def test_a_t3_declara_a_saturacao_do_regime_a_por_bloco(dados):
    """O teto de métrica do ISCX tem de aparecer na tabela, contado."""
    registros = dados["matriz_por_seed"][report.SEED_CANONICA]
    tex = report.tabela3_diagonal(dados)
    for corpus in report.ORDEM_CORPORA:
        n = sum(
            1 for r in registros
            if r["regime"] == "oficial" and r["diagonal"] and r["train_corpus"] == corpus
            and r["metrics"]["auroc"] >= report.LIMIAR_SATURACAO
        )
        rotulo = report.ROTULO_CORPUS[corpus]
        bloco = next(ln for ln in tex.splitlines() if f"\\textbf{{{rotulo}}}:" in ln)
        assert f"Regime A em {n} de 6" in bloco, bloco


def test_negativo_que_arredonda_a_zero_sai_sem_sinal():
    """`-0,0000` leria como queda onde as casas impressas não mostram nenhuma."""
    assert report._num(-0.00001) == "0,0000"
    assert report._num(-0.0001) == "-0,0001"
    assert report._num(0.0) == "0,0000"


# ---------------------------------------------------------------------------
# Cap. 6: tab:importancias com 5 postos por dataset (T4)
# ---------------------------------------------------------------------------


def test_a_t7_traz_os_cinco_postos_por_dataset_escapados(dados):
    tex = report.tabela7_importancias(dados)
    imp = dados["importances_por_seed"][report.SEED_CANONICA]
    linhas = tex.splitlines()
    for i in range(1, report.TOP_CARACTERISTICAS + 1):
        linha = next((ln for ln in linhas if ln.startswith(f"{i} & ")), None)
        assert linha is not None, f"posto {i} ausente"
        celulas = linha.removesuffix(" \\\\").split(" & ")[1:]
        assert len(celulas) == len(report.ORDEM_CORPORA)
        for corpus, celula in zip(report.ORDEM_CORPORA, celulas):
            nome = report.top_caracteristicas(imp, "etld1", corpus, i)[i - 1]
            # Nome escapado: `_` cru derruba a compilação fora do modo matemático.
            assert celula == "\\texttt{" + nome.replace("_", "\\_") + "}", (corpus, i, celula)
    assert sum(1 for ln in linhas if ln.startswith("Compart.")) == 1


# ---------------------------------------------------------------------------
# Cap. 6: tab:calibracao (T5)
# ---------------------------------------------------------------------------


def test_a_t9_tem_seis_modelos_e_a_media(dados):
    tex = report.tabela9_calibracao(dados)
    linhas = _linhas_de_modelo(tex)
    assert len(linhas) == 6 and all(ln.count(" & ") == 4 for ln in linhas)
    linhas_tex = tex.splitlines()
    media = [i for i, ln in enumerate(linhas_tex) if ln.startswith("\\textbf{Média} &")]
    assert len(media) == 1
    faixa = linhas_tex[media[0] + 1]
    assert faixa.startswith("\\quad faixa &") and faixa.count("[") == 4, (
        "média sem faixa entre sementes"
    )
    assert "\\label{tab:calibracao}" in tex and "sem\nrecalibração" in tex


def test_a_t9_recalcula_o_ece_fora_da_diagonal_do_catboost(dados):
    regs = _results_json("matrix.json")
    fora = [
        r["metrics"]["ece"] for r in regs
        if r["regime"] == "etld1" and r["model"] == "catboost"
        and r["train_corpus"] != r["test_corpus"]
    ]
    diag = [
        r["metrics"]["brier"] for r in regs
        if r["regime"] == "etld1" and r["model"] == "catboost"
        and r["train_corpus"] == r["test_corpus"]
    ]
    assert len(fora) == 12 and len(diag) == 4
    linha = next(ln for ln in _linhas_de_modelo(report.tabela9_calibracao(dados))
                 if ln.startswith("CatBoost &"))
    celulas = linha.removesuffix(" \\\\").split(" & ")
    assert celulas[1] == report._num(sum(diag) / len(diag))
    assert celulas[4] == report._num(sum(fora) / len(fora))


def test_a_media_da_t9_e_a_das_macros_de_calibracao(dados):
    """A prosa cita \\eceForaDiagonal ao lado da tabela: o mesmo número."""
    macros = _macros_de(report.numeros_prosa(dados))
    media = next(ln for ln in report.tabela9_calibracao(dados).splitlines()
                 if ln.startswith("\\textbf{Média} &"))
    pontos = [c.split(" ")[0] for c in media.removesuffix(" \\\\").split(" & ")[1:]]
    assert pontos == [macros["brierDiagonal"], macros["brierForaDiagonal"],
                      macros["eceDiagonal"], macros["eceForaDiagonal"]]


def test_os_bins_do_ece_na_legenda_sao_os_da_configuracao(dados):
    meta = _results_json("run_meta.json")
    n = meta["config"]["avaliacao"]["ece_bins"]
    assert f"ECE\n({n} \\emph{{bins}} de largura igual)" in report.tabela9_calibracao(dados)


# ---------------------------------------------------------------------------
# Spec 8, Apêndice B: configuração fixa e melhor configuração do HPO
# ---------------------------------------------------------------------------


def test_o_apendice_b1_tem_uma_linha_por_modelo_com_os_parametros_do_run_meta(dados):
    tex = report.apendiceB1_configuracao(dados)
    linhas = _linhas_de_modelo(tex)
    assert len(linhas) == 6
    assert all(ln.count(" & ") == 3 for ln in linhas)
    config = _results_json("run_meta.json")["config"]["modelos"]
    for nome, rotulo in report.ROTULO_MODELO.items():
        linha = next(ln for ln in linhas if ln.startswith(rotulo + " &"))
        params = config[nome]["params"]
        # Um `\texttt` por par chave = valor, na ordem em que o JSON os grava.
        pares = re.findall(r"\\texttt\{([^}]*)\}", linha)
        assert len(pares) == len(params), nome
        for par, (chave, valor) in zip(pares, params.items()):
            assert par.startswith(chave.replace("_", "\\_") + " = "), (nome, par)
        assert (" & sim & " if config[nome]["escala"] else " & não & ") in linha
    assert "\\label{tab:apB-configuracao}" in tex


def test_o_apendice_b1_grava_o_valor_como_o_codigo_o_le(dados):
    tex = report.apendiceB1_configuracao(dados)
    # `null` do JSON é `None` em Python; booleano e tupla idem; real sem arredondar.
    assert "\\texttt{max\\_depth = None}" in tex
    assert "\\texttt{early\\_stopping = True}" in tex
    assert "\\texttt{hidden\\_layer\\_sizes = (128, 64)}" in tex
    assert "\\texttt{tol = 0.0001}" in tex
    assert report._valor_codigo([128]) == "(128,)"
    assert report._valor_codigo(False) == "False"


def test_o_apendice_b2_tem_uma_linha_por_modelo_com_os_parametros_ajustados(dados):
    tex = report.apendiceB2_hpo(dados)
    linhas = _linhas_de_modelo(tex)
    assert len(linhas) == 6
    assert all(ln.count(" & ") == 1 for ln in linhas)
    hpo = _results_json("hpo.json")["modelos"]
    for nome, rotulo in report.ROTULO_MODELO.items():
        linha = next(ln for ln in linhas if ln.startswith(rotulo + " &"))
        params = hpo[nome]["params_ajustados"]
        # Um `\texttt` por par, na ordem em que o JSON grava os parâmetros.
        pares = re.findall(r"\\texttt\{([^}]*)\}", linha)
        assert len(pares) == len(params), nome
        for par, chave in zip(pares, params):
            assert par.startswith(chave.replace("_", "\\_") + " = "), (nome, par)
    assert "\\label{tab:apB-hpo}" in tex


def test_o_apendice_b2_imprime_os_reais_com_quatro_algarismos_significativos(dados):
    tex = report.apendiceB2_hpo(dados)
    hpo = _results_json("hpo.json")["modelos"]
    lr = hpo["xgboost"]["params_ajustados"]["learning_rate"]
    assert f"\\texttt{{learning\\_rate = {lr:.4g}}}" in tex
    # Inteiro fica inteiro; `None` e tupla como o código os lê.
    # Expectativa da rodada consolidada (AD-055, que substitui o AD-010): o HPO
    # escolheu n_estimators = 300 para o RF e o LightGBM (era 200 para o RF).
    assert "\\texttt{n\\_estimators = 300}" in tex
    assert "\\texttt{max\\_depth = None}" in tex
    assert "\\texttt{hidden\\_layer\\_sizes = (256, 128, 64)}" in tex
    # Notação científica do próprio formato `.4g`: tol da regressão logística.
    assert report._valor_ajustado(4.5016922222511204e-05) == "4.502e-05"
    assert report._valor_ajustado(43.35018585095888) == "43.35"
    assert report._valor_ajustado(200) == "200"


def test_a_legenda_do_apendice_b2_traz_corpus_regime_e_tentativas_do_hpo(dados):
    hpo = _results_json("hpo.json")
    legenda = " ".join(_palavras_de_legenda(report.apendiceB2_hpo(dados)))
    assert "Grambeddings" in legenda and "Regime B" in legenda
    assert f"entre {hpo['n_trials']} tentativas" in legenda


# ---------------------------------------------------------------------------
# Spec 8, Apêndice C: tabelas completas da matriz e do LOCO
# ---------------------------------------------------------------------------

_SUBDIR_SEMENTE = {42: "", 7: "seed_07", 13: "seed_13"}
_METRICAS_C = {"auroc": "auroc", "aucpr": "auc_pr", "f1": "f1", "mcc": "mcc"}
_REGIMES_C = {"a": "oficial", "b": "etld1"}
_CORPORA_C = ("grambeddings", "kaggle03", "phishstorm", "iscx2016")
_CELULA_C = re.compile(
    r"^(?:\\textbf\{)?(-?\d+,\d{4})\}? \[(-?\d+,\d{4})(?:--|;\\,)(-?\d+,\d{4})\]$"
)


def _json_da_semente(semente: int, nome: str):
    pasta = _SUBDIR_SEMENTE[semente]
    base = REPO_ROOT / "results" / pasta if pasta else REPO_ROOT / "results"
    return json.loads((base / nome).read_text(encoding="utf-8"))


def _numero(texto: str) -> float:
    return float(texto.replace(",", "."))


def _linhas_c(tex: str, rotulos: tuple[str, ...]) -> list[list[str]]:
    """Linhas de dados como `[rótulo, células...]`, sem a primeira coluna.

    A primeira coluna é o dataset de treino (matriz) ou de teste (LOCO),
    preenchido só na primeira linha de cada bloco; as outras começam em branco.
    A linha de dados é a que tem, no segundo campo, um dos `rotulos`.
    """
    saida = []
    for ln in _sem_comentarios(tex).splitlines():
        campos = ln.rstrip().removesuffix("\\\\").rstrip().split(" & ")
        if len(campos) > 2 and campos[1].removeprefix("\\textbf{").removesuffix("}") in rotulos:
            saida.append(campos[1:])
    return saida


def _matriz_c(dados, regime: str, sufixo: str) -> list[list[str]]:
    tex = report.TABELAS[f"apendiceC-matriz-{regime}-{sufixo}.tex"](dados)
    return _linhas_c(tex, tuple(report.ROTULO_CORPUS[c] for c in _CORPORA_C))


def test_o_apendice_c_tem_dez_arquivos_um_float_cada_com_o_rotulo_esperado(dados):
    rotulos_esperados = {
        f"apendiceC-matriz-{r}-{m}.tex": f"tab:apC-{r}-{m}"
        for r in _REGIMES_C for m in _METRICAS_C
    } | {f"apendiceC-loco-{r}.tex": f"tab:apC-loco-{r}" for r in _REGIMES_C}
    assert set(report.TABELAS_APENDICE_C) == set(rotulos_esperados)
    for nome, rotulo in rotulos_esperados.items():
        tex = report.TABELAS[nome](dados)
        corpo = _sem_comentarios(tex)
        assert corpo.count("\\begin{table}") == 1, nome
        assert corpo.count("\\caption{") == 1, nome
        assert f"\\label{{{rotulo}}}" in corpo, nome
        assert "\\fonte{" in corpo, nome
        assert "\\scriptsize" in corpo, nome
        # Página paisagem: a tabela sai em corpo `\scriptsize`, sem faixa em
        # corpo menor. A matriz só é escalada se passar da largura da página (o
        # MCC passa), e o LOCO nunca é escalado.
        assert "\\tiny" not in corpo, nome
        if "loco" in nome:
            assert "\\resizebox" not in corpo, nome
        else:
            assert corpo.count("\\resizebox") == 1, nome
            assert "\\ifdim\\wd0>\\linewidth\\resizebox{\\linewidth}{!}{\\copy0}" in corpo, nome


def test_a_matriz_do_apendice_c_tem_16_linhas_de_6_celulas_e_a_diagonal_em_negrito(dados):
    for regime in _REGIMES_C:
        for sufixo in _METRICAS_C:
            linhas = _matriz_c(dados, regime, sufixo)
            assert len(linhas) == 16, (regime, sufixo)
            assert all(len(ln) == 7 for ln in linhas), (regime, sufixo)
            em_negrito = [ln for ln in linhas if ln[0].startswith("\\textbf{")]
            assert len(em_negrito) == 4, (regime, sufixo)
            # Só o nome do dataset de teste vai em negrito na diagonal: o valor em
            # negrito alargaria a coluna e a tabela deixaria de caber na página.
            assert not any(c.startswith("\\textbf{") for ln in linhas for c in ln[1:])
            # A diagonal é a linha em que o teste é o dataset do bloco de treino.
            assert [ln[0] for ln in em_negrito] == [
                f"\\textbf{{{report.ROTULO_CORPUS[c]}}}" for c in _CORPORA_C
            ]
            assert [i for i, ln in enumerate(linhas) if ln in em_negrito] == [0, 5, 10, 15]
            # A coluna do treino traz os quatro datasets na ordem, uma vez cada.
            tex = report.TABELAS[f"apendiceC-matriz-{regime}-{sufixo}.tex"](dados)
            treinos = [
                ln.split(" & ")[0]
                for ln in _sem_comentarios(tex).splitlines()
                if ln.count(" & ") == 7 and not ln.startswith("Treino")
            ]
            assert [t for t in treinos if t] == [
                report.ROTULO_CORPUS[c] for c in _CORPORA_C
            ], (regime, sufixo)


_ROTULOS_METRICA_C = ("AUROC", "AUC-PR", "F1", "MCC")


def _loco_c(dados, letra: str) -> list[list[str]]:
    """Linhas do LOCO como `[métrica, 6 células]`: 4 datasets retidos x 4 métricas.

    A forma com uma linha por modelo (24 linhas) não cabe duas por página
    paisagem, e a tabela passou a ter as métricas nas linhas e os modelos nas
    colunas, como a matriz.
    """
    tex = report.TABELAS[f"apendiceC-loco-{letra}.tex"](dados)
    return _linhas_c(tex, _ROTULOS_METRICA_C)


def test_o_loco_do_apendice_c_tem_16_linhas_de_6_celulas_uma_por_modelo(dados):
    for letra in _REGIMES_C:
        linhas = _loco_c(dados, letra)
        assert len(linhas) == 16, letra
        assert all(len(ln) == 7 for ln in linhas), letra
        # Quatro métricas por dataset retido, na ordem do apêndice.
        assert [ln[0] for ln in linhas] == list(_ROTULOS_METRICA_C) * 4, letra
        # Os seis modelos são as colunas, na ordem do artigo.
        cabecalho = report.TABELAS[f"apendiceC-loco-{letra}.tex"](dados)
        assert " & ".join(report.ROTULO_MODELO.values()) in cabecalho, letra


def test_a_celula_da_matriz_do_apendice_c_e_o_valor_da_semente_42_do_json(dados):
    registros = _json_da_semente(42, "matrix.json")
    for letra, regime in _REGIMES_C.items():
        for sufixo, metrica in _METRICAS_C.items():
            linhas = _matriz_c(dados, letra, sufixo)
            por_chave = {
                (r["train_corpus"], r["test_corpus"], r["model"]): r["metrics"][metrica]
                for r in registros if r["regime"] == regime
            }
            i = 0
            for treino in _CORPORA_C:
                for teste in _CORPORA_C:
                    for j, modelo in enumerate(report.ROTULO_MODELO, start=1):
                        ponto = _CELULA_C.match(linhas[i][j]).group(1)
                        esperado = f"{por_chave[(treino, teste, modelo)]:.4f}".replace(".", ",")
                        assert ponto == esperado, (letra, sufixo, treino, teste, modelo)
                    i += 1


def test_a_celula_do_loco_do_apendice_c_e_o_valor_da_semente_42_do_json(dados):
    registros = _json_da_semente(42, "loco.json")
    for letra, regime in _REGIMES_C.items():
        linhas = _loco_c(dados, letra)
        por_chave = {
            (r["test_corpus"], r["model"]): r["metrics"]
            for r in registros if r["regime"] == regime
        }
        i = 0
        for retido in _CORPORA_C:
            for metrica in _METRICAS_C.values():
                for j, modelo in enumerate(report.ROTULO_MODELO, start=1):
                    ponto = _CELULA_C.match(linhas[i][j]).group(1)
                    esperado = f"{por_chave[(retido, modelo)][metrica]:.4f}".replace(".", ",")
                    assert ponto == esperado, (letra, retido, modelo, metrica)
                i += 1


def test_a_faixa_do_apendice_c_contem_o_ponto_e_e_o_minimo_e_o_maximo_das_tres_sementes(dados):
    matrizes = {s: _json_da_semente(s, "matrix.json") for s in _SUBDIR_SEMENTE}
    for letra, regime in _REGIMES_C.items():
        for sufixo, metrica in _METRICAS_C.items():
            por_semente = {
                s: {
                    (r["train_corpus"], r["test_corpus"], r["model"]): r["metrics"][metrica]
                    for r in regs if r["regime"] == regime
                }
                for s, regs in matrizes.items()
            }
            linhas = _matriz_c(dados, letra, sufixo)
            i = 0
            for treino in _CORPORA_C:
                for teste in _CORPORA_C:
                    for j, modelo in enumerate(report.ROTULO_MODELO, start=1):
                        ponto, lo, hi = (
                            _numero(g) for g in _CELULA_C.match(linhas[i][j]).groups()
                        )
                        assert lo <= ponto <= hi, (letra, sufixo, treino, teste, modelo)
                        tres = [por_semente[s][(treino, teste, modelo)] for s in por_semente]
                        assert lo == pytest.approx(min(tres), abs=5.1e-5)
                        assert hi == pytest.approx(max(tres), abs=5.1e-5)
                    i += 1


def test_a_faixa_do_loco_do_apendice_c_contem_o_ponto(dados):
    for letra in _REGIMES_C:
        for ln in _loco_c(dados, letra):
            for celula in ln[1:]:
                ponto, lo, hi = (_numero(g) for g in _CELULA_C.match(celula).groups())
                assert lo <= ponto <= hi, (letra, ln[0])


def test_o_apendice_c_nao_encosta_dois_hifens_no_sinal_de_menos(dados):
    """AD-041: só o MCC tem negativo, e a faixa dele usa `;\\,`, não `--`."""
    for nome, funcao in report.TABELAS_APENDICE_C.items():
        corpo = _sem_comentarios(funcao(dados))
        assert not re.search(r"--\s*-|-\d+,\d+--", corpo), nome
        if "mcc" in nome or "loco" in nome:
            assert ";\\," in corpo, nome
        else:
            assert ";\\," not in corpo, nome
    # O MCC fora da diagonal da matriz tem valor negativo: o caso que o AD-041 trata.
    mcc = report.TABELAS["apendiceC-matriz-a-mcc.tex"](dados) + report.TABELAS[
        "apendiceC-matriz-b-mcc.tex"
    ](dados)
    assert re.search(r"\[-\d+,\d{4};\\,", mcc)


def test_a_legenda_do_apendice_c_diz_regime_metrica_e_faixa_sem_o_jargao_proibido(dados):
    for nome, funcao in report.TABELAS_APENDICE_C.items():
        legenda = " ".join(_palavras_de_legenda(funcao(dados)))
        letra = nome.split("-")[2].removesuffix(".tex")
        assert ("Regime A" if letra == "a" else "Regime B") in legenda, nome
        assert "semente 42" in legenda and "sementes 42, 7 e 13" in legenda, nome
        assert len(legenda.split()) <= TETO_LEGENDA, nome
        for proibida in ("seed", "seeds", "repetições", "oficial"):
            assert proibida not in legenda.split(), (nome, proibida)


# ---------------------------------------------------------------------------
# Spec 8, Apêndice D: execuções e versões das bibliotecas
# ---------------------------------------------------------------------------


def _sem_quebras(tex: str) -> str:
    """O `.tex` sem os `\\allowbreak{}` e sem o `\\texttt{}` em volta de cada valor."""
    return re.sub(r"\\texttt\{([^{}]*)\}", r"\1", tex.replace("\\allowbreak{}", ""))


def _linhas_d1(dados) -> list[list[str]]:
    tex = _sem_quebras(_sem_comentarios(report.apendiceD1_execucao(dados)))
    saida = []
    for ln in tex.splitlines():
        if ln.rstrip().endswith("\\\\") and " & " in ln:
            saida.append(ln.rstrip().removesuffix("\\\\").rstrip().split(" & "))
    return saida


def test_o_apendice_d1_tem_uma_linha_por_execucao_com_os_valores_dos_json(dados):
    linhas = _linhas_d1(dados)
    execucoes = [ln for ln in linhas if ln[0].startswith(("Matriz e LOCO", "Ajuste"))]
    assert len(execucoes) == 4
    for linha, semente in zip(execucoes, (42, 7, 13)):
        meta = _json_da_semente(semente, "run_meta.json")
        assert linha[0] == f"Matriz e LOCO, semente {semente}"
        assert linha[1] == meta["commit"]
        assert linha[2] == ", ".join(meta["run_ids"])
        assert linha[3] == meta["config_fingerprint"]
        assert linha[4] == f"{meta['minutos_ajuste_total']:.1f}".replace(".", ",")
    hpo = _results_json("hpo.json")
    ajuste = execucoes[3]
    assert ajuste[0].startswith("Ajuste de hiperparâmetros")
    assert ajuste[1:] == [
        hpo["commit"], hpo["run_id"], hpo["config_fingerprint"], "não registrado"
    ]


def test_o_apendice_d1_traz_python_plataforma_e_gpu_iguais_nas_execucoes(dados):
    comuns = {ln[0]: ln[1] for ln in _linhas_d1(dados) if ln[0].startswith(("Python", "Plat", "GPU"))}
    metas = [_json_da_semente(s, "run_meta.json") for s in _SUBDIR_SEMENTE]
    assert {m["python"] for m in metas} == {"3.14.2"}
    assert comuns["Python (matriz e LOCO)"].removeprefix("\\multicolumn{4}{l}{").removesuffix("}") == "3.14.2"
    plataforma = metas[0]["plataforma"]
    assert plataforma in comuns["Plataforma (matriz e LOCO)"]
    gpu = metas[0]["gpu"]
    assert gpu in comuns["GPU (todas)"]
    assert _results_json("hpo.json")["gpu"] == gpu


def test_o_hpo_json_nao_grava_tempo_de_busca_e_a_tabela_diz_nao_registrado(dados):
    """Se o `hpo.json` passar a gravar o tempo da busca, a célula tem de usá-lo."""
    hpo = _results_json("hpo.json")
    assert not [k for k in hpo if "minuto" in k or "tempo" in k or "segundos" in k]
    assert "não registrado" in report.apendiceD1_execucao(dados)


def test_o_apendice_d1_recusa_python_diferente_entre_as_execucoes(dados):
    alterado = {**dados, "run_meta_por_seed": {
        s: {**m, "python": "3.14.9" if s == 7 else m["python"]}
        for s, m in dados["run_meta_por_seed"].items()
    }}
    with pytest.raises(report.ReportError, match="python difere"):
        report.apendiceD1_execucao(alterado)


def test_os_identificadores_do_d1_podem_quebrar_linha(dados):
    tex = report.apendiceD1_execucao(dados)
    assert "\\allowbreak{}" in tex
    meta = _json_da_semente(42, "run_meta.json")
    assert report._texttt_quebravel(meta["run_ids"][0]).count("\\allowbreak{}") >= 2
    assert _sem_quebras(report._texttt_quebravel("49869f3-20260805T220020Z")) == (
        "49869f3-20260805T220020Z"
    )


def _pinos_do_arquivo() -> tuple[str, dict]:
    texto = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")
    exigido = re.search(r"^# Python exigido: (.+)$", texto, re.M).group(1).strip()
    pinos = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)$", texto, re.M))
    return exigido, {k.lower(): v for k, v in pinos.items()}


def test_o_apendice_d2_traz_os_pinos_do_requirements_lock_na_ordem_fixa(dados):
    exigido, pinos = _pinos_do_arquivo()
    tex = _sem_quebras(_sem_comentarios(report.apendiceD2_versoes(dados)))
    linhas = [
        ln.rstrip().removesuffix("\\\\").rstrip().split(" & ")
        for ln in tex.splitlines()
        if ln.rstrip().endswith("\\\\") and " & " in ln and not ln.startswith("Componente")
    ]
    # Duas metades lado a lado: a esquerda traz as cinco primeiras entradas, a
    # direita as cinco últimas, na mesma ordem de leitura.
    assert len(linhas) == 5 and all(len(ln) == 4 for ln in linhas)
    entradas = [(ln[0], ln[1]) for ln in linhas] + [(ln[2], ln[3]) for ln in linhas]
    assert entradas[0] == ("Python exigido", exigido)
    assert [nome for nome, _ in entradas[1:]] == [
        "scikit-learn", "XGBoost", "LightGBM", "CatBoost", "Optuna",
        "NumPy", "pandas", "SciPy", "tldextract",
    ]
    for (nome, versao), chave in zip(
        entradas[1:],
        ["scikit-learn", "xgboost", "lightgbm", "catboost", "optuna",
         "numpy", "pandas", "scipy", "tldextract"],
    ):
        assert versao == pinos[chave], nome
    assert "3.14.3" in exigido  # a ressalva do Apêndice D (CAP8-51)


def test_o_apendice_d2_recusa_pino_faltante_e_cabecalho_faltante(tmp_path):
    raiz = tmp_path / "pipeline"
    (raiz / "results").mkdir(parents=True)
    dados = {"results_dir": raiz / "results"}
    completo = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")

    (raiz / "requirements.lock").write_text(
        "\n".join(ln for ln in completo.splitlines() if not ln.startswith("scipy==")),
        encoding="utf-8",
    )
    with pytest.raises(report.ReportError, match="scipy"):
        report.apendiceD2_versoes(dados)

    (raiz / "requirements.lock").write_text(
        "\n".join(ln for ln in completo.splitlines() if "Python exigido" not in ln),
        encoding="utf-8",
    )
    with pytest.raises(report.ReportError, match="Python exigido"):
        report.apendiceD2_versoes(dados)

    (raiz / "requirements.lock").unlink()
    with pytest.raises(report.ReportError, match="requirements.lock"):
        report.apendiceD2_versoes(dados)
