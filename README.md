# tcc-phishing-pipeline

Código e resultados do *pipeline* da monografia de TCC II "Detecção de Phishing por Características Léxicas de URL: Comparação de Modelos Tradicionais sob Avaliação Livre de Vazamento", do curso de Sistemas de Informação da Universidade Federal de Viçosa (UFV), *Campus* Rio Paranaíba. Autores: Julio Cezar Lopes Cardoso e Lucas Souza e Silva. Orientador: Pedro Henrique Aparecido Damaso de Melo.

O *pipeline* carrega seis *datasets* de URLs, extrai 41 características léxicas, treina seis modelos tradicionais de aprendizado de máquina em dois regimes de partição (um que respeita o corte de referência de cada *dataset* e outro que agrupa as URLs por eTLD+1) e grava as métricas em `pipeline/results/`. A monografia descreve o método, o código e os resultados.

## Identidade da rodada

Os resultados deste repositório são os da rodada consolidada de novembro de 2026. A rodada é identificada pelo *commit* de código `a56dab7` (`a56dab79264566926a2ae6af5c58216cebf527b1`), gravado como `commit_codigo` em `pipeline/results/run_meta.json`, e pela tag `rodada-consolidada-2026-11`. Esse *commit* pertence ao repositório de desenvolvimento do trabalho, que é privado. Este repositório não tem o histórico dele: o *commit* inicial reproduz a árvore de `pipeline/` do repositório de desenvolvimento, e a tag marca esse *commit* inicial.

Os arquivos de `pipeline/src/`, `pipeline/scripts/` e `pipeline/config/` e o `pipeline/requirements.lock` são idênticos, byte a byte, aos do *commit* `a56dab7`, com três exceções: `src/report.py`, `src/analises.py` e `src/diagnostics.py`. Esses três módulos só leem `pipeline/results/` ou os *datasets* e não treinam modelos; mudaram depois da rodada, sem alterar o que foi treinado. O `run_id` de cada registro de resultado começa pelo *commit* do repositório de desenvolvimento no momento da execução, que difere do `commit_codigo` apenas em arquivos fora da identidade da rodada.

## Estrutura

```
pipeline/
  config/        experiment.yml, features.json, lista de sufixos públicos (psl/)
  corpora/       manifest.json e fontes_brutas.json (sem os dados)
  results/       métricas, curvas, diagnósticos e metadados da rodada
  scripts/       run_all.py, run_hpo.py, run_smoke.py, reconstruir_corpora.py e auxiliares
  src/           módulos do pipeline
  tests/         suíte de testes
```

## Datasets

Os *datasets* não estão neste repositório. Cada um segue a licença da sua fonte, e a redistribuição não foi verificada.

- ISCX-URL2016, PhishStorm, PDRCNN, Kaggle-01 e Kaggle-03: os CSVs vêm do repositório do GramBeddings (`https://github.com/fcdalgic/GramBeddings`, pasta `data/`, *commit* `a47f863`). `pipeline/corpora/fontes_brutas.json` registra a URL e o sha256 de cada CSV. A partir de `pipeline/`, `python scripts/reconstruir_corpora.py --baixar` baixa os arquivos, confere o sha256 de cada um e reconstrói os *datasets*.
- Grambeddings: a pasta `data/grambeddings` do *commit* `a47f863` só tem um `ReadMe.md`, que remete a `https://web.cs.hacettepe.edu.tr/~selman/grambeddings-dataset/`. Os dados vêm de um arquivo protegido por senha baixado desse site e ficam fora do script de reconstrução. `pipeline/corpora/manifest.json` traz o sha256 do arquivo usado e as contagens de cada *dataset*.

Os testes que leem os *datasets* precisam deles em `pipeline/corpora/<nome>/data.parquet`.

## Predições da rodada

Os escores por amostra e as partições (`pipeline/results/**/predicoes/`) não estão no repositório, por causa do tamanho. Ficam como anexos da *release* `rodada-consolidada-2026-11` deste repositório, com o `SHA256SUMS.txt` dos arquivos e o `LEIAME-PREDICOES.md`, que descreve cada um. Para restaurar, na tag `rodada-consolidada-2026-11`:

```bash
cd pipeline/results
for z in predicoes-*.zip; do unzip -o "$z"; done
```

Antes, confira os arquivos baixados com `sha256sum -c SHA256SUMS.txt`. Com as predições restauradas, `python -m src.offline` (a partir de `pipeline/`) refaz a filtragem da sobreposição e o *bootstrap* pareado sem treinar de novo.

## Ambiente e execução

O *pipeline* exige Python 3.14. O ambiente fixado está em `pipeline/requirements.lock`:

```bash
cd pipeline
python3.14 -m venv venv
venv/bin/python -m pip install -r requirements.lock
venv/bin/python scripts/verify_environment.py
venv/bin/python -m pytest tests/
```

No Windows, o interpretador do ambiente virtual fica em `venv\Scripts\python.exe`.

A rodada consolidada foi treinada numa máquina com GPU NVIDIA. XGBoost e CatBoost usam a GPU quando ela está disponível, e os outros quatro modelos rodam na CPU. Com GPU, os resultados do CatBoost não se repetem bit a bit entre execuções, e a monografia mede essa variação.

## Licença

O código e os resultados deste repositório estão sob a licença MIT (arquivo `LICENSE`). Os *datasets* seguem as licenças das fontes originais.
