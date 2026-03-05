# B3 Default Prediction

Pipeline de ingestão, processamento e análise de dados da B3 e CVM para modelagem de **Probabilidade de Default (PD)** de empresas listadas, utilizando o modelo estrutural de Merton e indicadores de distress financeiro.

---

## Visão Geral

O sistema coleta dados públicos de mercado (B3) e demonstrações financeiras (CVM), aplica um pipeline de qualidade e normalizaçao, e calcula a PD por empresa usando o modelo de Merton. Também detecta eventos de default observados (recuperação judicial, suspensão prolongada, queda extrema de preço) para validação e backtesting.

**Fontes de dados:**
- **B3** — cotações históricas (COTAHIST), eventos corporativos (dividendos, splits, suspensões)
- **CVM** — DFP (demonstrações anuais auditadas), ITR (trimestrais), FRE, cadastro de companhias, fatos relevantes

---

## Arquitetura do Pipeline

```
Extractors  →  Parsers  →  Storage  →  Quality  →  Models  →  Analytics
   B3/CVM       Norm.      Postgres    Validação    Merton     PD / SHI
                           Parquet     & Scoring    Default    Relatórios
```

### Módulos

| Módulo | Descrição |
|---|---|
| `src/extractors/` | Ingestão de dados B3 e CVM com retry, rate limiting e versionamento |
| `src/parsers/` | Normalização e padronização de demonstrações financeiras |
| `src/storage/` | Persistência em PostgreSQL (upsert versionado) e Parquet particionado |
| `src/quality/` | Validação de consistência, completude e anomalias |
| `src/default_detection/` | Detecção de eventos de default via múltiplas fontes |
| `src/models/` | Modelo estrutural de Merton para cálculo de PD |
| `src/analytics/` | Agregação por setor, índice de saúde setorial, validação de modelos |
| `src/pipeline/` | Orquestração e controle de execução dos pipelines |
| `src/utils/` | Logging estruturado, retry exponencial, rate limiter |

---

## Modelo de Merton

O modelo de Merton trata o capital da empresa como uma opção de compra europeia sobre seus ativos. O default ocorre quando o valor dos ativos cai abaixo do valor da dívida no vencimento.

**Equações centrais:**

```
E  = V · Φ(d1) − D · e^(−rT) · Φ(d2)
d1 = [ln(V/D) + (μ + 0.5σV²)T] / (σV√T)
d2 = d1 − σV√T

DD = [ln(V/D) + (μ − 0.5σV²)T] / (σV√T)
PD = Φ(−DD)
```

Onde `V` = valor dos ativos, `D` = barreira de dívida, `σV` = volatilidade dos ativos, `T` = horizonte temporal.

O sistema resolve o sistema não-linear iterativamente até convergência (tolerância 1e-8) e suporta três modos de drift: `RISK_NEUTRAL`, `HISTORICAL` e `CONSERVATIVE`.

**Classificação de rating:**

| Rating | PD |
|---|---|
| AAA | < 0.01% |
| AA | 0.01% – 0.05% |
| A | 0.05% – 0.10% |
| BBB | 0.10% – 0.50% |
| BB | 0.50% – 2.00% |
| B | 2.00% – 8.00% |
| CCC | 8.00% – 20.00% |
| CC | 20.00% – 50.00% |
| D | > 50.00% |

> Empresas do setor financeiro (bancos, seguradoras) recebem ajuste na barreira de dívida (`FINANCIAL_DEBT_BARRIER_FACTOR = 0.20`) para evitar distorções estruturais.

---

## Detecção de Eventos de Default

O `DefaultDetector` classifica eventos por cinco critérios:

| Critério | Descrição |
|---|---|
| Recuperação Judicial | Palavras-chave no cadastro/fatos relevantes CVM |
| Cancelamento de Registro | Cancelamento associado a keywords de insolvência |
| Suspensão Prolongada | Ausência de negociação por > 30 dias |
| Queda Extrema de Preço | Queda ≤ −80% em janela de 252 dias |
| Distress Score | Altman Z-score < 2.99 com score composto > 0.7 |

Cada evento recebe um `confidence` (0.0–1.0) e é armazenado em `defaults.events`.

---

## Pré-requisitos

- Python 3.10+
- PostgreSQL 14+
- Chromium (para Playwright, opcional)

---

## Instalação

```bash
git clone https://github.com/junipac/b3-default-prediction.git
cd b3-default-prediction

pip install -r requirements.txt
python setup.py develop

cp .env.example .env
# edite .env com suas credenciais
```

Criar o schema do banco de dados:

```bash
psql -U b3_user -d b3_ingestion -f migrations/001_schema.sql
```

---

## Configuração

Todas as variáveis são carregadas via `.env`. Copie `.env.example` como base.

| Variável | Padrão | Descrição |
|---|---|---|
| `DB_HOST` | `localhost` | Host do PostgreSQL |
| `DB_PORT` | `5432` | Porta |
| `DB_NAME` | `b3_ingestion` | Nome do banco |
| `DB_USER` | `b3_user` | Usuário |
| `DB_PASSWORD` | — | Senha (obrigatória) |
| `RETRY_MAX_ATTEMPTS` | `5` | Tentativas máximas por requisição |
| `B3_RPM` | `30` | Requisições/min para B3 |
| `CVM_RPM` | `60` | Requisições/min para CVM |
| `MIN_COMPLETENESS` | `0.75` | Score mínimo de completude aceito |
| `DISTRESS_THRESHOLD` | `0.70` | Limiar para classificar distress |
| `LOG_LEVEL` | `INFO` | Nível de logging |

---

## Uso

### Pipeline principal (CLI)

```bash
# Ingestão diária: cotações B3 + eventos corporativos
python main.py daily

# Pipeline trimestral: ITR (demonstrações intermediárias)
python main.py quarterly --year 2024

# Pipeline anual: DFP + detecção de defaults
python main.py annual --year 2023

# Verificar disponibilidade das fontes
python main.py validate

# Relatório de inconsistências das últimas execuções
python main.py report

# Reprocessar revisões de uma empresa
python main.py reprocess --cnpj 33000167000101
```

### Relatórios analíticos

```bash
# Relatório de PD por setor e rating (CSV + HTML)
python generate_report.py

# Relatório Merton completo com visualizações
python generate_merton_report.py

# Demo ao vivo com dados reais da CVM (sem banco de dados)
python demo_real_data.py
```

Os relatórios são gerados em `data/reports/`.

---

## Estrutura do Projeto

```
b3-default-prediction/
├── config/
│   └── settings.py          # Configurações centralizadas (carrega .env)
├── data/
│   ├── raw/                 # Dados brutos versionados (ignorado no git)
│   ├── processed/           # Datasets processados
│   │   └── pd_dataset_sample.csv  # Amostra dos dados
│   └── reports/             # Relatórios gerados (ignorado no git)
├── migrations/
│   └── 001_schema.sql       # Schema completo do PostgreSQL
├── src/
│   ├── analytics/           # Agregação, SHI, validação de modelos
│   ├── default_detection/   # Detecção de eventos de default
│   ├── extractors/          # Ingestão B3 e CVM
│   ├── models/              # Modelo de Merton
│   ├── parsers/             # Normalização de demonstrações financeiras
│   ├── pipeline/            # Orquestração
│   ├── quality/             # Validação de dados
│   ├── storage/             # PostgreSQL e Parquet
│   └── utils/               # Logging, retry, rate limiter
├── tests/                   # Testes unitários e de integração
├── main.py                  # CLI principal
├── generate_report.py       # Relatório analítico de PD
├── generate_merton_report.py# Relatório completo Merton
├── demo_real_data.py        # Demo sem banco de dados
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Banco de Dados

O schema PostgreSQL é dividido em cinco schemas lógicos:

| Schema | Tabelas principais |
|---|---|
| `public` | `company_register`, `company_name_history`, `ticker_history` |
| `market` | `daily` (OHLCV), `corporate_events` |
| `financials` | `accounts` (dados normalizados), `analytical` (indicadores derivados) |
| `defaults` | `events`, `merton_output` |
| `audit` | `pipeline_runs` |

**Princípios de design:**
- Dados brutos imutáveis, versionados por hash SHA-256
- Anti-survivorship bias: inclui empresas canceladas
- Upsert com rastreamento de versão (`_updated_at`, `_row_hash`)
- Log de auditoria com retenção de 10 anos

---

## Testes

```bash
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html
```

| Arquivo | Cobertura |
|---|---|
| `test_extractors.py` | Parsing de ZIP, encoding, rate limiting, schema drift |
| `test_quality.py` | Reconciliação de balanço, completude, outliers |
| `test_default_detector.py` | Keywords RJ, thresholds de preço e suspensão |
| `test_merton.py` | Convergência, bounds de PD, efeitos de alavancagem |
| `test_analytics.py` | Agregação setorial, índice de saúde, backtesting |

---

## Dados de Exemplo

O arquivo `data/processed/pd_dataset_sample.csv` contém uma amostra dos dados processados para exploração sem necessidade de executar o pipeline completo.

---

## Aviso Legal

Este projeto utiliza exclusivamente dados públicos disponibilizados pela B3 e CVM. Os resultados não constituem recomendação de investimento.
