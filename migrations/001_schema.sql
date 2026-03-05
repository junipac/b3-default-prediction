-- =============================================================================
-- Schema Principal do Sistema de Ingestão B3/CVM
-- Versão: 001
-- Descrição: Criação das tabelas base com suporte a:
--   - Versionamento de registros
--   - Auditoria imutável
--   - Controle de survivorship bias
--   - Rastreabilidade completa de dados
-- =============================================================================

-- Extensões necessárias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- busca textual em nomes de empresa
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- índices compostos eficientes

-- Schemas separados por domínio
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS financials;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS defaults;

-- =============================================================================
-- CADASTRO DE EMPRESAS
-- Inclui TODAS as empresas: ativas, canceladas, fundidas
-- CRÍTICO para evitar survivorship bias
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.company_register (
    id                  BIGSERIAL PRIMARY KEY,
    cnpj                CHAR(14) NOT NULL,
    cnpj_root           CHAR(8)  NOT NULL,            -- primeiros 8 dígitos do CNPJ
    company_name        TEXT     NOT NULL,
    trade_name          TEXT,
    cvm_code            TEXT,                          -- código CVM (CD_CVM)
    registration_date   DATE,
    constitution_date   DATE,
    cancellation_date   DATE,                          -- NULL = ainda ativa
    cancellation_reason TEXT,
    status              TEXT     NOT NULL DEFAULT 'ATIVO',
    status_since        DATE,
    sector              TEXT,
    market_type         TEXT,
    registration_category TEXT,
    issuer_status       TEXT,
    is_active           BOOLEAN  NOT NULL DEFAULT TRUE,
    is_cancelled        BOOLEAN  NOT NULL DEFAULT FALSE,
    city                TEXT,
    state               CHAR(2),
    country             TEXT     DEFAULT 'BR',
    email               TEXT,
    phone               TEXT,
    -- Rastreabilidade
    source              TEXT     NOT NULL DEFAULT 'cvm_cadastro',
    _updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _row_hash           TEXT,
    CONSTRAINT company_register_cnpj_uq UNIQUE (cnpj)
);

-- Histórico de razão social (para reconciliação de CNPJ com nome)
CREATE TABLE IF NOT EXISTS public.company_name_history (
    id              BIGSERIAL PRIMARY KEY,
    cnpj            CHAR(14) NOT NULL,
    company_name    TEXT     NOT NULL,
    valid_from      DATE     NOT NULL,
    valid_to        DATE,                    -- NULL = atual
    source          TEXT,
    _created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (cnpj) REFERENCES public.company_register(cnpj)
        ON DELETE RESTRICT
);

-- Histórico de tickers (crítico: tickers são reutilizados pela B3)
CREATE TABLE IF NOT EXISTS public.ticker_history (
    id              BIGSERIAL PRIMARY KEY,
    cnpj            CHAR(14),
    old_ticker      TEXT     NOT NULL,
    new_ticker      TEXT,
    change_date     DATE,
    reason          TEXT,
    source          TEXT     NOT NULL DEFAULT 'b3',
    _created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- DADOS DE MERCADO
-- =============================================================================

CREATE TABLE IF NOT EXISTS market.daily (
    id                  BIGSERIAL PRIMARY KEY,
    ticker              TEXT     NOT NULL,
    date                DATE     NOT NULL,
    codbdi              TEXT,
    company_name        TEXT,
    spec                TEXT,
    open                NUMERIC(18, 6),
    high                NUMERIC(18, 6),
    low                 NUMERIC(18, 6),
    avg                 NUMERIC(18, 6),
    close               NUMERIC(18, 6),
    trades              BIGINT,
    volume_shares       BIGINT,
    volume_financial    NUMERIC(22, 2),
    isin                TEXT,
    factor              INT      DEFAULT 1,
    suspended           BOOLEAN  DEFAULT FALSE,  -- dia sem negociação
    daily_return        NUMERIC(12, 8),
    -- Rastreabilidade
    source              TEXT     NOT NULL DEFAULT 'b3_cotahist',
    source_file_sha256  TEXT,
    _updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _row_hash           TEXT,
    CONSTRAINT market_daily_ticker_date_uq UNIQUE (ticker, date)
);

-- Eventos corporativos (proventos, splits, grupamentos)
CREATE TABLE IF NOT EXISTS market.corporate_events (
    id                  BIGSERIAL PRIMARY KEY,
    ticker              TEXT     NOT NULL,
    cnpj                CHAR(14),
    company_name        TEXT,
    event_type          TEXT     NOT NULL,  -- DIVIDENDO, JCP, BONIFICACAO, DESDOBRAMENTO, etc.
    ex_date             DATE,
    payment_date        DATE,
    value_per_share     NUMERIC(18, 8),
    currency            CHAR(3)  DEFAULT 'BRL',
    ratio               NUMERIC(10, 6),     -- para splits/grupamentos
    asset_issued        TEXT,
    remarks             TEXT,
    source              TEXT     NOT NULL DEFAULT 'b3',
    _updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT corporate_events_uq UNIQUE (ticker, event_type, ex_date)
);

-- =============================================================================
-- DEMONSTRAÇÕES FINANCEIRAS — DADOS BRUTOS NORMALIZADOS
-- =============================================================================

-- Tabela genérica para todas as contas (DFP + ITR)
CREATE TABLE IF NOT EXISTS financials.accounts (
    id                  BIGSERIAL PRIMARY KEY,
    cnpj_cia            CHAR(14)  NOT NULL,
    reference_date      DATE      NOT NULL,
    version             INT       NOT NULL DEFAULT 1,
    is_latest_version   BOOLEAN   NOT NULL DEFAULT TRUE,
    doc_type            TEXT      NOT NULL,   -- DFP, ITR
    report_group        TEXT,                 -- DFP_CIA_ABERTA, etc.
    period_type         TEXT,                 -- ÚLTIMO, PENÚLTIMO
    period_start        DATE,
    period_end          DATE,
    consolidation_type  TEXT,                 -- con, ind
    account_code        TEXT      NOT NULL,
    account_standard    TEXT,                 -- mapeamento interno padronizado
    account_description TEXT,
    value_raw           NUMERIC(22, 2),       -- valor original na escala CVM
    value_scaled        NUMERIC(22, 2),       -- valor em R$ absolutos
    currency            CHAR(3)   DEFAULT 'BRL',
    scale               TEXT,
    fixed_account       TEXT,
    source_year         INT,
    source_file         TEXT,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT financials_accounts_uq
        UNIQUE (cnpj_cia, reference_date, version, doc_type, consolidation_type, account_code)
);

-- Dataset analítico — um registro por empresa/período
CREATE TABLE IF NOT EXISTS financials.analytical (
    id                      BIGSERIAL PRIMARY KEY,
    cnpj_cia                CHAR(14)  NOT NULL,
    reference_date          DATE      NOT NULL,
    company_name            TEXT,
    cvm_code                TEXT,
    doc_type                TEXT,
    -- Balanço Patrimonial
    total_assets            NUMERIC(22, 2),
    current_assets          NUMERIC(22, 2),
    cash_equivalents        NUMERIC(22, 2),
    inventories             NUMERIC(22, 2),
    receivables             NUMERIC(22, 2),
    non_current_assets      NUMERIC(22, 2),
    ppe                     NUMERIC(22, 2),
    intangibles             NUMERIC(22, 2),
    investments             NUMERIC(22, 2),
    current_liabilities     NUMERIC(22, 2),
    short_term_debt         NUMERIC(22, 2),
    non_current_liabilities NUMERIC(22, 2),
    long_term_debt          NUMERIC(22, 2),
    equity                  NUMERIC(22, 2),
    paid_in_capital         NUMERIC(22, 2),
    retained_earnings       NUMERIC(22, 2),
    total_debt              NUMERIC(22, 2),
    net_debt                NUMERIC(22, 2),
    -- DRE
    net_revenue             NUMERIC(22, 2),
    cost_of_goods_sold      NUMERIC(22, 2),
    gross_profit            NUMERIC(22, 2),
    operating_expenses      NUMERIC(22, 2),
    ebit                    NUMERIC(22, 2),
    financial_result        NUMERIC(22, 2),
    ebt                     NUMERIC(22, 2),
    income_tax              NUMERIC(22, 2),
    net_income              NUMERIC(22, 2),
    -- DFC
    cfo                     NUMERIC(22, 2),
    cfi                     NUMERIC(22, 2),
    cff                     NUMERIC(22, 2),
    -- Indicadores Calculados
    current_ratio           NUMERIC(10, 6),
    quick_ratio             NUMERIC(10, 6),
    cash_ratio              NUMERIC(10, 6),
    debt_to_equity          NUMERIC(10, 6),
    net_debt_ebitda         NUMERIC(10, 6),
    ebitda_approx           NUMERIC(22, 2),
    ebitda_margin           NUMERIC(10, 6),
    gross_margin            NUMERIC(10, 6),
    ebit_margin             NUMERIC(10, 6),
    net_margin              NUMERIC(10, 6),
    roe                     NUMERIC(10, 6),
    roa                     NUMERIC(10, 6),
    roic_approx             NUMERIC(10, 6),
    -- Altman Z-Score
    altman_zscore           NUMERIC(10, 6),
    altman_zone             TEXT,
    -- Data Quality
    completeness_score      NUMERIC(5, 4),
    missing_pct             NUMERIC(5, 4),
    bs_balanced             BOOLEAN,
    bs_check_pct            NUMERIC(10, 6),
    -- Rastreabilidade
    _updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _row_hash               TEXT,
    CONSTRAINT financials_analytical_uq UNIQUE (cnpj_cia, reference_date, doc_type)
);

-- =============================================================================
-- DETECÇÃO DE DEFAULT
-- =============================================================================

CREATE TABLE IF NOT EXISTS defaults.events (
    id              BIGSERIAL PRIMARY KEY,
    cnpj            CHAR(14)  NOT NULL,
    ticker          TEXT,
    company_name    TEXT,
    default_flag    SMALLINT  NOT NULL DEFAULT 0,
    event_type      TEXT,
    event_date      DATE,
    source          TEXT,
    description     TEXT,
    confidence      NUMERIC(5, 4),
    raw_text        TEXT,
    -- Score de distress pré-evento
    distress_score  NUMERIC(5, 4),
    distress_zone   TEXT,
    altman_zscore   NUMERIC(10, 6),
    current_ratio   NUMERIC(10, 6),
    net_debt_ebitda NUMERIC(10, 6),
    distress_alert  SMALLINT  DEFAULT 0,
    identified_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT defaults_events_cnpj_uq UNIQUE (cnpj)
);

-- =============================================================================
-- AUDITORIA IMUTÁVEL
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit.quality_reports (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT      NOT NULL,
    dataset         TEXT      NOT NULL,
    started_at      TIMESTAMPTZ,
    total_records   BIGINT,
    total_companies BIGINT,
    n_errors        INT,
    n_warnings      INT,
    overall_score   NUMERIC(5, 4),
    passed          BOOLEAN,
    details         JSONB,
    _created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT      NOT NULL UNIQUE,
    pipeline_type       TEXT      NOT NULL,
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    status              TEXT,
    records_processed   BIGINT    DEFAULT 0,
    companies_processed BIGINT    DEFAULT 0,
    n_errors            INT       DEFAULT 0,
    n_warnings          INT       DEFAULT 0,
    quality_score       NUMERIC(5, 4),
    schema_drifts       INT       DEFAULT 0,
    _created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabela de mudanças estruturais detectadas
CREATE TABLE IF NOT EXISTS audit.schema_drift (
    id              BIGSERIAL PRIMARY KEY,
    schema_key      TEXT      NOT NULL,
    detected_at     TIMESTAMPTZ NOT NULL,
    columns_added   JSONB,
    columns_removed JSONB,
    previous_count  INT,
    current_count   INT,
    source          TEXT,
    _created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- DATASET FINAL CONSOLIDADO (view para modelo de PD)
-- =============================================================================

CREATE OR REPLACE VIEW public.pd_dataset AS
SELECT
    cr.cnpj,
    cr.cnpj_root,
    cr.company_name,
    cr.sector,
    cr.is_active,
    cr.is_cancelled,
    cr.cancellation_date,
    fa.reference_date,
    fa.doc_type,
    -- Indicadores de mercado (mais recente disponível)
    -- (join com market.daily seria via subquery separada)
    -- Indicadores Financeiros
    fa.total_assets,
    fa.equity,
    fa.net_revenue,
    fa.net_income,
    fa.ebit,
    fa.ebitda_approx,
    fa.total_debt,
    fa.net_debt,
    fa.current_ratio,
    fa.debt_to_equity,
    fa.net_debt_ebitda,
    fa.ebitda_margin,
    fa.gross_margin,
    fa.net_margin,
    fa.roe,
    fa.roa,
    fa.altman_zscore,
    fa.altman_zone,
    fa.completeness_score,
    fa.missing_pct,
    -- Default
    COALESCE(de.default_flag, 0) AS default_flag,
    de.event_type              AS default_event_type,
    de.event_date              AS default_event_date,
    de.distress_score,
    de.distress_zone,
    de.distress_alert,
    -- Rastreabilidade
    fa._updated_at             AS financial_updated_at,
    'cvm+b3'                   AS source
FROM
    public.company_register cr
    LEFT JOIN financials.analytical fa
           ON fa.cnpj_cia = cr.cnpj
    LEFT JOIN defaults.events de
           ON de.cnpj = cr.cnpj;

-- Comentário
COMMENT ON VIEW public.pd_dataset IS
'Dataset consolidado para modelagem de Probabilidade de Default (PD). '
'Inclui TODAS as empresas (ativas e deslistadas) para evitar survivorship bias. '
'Cada linha = empresa + período de referência financeira.';

-- =============================================================================
-- ÍNDICES DE PERFORMANCE
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_market_daily_ticker_date
    ON market.daily (ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_market_daily_date
    ON market.daily (date DESC);

CREATE INDEX IF NOT EXISTS idx_financials_accounts_cnpj_ref
    ON financials.accounts (cnpj_cia, reference_date DESC);

CREATE INDEX IF NOT EXISTS idx_financials_analytical_cnpj_ref
    ON financials.analytical (cnpj_cia, reference_date DESC);

CREATE INDEX IF NOT EXISTS idx_company_register_cnpj_root
    ON public.company_register (cnpj_root);

CREATE INDEX IF NOT EXISTS idx_defaults_events_cnpj
    ON defaults.events (cnpj);

CREATE INDEX IF NOT EXISTS idx_company_register_status
    ON public.company_register (is_active, is_cancelled);

-- Índice GIN para busca textual em nomes de empresa
CREATE INDEX IF NOT EXISTS idx_company_name_trgm
    ON public.company_register USING GIN (company_name gin_trgm_ops);

-- =============================================================================
-- CONTROLE DE ACESSO POR PERFIL
-- =============================================================================

-- Role somente leitura
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'b3_readonly') THEN
        CREATE ROLE b3_readonly;
    END IF;
END $$;

GRANT USAGE ON SCHEMA public, market, financials, defaults TO b3_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO b3_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA market TO b3_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA financials TO b3_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA defaults TO b3_readonly;
-- Sem acesso a audit (logs imutáveis protegidos)

-- Role de analista
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'b3_analyst') THEN
        CREATE ROLE b3_analyst;
    END IF;
END $$;

GRANT b3_readonly TO b3_analyst;
-- Analista pode inserir na fila de reprocessamento (tabela específica)

-- Role de admin (acesso total para o pipeline)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'b3_admin') THEN
        CREATE ROLE b3_admin;
    END IF;
END $$;

GRANT ALL ON SCHEMA public, market, financials, defaults, audit TO b3_admin;
GRANT ALL ON ALL TABLES IN SCHEMA public TO b3_admin;
GRANT ALL ON ALL TABLES IN SCHEMA market TO b3_admin;
GRANT ALL ON ALL TABLES IN SCHEMA financials TO b3_admin;
GRANT ALL ON ALL TABLES IN SCHEMA defaults TO b3_admin;
GRANT ALL ON ALL TABLES IN SCHEMA audit TO b3_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO b3_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA market TO b3_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA financials TO b3_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA defaults TO b3_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA audit TO b3_admin;
