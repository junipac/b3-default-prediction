"""
Configurações centralizadas do sistema de ingestão B3/CVM.
Todas as variáveis de ambiente são carregadas via .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ─── Banco de Dados ────────────────────────────────────────────────────────────
DATABASE = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "name": os.getenv("DB_NAME", "b3_ingestion"),
    "user": os.getenv("DB_USER", "b3_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "pool_size": int(os.getenv("DB_POOL_SIZE", 10)),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 20)),
}

DATABASE_URL = (
    f"postgresql+psycopg2://{DATABASE['user']}:{DATABASE['password']}"
    f"@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['name']}"
)

# ─── Storage ──────────────────────────────────────────────────────────────────
STORAGE = {
    "raw_dir": BASE_DIR / "data" / "raw",
    "processed_dir": BASE_DIR / "data" / "processed",
    "parquet_dir": BASE_DIR / "data" / "parquet",
    "max_raw_versions": int(os.getenv("MAX_RAW_VERSIONS", 30)),
}

# ─── Fontes de Dados ──────────────────────────────────────────────────────────
SOURCES = {
    "b3": {
        "base_url": "https://sistemaswebb3-listados.b3.com.br",
        "market_data_url": "https://bvmf.bmfbovespa.com.br/InstDados/SerHist",
        "cotacoes_url": "https://bvmf.bmfbovespa.com.br/Cotacoes2000",
        "instruments_url": "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetInitialCompanies",
        "events_url": "https://sistemaswebb3-listados.b3.com.br/corporateEventsProxy",
        "timeout": 60,
        "enabled": True,
    },
    "cvm": {
        "base_url": "https://dados.cvm.gov.br",
        "dfp_url": "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS",
        "itr_url": "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS",
        "fre_url": "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS",
        "rad_url": "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD",
        "fatos_relevantes_url": "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FATO_RELEVANTE/DADOS",
        "timeout": 120,
        "enabled": True,
    },
    "diario_oficial": {
        "base_url": "https://www.in.gov.br",
        "search_url": "https://www.in.gov.br/consulta/-/buscar/dou",
        "timeout": 45,
        "enabled": True,
    },
}

# ─── Retry & Rate Limiting ────────────────────────────────────────────────────
RETRY = {
    "max_attempts": int(os.getenv("RETRY_MAX_ATTEMPTS", 5)),
    "initial_wait": float(os.getenv("RETRY_INITIAL_WAIT", 2.0)),
    "max_wait": float(os.getenv("RETRY_MAX_WAIT", 120.0)),
    "multiplier": float(os.getenv("RETRY_MULTIPLIER", 2.0)),
    "jitter": True,
}

RATE_LIMIT = {
    "b3_requests_per_minute": int(os.getenv("B3_RPM", 30)),
    "cvm_requests_per_minute": int(os.getenv("CVM_RPM", 60)),
    "diario_oficial_requests_per_minute": int(os.getenv("DOU_RPM", 10)),
    "concurrent_requests": int(os.getenv("CONCURRENT_REQUESTS", 5)),
}

# ─── User Agents (rotação) ────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
    "Gecko/20100101 Firefox/124.0",
]

# ─── Data Quality Thresholds ──────────────────────────────────────────────────
QUALITY = {
    "min_completeness_score": float(os.getenv("MIN_COMPLETENESS", 0.75)),
    "max_missing_pct": float(os.getenv("MAX_MISSING_PCT", 0.25)),
    "balance_sheet_tolerance": float(os.getenv("BS_TOLERANCE", 0.01)),
    "price_spike_threshold": float(os.getenv("PRICE_SPIKE_THRESHOLD", 0.50)),
    "min_trading_days_per_year": int(os.getenv("MIN_TRADING_DAYS", 180)),
}

# ─── Default Detection ────────────────────────────────────────────────────────
DEFAULT_DETECTION = {
    "suspension_days_threshold": int(os.getenv("SUSPENSION_DAYS", 30)),
    "price_drop_threshold": float(os.getenv("PRICE_DROP_THRESHOLD", -0.80)),
    "price_drop_window_days": int(os.getenv("PRICE_DROP_WINDOW", 252)),
    "distress_score_threshold": float(os.getenv("DISTRESS_THRESHOLD", 0.7)),
    "keywords_recuperacao_judicial": [
        "recuperação judicial",
        "recuperacao judicial",
        "falência",
        "falencia",
        "concordata",
        "insolvência",
        "liquidação extrajudicial",
    ],
}

# ─── Pipeline Schedule ────────────────────────────────────────────────────────
PIPELINE = {
    "daily_market_cron": os.getenv("DAILY_CRON", "0 20 * * 1-5"),
    "quarterly_itr_cron": os.getenv("QUARTERLY_CRON", "0 6 * * 1"),
    "annual_dfp_cron": os.getenv("ANNUAL_CRON", "0 4 1 * *"),
    "quality_check_cron": os.getenv("QUALITY_CRON", "0 22 * * 1-5"),
    "max_parallel_jobs": int(os.getenv("MAX_PARALLEL_JOBS", 4)),
}

# ─── Segurança ────────────────────────────────────────────────────────────────
SECURITY = {
    "secret_key": os.getenv("SECRET_KEY", "CHANGE_IN_PRODUCTION"),
    "encrypt_sensitive": os.getenv("ENCRYPT_SENSITIVE", "true").lower() == "true",
    "immutable_logs": os.getenv("IMMUTABLE_LOGS", "true").lower() == "true",
    "retention_days": int(os.getenv("RETENTION_DAYS", 3650)),
    "allowed_roles": ["admin", "analyst", "readonly"],
}

# ─── Logging ──────────────────────────────────────────────────────────────────
LOGGING = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "log_dir": BASE_DIR / "logs",
    "structured": True,
    "include_caller": True,
}

# Garantir que diretórios existam
for path in [
    STORAGE["raw_dir"],
    STORAGE["processed_dir"],
    STORAGE["parquet_dir"],
    LOGGING["log_dir"],
]:
    path.mkdir(parents=True, exist_ok=True)
