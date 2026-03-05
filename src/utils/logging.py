"""
Logger estruturado e imutável para auditoria.
Usa structlog para JSON formatado + rotação de arquivos.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import structlog
from config.settings import LOGGING, SECURITY


def _add_timestamp(logger: Any, method: str, event_dict: dict) -> dict:
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _add_source_info(logger: Any, method: str, event_dict: dict) -> dict:
    frame = sys._getframe(6)
    event_dict["caller"] = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    return event_dict


def configure_logging() -> None:
    log_dir = Path(LOGGING["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Handler de arquivo com rotação diária e retenção de 90 dias
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "ingestion.log",
        when="midnight",
        interval=1,
        backupCount=90,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setLevel(logging.DEBUG)

    # Handler separado imutável para auditoria (append-only conceptual)
    audit_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "audit.log",
        when="midnight",
        interval=1,
        backupCount=3650,  # 10 anos de retenção de auditoria
        encoding="utf-8",
        utc=True,
    )
    audit_handler.setLevel(logging.WARNING)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if LOGGING["include_caller"]:
        processors.append(_add_source_info)

    processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOGGING["level"]))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(audit_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


configure_logging()
