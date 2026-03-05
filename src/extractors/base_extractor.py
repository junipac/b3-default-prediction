"""
Extrator base abstrato.
Define contrato e comportamento comum: retry, rate limit,
detecção de bloqueio, versionamento de raw, hashing e schema drift.
"""

import hashlib
import json
import time
import random
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import SOURCES, USER_AGENTS, STORAGE
from src.utils.logging import get_logger
from src.utils.retry import is_blocked_response, HTTPRetryError, BlockedIPError, RETRYABLE_STATUS_CODES
from src.utils.rate_limiter import RateLimiterRegistry

logger = get_logger(__name__)


class ExtractionResult:
    """Encapsula resultado de uma extração com metadados completos."""

    def __init__(
        self,
        source: str,
        url: str,
        content: bytes,
        content_type: str,
        status_code: int,
        headers: Dict[str, str],
    ):
        self.source = source
        self.url = url
        self.content = content
        self.content_type = content_type
        self.status_code = status_code
        self.headers = headers
        self.timestamp = datetime.now(timezone.utc)
        self.sha256 = hashlib.sha256(content).hexdigest()
        self.size_bytes = len(content)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "status_code": self.status_code,
            "content_type": self.content_type,
        }


class BaseExtractor(ABC):
    """
    Classe base para todos os extratores.
    Fornece:
    - Sessão HTTP com retry automático
    - Rotação de User-Agent
    - Rate limiting por fonte
    - Persistência imutável de raw data com hash
    - Detecção de schema drift
    - Logging estruturado de cada operação
    """

    SOURCE_NAME: str = "base"

    def __init__(self):
        self.source_config = SOURCES.get(self.SOURCE_NAME, {})
        self.timeout = self.source_config.get("timeout", 60)
        self._session: Optional[requests.Session] = None
        self._user_agent_index = 0
        self._schema_snapshots: Dict[str, Any] = {}
        self.rate_limiter = RateLimiterRegistry.get_sync(self.SOURCE_NAME)
        self._raw_dir = Path(STORAGE["raw_dir"]) / self.SOURCE_NAME
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = self._build_session()
        return self._session

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=0,  # retry gerenciado manualmente
            backoff_factor=0,
            status_forcelist=[],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(self._build_headers())
        return session

    def _build_headers(self) -> Dict[str, str]:
        ua = USER_AGENTS[self._user_agent_index % len(USER_AGENTS)]
        return {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def _rotate_user_agent(self) -> None:
        self._user_agent_index += 1
        if self._session:
            self._session.headers.update(self._build_headers())
        logger.debug(
            "user_agent_rotated",
            source=self.SOURCE_NAME,
            new_index=self._user_agent_index % len(USER_AGENTS),
        )

    def _get(
        self,
        url: str,
        params: Optional[Dict] = None,
        stream: bool = False,
        max_attempts: Optional[int] = None,
    ) -> requests.Response:
        """
        GET com retry exponencial, rate limiting e detecção de bloqueio.
        Registra cada tentativa no log estruturado.
        """
        from config.settings import RETRY
        max_attempts = max_attempts or RETRY["max_attempts"]
        initial_wait = RETRY["initial_wait"]
        max_wait = RETRY["max_wait"]
        multiplier = RETRY["multiplier"]

        wait = initial_wait
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            self.rate_limiter.acquire()
            try:
                start = time.monotonic()
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    stream=stream,
                )
                elapsed = time.monotonic() - start

                logger.info(
                    "http_get",
                    source=self.SOURCE_NAME,
                    url=url,
                    status=response.status_code,
                    elapsed_ms=round(elapsed * 1000),
                    attempt=attempt,
                    size_bytes=len(response.content) if not stream else "streaming",
                )

                if is_blocked_response(response):
                    logger.warning(
                        "ip_blocked_detected",
                        source=self.SOURCE_NAME,
                        url=url,
                        status=response.status_code,
                    )
                    self._rotate_user_agent()
                    raise BlockedIPError(f"Bloqueio detectado: {url}")

                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise requests.exceptions.HTTPError(
                        f"Status {response.status_code}",
                        response=response,
                    )

                response.raise_for_status()
                return response

            except (BlockedIPError, requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout, requests.exceptions.HTTPError,
                    IOError) as e:
                last_exception = e
                if attempt >= max_attempts:
                    break
                jitter = random.uniform(0, wait * 0.15)
                sleep_time = min(wait + jitter, max_wait)
                logger.warning(
                    "http_retry",
                    source=self.SOURCE_NAME,
                    url=url,
                    attempt=attempt,
                    next_wait_s=round(sleep_time, 1),
                    error=str(e),
                )
                time.sleep(sleep_time)
                wait = min(wait * multiplier, max_wait)

        logger.error(
            "http_get_failed",
            source=self.SOURCE_NAME,
            url=url,
            attempts=max_attempts,
            error=str(last_exception),
        )
        raise last_exception

    def persist_raw(
        self,
        content: bytes,
        filename: str,
        metadata: Dict[str, Any],
    ) -> Path:
        """
        Persiste conteúdo bruto de forma imutável com hash e metadados.
        Nunca sobrescreve — cria versão nova se conteúdo difere.
        Retorna o caminho do arquivo salvo.
        """
        sha256 = hashlib.sha256(content).hexdigest()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(filename).stem
        suffix = Path(filename).suffix or ".bin"

        versioned_name = f"{stem}__{timestamp}__{sha256[:12]}{suffix}"
        target_path = self._raw_dir / versioned_name

        # Verifica se já existe arquivo com mesmo hash (deduplicação)
        existing = list(self._raw_dir.glob(f"{stem}__*__{sha256[:12]}{suffix}"))
        if existing:
            logger.debug(
                "raw_deduplicated",
                source=self.SOURCE_NAME,
                filename=filename,
                sha256=sha256[:12],
            )
            return existing[0]

        target_path.write_bytes(content)

        meta_path = target_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {**metadata, "sha256": sha256, "filename": versioned_name, "size_bytes": len(content)},
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "raw_persisted",
            source=self.SOURCE_NAME,
            path=str(target_path),
            sha256=sha256,
            size_bytes=len(content),
        )
        return target_path

    def detect_schema_drift(
        self,
        schema_key: str,
        current_columns: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Compara schema atual com snapshot anterior.
        Retorna diff se houve mudança, None se igual.
        """
        snapshot_path = self._raw_dir / f"schema__{schema_key}.json"

        if snapshot_path.exists():
            previous = json.loads(snapshot_path.read_text())
            prev_cols = set(previous.get("columns", []))
            curr_cols = set(current_columns)

            added = list(curr_cols - prev_cols)
            removed = list(prev_cols - curr_cols)

            if added or removed:
                drift = {
                    "schema_key": schema_key,
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "columns_added": added,
                    "columns_removed": removed,
                    "previous_count": len(prev_cols),
                    "current_count": len(curr_cols),
                }
                logger.warning("schema_drift_detected", **drift)

                # Salva snapshot atualizado
                snapshot_path.write_text(
                    json.dumps({"columns": current_columns, "updated_at": drift["detected_at"]}),
                    encoding="utf-8",
                )
                return drift
        else:
            # Primeiro snapshot
            snapshot_path.write_text(
                json.dumps({
                    "columns": current_columns,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }),
                encoding="utf-8",
            )

        return None

    @abstractmethod
    def extract(self, **kwargs) -> Any:
        """Método principal de extração. Cada fonte implementa o seu."""
        ...

    @abstractmethod
    def validate_source(self) -> bool:
        """Verifica se a fonte está acessível e responsiva."""
        ...
