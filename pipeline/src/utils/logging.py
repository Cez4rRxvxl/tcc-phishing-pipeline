"""Configuração estruturada de logging para o pipeline."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import os

# Carrega variáveis de ambiente de um .env, se python-dotenv estiver disponível.
# A dependência é opcional: ela só popula LOG_LEVEL e LOG_FORMAT, que já têm
# default próprio mais abaixo. Ver SDD.md §11 (poda de dependências na T1.1).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover
    pass


class StructuredFormatter(logging.Formatter):
    """Formatter para logs estruturados em JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Formata o log record como JSON estruturado."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Adiciona informações extras se disponíveis
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Adiciona informações de exceção se presentes
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class PipelineLogger:
    """Logger estruturado para o pipeline de detecção de phishing."""

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_format: str = "json",
        log_file: Optional[Path] = None,
    ):
        """
        Inicializa o logger.

        Args:
            name: Nome do logger
            level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Formato do log ('json' ou 'text')
            log_file: Arquivo para salvar logs (opcional)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Configura handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        
        if log_format.lower() == "json":
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        
        self.logger.addHandler(console_handler)

        # Configura handler para arquivo se especificado
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log de debug com dados extras."""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log de informação com dados extras."""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log de warning com dados extras."""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log de erro com dados extras."""
        self._log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log crítico com dados extras."""
        self._log(logging.CRITICAL, message, kwargs)

    def _log(self, level: int, message: str, extra_data: Dict[str, Any]) -> None:
        """Método interno para logging com dados extras."""
        # Usa a API padrão do logging com extra
        if extra_data:
            self.logger.log(level, message, extra={"extra_data": extra_data})
        else:
            self.logger.log(level, message)

    def log_pipeline_step(
        self, 
        step: str, 
        action: str, 
        status: str = "started",
        **kwargs: Any
    ) -> None:
        """
        Log específico para etapas do pipeline.

        Args:
            step: Nome da etapa (ex: 'canonicalization', 'deduplication')
            action: Ação sendo executada
            status: Status da ação ('started', 'completed', 'failed')
            **kwargs: Dados extras para logging
        """
        self.info(
            f"Pipeline step {action}",
            step=step,
            action=action,
            status=status,
            **kwargs
        )

    def log_data_stats(
        self,
        operation: str,
        before_count: int,
        after_count: int,
        **kwargs: Any
    ) -> None:
        """
        Log específico para estatísticas de dados.

        Args:
            operation: Operação realizada
            before_count: Contagem antes da operação
            after_count: Contagem após a operação
            **kwargs: Dados extras
        """
        change = after_count - before_count
        change_pct = (change / before_count * 100) if before_count > 0 else 0
        
        self.info(
            f"Data operation: {operation}",
            operation=operation,
            before_count=before_count,
            after_count=after_count,
            change=change,
            change_percentage=round(change_pct, 2),
            **kwargs
        )


def get_logger(
    name: str, 
    level: Optional[str] = None, 
    log_format: Optional[str] = None
) -> PipelineLogger:
    """
    Cria um logger configurado com as variáveis de ambiente.

    Args:
        name: Nome do logger
        level: Nível de log (sobrescreve variável de ambiente)
        log_format: Formato do log (sobrescreve variável de ambiente)

    Returns:
        Instância configurada do PipelineLogger
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_format = log_format or os.getenv("LOG_FORMAT", "json")
    
    return PipelineLogger(
        name=name,
        level=level,
        log_format=log_format
    )


def log_execution_time(logger: PipelineLogger):
    """
    Decorator para logar tempo de execução de funções.

    Args:
        logger: Instância do logger

    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            logger.info(
                f"Starting {func.__name__}",
                function=func.__name__,
                status="started"
            )
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.info(
                    f"Completed {func.__name__}",
                    function=func.__name__,
                    status="completed",
                    execution_time_seconds=execution_time
                )
                return result
                
            except Exception as e:
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.error(
                    f"Failed {func.__name__}",
                    function=func.__name__,
                    status="failed",
                    execution_time_seconds=execution_time,
                    error=str(e)
                )
                raise
                
        return wrapper
    return decorator
