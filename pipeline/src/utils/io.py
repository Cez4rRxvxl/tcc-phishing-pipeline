"""Utilitários para leitura e escrita de dados com validações."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from pydantic import BaseModel

from .logging import get_logger

logger = get_logger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """Converte tipos numpy e outros tipos não-serializáveis para JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(obj.__dict__)
    else:
        return obj


class DataFrameInfo(BaseModel):
    """Informações sobre um DataFrame."""
    
    shape: tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    memory_usage_mb: float


class CSVConfig(BaseModel):
    """Configuração para leitura/escrita de CSV."""
    
    encoding: str = "utf-8"
    sep: str = ","
    quotechar: str = '"'
    escapechar: Optional[str] = None
    low_memory: bool = False
    na_values: List[str] = ["", "NA", "N/A", "null", "NULL", "nan", "NaN"]


def read_csv_safe(
    file_path: Union[str, Path],
    config: Optional[CSVConfig] = None,
    required_columns: Optional[List[str]] = None,
    dtype_mapping: Optional[Dict[str, str]] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Lê arquivo CSV com validações e configurações seguras.

    Args:
        file_path: Caminho para o arquivo CSV
        config: Configuração de leitura
        required_columns: Colunas obrigatórias
        dtype_mapping: Mapeamento de tipos de dados
        chunk_size: Tamanho do chunk para leitura em blocos

    Returns:
        DataFrame carregado

    Raises:
        FileNotFoundError: Se arquivo não existe
        ValueError: Se colunas obrigatórias não existem
    """
    file_path = Path(file_path)
    config = config or CSVConfig()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    logger.info(
        "Reading CSV file",
        file_path=str(file_path),
        file_size_mb=round(file_path.stat().st_size / 1024 / 1024, 2)
    )
    
    read_kwargs = {
        "encoding": config.encoding,
        "sep": config.sep,
        "quotechar": config.quotechar,
        "escapechar": config.escapechar,
        "low_memory": config.low_memory,
        "na_values": config.na_values,
    }
    
    if dtype_mapping:
        read_kwargs["dtype"] = dtype_mapping
    
    if chunk_size:
        read_kwargs["chunksize"] = chunk_size
        return pd.read_csv(file_path, **read_kwargs)
    
    df = pd.read_csv(file_path, **read_kwargs)
    
    # Valida colunas obrigatórias
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Colunas obrigatórias não encontradas: {missing_cols}"
            )
    
    logger.info(
        "CSV file loaded successfully",
        shape=df.shape,
        columns=list(df.columns),
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    )
    
    return df


def write_csv_safe(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    config: Optional[CSVConfig] = None,
    compress: bool = False,
    create_dirs: bool = True,
) -> None:
    """
    Escreve DataFrame em CSV com configurações seguras.

    Args:
        df: DataFrame para salvar
        file_path: Caminho de destino
        config: Configuração de escrita
        compress: Se deve comprimir com gzip
        create_dirs: Se deve criar diretórios se não existem
    """
    file_path = Path(file_path)
    config = config or CSVConfig()
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Adiciona .gz se compressão está habilitada
    if compress and not file_path.suffix.endswith('.gz'):
        file_path = file_path.with_suffix(file_path.suffix + '.gz')
    
    logger.info(
        "Writing CSV file",
        file_path=str(file_path),
        shape=df.shape,
        compressed=compress
    )
    
    write_kwargs = {
        "encoding": config.encoding,
        "sep": config.sep,
        "quotechar": config.quotechar,
        "escapechar": config.escapechar,
        "index": False,
    }
    
    if compress:
        write_kwargs["compression"] = "gzip"
    
    df.to_csv(file_path, **write_kwargs)
    
    file_size_mb = round(file_path.stat().st_size / 1024 / 1024, 2)
    logger.info(
        "CSV file written successfully",
        file_size_mb=file_size_mb
    )


def get_dataframe_info(df: pd.DataFrame) -> DataFrameInfo:
    """
    Extrai informações estruturadas sobre um DataFrame.

    Args:
        df: DataFrame para analisar

    Returns:
        Informações estruturadas do DataFrame
    """
    return DataFrameInfo(
        shape=df.shape,
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        null_counts=df.isnull().sum().to_dict(),
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    )


def list_csv_files(directory: Union[str, Path]) -> List[Path]:
    """
    Lista todos os arquivos CSV em um diretório.

    Args:
        directory: Diretório para buscar

    Returns:
        Lista de caminhos para arquivos CSV
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    csv_files = []
    
    # Busca arquivos .csv e .tsv
    for pattern in ["*.csv", "*.tsv", "*.CSV", "*.TSV"]:
        csv_files.extend(directory.glob(pattern))
    
    # Ordena por nome
    csv_files.sort()
    
    logger.info(
        "CSV files found",
        directory=str(directory),
        count=len(csv_files),
        files=[f.name for f in csv_files]
    )
    
    return csv_files


def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
    create_dirs: bool = True,
) -> None:
    """
    Salva dados em arquivo JSON.

    Args:
        data: Dados para salvar
        file_path: Caminho de destino
        indent: Indentação para formatação
        ensure_ascii: Se deve garantir ASCII
        create_dirs: Se deve criar diretórios
    """
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sanitiza dados para JSON (converte tipos numpy, etc.)
    sanitized_data = sanitize_for_json(data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sanitized_data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
    
    logger.info(f"JSON saved to {file_path}")


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Carrega dados de arquivo JSON.

    Args:
        file_path: Caminho do arquivo

    Returns:
        Dados carregados

    Raises:
        FileNotFoundError: Se arquivo não existe
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"JSON loaded from {file_path}")
    return data


def create_directory_structure(base_path: Union[str, Path], subdirs: List[str]) -> None:
    """
    Cria estrutura de diretórios.

    Args:
        base_path: Caminho base
        subdirs: Lista de subdiretórios para criar
    """
    base_path = Path(base_path)
    
    for subdir in subdirs:
        dir_path = base_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created: {dir_path}")
    
    logger.info(
        "Directory structure created",
        base_path=str(base_path),
        subdirs=subdirs
    )


def check_file_permissions(file_path: Union[str, Path]) -> Dict[str, bool]:
    """
    Verifica permissões de arquivo.

    Args:
        file_path: Caminho do arquivo

    Returns:
        Dicionário com permissões (read, write, execute)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"read": False, "write": False, "execute": False, "exists": False}
    
    import os
    
    permissions = {
        "exists": True,
        "read": os.access(file_path, os.R_OK),
        "write": os.access(file_path, os.W_OK),
        "execute": os.access(file_path, os.X_OK),
    }
    
    return permissions


def get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Obtém estatísticas de arquivo.

    Args:
        file_path: Caminho do arquivo

    Returns:
        Dicionário com estatísticas do arquivo
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"exists": False}
    
    stat = file_path.stat()
    
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
        "modified_time": stat.st_mtime,
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "suffix": file_path.suffix,
        "name": file_path.name,
    }
