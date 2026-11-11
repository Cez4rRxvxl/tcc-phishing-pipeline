"""Utilitários para validação de dados e schema."""

from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd
from pydantic import BaseModel, Field, validator
import numpy as np

from .logging import get_logger

logger = get_logger(__name__)


class ColumnValidation(BaseModel):
    """Validação para uma coluna específica."""
    
    name: str
    dtype: str
    required: bool = True
    nullable: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[Set[Any]] = None
    regex_pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class DataFrameSchema(BaseModel):
    """Schema para validação de DataFrame."""
    
    columns: List[ColumnValidation]
    min_rows: int = Field(default=1, ge=0)
    max_rows: Optional[int] = Field(default=None, ge=0)
    allow_extra_columns: bool = True
    
    @validator('max_rows')
    def validate_max_rows(cls, v, values):
        if v is not None and 'min_rows' in values and v < values['min_rows']:
            raise ValueError('max_rows deve ser maior ou igual a min_rows')
        return v


class ValidationResult(BaseModel):
    """Resultado de validação."""
    
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    column_stats: Dict[str, Any] = {}
    
    def add_error(self, error: str) -> None:
        """Adiciona um erro."""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str) -> None:
        """Adiciona um warning."""
        self.warnings.append(warning)


def validate_dataframe(df: pd.DataFrame, schema: DataFrameSchema) -> ValidationResult:
    """
    Valida DataFrame contra um schema.

    Args:
        df: DataFrame para validar
        schema: Schema de validação

    Returns:
        Resultado da validação
    """
    result = ValidationResult(is_valid=True)
    
    logger.info(
        "Starting DataFrame validation",
        shape=df.shape,
        columns=list(df.columns)
    )
    
    # Valida número de linhas
    if df.shape[0] < schema.min_rows:
        result.add_error(f"DataFrame tem {df.shape[0]} linhas, mínimo requerido: {schema.min_rows}")
    
    if schema.max_rows and df.shape[0] > schema.max_rows:
        result.add_error(f"DataFrame tem {df.shape[0]} linhas, máximo permitido: {schema.max_rows}")
    
    # Valida colunas
    df_columns = set(df.columns)
    schema_columns = {col.name for col in schema.columns}
    required_columns = {col.name for col in schema.columns if col.required}
    
    # Verifica colunas obrigatórias
    missing_required = required_columns - df_columns
    if missing_required:
        result.add_error(f"Colunas obrigatórias ausentes: {missing_required}")
    
    # Verifica colunas extras
    extra_columns = df_columns - schema_columns
    if extra_columns and not schema.allow_extra_columns:
        result.add_error(f"Colunas não permitidas: {extra_columns}")
    elif extra_columns:
        result.add_warning(f"Colunas extras encontradas: {extra_columns}")
    
    # Valida cada coluna presente
    for col_validation in schema.columns:
        if col_validation.name not in df.columns:
            continue
            
        col_result = _validate_column(df[col_validation.name], col_validation)
        result.column_stats[col_validation.name] = col_result
        
        if col_result.get('errors'):
            for error in col_result['errors']:
                result.add_error(f"Coluna '{col_validation.name}': {error}")
        
        if col_result.get('warnings'):
            for warning in col_result['warnings']:
                result.add_warning(f"Coluna '{col_validation.name}': {warning}")
    
    logger.info(
        "DataFrame validation completed",
        is_valid=result.is_valid,
        errors_count=len(result.errors),
        warnings_count=len(result.warnings)
    )
    
    return result


def _validate_column(series: pd.Series, validation: ColumnValidation) -> Dict[str, Any]:
    """Valida uma coluna específica."""
    stats = {
        'name': validation.name,
        'count': len(series),
        'null_count': series.isnull().sum(),
        'null_percentage': round(series.isnull().sum() / len(series) * 100, 2),
        'unique_count': series.nunique(),
        'errors': [],
        'warnings': []
    }
    
    # Valida nulos
    if not validation.nullable and stats['null_count'] > 0:
        stats['errors'].append(f"{stats['null_count']} valores nulos encontrados (não permitido)")
    
    # Remove nulos para validações subsequentes
    non_null_series = series.dropna()
    
    if len(non_null_series) == 0:
        return stats
    
    # Valida tipo de dados
    try:
        if validation.dtype in ['int', 'int64', 'int32']:
            pd.to_numeric(non_null_series, errors='raise', downcast='integer')
        elif validation.dtype in ['float', 'float64', 'float32']:
            pd.to_numeric(non_null_series, errors='raise')
        elif validation.dtype == 'datetime':
            pd.to_datetime(non_null_series, errors='raise')
        elif validation.dtype == 'string':
            non_null_series.astype(str)
    except (ValueError, TypeError) as e:
        stats['errors'].append(f"Erro de tipo de dados: {str(e)}")
    
    # Valida comprimento para strings
    if validation.dtype == 'string' and (validation.min_length or validation.max_length):
        str_lengths = non_null_series.astype(str).str.len()
        
        if validation.min_length and (str_lengths < validation.min_length).any():
            min_violations = (str_lengths < validation.min_length).sum()
            stats['errors'].append(f"{min_violations} valores com comprimento menor que {validation.min_length}")
        
        if validation.max_length and (str_lengths > validation.max_length).any():
            max_violations = (str_lengths > validation.max_length).sum()
            stats['errors'].append(f"{max_violations} valores com comprimento maior que {validation.max_length}")
    
    # Valida valores permitidos
    if validation.allowed_values:
        invalid_values = set(non_null_series) - validation.allowed_values
        if invalid_values:
            stats['errors'].append(f"Valores não permitidos encontrados: {invalid_values}")
    
    # Valida regex
    if validation.regex_pattern and validation.dtype == 'string':
        import re
        pattern = re.compile(validation.regex_pattern)
        str_series = non_null_series.astype(str)
        invalid_matches = ~str_series.str.match(pattern, na=False)
        if invalid_matches.any():
            invalid_count = invalid_matches.sum()
            stats['errors'].append(f"{invalid_count} valores não atendem ao padrão regex")
    
    # Valida valores numéricos
    if validation.dtype in ['int', 'float', 'int64', 'int32', 'float64', 'float32']:
        numeric_series = pd.to_numeric(non_null_series, errors='coerce')
        
        if validation.min_value is not None:
            below_min = (numeric_series < validation.min_value).sum()
            if below_min > 0:
                stats['errors'].append(f"{below_min} valores abaixo do mínimo {validation.min_value}")
        
        if validation.max_value is not None:
            above_max = (numeric_series > validation.max_value).sum()
            if above_max > 0:
                stats['errors'].append(f"{above_max} valores acima do máximo {validation.max_value}")
        
        # Adiciona estatísticas numéricas
        stats.update({
            'min': float(numeric_series.min()),
            'max': float(numeric_series.max()),
            'mean': float(numeric_series.mean()),
            'std': float(numeric_series.std())
        })
    
    return stats


def validate_url_dataset_schema(df: pd.DataFrame) -> ValidationResult:
    """
    Valida schema específico para datasets de URL.

    Args:
        df: DataFrame com dados de URL

    Returns:
        Resultado da validação
    """
    schema = DataFrameSchema(
        columns=[
            ColumnValidation(
                name="url",
                dtype="string",
                required=True,
                nullable=False,
                min_length=10,
                max_length=2048
            ),
            ColumnValidation(
                name="label",
                dtype="int",
                required=True,
                nullable=False,
                allowed_values={0, 1},
                min_value=0,
                max_value=1
            ),
            ColumnValidation(
                name="source",
                dtype="string",
                required=True,
                nullable=False,
                min_length=3,
                max_length=50
            ),
            ColumnValidation(
                name="url_norm",
                dtype="string",
                required=False,
                nullable=True
            ),
            ColumnValidation(
                name="domain",
                dtype="string",
                required=False,
                nullable=True
            ),
            ColumnValidation(
                name="etld_plus_one",
                dtype="string",
                required=False,
                nullable=True
            ),
        ],
        min_rows=1,
        allow_extra_columns=True
    )
    
    return validate_dataframe(df, schema)


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa qualidade geral dos dados.

    Args:
        df: DataFrame para analisar

    Returns:
        Relatório de qualidade dos dados
    """
    quality_report = {
        'shape': df.shape,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'total_null_count': int(df.isnull().sum().sum()),
        'total_null_percentage': round(df.isnull().sum().sum() / df.size * 100, 2),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_percentage': round(df.duplicated().sum() / len(df) * 100, 2),
        'columns': {}
    }
    
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
            'unique_count': int(df[col].nunique()),
            'unique_percentage': round(df[col].nunique() / len(df) * 100, 2),
        }
        
        # Adiciona estatísticas específicas por tipo
        if df[col].dtype in ['int64', 'float64']:
            col_stats.update({
                'min': float(df[col].min()) if not df[col].isnull().all() else None,
                'max': float(df[col].max()) if not df[col].isnull().all() else None,
                'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                'std': float(df[col].std()) if not df[col].isnull().all() else None,
            })
        elif df[col].dtype == 'object':
            non_null_col = df[col].dropna()
            if len(non_null_col) > 0:
                col_stats.update({
                    'avg_length': float(non_null_col.astype(str).str.len().mean()),
                    'min_length': int(non_null_col.astype(str).str.len().min()),
                    'max_length': int(non_null_col.astype(str).str.len().max()),
                })
        
        quality_report['columns'][col] = col_stats
    
    logger.info(
        "Data quality check completed",
        **quality_report
    )
    
    return quality_report


def detect_outliers(
    series: pd.Series, 
    method: str = 'iqr', 
    threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Detecta outliers em uma série numérica.

    Args:
        series: Série para analisar
        method: Método de detecção ('iqr', 'zscore')
        threshold: Threshold para detecção

    Returns:
        Informações sobre outliers
    """
    if not pd.api.types.is_numeric_dtype(series):
        return {'error': 'Série deve ser numérica'}
    
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {'error': 'Série vazia após remoção de nulos'}
    
    if method == 'iqr':
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
        
    elif method == 'zscore':
        z_scores = np.abs((clean_series - clean_series.mean()) / clean_series.std())
        outliers = clean_series[z_scores > threshold]
        lower_bound = clean_series.mean() - threshold * clean_series.std()
        upper_bound = clean_series.mean() + threshold * clean_series.std()
        
    else:
        return {'error': f'Método não suportado: {method}'}
    
    return {
        'method': method,
        'threshold': threshold,
        'outlier_count': len(outliers),
        'outlier_percentage': round(len(outliers) / len(clean_series) * 100, 2),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outlier_values': outliers.tolist()[:10],  # Apenas os primeiros 10
    }
