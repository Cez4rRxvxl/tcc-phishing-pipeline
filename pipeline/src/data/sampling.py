"""Estratégias de amostragem e split por domínio para evitar vazamento."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import hashlib

# Adiciona src ao path para imports relativos
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import get_logger

logger = get_logger(__name__)


class DomainAwareSplitter:
    """Classe para split PURO por domínio sem filtros, garantindo 100% dos dados."""
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa o splitter.
        
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        
        logger.info(
            "Pure domain-aware splitter initialized",
            random_state=random_state,
            filter_policy="NO_FILTERS_PURE_SPLIT"
        )
    
    def split_by_etld_pure(self, 
                          df: pd.DataFrame,
                          domain_column: str = 'etld_plus_one',
                          label_column: str = 'label',
                          test_size: float = 0.2,
                          val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Split PURO por domínio usando GroupShuffleSplit - 100% dos dados particionados.
        
        Args:
            df: DataFrame com dados
            domain_column: Coluna com domínios (eTLD+1)
            label_column: Coluna com labels
            test_size: Proporção para teste
            val_size: Proporção para validação
            
        Returns:
            Dicionário com DataFrames de train, val e test
        """
        if domain_column not in df.columns:
            raise ValueError(f"Coluna '{domain_column}' não encontrada")
        
        # === DIAGNÓSTICOS PRÉ-SPLIT ===
        total_input_rows = len(df)
        rows_with_missing_etld = df[domain_column].isna().sum()
        unique_domains_original = df[domain_column].nunique()
        
        logger.info(
            "Starting PURE domain split (no filters)",
            total_input_rows=total_input_rows,
            rows_with_missing_etld=rows_with_missing_etld,
            unique_domains_original=unique_domains_original,
            test_size=test_size,
            val_size=val_size
        )
        
        # === TRATAMENTO DE DOMÍNIOS FALTANTES ===
        df_clean = df.copy()
        
        # Se há domínios faltantes, atribui um domínio único baseado em hash
        if rows_with_missing_etld > 0:
            logger.warning(f"Found {rows_with_missing_etld} rows with missing domain")
            
            for idx in df_clean[df_clean[domain_column].isna()].index:
                # Cria domínio único baseado no hash da URL ou índice
                url_hash = hashlib.md5(str(df_clean.loc[idx, 'url']).encode()).hexdigest()[:8]
                missing_domain = f"_missing_domain_{url_hash}"
                df_clean.loc[idx, domain_column] = missing_domain
                
            logger.info(f"Assigned unique domains to {rows_with_missing_etld} missing entries")
        
        rows_dropped_by_filter = 0  # Zero filtros aplicados!
        
        # === SPLIT PURO POR GRUPOS ===
        unique_domains = df_clean[domain_column].unique()
        groups_total = len(unique_domains)
        
        # Primeira divisão: treino+val vs teste (GroupShuffleSplit nos domínios)
        gss_test = GroupShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        # Dummy y para GroupShuffleSplit (não usado, mas obrigatório)
        dummy_y = np.zeros(len(df_clean))
        
        # Split treino+val vs teste
        train_val_idx, test_idx = next(gss_test.split(
            X=df_clean, 
            y=dummy_y, 
            groups=df_clean[domain_column]
        ))
        
        df_train_val = df_clean.iloc[train_val_idx]
        df_test = df_clean.iloc[test_idx]
        
        test_domains = set(df_test[domain_column].unique())
        groups_test = len(test_domains)
        
        # Segunda divisão: treino vs validação (do conjunto treino+val)
        actual_val_size = val_size / (1 - test_size)
        
        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=actual_val_size,
            random_state=self.random_state
        )
        
        dummy_y_train_val = np.zeros(len(df_train_val))
        
        train_idx, val_idx = next(gss_val.split(
            X=df_train_val,
            y=dummy_y_train_val,
            groups=df_train_val[domain_column]
        ))
        
        df_train = df_train_val.iloc[train_idx]
        df_val = df_train_val.iloc[val_idx]
        
        train_domains = set(df_train[domain_column].unique())
        val_domains = set(df_val[domain_column].unique())
        groups_train = len(train_domains)
        groups_val = len(val_domains)
        
        # === CRIAÇÃO DOS SPLITS ===
        splits = {
            'train': df_train,
            'val': df_val,
            'test': df_test
        }
        
        # === ASSERTS OBRIGATÓRIOS DE INTEGRIDADE ===
        total_split_rows = len(df_train) + len(df_val) + len(df_test)
        
        # Assert 1: Todas as linhas devem estar em exatamente um split
        assert total_split_rows == total_input_rows, f"FALHA CRÍTICA: {total_split_rows} != {total_input_rows} (perda de dados!)"
        
        # Assert 2: Domínios devem ser exclusivos entre splits
        all_splits_domains = train_domains | val_domains | test_domains
        domain_intersections = [
            len(train_domains & val_domains),
            len(train_domains & test_domains), 
            len(val_domains & test_domains)
        ]
        
        assert all(intersection == 0 for intersection in domain_intersections), f"FALHA CRÍTICA: Sobreposição de domínios {domain_intersections}"
        
        # Assert 3: União dos domínios deve cobrir todos os domínios originais
        union_domains = len(all_splits_domains)
        assert union_domains == unique_domains_original + (rows_with_missing_etld if rows_with_missing_etld > 0 else 0), \
            f"FALHA CRÍTICA: Domínios perdidos {union_domains} vs esperado {unique_domains_original}"
        
        # === DIAGNÓSTICOS PÓS-SPLIT ===
        logger.info(
            "Pure domain split completed successfully",
            total_input_rows=total_input_rows,
            rows_with_missing_etld=rows_with_missing_etld,
            rows_dropped_by_filter=rows_dropped_by_filter,
            groups_total=groups_total,
            groups_train=groups_train,
            groups_val=groups_val,
            groups_test=groups_test,
            INTEGRITY_CHECK="PASSED"
        )
        
        # Log estatísticas finais
        for split_name, split_df in splits.items():
            label_dist = split_df[label_column].value_counts() if label_column in split_df.columns else {}
            logger.info(
                f"Split {split_name} statistics",
                samples=len(split_df),
                domains=split_df[domain_column].nunique(),
                label_0=label_dist.get(0, 0),
                label_1=label_dist.get(1, 0)
            )
        
        return splits
    
    # Métodos antigos removidos - agora usamos split PURO por grupos
    
    def analyze_split_quality(self, 
                             splits: Dict[str, pd.DataFrame],
                             original_df: pd.DataFrame,
                             domain_column: str = 'etld_plus_one',
                             label_column: str = 'label',
                             source_column: str = 'source') -> Dict:
        """
        Analisa qualidade do split com ASSERTS OBRIGATÓRIOS de integridade.
        
        Args:
            splits: Dicionário com DataFrames de splits
            original_df: DataFrame original para verificar integridade
            domain_column: Coluna com domínios
            label_column: Coluna com labels
            source_column: Coluna com sources
            
        Returns:
            Relatório de qualidade do split com verificações críticas
        """
        # === ASSERTS CRÍTICOS DE INTEGRIDADE ===
        total_original = len(original_df)
        total_splits = sum(len(df) for df in splits.values())
        
        # Assert 1: TOTAL DE REGISTROS DEVE SER IDÊNTICO
        assert total_splits == total_original, \
            f"FALHA CRÍTICA: {total_splits} registros nos splits != {total_original} original"
        
        # === ANÁLISE DE DOMÍNIOS ===
        all_domains = {}
        for split_name, split_df in splits.items():
            all_domains[split_name] = set(split_df[domain_column].unique())
        
        # Assert 2: DOMÍNIOS DEVEM SER EXCLUSIVOS
        domain_overlaps = {}
        for split1 in all_domains:
            for split2 in all_domains:
                if split1 != split2:
                    overlap = all_domains[split1] & all_domains[split2]
                    overlap_count = len(overlap)
                    domain_overlaps[f"{split1}_vs_{split2}"] = {
                        'overlap_count': overlap_count,
                        'overlap_domains': list(overlap)[:10]
                    }
                    
                    # Assert crítico
                    assert overlap_count == 0, \
                        f"FALHA CRÍTICA: {overlap_count} domínios sobrepostos entre {split1} e {split2}"
        
        # Assert 3: UNIÃO DOS DOMÍNIOS DEVE SER COMPLETA
        union_domains = set()
        for domains in all_domains.values():
            union_domains.update(domains)
        
        original_domains = set(original_df[domain_column].unique())
        
        assert union_domains == original_domains, \
            f"FALHA CRÍTICA: Domínios perdidos! Original: {len(original_domains)}, União: {len(union_domains)}"
        
        # === CONSTRUÇÃO DO RELATÓRIO ===
        analysis = {
            'integrity_checks': {
                'total_records_preserved': total_splits == total_original,
                'domains_exclusive': all(info['overlap_count'] == 0 for info in domain_overlaps.values()),
                'all_domains_preserved': len(union_domains) == len(original_domains),
                'original_records': total_original,
                'split_records': total_splits,
                'original_domains': len(original_domains),
                'union_domains': len(union_domains)
            },
            'domain_overlap': domain_overlaps,
            'label_distribution': {},
            'source_distribution': {},
            'balance_metrics': {}
        }
        
        # Analisa distribuição de labels
        for split_name, split_df in splits.items():
            label_counts = split_df[label_column].value_counts() if label_column in split_df.columns else {}
            total = len(split_df)
            
            analysis['label_distribution'][split_name] = {
                'total_samples': total,
                'label_0': int(label_counts.get(0, 0)),
                'label_1': int(label_counts.get(1, 0)),
                'label_0_pct': round(label_counts.get(0, 0) / total * 100, 2) if total > 0 else 0,
                'label_1_pct': round(label_counts.get(1, 0) / total * 100, 2) if total > 0 else 0,
                'balance_ratio': round(
                    min(label_counts.get(0, 0), label_counts.get(1, 0)) / 
                    max(label_counts.get(0, 0), label_counts.get(1, 0)), 3
                ) if max(label_counts.get(0, 0), label_counts.get(1, 0)) > 0 else 0
            }
        
        # Analisa distribuição de sources
        if source_column in splits['train'].columns:
            for split_name, split_df in splits.items():
                source_counts = split_df[source_column].value_counts()
                analysis['source_distribution'][split_name] = {
                    'unique_sources': len(source_counts),
                    'top_sources': source_counts.head(10).to_dict()
                }
        
        # Métricas de balanceamento geral
        analysis['balance_metrics'] = {
            'total_samples': total_splits,
            'train_pct': round(len(splits['train']) / total_splits * 100, 2),
            'val_pct': round(len(splits['val']) / total_splits * 100, 2),
            'test_pct': round(len(splits['test']) / total_splits * 100, 2),
            'unique_domains': {
                'train': splits['train'][domain_column].nunique(),
                'val': splits['val'][domain_column].nunique(), 
                'test': splits['test'][domain_column].nunique()
            }
        }
        
        logger.info("Split quality analysis completed with INTEGRITY CHECKS PASSED")
        
        return analysis


def create_cross_validation_splits(df: pd.DataFrame,
                                  domain_column: str = 'etld_plus_one',
                                  label_column: str = 'label',
                                  n_splits: int = 5,
                                  random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Cria splits para validação cruzada respeitando domínios.
    
    Args:
        df: DataFrame com dados
        domain_column: Coluna com domínios
        label_column: Coluna com labels
        n_splits: Número de folds
        random_state: Seed para reprodutibilidade
        
    Returns:
        Lista de tuples (train_df, val_df) para cada fold
    """
    logger.info(f"Creating {n_splits}-fold cross-validation splits")
    
    # Prepara dados por domínio
    domains = df[domain_column].unique()
    domain_labels = []
    
    for domain in domains:
        domain_df = df[df[domain_column] == domain]
        majority_label = domain_df[label_column].mode()[0]
        domain_labels.append(majority_label)
    
    # GroupShuffleSplit para respeitar domínios
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)
    
    splits = []
    for train_domains_idx, val_domains_idx in gss.split(domains, domain_labels, domains):
        train_domains = domains[train_domains_idx]
        val_domains = domains[val_domains_idx]
        
        train_df = df[df[domain_column].isin(train_domains)]
        val_df = df[df[domain_column].isin(val_domains)]
        
        splits.append((train_df, val_df))
    
    logger.info(f"Created {len(splits)} CV splits")
    return splits


def generate_split_report(analysis: Dict, output_file: Optional[str] = None) -> str:
    """
    Gera relatório em markdown da análise de qualidade do split com CHECKS DE INTEGRIDADE.
    
    Args:
        analysis: Análise de qualidade do split
        output_file: Arquivo de saída (opcional)
        
    Returns:
        Conteúdo do relatório em markdown
    """
    report_lines = [
        "# Relatório de Qualidade do Split por Domínio",
        "",
        "## ✅ CHECKS DE INTEGRIDADE OBRIGATÓRIOS",
        "",
    ]
    
    # Checks de integridade
    if 'integrity_checks' in analysis:
        checks = analysis['integrity_checks']
        
        # Total de registros preservado
        if checks['total_records_preserved']:
            report_lines.append(f"✅ **Total de registros preservado:** {checks['split_records']:,} = {checks['original_records']:,} ✓")
        else:
            report_lines.append(f"❌ **FALHA CRÍTICA:** {checks['split_records']:,} ≠ {checks['original_records']:,} registros!")
        
        # Domínios exclusivos
        if checks['domains_exclusive']:
            report_lines.append("✅ **Domínios exclusivos entre splits:** Nenhuma sobreposição ✓")
        else:
            report_lines.append("❌ **FALHA CRÍTICA:** Sobreposição de domínios detectada!")
        
        # Todos os domínios preservados
        if checks['all_domains_preserved']:
            report_lines.append(f"✅ **Todos os domínios preservados:** {checks['union_domains']:,} = {checks['original_domains']:,} ✓")
        else:
            report_lines.append(f"❌ **FALHA CRÍTICA:** {checks['union_domains']:,} ≠ {checks['original_domains']:,} domínios!")
    
    report_lines.extend([
        "",
        "## Métricas Gerais",
        "",
        f"**Total de amostras:** {analysis['balance_metrics']['total_samples']:,}",
        f"**Treino:** {analysis['balance_metrics']['train_pct']}%",
        f"**Validação:** {analysis['balance_metrics']['val_pct']}%",
        f"**Teste:** {analysis['balance_metrics']['test_pct']}%",
        "",
    ])
    
    # Verificação de vazamento
    report_lines.extend([
        "## Verificação de Vazamento de Domínios",
        "",
    ])
    
    has_overlap = False
    for comparison, overlap_info in analysis['domain_overlap'].items():
        if overlap_info['overlap_count'] > 0:
            has_overlap = True
            report_lines.append(f"⚠️ **{comparison}:** {overlap_info['overlap_count']} domínios sobrepostos")
    
    if not has_overlap:
        report_lines.append("✅ **Nenhum vazamento detectado** - domínios exclusivos entre splits")
    
    report_lines.append("")
    
    # Distribuição de labels
    report_lines.extend([
        "## Distribuição de Labels",
        "",
        "| Split | Total | Benignos (0) | Maliciosos (1) | Balanceamento |",
        "|-------|-------|--------------|----------------|---------------|",
    ])
    
    for split_name, dist in analysis['label_distribution'].items():
        report_lines.append(
            f"| {split_name.capitalize()} | {dist['total_samples']:,} | "
            f"{dist['label_0']:,} ({dist['label_0_pct']}%) | "
            f"{dist['label_1']:,} ({dist['label_1_pct']}%) | "
            f"{dist['balance_ratio']:.3f} |"
        )
    
    report_lines.append("")
    
    # Distribuição de domínios
    report_lines.extend([
        "## Distribuição de Domínios",
        "",
    ])
    
    for split_name, count in analysis['balance_metrics']['unique_domains'].items():
        report_lines.append(f"- **{split_name.capitalize()}:** {count:,} domínios únicos")
    
    report_lines.append("")
    
    # Distribuição de sources
    if 'source_distribution' in analysis and analysis['source_distribution']:
        report_lines.extend([
            "## Distribuição de Sources",
            "",
        ])
        
        for split_name, source_info in analysis['source_distribution'].items():
            report_lines.append(f"### {split_name.capitalize()}")
            report_lines.append(f"- Sources únicos: {source_info['unique_sources']}")
            report_lines.append("- Top sources:")
            
            for source, count in source_info['top_sources'].items():
                report_lines.append(f"  - {source}: {count:,}")
            
            report_lines.append("")
    
    report_content = '\n'.join(report_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"Split report saved to {output_file}")
    
    return report_content
