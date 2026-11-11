"""Extração de características léxicas de URLs para detecção de phishing."""

import sys
from pathlib import Path
import re
import string
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse
import pandas as pd
import numpy as np

# Adiciona src ao path para imports relativos
sys.path.insert(0, str(Path(__file__).parent.parent))

import psl
from utils.logging import get_logger

logger = get_logger(__name__)

# --- Correção B4 (SDD §3.1) --------------------------------------------------
# `urlparse()` só separa corretamente `netloc` de `path` quando a URL tem um
# scheme explícito (`scheme://...`). Sem scheme, `urlparse("google.com/x")`
# devolve `netloc=''` e joga a URL inteira em `path` — o que zera todas as
# features de domínio (domain_length, subdomain_count, has_ip, has_port,
# has_suspicious_tld, has_url_shortener, domain_entropy, subdomain_ratio).
#
# Regra: se a URL não casar com `^[a-zA-Z][a-zA-Z0-9+.-]*://`, prefixamos
# `http://` antes de chamar `urlparse`. O scheme injetado é usado **apenas**
# para obter um `ParseResult` com `netloc`/`path`/`query`/`fragment`
# corretos — nunca para calcular features de texto corrido (url_length,
# contagens de caracteres, entropia), que continuam operando sobre a URL
# original (`url_clean`), sem o prefixo artificial.
_SCHEME_RE = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.\-]*)://')


def _split_explicit_scheme(url: str) -> Optional[str]:
    """
    Retorna o scheme explícito (em minúsculas) se a URL já tiver um, ou
    None caso contrário. Não decide nada sobre validade da URL — apenas
    detecta a presença literal de `scheme://` no início da string.
    """
    match = _SCHEME_RE.match(url)
    return match.group(1).lower() if match else None


def ensure_scheme(url: str) -> str:
    """
    Garante que a URL tenha um scheme antes de ser passada a `urlparse`.

    Não normaliza nada além disso (não mexe em www, portas padrão,
    tracking params etc. — isso é responsabilidade de
    `src/data/url_canonicalization.py`, usado para outros fins).
    """
    return url if _split_explicit_scheme(url) is not None else f'http://{url}'


# --- Decisão deliberada: sem feature `had_explicit_scheme` -------------------
# O SDD (§3.2) mede que a *presença de scheme* é, em vários corpora públicos,
# um artefato de como o dataset foi construído (gap de scheme de até ~98 p.p.
# entre classes no `dataset.csv` legado) — não um sinal genuíno de phishing.
# Expor "a URL tinha scheme explícito?" como feature de treino replicaria
# esse viés de construção dentro do próprio vetor de features, na contramão
# do que este SDD tenta eliminar (C1/C2, §1.3). Por isso a correção de B4 é
# só estrutural (garantir parsing correto), e deliberadamente NÃO adiciona
# `had_explicit_scheme` (nem qualquer variante) às 41 features de treino.
# Se essa informação for necessária para auditoria de viés de corpus, ela
# deve ser calculada em `diagnostics.py` diretamente sobre a URL crua — nunca
# dentro do vetor de features usado pelos modelos.


class LexicalFeatureExtractor:
    """Extrator de características léxicas de URLs."""
    
    def __init__(self, strip_scheme: bool = False):
        """Inicializa o extrator de características.

        Args:
            strip_scheme: SENS-01. Com True, as características calculadas sobre o
                texto inteiro da URL (comprimentos, contagens de caractere,
                `www_count`, palavras-chave, entropia da URL e as razões) ignoram
                o prefixo `esquema://`. `has_https` e as características de
                domínio, caminho e consulta (calculadas sobre `parsed`) não mudam.
        """
        self.strip_scheme = strip_scheme

        # Extensões suspeitas
        self.suspicious_extensions = {
            'exe', 'scr', 'bat', 'cmd', 'com', 'pif', 'jar', 'zip', 
            'rar', '7z', 'tar', 'gz', 'apk', 'deb', 'rpm'
        }
        
        # TLDs suspeitos
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'click', 'download', 'zip',
            'racing', 'space', 'review', 'country', 'kim', 'loan'
        }
        
        # Palavras-chave suspeitas
        self.suspicious_keywords = {
            'secure', 'update', 'verify', 'confirm', 'account', 'login',
            'signin', 'banking', 'paypal', 'amazon', 'microsoft', 'apple',
            'google', 'facebook', 'twitter', 'instagram', 'linkedin',
            'phishing', 'malware', 'virus', 'hack', 'crack', 'free'
        }
        
        # Domínios de encurtadores
        self.url_shorteners = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
            'is.gd', 'buff.ly', 'ift.tt', 'dlvr.it', 'soo.gd',
            'short.link', 'tiny.cc', 'rebrand.ly'
        }
        
        logger.info("Lexical feature extractor initialized")
    
    def extract_all_features(self, url: str) -> Dict[str, Union[int, float, bool]]:
        """
        Extrai todas as características léxicas de uma URL.
        
        Args:
            url: URL para análise
            
        Returns:
            Dicionário com todas as características extraídas
        """
        if not url or not isinstance(url, str):
            return self._get_empty_features()

        url_clean = url.strip()
        if not url_clean:
            return self._get_empty_features()

        try:
            # Correção B4: garante scheme apenas para o parsing estrutural.
            # `explicit_scheme` é o scheme *original* (ou None se ausente);
            # `parsed` vem da URL com scheme garantido (`ensure_scheme`),
            # mas todas as features de texto corrido abaixo recebem
            # `url_clean` (a URL original, sem o prefixo artificial), nunca
            # a versão parseada.
            explicit_scheme = _split_explicit_scheme(url_clean)
            parsed = urlparse(ensure_scheme(url_clean))
            features = {}

            # SENS-01: texto das características de texto inteiro. Com
            # `strip_scheme`, sem o prefixo `esquema://`; `explicit_scheme` e
            # `parsed` já foram calculados sobre a URL original.
            texto = _SCHEME_RE.sub('', url_clean, count=1) if self.strip_scheme else url_clean

            # Características de comprimento
            features.update(self._extract_length_features(texto, parsed))

            # Características de contagem
            features.update(self._extract_count_features(texto, parsed))

            # Características booleanas
            features.update(self._extract_boolean_features(texto, parsed, explicit_scheme))

            # Características de estrutura
            features.update(self._extract_structure_features(texto, parsed))

            # Características de domínio
            features.update(self._extract_domain_features(parsed))

            # Características de entropia
            features.update(self._extract_entropy_features(texto, parsed))

            return features

        except Exception as e:
            logger.debug(f"Error extracting features from URL '{url}': {e}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> Dict[str, Union[int, float, bool]]:
        """Retorna dicionário com características vazias/padrão."""
        return {
            # Comprimentos
            'url_length': 0,
            'domain_length': 0,
            'path_length': 0,
            'query_length': 0,
            'fragment_length': 0,
            'max_token_length': 0,
            
            # Contagens de caracteres
            'digit_count': 0,
            'letter_count': 0,
            'special_char_count': 0,
            'dot_count': 0,
            'slash_count': 0,
            'hyphen_count': 0,
            'underscore_count': 0,
            'percent_count': 0,
            'at_count': 0,
            'question_count': 0,
            'equal_count': 0,
            'ampersand_count': 0,
            'hash_count': 0,
            
            # Contagens estruturais
            'subdomain_count': 0,
            'path_tokens': 0,
            'query_params': 0,
            'www_count': 0,
            
            # Características booleanas
            'has_ip': False,
            'has_https': False,
            'has_www': False,
            'has_suspicious_tld': False,
            'has_suspicious_extension': False,
            'has_suspicious_keywords': False,
            'has_url_shortener': False,
            'has_double_slash': False,
            'has_at_symbol': False,
            'has_port': False,
            'has_query': False,
            'has_fragment': False,
            
            # Ratios
            'digit_letter_ratio': 0.0,
            'special_char_ratio': 0.0,
            'subdomain_ratio': 0.0,
            'path_domain_ratio': 0.0,
            
            # Entropia
            'url_entropy': 0.0,
            'domain_entropy': 0.0,
        }
    
    @staticmethod
    def _host_sem_porta(netloc: str) -> str:
        """Devolve o host de um `netloc`, sem userinfo e sem `:porta`.

        Feito com fatiamento de string, e **não** com `ParseResult.hostname` /
        `.port`: no Python 3.14 essas propriedades reinterpretam host entre
        colchetes como IPv6 e levantam `ValueError` em host malformado — e os
        corpora têm linhas assim (§3.2: 18 no sid321axn). Um extrator não pode
        morrer por causa do lixo que ele existe para descrever. Mesma razão da
        decisão de 04/08/2026 sobre o `diagnostics.py` medir com string.
        """
        if not netloc:
            return ''
        host = netloc.rsplit('@', 1)[-1]          # descarta userinfo
        if host.startswith('['):                  # literal IPv6: [::1]:8080
            fim = host.find(']')
            return host[:fim + 1] if fim != -1 else host
        cabeca, sep, cauda = host.rpartition(':')
        return cabeca if sep and cauda.isdigit() else host

    @classmethod
    def _has_porta_nao_padrao(cls, netloc: str) -> bool:
        """FEAT-01 e FEAT-02: há `:dígitos` depois do host e a porta não é 80 nem 443.

        O userinfo (`user:pw@`) e os dois pontos de um literal IPv6 não são porta.
        """
        if not netloc:
            return False
        sem_userinfo = netloc.rsplit('@', 1)[-1]
        resto = sem_userinfo[len(cls._host_sem_porta(netloc)):]
        if not resto.startswith(':') or not resto[1:].isdigit():
            return False
        return int(resto[1:]) not in (80, 443)

    def _extract_length_features(self, url: str, parsed) -> Dict[str, int]:
        """Extrai características de comprimento."""
        features = {
            'url_length': len(url),
            # T3.4-b: o `netloc` cru inclui os dígitos da porta, então
            # `site.com:8080` media 13 em vez de 8. Porta não é domínio.
            'domain_length': len(self._host_sem_porta(parsed.netloc)),
            'path_length': len(parsed.path) if parsed.path else 0,
            'query_length': len(parsed.query) if parsed.query else 0,
            'fragment_length': len(parsed.fragment) if parsed.fragment else 0,
        }
        
        # Comprimento máximo de token
        tokens = re.split(r'[./\-_]', url)
        features['max_token_length'] = max(len(token) for token in tokens) if tokens else 0
        
        return features
    
    def _extract_count_features(self, url: str, parsed) -> Dict[str, int]:
        """Extrai características de contagem de caracteres."""
        features = {
            'digit_count': sum(c.isdigit() for c in url),
            'letter_count': sum(c.isalpha() for c in url),
            'special_char_count': sum(c in string.punctuation for c in url),
            'dot_count': url.count('.'),
            'slash_count': url.count('/'),
            'hyphen_count': url.count('-'),
            'underscore_count': url.count('_'),
            'percent_count': url.count('%'),
            'at_count': url.count('@'),
            'question_count': url.count('?'),
            'equal_count': url.count('='),
            'ampersand_count': url.count('&'),
            'hash_count': url.count('#'),
        }
        
        # Contagens estruturais
        if parsed.netloc:
            # T3.4-c: contar pontos ingenuamente trata o sufixo composto como
            # subdomínio — `bbc.co.uk` dava 1 e `a.b.bbc.co.uk` dava 3. Medido
            # em 109.965 linhas reais: erro em 10,0% delas, com taxa variando
            # 2,6x entre corpora (PhishStorm 14,8%, kaggle03 5,6%), o que é
            # confundidor direto da C2. O `tldextract` usa a PSL e é a mesma
            # definição que `splits.py` e `diagnostics.py` já usam.
            try:
                subdominio = psl.extrator()(
                    self._host_sem_porta(parsed.netloc)
                ).subdomain
            except Exception as e:  # noqa: BLE001
                logger.debug(f"tldextract falhou em {parsed.netloc}: {e}")
                subdominio = ''
            features['subdomain_count'] = len(
                [rotulo for rotulo in subdominio.split('.') if rotulo]
            )
            features['www_count'] = url.lower().count('www')
        else:
            features['subdomain_count'] = 0
            features['www_count'] = 0
        
        # Tokens no path
        if parsed.path:
            path_tokens = [token for token in parsed.path.split('/') if token]
            features['path_tokens'] = len(path_tokens)
        else:
            features['path_tokens'] = 0
        
        # Parâmetros na query
        if parsed.query:
            features['query_params'] = len(parsed.query.split('&'))
        else:
            features['query_params'] = 0
        
        return features
    
    def _extract_boolean_features(self, url: str, parsed, explicit_scheme: Optional[str] = None) -> Dict[str, bool]:
        """Extrai características booleanas.

        Args:
            url: URL original (já limpa/stripped), sem o scheme injetado
                pela correção de B4.
            parsed: resultado de `urlparse` sobre a URL com scheme garantido.
            explicit_scheme: scheme literal da URL *original* (minúsculo),
                ou None se a URL não tinha scheme. Usado para `has_https` —
                antes o código fazia `url.lower().startswith('https')`, que
                dava falso positivo para hosts como `httpsecure.com` (sem
                scheme, mas cujo texto começa com "https") e sempre refletia
                o texto bruto, nunca o scheme real da URL.
        """
        features = {
            'has_https': explicit_scheme == 'https',
            # T3.4-a: `'www.' in url` disparava para `site.com/www.algo`, que
            # não tem www no host. Medido: falso positivo em 0,8% das linhas.
            # `www_count` é outra feature e continua contando na URL inteira.
            'has_www': self._host_sem_porta(parsed.netloc).lower().startswith('www.'),
            'has_double_slash': '//' in parsed.path if parsed.path else False,
            'has_at_symbol': '@' in url,
            'has_port': self._has_porta_nao_padrao(parsed.netloc),
            'has_query': bool(parsed.query),
            'has_fragment': bool(parsed.fragment),
        }

        return features
    
    def _extract_structure_features(self, url: str, parsed) -> Dict[str, bool]:
        """Extrai características estruturais específicas."""
        features = {}
        
        # Verifica IP no domínio
        features['has_ip'] = self._is_ip_address(self._host_sem_porta(parsed.netloc))
        
        # Verifica extensão suspeita
        if parsed.path:
            path_lower = parsed.path.lower()
            features['has_suspicious_extension'] = any(
                path_lower.endswith(f'.{ext}') for ext in self.suspicious_extensions
            )
        else:
            features['has_suspicious_extension'] = False
        
        # Verifica palavras-chave suspeitas
        url_lower = url.lower()
        features['has_suspicious_keywords'] = any(
            keyword in url_lower for keyword in self.suspicious_keywords
        )
        
        return features
    
    def _extract_domain_features(self, parsed) -> Dict[str, bool]:
        """Extrai características específicas do domínio."""
        features = {
            'has_suspicious_tld': False,
            'has_url_shortener': False,
        }
        
        if parsed.netloc:
            try:
                # Extrai informações do domínio
                extracted = psl.extrator()(parsed.netloc)
                
                # Verifica TLD suspeito
                if extracted.suffix:
                    features['has_suspicious_tld'] = extracted.suffix.lower() in self.suspicious_tlds
                
                # Verifica encurtador de URL
                if extracted.domain and extracted.suffix:
                    domain_full = f"{extracted.domain}.{extracted.suffix}"
                    features['has_url_shortener'] = domain_full.lower() in self.url_shorteners
                
            except Exception as e:
                logger.debug(f"Error extracting domain features from {parsed.netloc}: {e}")
        
        return features
    
    def _extract_entropy_features(self, url: str, parsed) -> Dict[str, float]:
        """Extrai características de entropia."""
        features = {
            'url_entropy': self._calculate_entropy(url),
            # FEAT-03: entropia do host, sem userinfo nem porta.
            'domain_entropy': self._calculate_entropy(self._host_sem_porta(parsed.netloc)),
        }
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calcula entropia de Shannon de um texto."""
        if not text:
            return 0.0
        
        # Conta frequência de caracteres
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calcula probabilidades
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return round(entropy, 4)
    
    def _is_ip_address(self, host: str) -> bool:
        """Verifica se o host (já sem userinfo e porta) é um endereço IP.

        Aceita IPv6 entre colchetes. Colchete sem fechamento não é IP.
        """
        if not host:
            return False

        if host.startswith('['):
            if not host.endswith(']'):
                return False
            host = host[1:-1]

        try:
            import ipaddress
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False
    
    def extract_ratio_features(self, features: Dict) -> Dict[str, float]:
        """
        Calcula características de ratio baseadas nas características básicas.
        
        Args:
            features: Dicionário com características básicas
            
        Returns:
            Dicionário com características de ratio
        """
        ratios = {}
        
        # Ratio dígito/letra
        total_alphanum = features['digit_count'] + features['letter_count']
        if total_alphanum > 0:
            ratios['digit_letter_ratio'] = round(features['digit_count'] / total_alphanum, 4)
        else:
            ratios['digit_letter_ratio'] = 0.0
        
        # Ratio caracteres especiais
        url_length = features['url_length']
        if url_length > 0:
            ratios['special_char_ratio'] = round(features['special_char_count'] / url_length, 4)
        else:
            ratios['special_char_ratio'] = 0.0
        
        # Ratio subdomínio
        domain_length = features['domain_length']
        if domain_length > 0:
            ratios['subdomain_ratio'] = round(features['subdomain_count'] / domain_length, 4)
        else:
            ratios['subdomain_ratio'] = 0.0
        
        # Ratio path/domain
        if domain_length > 0:
            ratios['path_domain_ratio'] = round(features['path_length'] / domain_length, 4)
        else:
            ratios['path_domain_ratio'] = 0.0
        
        return ratios


def extract_lexical_features(url: str) -> Dict[str, Union[int, float, bool]]:
    """
    Função conveniente para extrair características léxicas de uma URL.
    
    Args:
        url: URL para análise
        
    Returns:
        Dicionário com características extraídas
    """
    extractor = LexicalFeatureExtractor()
    features = extractor.extract_all_features(url)
    
    # Adiciona características de ratio
    ratios = extractor.extract_ratio_features(features)
    features.update(ratios)
    
    return features


def extract_features_batch(urls: List[str], 
                          chunk_size: int = 1000) -> pd.DataFrame:
    """
    Extrai características léxicas em lote.
    
    Args:
        urls: Lista de URLs
        chunk_size: Tamanho do chunk para processamento
        
    Returns:
        DataFrame com características extraídas
    """
    logger.info(f"Starting batch feature extraction for {len(urls)} URLs")
    
    extractor = LexicalFeatureExtractor()
    all_features = []
    
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        
        chunk_features = []
        for url in chunk:
            features = extractor.extract_all_features(url)
            ratios = extractor.extract_ratio_features(features)
            features.update(ratios)
            chunk_features.append(features)
        
        all_features.extend(chunk_features)
        
        if (i // chunk_size + 1) % 10 == 0:  # Log a cada 10 chunks
            logger.info(f"Processed {min(i + chunk_size, len(urls))} / {len(urls)} URLs")
    
    df = pd.DataFrame(all_features)
    
    logger.info(
        "Batch feature extraction completed",
        urls_processed=len(urls),
        features_extracted=len(df.columns) if len(df) > 0 else 0
    )
    
    return df


def get_feature_names() -> List[str]:
    """
    Retorna lista com nomes de todas as características.
    
    Returns:
        Lista com nomes das características
    """
    # Cria instância temporária para obter nomes das características
    extractor = LexicalFeatureExtractor()
    dummy_features = extractor._get_empty_features()
    
    # Adiciona características de ratio
    ratios = extractor.extract_ratio_features(dummy_features)
    dummy_features.update(ratios)
    
    return list(dummy_features.keys())


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Retorna características agrupadas por categoria.
    
    Returns:
        Dicionário com grupos de características
    """
    return {
        'length': [
            'url_length', 'domain_length', 'path_length', 
            'query_length', 'fragment_length', 'max_token_length'
        ],
        'counts': [
            'digit_count', 'letter_count', 'special_char_count',
            'dot_count', 'slash_count', 'hyphen_count', 'underscore_count',
            'percent_count', 'at_count', 'question_count', 'equal_count',
            'ampersand_count', 'hash_count', 'subdomain_count',
            'path_tokens', 'query_params', 'www_count'
        ],
        'boolean': [
            'has_ip', 'has_https', 'has_www', 'has_suspicious_tld',
            'has_suspicious_extension', 'has_suspicious_keywords',
            'has_url_shortener', 'has_double_slash', 'has_at_symbol',
            'has_port', 'has_query', 'has_fragment'
        ],
        'ratios': [
            'digit_letter_ratio', 'special_char_ratio',
            'subdomain_ratio', 'path_domain_ratio'
        ],
        'entropy': [
            'url_entropy', 'domain_entropy'
        ]
    }
