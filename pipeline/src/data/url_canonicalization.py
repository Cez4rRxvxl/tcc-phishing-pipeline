"""Canonização de URLs para padronização e normalização."""

import re
import urllib.parse
from typing import Dict, Optional, Union
from urllib.parse import urlparse, parse_qs, urlencode, unquote
import ipaddress

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psl
from utils.logging import get_logger

logger = get_logger(__name__)


class URLCanonicalizer:
    """Classe para canonização e normalização de URLs."""
    
    def __init__(self):
        """Inicializa o canonizador."""
        # Portas padrão por protocolo
        self.default_ports = {
            'http': 80,
            'https': 443,
            'ftp': 21,
        }
        
        # Parâmetros de query a serem removidos (UTM, tracking, etc.)
        self.tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'gclsrc', 'dclid', 'gbraid', 'wbraid',
            'ref', 'source', 'medium', 'campaign',
            '_ga', '_gac', '_gid', '_gcl_au', '_gcl_aw', '_gcl_dc',
            'mc_cid', 'mc_eid',  # MailChimp
            'msclkid',  # Microsoft Advertising
            'twclid',   # Twitter
            'li_fat_id',  # LinkedIn
            'igshid',   # Instagram
        }
        
        logger.info("URL Canonicalizer initialized")
    
    def canonicalize_url(self, url: str) -> Dict[str, Optional[Union[str, bool]]]:
        """
        Canoniza uma URL extraindo e normalizando seus componentes.
        
        Args:
            url: URL original para canonizar
            
        Returns:
            Dicionário com componentes canonizados da URL e flag de validade
        """
        if not url or not isinstance(url, str):
            return self._invalid_result('empty')
        
        # Limpeza robusta da URL antes do processamento
        url = self._clean_url(url)
        if not url:
            return self._invalid_result('empty')
        
        try:
            # Adiciona protocolo se ausente
            if not url.startswith(('http://', 'https://', 'ftp://')):
                url = 'http://' + url
            
            # Parse inicial da URL
            parsed = urlparse(url)
            
            # Verifica se tem host
            if not parsed.netloc:
                return self._invalid_result('no_host')
            
            # Extrai host limpo (sem porta)
            host = self._extract_clean_host(parsed.netloc)
            if not host:
                return self._invalid_result('malformed')
            
            # Extrai componentes
            domain = self._extract_domain_robust(parsed, host)
            etld_plus_one = self._extract_etld_plus_one_robust(parsed, host)
            
            # Verifica se domínio é válido
            if not domain or not etld_plus_one:
                return self._invalid_result('invalid_domain')
            
            result = {
                'url_norm': self._normalize_url(parsed),
                'domain': domain,
                'etld_plus_one': etld_plus_one,
                'path': self._normalize_path(parsed.path),
                'query': self._normalize_query(parsed.query),
                'fragment': self._normalize_fragment(parsed.fragment),
                'invalid_url': False,
                'invalid_reason': None,
            }
            
            logger.debug(f"URL canonicalized: {url} -> {result['url_norm']}")
            return result
            
        except Exception as e:
            logger.warning(f"Error canonicalizing URL '{url}': {e}")
            return self._invalid_result('exception')
    
    def _empty_result(self) -> Dict[str, Optional[Union[str, bool]]]:
        """Retorna resultado vazio."""
        return {
            'url_norm': None,
            'domain': None,
            'etld_plus_one': None,
            'path': None,
            'query': None,
            'fragment': None,
            'invalid_url': False,
            'invalid_reason': None,
        }
    
    def _invalid_result(self, reason: str) -> Dict[str, Optional[Union[str, bool]]]:
        """Retorna resultado com flag de URL inválida."""
        return {
            'url_norm': None,
            'domain': None,
            'etld_plus_one': None,
            'path': None,
            'query': None,
            'fragment': None,
            'invalid_url': True,
            'invalid_reason': reason,
        }
    
    def _clean_url(self, url: str) -> str:
        """
        Limpeza robusta da URL antes do processamento.
        
        Args:
            url: URL original
            
        Returns:
            URL limpa ou string vazia se inválida
        """
        if not url:
            return ''
        
        # Aplica strip
        url = url.strip()
        
        # Remove aspas simples/duplas ao redor
        if len(url) >= 2:
            if (url.startswith('"') and url.endswith('"')) or \
               (url.startswith("'") and url.endswith("'")):
                url = url[1:-1].strip()
        
        # Colapsa espaços internos
        url = re.sub(r'\s+', ' ', url)
        url = url.replace(' ', '')  # Remove todos os espaços
        
        return url.strip()
    
    def _extract_clean_host(self, netloc: str) -> str:
        """
        Extrai host limpo do netloc.
        
        Args:
            netloc: Network location da URL
            
        Returns:
            Host limpo ou string vazia se inválido
        """
        if not netloc:
            return ''
        
        # Remove porta se presente
        if ':' in netloc:
            host = netloc.split(':')[0]
        else:
            host = netloc
        
        # Remove www.
        if host.lower().startswith('www.'):
            host = host[4:]
        
        return host.lower().strip()
    
    def _extract_domain_robust(self, parsed: urllib.parse.ParseResult, host: str) -> Optional[str]:
        """
        Extrai domínio com robustez para IPs e casos especiais.
        
        Args:
            parsed: URL parseada
            host: Host limpo
            
        Returns:
            Domínio ou None se inválido
        """
        if not host:
            return None
        
        # Verifica se é IP
        if self._is_ip_address(host):
            return host  # Para IPs, domain = ip
        
        # Para domínios normais
        return host if host else None
    
    def _extract_etld_plus_one_robust(self, parsed: urllib.parse.ParseResult, host: str) -> Optional[str]:
        """
        Extrai eTLD+1 com robustez para IPs e tratamento de falhas.
        
        Args:
            parsed: URL parseada
            host: Host limpo
            
        Returns:
            eTLD+1 ou None se inválido
        """
        if not host:
            return None
        
        # Verifica se é IP
        if self._is_ip_address(host):
            return host  # Para IPs, etld_plus_one = ip
        
        try:
            # Tenta extrair eTLD+1 com tldextract
            extracted = psl.extrator()(host)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}".lower()
            
            # Se tldextract falhou ou retornou vazio, tenta punycode
            try:
                host_punycode = host.encode('idna').decode('ascii')
                extracted = psl.extrator()(host_punycode)
                if extracted.domain and extracted.suffix:
                    return f"{extracted.domain}.{extracted.suffix}".lower()
            except (UnicodeError, UnicodeDecodeError):
                pass
            
            # Remove caracteres não-imprimíveis e tenta novamente
            host_clean = ''.join(char for char in host if char.isprintable())
            if host_clean != host:
                extracted = psl.extrator()(host_clean)
                if extracted.domain and extracted.suffix:
                    return f"{extracted.domain}.{extracted.suffix}".lower()
            
            # Se ainda assim falhou, mas temos um host válido, usa como fallback
            if '.' in host and len(host) > 3:
                parts = host.split('.')
                if len(parts) >= 2:
                    return f"{parts[-2]}.{parts[-1]}".lower()
            
        except Exception as e:
            logger.debug(f"Error extracting eTLD+1 from {host}: {e}")
        
        return None
    
    def _is_ip_address(self, host: str) -> bool:
        """
        Verifica se host é um endereço IP (IPv4 ou IPv6).
        
        Args:
            host: Host para verificar
            
        Returns:
            True se for IP, False caso contrário
        """
        try:
            # Remove colchetes do IPv6 se presentes
            clean_host = host.strip('[]')
            ipaddress.ip_address(clean_host)
            return True
        except ValueError:
            return False
    
    def _normalize_url(self, parsed: urllib.parse.ParseResult) -> str:
        """Normaliza URL completa."""
        # Normaliza esquema
        scheme = parsed.scheme.lower() if parsed.scheme else 'http'
        
        # Normaliza netloc
        netloc = self._normalize_netloc(parsed.netloc)
        
        # Normaliza path
        path = self._normalize_path(parsed.path)
        
        # Normaliza query
        query = self._normalize_query(parsed.query)
        
        # Reconstrói URL
        url_parts = [scheme, '://', netloc]
        
        if path and path != '/':
            url_parts.append(path)
        
        if query:
            url_parts.extend(['?', query])
        
        url_norm = ''.join(url_parts)
        
        # Remove trailing slash se não for root
        if url_norm.endswith('/') and len(url_norm) > len(scheme) + 4:  # mais que scheme://
            url_norm = url_norm.rstrip('/')
        
        return url_norm
    
    def _normalize_netloc(self, netloc: str) -> str:
        """Normaliza netloc (host:port)."""
        if not netloc:
            return ''
        
        # Remove www. prefix
        if netloc.lower().startswith('www.'):
            netloc = netloc[4:]
        
        # Separa host e porta
        if ':' in netloc:
            host, port = netloc.rsplit(':', 1)
            try:
                port_num = int(port)
                # Remove porta padrão
                if port_num in self.default_ports.values():
                    scheme_for_port = next(
                        (k for k, v in self.default_ports.items() if v == port_num), 
                        None
                    )
                    if scheme_for_port:
                        netloc = host
                    else:
                        netloc = f"{host}:{port}"
                else:
                    netloc = f"{host}:{port}"
            except ValueError:
                # Porta inválida, mantém original
                pass
        else:
            host = netloc
        
        # Converte para lowercase
        host = host.lower()
        
        # Converte para punycode se necessário
        try:
            host = host.encode('idna').decode('ascii')
        except (UnicodeError, UnicodeDecodeError):
            # Se falhar, mantém original
            pass
        
        return netloc.lower()
    
    def _extract_domain(self, parsed: urllib.parse.ParseResult) -> Optional[str]:
        """Extrai FQDN (domínio completo)."""
        netloc = parsed.netloc
        if not netloc:
            return None
        
        # Remove porta se presente
        if ':' in netloc:
            netloc = netloc.split(':')[0]
        
        # Remove www.
        if netloc.lower().startswith('www.'):
            netloc = netloc[4:]
        
        return netloc.lower() if netloc else None
    
    def _extract_etld_plus_one(self, parsed: urllib.parse.ParseResult) -> Optional[str]:
        """Extrai eTLD+1 (domínio registrável)."""
        netloc = parsed.netloc
        if not netloc:
            return None
        
        # Remove porta se presente
        if ':' in netloc:
            netloc = netloc.split(':')[0]
        
        try:
            extracted = psl.extrator()(netloc)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}".lower()
        except Exception as e:
            logger.debug(f"Error extracting eTLD+1 from {netloc}: {e}")
        
        return None
    
    def _normalize_path(self, path: str) -> Optional[str]:
        """Normaliza caminho da URL."""
        if not path or path == '/':
            return None
        
        # URL decode
        try:
            path = unquote(path)
        except Exception:
            pass
        
        # Remove múltiplas barras
        path = re.sub(r'/+', '/', path)
        
        # Remove trailing slash
        if path.endswith('/') and len(path) > 1:
            path = path.rstrip('/')
        
        return path if path != '/' else None
    
    def _normalize_query(self, query: str) -> Optional[str]:
        """Normaliza query string."""
        if not query:
            return None
        
        try:
            # Parse query parameters
            params = parse_qs(query, keep_blank_values=True)
            
            # Remove parâmetros de tracking
            filtered_params = {
                k: v for k, v in params.items() 
                if k.lower() not in self.tracking_params
            }
            
            if not filtered_params:
                return None
            
            # Ordena parâmetros por chave
            sorted_params = sorted(filtered_params.items())
            
            # Reconstrói query string
            query_parts = []
            for key, values in sorted_params:
                for value in values:
                    if value:  # Remove valores vazios
                        query_parts.append(f"{key}={value}")
                    else:
                        query_parts.append(key)
            
            return '&'.join(query_parts) if query_parts else None
            
        except Exception as e:
            logger.debug(f"Error normalizing query '{query}': {e}")
            return query
    
    def _normalize_fragment(self, fragment: str) -> Optional[str]:
        """Normaliza fragmento da URL."""
        if not fragment:
            return None
        
        # URL decode
        try:
            fragment = unquote(fragment)
        except Exception:
            pass
        
        return fragment.strip() if fragment.strip() else None
    
    def is_ip_address(self, host: str) -> bool:
        """Verifica se host é um endereço IP (método público)."""
        return self._is_ip_address(host)
    
    def extract_file_extension(self, path: str) -> Optional[str]:
        """Extrai extensão do arquivo da URL."""
        if not path:
            return None
        
        # Remove query e fragment se presentes
        path = path.split('?')[0].split('#')[0]
        
        if '.' in path:
            extension = path.split('.')[-1].lower()
            # Valida se é uma extensão válida (apenas letras/números)
            if re.match(r'^[a-z0-9]+$', extension) and len(extension) <= 10:
                return extension
        
        return None
    
    def has_suspicious_patterns(self, url: str) -> Dict[str, bool]:
        """
        Detecta padrões suspeitos na URL.
        
        Args:
            url: URL para analisar
            
        Returns:
            Dicionário com flags de padrões suspeitos
        """
        flags = {
            'has_ip_in_domain': False,
            'has_double_slash': False,
            'has_at_symbol': False,
            'has_dash_in_domain': False,
            'has_long_subdomain': False,
            'has_suspicious_tld': False,
            'has_url_shortener': False,
            'has_suspicious_keywords': False,
        }
        
        try:
            parsed = urlparse(url)
            domain = self._extract_domain(parsed)
            
            if domain:
                # IP no domínio
                flags['has_ip_in_domain'] = self.is_ip_address(domain)
                
                # Hífen no domínio
                flags['has_dash_in_domain'] = '-' in domain
                
                # Subdomínio longo
                parts = domain.split('.')
                if len(parts) > 2:
                    max_subdomain_len = max(len(part) for part in parts[:-2])
                    flags['has_long_subdomain'] = max_subdomain_len > 20
                
                # TLD suspeito
                extracted = psl.extrator()(domain)
                suspicious_tlds = {
                    'tk', 'ml', 'ga', 'cf', 'click', 'download', 'zip',
                    'racing', 'space', 'review', 'country', 'kim'
                }
                flags['has_suspicious_tld'] = extracted.suffix in suspicious_tlds
                
                # URL shorteners
                shortener_domains = {
                    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
                    'is.gd', 'buff.ly', 'ift.tt', 'dlvr.it', 'soo.gd'
                }
                flags['has_url_shortener'] = any(
                    shortener in domain for shortener in shortener_domains
                )
            
            # Dupla barra no path
            if parsed.path:
                flags['has_double_slash'] = '//' in parsed.path
            
            # Símbolo @ na URL
            flags['has_at_symbol'] = '@' in url
            
            # Palavras-chave suspeitas
            suspicious_keywords = [
                'phishing', 'secure', 'update', 'verify', 'confirm',
                'account', 'login', 'signin', 'banking', 'paypal',
                'amazon', 'microsoft', 'apple', 'google', 'facebook'
            ]
            url_lower = url.lower()
            flags['has_suspicious_keywords'] = any(
                keyword in url_lower for keyword in suspicious_keywords
            )
            
        except Exception as e:
            logger.debug(f"Error detecting suspicious patterns in '{url}': {e}")
        
        return flags


# Instância global do canonizador
canonicalizer = URLCanonicalizer()


def canonicalize_url(url: str) -> Dict[str, Optional[Union[str, bool]]]:
    """
    Função conveniente para canonizar URL.
    
    Args:
        url: URL para canonizar
        
    Returns:
        Dicionário com componentes canonizados e flag de validade
    """
    return canonicalizer.canonicalize_url(url)


def batch_canonicalize_urls(urls: list[str], chunk_size: int = 1000) -> list[Dict[str, Optional[Union[str, bool]]]]:
    """
    Canoniza URLs em lote.
    
    Args:
        urls: Lista de URLs para canonizar
        chunk_size: Tamanho do chunk para processamento
        
    Returns:
        Lista de dicionários com componentes canonizados e flags de validade
    """
    results = []
    
    logger.info(f"Starting batch canonicalization of {len(urls)} URLs")
    
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunk_results = [canonicalizer.canonicalize_url(url) for url in chunk]
        results.extend(chunk_results)
        
        if (i // chunk_size + 1) % 10 == 0:  # Log a cada 10 chunks
            logger.info(f"Processed {min(i + chunk_size, len(urls))} / {len(urls)} URLs")
    
    logger.info(f"Batch canonicalization completed for {len(urls)} URLs")
    return results
