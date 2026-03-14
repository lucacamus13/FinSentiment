import os
import re
from typing import Optional
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup

class SECLoader:
    """
    Módulo encargado de la interacción con la SEC y la ingesta de documentos crudos.
    """
    def __init__(self, data_dir: str, email_contact: str = "user@example.com"):
        """
        Inicializa el descargador SEC.
        Args:
            data_dir (str): Directorio raíz para datos.
            email_contact (str): Email requerido por la SEC para el User-Agent.
        """
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.downloader = Downloader("FinSentimentApp", email_contact, self.raw_dir)
        
        # Asegurar existencia de directorios
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_filings(self, ticker: str, amount: int = 2) -> None:
        """
        Descarga los últimos N reportes 10-K y 10-Q para un ticker.
        """
        print(f"[>] [Ingesta] Iniciando descarga para {ticker}...")
        
        try:
            # Descargar 10-K (Anuales)
            print(f"   - Descargando {amount} últimos 10-K...")
            self.downloader.get("10-K", ticker, limit=amount)
            
            # Descargar 10-Q (Trimestrales)
            print(f"   - Descargando {amount} últimos 10-Q...")
            self.downloader.get("10-Q", ticker, limit=amount)
            
            print(f"[+] Descarga completa para {ticker}.")
            
        except Exception as e:
            print(f"[!] Error en descarga para {ticker}: {e}")

    def extract_mda(self, html_content: str) -> Optional[str]:
        """
        Extrae la sección 'Item 7. Management's Discussion and Analysis' del HTML crudo.
        Lógica adaptada para ser tolerante a fallos.
        """
        # Parsing inicial con BeautifulSoup
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except:
            soup = BeautifulSoup(html_content, 'html.parser')
            
        text = soup.get_text(separator=' ')
        
        # Regex Patterns
        # Buscamos variaciones de "Item 7" seguido de "Management's..."
        start_pattern = re.compile(r'item\s+7\.?\s+management', re.IGNORECASE)
        # El fin suele ser "Item 7A" (Riesgo de Mercado) o "Item 8" (Estados Financieros)
        end_pattern_7a = re.compile(r'item\s+7a\.?\s+quantitative', re.IGNORECASE)
        end_pattern_8 = re.compile(r'item\s+8\.?\s+financial', re.IGNORECASE)

        matches_start = list(start_pattern.finditer(text))
        if not matches_start:
            # Fallback para 10-Q: Item 2 es MD&A
            start_pattern_q = re.compile(r'item\s+2\.?\s+management', re.IGNORECASE)
            matches_start = list(start_pattern_q.finditer(text))
            
            if not matches_start:
                return None

        # Estrategia: Buscar desde el final (para evitar índices al principio)
        best_text = None
        
        for m_start in reversed(matches_start):
            start_idx = m_start.end()
            
            # Buscar puntos de corte
            m_end_7a = end_pattern_7a.search(text, start_idx)
            print(f"[!] Error en descarga: {e}")

    def extract_date(self, content: str) -> str:
        patterns = [
            r'FILED AS OF DATE:\s+(\d{8})',
            r'CONFORMED PERIOD OF REPORT:\s+(\d{8})'
        ]
        for p in patterns:
            match = re.search(p, content)
            if match:
                date_str = match.group(1)
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return None

    def extract_mda(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator='\n')
        patterns = [r'Item\s+7\.\s+Management', r"Management's\s+Discussion", r'Item\s+7\.']
        start_idx = -1
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match: start_idx = match.start(); break
        if start_idx == -1: return text[:50000]
        return text[start_idx:start_idx+30000]

    def process_filings(self, ticker: str):
        raw_path = os.path.join(self.data_dir, "raw", "sec-edgar-filings", ticker)
        processed_data = []
        for root, _, files in os.walk(raw_path):
            for file in files:
                if file.lower().endswith(".txt") and "primary" not in file:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        mda = self.extract_mda(content)
                        date = self.extract_date(content)
                        if len(mda) > 500:
                            processed_data.append({'text': mda, 'date': date, 'accession': file})
                    except: pass
        return processed_data
