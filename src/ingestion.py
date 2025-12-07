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
            m_end_8 = end_pattern_8.search(text, start_idx)
            
            # También para 10-Q el corte podría ser Item 3
            end_pattern_3 = re.compile(r'item\s+3\.?\s+quantitative', re.IGNORECASE)
            m_end_3 = end_pattern_3.search(text, start_idx)
            
            indices = []
            if m_end_7a: indices.append(m_end_7a.start())
            if m_end_8: indices.append(m_end_8.start())
            if m_end_3: indices.append(m_end_3.start())
            
            if indices:
                end_idx = min(indices) # El cierre más cercano
                candidate = text[start_idx:end_idx]
                
                # Filtro de calidad simple: longitud mínima
                if len(candidate) > 1000:
                    best_text = candidate
                    break
        
        return best_text

    def process_local_filings(self, ticker: str):
        """
        Recorre los archivos descargados para el ticker, extrae MD&A y guarda en processed.
        """
        ticker_path = os.path.join(self.raw_dir, "sec-edgar-filings", ticker)
        if not os.path.exists(ticker_path):
            print(f"⚠️ No se encontraron datos crudos para {ticker}. Ejecuta download primero.")
            return

        print(f"[*] [Procesamiento] Extrayendo MD&A para {ticker}...")
        
        for root, dirs, files in os.walk(ticker_path):
            for file in files:
                if file.endswith(".txt"): # El formato que baja sec-edgar-downloader
                    full_path = os.path.join(root, file)
                    
                    # Identificar tipo (10-K o 10-Q) y año/periodo aproximado del path
                    # Path: .../10-K/ACCESSION-NUMBER/full-submission.txt
                    parts = full_path.split(os.sep)
                    doc_type = "UNKNOWN"
                    if "10-K" in parts: doc_type = "10-K"
                    elif "10-Q" in parts: doc_type = "10-Q"
                    
                    # ID único del reporte (Accession Number)
                    accession = parts[-2]
                    
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_content = f.read()
                    
                    mda_text = self.extract_mda(raw_content)
                    
                    if mda_text:
                        out_name = f"{ticker}_{doc_type}_{accession}_MDA.txt"
                        out_path = os.path.join(self.processed_dir, out_name)
                        
                        with open(out_path, 'w', encoding='utf-8') as f_out:
                            f_out.write(mda_text)
                            
                        print(f"   Reference: {doc_type} {accession} -> Guardado ({len(mda_text)} chars)")
                    else:
                        print(f"   [!] No se pudo extraer MD&A de {doc_type} {accession}")

if __name__ == "__main__":
    # Prueba unitaria del módulo
    import sys
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data")
    
    loader = SECLoader(data_dir=data_path)
    
    # Prueba con Tesla (TSLA)
    TICKER = "TSLA"
    loader.download_filings(TICKER, amount=1)
    loader.process_local_filings(TICKER)
