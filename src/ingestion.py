import os
import re
from typing import Optional
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup

class SECLoader:
    """
    Módulo encargado de la interacción con la SEC y la ingesta de documentos crudos.
    """
    def __init__(self, data_dir: str = "data", email_contact: str = "user@example.com"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # sec-edgar-downloader guarda en raw_dir/sec-edgar-filings
        self.downloader = Downloader("FinSentimentApp", email_contact, self.raw_dir)
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_filings(self, ticker: str, amount: int = 2) -> None:
        print(f"[>] [Ingesta] Iniciando descarga para {ticker}...")
        try:
            half = max(1, amount // 2)
            self.downloader.get("10-K", ticker, limit=half)
            self.downloader.get("10-Q", ticker, limit=amount - half)
            print(f"[+] Descarga completa para {ticker}.")
        except Exception as e:
            print(f"[!] Error en descarga para {ticker}: {e}")

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
        return "2023-01-01"

    def extract_mda(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        text = soup.get_text(separator='\n')
        # Buscamos variaciones de "Item 7" o "Management's Discussion"
        patterns = [
            r'Item\s+7\.\s+Management', 
            r"Management's\s+Discussion", 
            r'Item\s+7\.'
        ]
        start_idx = -1
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match: 
                start_idx = match.start()
                break
                
        if start_idx == -1: 
            return text[:50000] # Fallback: return first 50k chars
            
        return text[start_idx:start_idx+30000] # Assume MD&A is within 30k chars

    def process_filings(self, ticker: str):
        raw_path = os.path.join(self.raw_dir, "sec-edgar-filings", ticker)
        processed_data = []
        
        if not os.path.exists(raw_path):
            return processed_data
            
        for root, _, files in os.walk(raw_path):
            for file in files:
                if file.lower().endswith(".txt") and "primary" not in file:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        mda_text = self.extract_mda(content)
                        date_str = self.extract_date(content)
                        
                        if len(mda_text) > 500:
                            processed_data.append({
                                'text': mda_text, 
                                'date': date_str, 
                                'accession': file
                            })
                    except Exception as e: 
                        print(f"Error procesando {file}: {e}")
                        
        return processed_data
