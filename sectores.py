# @title 1. Instalación de Dependencias
!pip install sec-edgar-downloader transformers torch pandas numpy matplotlib seaborn beautifulsoup4 yfinance
# @title 2. Definición del Motor (Core Engine v2.1)

import os
import re
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from collections import Counter
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from transformers import BertTokenizer, BertForSequenceClassification

# Configurar estilos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [14, 7]

# --- MÓDULO 1: INGESTA ---
class SECLoader:
    def __init__(self, data_dir="data", email="research@example.com", company="Personal Research"):
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        self.downloader = Downloader(company, email, os.path.join(data_dir, "raw"))

    def download_filings(self, ticker: str, amount: int = 1):
        path = os.path.join(self.data_dir, "raw", "sec-edgar-filings", ticker)
        print(f"[>] ({ticker}) Iniciando descarga de hasta {amount} reportes...")
        try:
            self.downloader.get("10-K", ticker, limit=amount)
        except Exception as e:
            print(f"[!] Error en descarga {ticker}: {e}")

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
        # Ordenar por fecha reciente
        return sorted(processed_data, key=lambda x: x.get('date', '1900'), reverse=True)

# --- MÓDULO 2 NOISE FILTER ---
class TextPreprocessor:
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return "".join(ch for ch in text if ch.isprintable()).strip()
    
    def is_legal_noise(self, sentence: str) -> bool:
        legal_keywords = [
            'forward-looking', 'safe harbor', 'uncertainty', 'may differ', 
            'subject to error', 'actual results', 'factors that could cause',
            'statements regarding', 'cautionary note', 'risk factors', 
            'include but are not limited to', 'assumptions'
        ]
        return any(kw in sentence.lower() for kw in legal_keywords)

    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and len(s.split()) >= 4:
                if not self.is_legal_noise(s):
                    valid_sentences.append(s)
        return valid_sentences

# --- MÓDULO 3 FINBERT ---
class FinBertModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Cargando FinBERT en {self.device}...")
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        self.labels = {0: 'positive', 1: 'negative', 2: 'neutral'}

    def predict(self, sentences):
        if not sentences: return pd.DataFrame()
        batch_size = 32
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            for j, s in enumerate(batch):
                results.append({
                    "sentence": s,
                    "pos_val": probs[j][0],
                    "neg_val": probs[j][1]
                })
        return pd.DataFrame(results)

# @title 3. Ejecución Batch con Selector de Tickers

# Inicializar Modelos
print("[*] Inicializando Motor NLP...")
loader = SECLoader()
model = FinBertModel()
prep = TextPreprocessor()

# --- CONFIGURACIÓN DE USUARIO ---
TICKERS_INPUT = "META, AAPL, MSFT, GOOGL, AMZN" # @param {type:"string"}
NUM_REPORTS = 5 # @param {type:"integer"}

# Parsear tickers (separados por coma)
TICKERS = [t.strip().upper() for t in TICKERS_INPUT.split(',') if t.strip()]
print(f"\n📝 TARGETS: {TICKERS}")
print(f"📑 REPORTES SOLICITADOS (HISTÓRICO): {NUM_REPORTS}")

def analyze_ticker_history(ticker_symbol, amount=1):
    """Función Maestra: Procesa múltiples reportes históricos para un ticker."""
    print(f"\n" + "-"*50)
    print(f" 🏢 PROCESANDO AGENTE: {ticker_symbol} (Historic)")
    print("-"*50)
    try:
        # 1. Download
        loader.download_filings(ticker_symbol, amount=amount)
        
        # 2. Process
        docs = loader.process_filings(ticker_symbol)
        if not docs:
            print(f"[!] No se encontraron reportes legibles para {ticker_symbol}")
            return []
            
        reports_to_scan = docs[:amount]
        print(f"[>] Se encontraron {len(reports_to_scan)} reportes para analizar.")
        
        history_results = []
        
        for doc in reports_to_scan:
            report_date = doc.get('date', 'Unknown')
            print(f"   --> Analizando reporte del {report_date}...")
            
            # 3. Clean & Predict
            sentences = prep.split_sentences(prep.clean_text(doc['text']))
            if len(sentences) < 10:
                continue # Skip low-quality docs
                
            df = model.predict(sentences)
            
            if not df.empty:
                # 4. Calculate Metrics
                net_score = df['pos_val'].mean() - df['neg_val'].mean()
                history_results.append({
                    'ticker': ticker_symbol,
                    'date': report_date,
                    'net_score': net_score
                })
        
        return history_results
        
    except Exception as e:
        print(f"[ERROR] Fallo crítico en {ticker_symbol}: {e}")
        return []

# --- BUCLE PRINCIPAL ---
full_history_data = []

for t in TICKERS:
    results = analyze_ticker_history(t, amount=NUM_REPORTS)
    full_history_data.extend(results)

# Construir DataFrame Maestro
df_history = pd.DataFrame(full_history_data)

if not df_history.empty:
    # --- DATA PREP PARA VISUALIZACIÓN ---
    df_history['date_obj'] = pd.to_datetime(df_history['date'])
    df_history['year'] = df_history['date_obj'].dt.year
    
    # Calcular Z-Score global (para comparabilidad)
    global_mean = df_history['net_score'].mean()
    global_std = df_history['net_score'].std()
    if global_std == 0: global_std = 1
    df_history['z_score'] = (df_history['net_score'] - global_mean) / global_std

    # --- VISUALIZACIÓN 1: MACRO HEATMAP (Sentimiento Histórico) ---
    print("\n" + "="*40)
    print(" MACRO SENTIMENT HEATMAP")
    print("="*40)
    
    # Pivot Table: Index=Ticker, Columns=Year, Values=Z-Score
    # Agrupamos por año tomando el promedio si hay multiples reportes en un año
    pivot_table = df_history.pivot_table(index='ticker', columns='year', values='z_score', aggfunc='mean')
    
    plt.figure(figsize=(12, len(TICKERS)*1.2 + 2))
    sns.heatmap(
        pivot_table, 
        cmap='RdYlGn', 
        center=0, 
        annot=True, 
        fmt=".2f", 
        linewidths=.5, 
        cbar_kws={'label': 'Z-Score (Desviación del Promedio)'}
    )
    plt.title('Evolución Histórica del Sentimiento Sectorial', fontsize=16, fontweight='bold')
    plt.ylabel('Empresa')
    plt.xlabel('Año Fiscal')
    plt.show()

    # --- VISUALIZACIÓN 2: ALPHA HUNTER (Solo último año disponible) ---
    # Filtramos para quedarnos con el registro más reciente de cada ticker para el scatter plot
    latest_indices = df_history.groupby('ticker')['date_obj'].idxmax()
    df_sector_latest = df_history.loc[latest_indices].copy()
    
    print("\n[*] Calculando Alpha (Market Returns) para el periodo más reciente...")
    
    # Descarga optimizada de precios
    min_date = df_sector_latest['date_obj'].min() - timedelta(days=5)
    max_date = datetime.now() + timedelta(days=1)
    tickers_list = df_sector_latest['ticker'].tolist()
    
    try:
        market_data = yf.download(tickers_list, start=min_date, end=max_date, progress=False)['Close']
        if len(tickers_list) == 1: market_data = pd.DataFrame({tickers_list[0]: market_data})

        perf_list = []
        for idx, row in df_sector_latest.iterrows():
            t = row['ticker']
            d_start = row['date_obj']
            d_end = d_start + timedelta(days=180)
            if d_end > datetime.now(): d_end = datetime.now() - timedelta(days=1)
            
            if t in market_data.columns:
                ts = market_data[t].dropna()
                future_prices = ts[ts.index >= d_start]
                past_prices = ts[ts.index <= d_end]
                
                if not future_prices.empty and not past_prices.empty:
                    p_start = future_prices.iloc[0]
                    p_end = past_prices.iloc[-1]
                    perf_list.append((p_end - p_start) / p_start * 100)
                else:
                    perf_list.append(0.0)
            else:
                perf_list.append(0.0)
                
        df_sector_latest['price_return_6m'] = perf_list
            
    except Exception as e:
        print(f"[!] Error Market Data: {e}")
        df_sector_latest['price_return_6m'] = 0.0

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_sector_latest, x='z_score', y='price_return_6m', 
        s=150, color='#2E86AB', edgecolor='black', alpha=0.8
    )
    for i in range(df_sector_latest.shape[0]):
        plt.text(
            x=df_sector_latest.z_score.iloc[i]+0.02, 
            y=df_sector_latest.price_return_6m.iloc[i]+0.5, 
            s=df_sector_latest.ticker.iloc[i], 
            fontweight='bold', fontsize=10
        )
    plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.title('Alpha Hunter: IA Sentiment vs 6-Month Post-Filing Return (Latest Report)', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Z-Score (IA)', fontsize=12)
    plt.ylabel('Post-Filing Price Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("[!] No data for visualization.")