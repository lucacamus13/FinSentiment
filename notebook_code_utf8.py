# @title 1. Instalaci¾n de Dependencias
!pip install sec-edgar-downloader transformers torch pandas numpy matplotlib seaborn beautifulsoup4 yfinance
# @title 2. Definici¾n del Motor (Core Engine v2.1)

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

# --- MËDULO 1: INGESTA ---
class SECLoader:
    def __init__(self, data_dir="data", email="research@example.com", company="Personal Research"):
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        self.downloader = Downloader(company, email, os.path.join(data_dir, "raw"))

    def download_filings(self, ticker: str, amount: int = 2):
        path = os.path.join(self.data_dir, "raw", "sec-edgar-filings", ticker)
        if os.path.exists(path):
             print(f"[>] Archivos para {ticker} ya descargados.")
             return

        print(f"[>] Descargando {amount} reportes para {ticker}...")
        try:
            self.downloader.get("10-K", ticker, limit=amount)
            print("[+] Descarga completa.")
        except Exception as e:
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

# --- MËDULO 2 PREPROCESAMIENTO (CON FILTRO LEGAL) ---
class TextPreprocessor:
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return "".join(ch for ch in text if ch.isprintable()).strip()

    def is_legal_noise(self, sentence: str) -> bool:
        # Palabras clave de disclaimers y riesgo legal
        legal_keywords = [
            'forward-looking', 'safe harbor', 'uncertainty', 'may differ',
            'subject to error', 'actual results', 'factors that could cause',
            'statements regarding', 'cautionary note', 'risk factors',
            'include but are not limited to', 'assumptions'
        ]
        s_lower = sentence.lower()
        return any(kw in s_lower for kw in legal_keywords)

    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            # Filtro 1: Longitud
            if len(s) > 20 and len(s.split()) >= 4:
                # Filtro 2: Ruido Legal (Falso Negativo Commmon Source)
                if not self.is_legal_noise(s):
                    valid_sentences.append(s)
        return valid_sentences

# --- MËDULO 3: MODELO FINBERT ---
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
                idx = np.argmax(probs[j])
                results.append({
                    "sentence": s,
                    "sentiment": self.labels[idx],
                    "pos_val": probs[j][0],
                    "neg_val": probs[j][1]
                })
        return pd.DataFrame(results)

# --- MËDULO 4: DATA Y ANALYTICS ---
def get_market_data(ticker, start_date, end_date):
    print(f"[*] Descargando precios de {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df['Close']

def analyze_keywords(sentences, top_k=10):
    # Stopwords expandidas (Legal + Financiero GenÚrico)
    stopwords = set([
        'the', 'and', 'of', 'to', 'in', 'a', 'that', 'for', 'is', 'on', 'with', 'as',
        'our', 'we', 'are', 'by', 'it', 'from', 'an', 'be', 'files', 'company',
        # Solicitadas por el usuario:
        'may', 'such', 'period', 'results', 'year', 'quarter', 'other', 'have', 'million', 'billion'
    ])
    words = []
    for s in sentences:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', s)
        words.extend([w.lower() for w in tokens if w.lower() not in stopwords])
    return Counter(words).most_common(top_k)

def show_dashboard(ticker, results_df, market_data=None):
    print("\n" + "="*60 + f"\n SCOREBOARD: {ticker}\n" + "="*60)

    # 1. Normalizaci¾n (Z-Score)
    summary = results_df.groupby('date')[['pos_val', 'neg_val']].mean()
    summary['net_score'] = summary['pos_val'] - summary['neg_val']

    # Calcular Z-Score para ver cambios relativos
    mean_score = summary['net_score'].mean()
    std_score = summary['net_score'].std()
    if std_score == 0: std_score = 1 # Evitar Div0
    summary['z_score'] = (summary['net_score'] - mean_score) / std_score

    # Mostrar tabla coloreada por Z-Score (lo importante es si sube o baja vs media)
    print("[MÚtrica Clave] Z-Score: Desviaci¾n Estßndar respecto a la media hist¾rica.")
    display(summary.style.background_gradient(cmap='RdYlGn', subset=['z_score']))

    # 2. Visualizaci¾n Precio vs Sentimiento Normalizado
    if market_data is not None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = '#2E86AB'
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Sentiment Z-Score (Normalizado)', color=color, fontweight='bold')
        # Graficar Z-Score en lugar de raw score
        ax1.plot(pd.to_datetime(summary.index), summary['z_score'], 'o-', color=color, linewidth=2)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Media Hist¾rica')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = '#F24236'
        ax2.set_ylabel(f'Precio {ticker} ($)', color=color, fontweight='bold')
        ax2.plot(market_data.index, market_data, color=color, alpha=0.6, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"{ticker}: Precio vs Sentimiento Relativo (Z-Score)", fontsize=14)
        plt.show()

    # 3. Deep Dive (┌ltimo reporte)
    latest_date = results_df['date'].max()
    latest_df = results_df[results_df['date'] == latest_date]

    print(f"\n>>> DEEP DIVE (Reporte {latest_date})")
    print("\n[TOP 5 POSITIVAS]")
    for _, r in latest_df.nlargest(5, 'pos_val').iterrows():
        print(f"(+) {r['pos_val']:.2f}: {r['sentence'][:150]}...")

    print("\n[TOP 5 NEGATIVAS - FILTRADAS]")
    for _, r in latest_df.nlargest(5, 'neg_val').iterrows():
        print(f"(-) {r['neg_val']:.2f}: {r['sentence'][:150]}...")

    pos_text = latest_df[latest_df['sentiment']=='positive']['sentence']
    neg_text = latest_df[latest_df['sentiment']=='negative']['sentence']
    print(f"\n[KEYWORDS FRECUENTES] (Sin stopwords financieras)")
    print(f"Positivas: {[k for k,v in analyze_keywords(pos_text)]}")
    print(f"Negativas: {[k for k,v in analyze_keywords(neg_text)]}")
# @title 3. EJECUCIËN MAESTRA
TICKER = "ADBE" # @param {type:"string"}
NUM_REPORTS = 6 # @param {type:"integer"}

# 1. Ingesta
loader = SECLoader()
loader.download_filings(TICKER, NUM_REPORTS)
raw_docs = loader.process_filings(TICKER)

if not raw_docs:
    print("[!] No se encontraron datos.")
else:
    # 2. Anßlisis Robustos
    model = FinBertModel()
    prep = TextPreprocessor()

    all_results = []
    print("\n[*] Iniciando anßlisis de sentimeinto...")
    for doc in raw_docs:
        # Ahora split_sentences aplica el FILTRO LEGAL automßticamente
        sentences = prep.split_sentences(prep.clean_text(doc['text']))
        print(f"Oraciones vßlidas en {doc.get('date')}: {len(sentences)}")

        df_sent = model.predict(sentences)
        if not df_sent.empty:
            df_sent['date'] = doc.get('date', '2023-01-01')
            all_results.append(df_sent)

    # 3. Dashboard + Market Data
    if all_results:
        final_df = pd.concat(all_results)
        dates = pd.to_datetime(final_df['date']).sort_values()
        start = dates.min() - timedelta(days=30)
        end = dates.max() + timedelta(days=30)

        try:
            prices = get_market_data(TICKER, start, end)
            show_dashboard(TICKER, final_df, prices)
        except Exception as e:
            print(f"[!] Fall¾ descarga de precios: {e}")
            show_dashboard(TICKER, final_df)
