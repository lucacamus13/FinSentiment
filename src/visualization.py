import matplotlib.pyplot as plt
import seaborn as pd_sns
import pandas as pd
import numpy as np
import yfinance as yf
import os
import re
from collections import Counter

class SentimentVisualizer:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Configurar estilos
        pd_sns.set_style("whitegrid")

    def get_market_data(self, ticker, start_date, end_date):
        print(f"[*] Descargando precios de {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df['Close']

    def analyze_keywords(self, sentences, top_k=10):
        # Stopwords expandidas (Legal + Financiero Genérico)
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

    def plot_sentiment_trend(self, df_results, ticker):
        """
        Calcula Z-Score y cruza contra YFinance. 
        df_results debe tener 'date', 'pos_val', 'neg_val'
        """
        if df_results.empty or 'date' not in df_results.columns:
            return None, None, False
            
        # 1. Normalización (Z-Score)
        summary = df_results.groupby('date')[['pos_val', 'neg_val']].mean()
        summary['net_score'] = summary['pos_val'] - summary['neg_val']
    
        mean_score = summary['net_score'].mean()
        std_score = summary['net_score'].std()
        if std_score == 0: std_score = 1 # Evitar Div0
        summary['z_score'] = (summary['net_score'] - mean_score) / std_score
        
        # 2. Descarga Precios
        dates = pd.to_datetime(summary.index).sort_values()
        start = dates.min() - pd.Timedelta(days=30)
        end = dates.max() + pd.Timedelta(days=30)
        
        mark_data_success = False
        try:
            # Formateamos las fechas a string para evitar conflictos al invocar yfinance
            market_df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
            if not market_df.empty:
                market_data = market_df['Close']
                # Si yfinance devuelve un DataFrame multidimensional, tomamos la primera serie
                if isinstance(market_data, pd.DataFrame):
                    market_data = market_data.iloc[:, 0]
                # QUITAR TIMEZONE: Matplotlib ignora series con Tz si el eje principal es Naive
                market_data.index = market_data.index.tz_localize(None)
                mark_data_success = True
            else:
                market_data = None
        except Exception as e:
            print(f"[!] Error bajando precios: {e}")
            market_data = None
            
        # 3. Gráfico Doble Eje
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color = '#2E86AB'
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Sentiment Z-Score (Normalizado)', color=color, fontweight='bold')
        ax1.plot(pd.to_datetime(summary.index), summary['z_score'], 'o-', color=color, linewidth=2)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Media Histórica')
        ax1.tick_params(axis='y', labelcolor=color)
        
        if mark_data_success and market_data is not None:
            ax2 = ax1.twinx()
            color = '#F24236'
            ax2.set_ylabel(f'Precio {ticker} ($)', color=color, fontweight='bold')
            ax2.plot(market_data.index, market_data, color=color, alpha=0.6, linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            # MARCA DE AGUA SI YFINANCE FALLA SILENCIOSAMENTE DESDE LA NUBE
            ax1.text(0.5, 0.5, f"⚠️ Yahoo Finance bloqueó la descarga del precio de {ticker}", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, color='red', alpha=0.5, fontsize=12)
            
        plt.title(f"{ticker}: Precio vs Sentimiento Relativo (Z-Score)", fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{ticker}_hybrid_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path, summary, mark_data_success


if __name__ == "__main__":
    # Test con datos sintéticos
    print("[*] Ejecutando prueba de visualización...")
    
    # Crear datos de ejemplo
    dates = pd.date_range(start='2020-01-01', periods=8, freq='3M')
    scores = np.array([0.15, 0.22, 0.10, -0.05, -0.12, 0.08, 0.25, 0.30])
    
    df_test = pd.DataFrame({
        'date': dates,
        'sentiment_score': scores
    })
    
    print("\nDatos de prueba:")
    print(df_test)
    
    # Crear visualizador
    viz = SentimentVisualizer()
    
    # Generar gráfico
    output = viz.plot_sentiment_trend(df_test, ticker="TEST", window=3)
    
    if output:
        print(f"\n[SUCCESS] Prueba completada. Revisa el archivo: {output}")
