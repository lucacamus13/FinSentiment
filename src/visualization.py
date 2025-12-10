import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from collections import Counter
from datetime import datetime
import os

class SentimentVisualizer:
    """
    Módulo para visualizar la evolución temporal del sentimiento financiero.
    """
    
    def __init__(self, output_dir="reports/figures"):
        """
        Args:
            output_dir: Directorio donde se guardarán los gráficos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar estilo seaborn para gráficos profesionales
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

    def analyze_keywords(self, sentences, top_k=10):
        """
        Analiza frecuencia de palabras excluyendo stopwords financieras.
        """
        # Stopwords expandidas (General + Financieras + 'Legal')
        stopwords = set([
            'the', 'and', 'of', 'to', 'in', 'a', 'that', 'for', 'is', 'on', 'with', 'as', 
            'our', 'we', 'are', 'by', 'it', 'from', 'an', 'be', 'files', 'company',
            # Nuevas solicitadas por usuario
            'may', 'such', 'period', 'results', 'year', 'quarter', 'other', 'have'
        ])
        
        words = []
        for s in sentences:
            # Tokenización simple: solo letras, minúsculas, >3 caracteres
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', s)
            words.extend([w.lower() for w in tokens if w.lower() not in stopwords])
            
        return Counter(words).most_common(top_k)
    
    def normalize_scores(self, df):
        """
        Calcula Z-Score del sentimiento para resaltar cambios relativos.
        """
        if len(df) < 2: return df
        msg = df['sentiment_score'].mean()
        std = df['sentiment_score'].std()
        
        if std == 0: std = 1 # Evitar división por cero
        
        df['z_score'] = (df['sentiment_score'] - msg) / std
        return df
    
    def calculate_moving_average(self, series, window=3):
        """
        Calcula la media móvil simple.
        
        Args:
            series: Serie de pandas con los datos
            window: Ventana para la media móvil
        """
        return series.rolling(window=window, min_periods=1).mean()
    
    def plot_sentiment_trend(self, df, ticker, window=3):
        """
        Genera gráfico de evolución del sentimiento a lo largo del tiempo.
        
        Args:
            df: DataFrame con columnas 'date' y 'sentiment_score'
            ticker: Símbolo del ticker (ej. 'AAPL')
            window: Ventana para media móvil
            
        Returns:
            Path al archivo PNG generado
        """
        # Validaciones
        if df.empty:
            print("[!] DataFrame vacío, no se puede generar gráfico")
            return None
            
        if 'date' not in df.columns or 'sentiment_score' not in df.columns:
            print("[!] DataFrame debe tener columnas 'date' y 'sentiment_score'")
            return None
        
        # Ordenar por fecha
        df = df.sort_values('date').copy()
        
        # Calcular media móvil
        df['ma'] = self.calculate_moving_average(df['sentiment_score'], window)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot del sentimiento real
        ax.plot(df['date'], df['sentiment_score'], 
                marker='o', linewidth=2, markersize=8,
                label='Sentiment Score', color='#2E86AB', alpha=0.7)
        
        # Plot de la media móvil
        ax.plot(df['date'], df['ma'], 
                linewidth=2.5, label=f'Media Móvil ({window} períodos)',
                color='#F24236', linestyle='--', alpha=0.8)
        
        # Línea de referencia en 0 (sentimiento neutral)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Sombreado de área según sentimiento
        ax.fill_between(df['date'], df['sentiment_score'], 0,
                        where=(df['sentiment_score'] > 0),
                        alpha=0.1, color='green', label='Sentimiento Positivo')
        ax.fill_between(df['date'], df['sentiment_score'], 0,
                        where=(df['sentiment_score'] <= 0),
                        alpha=0.1, color='red', label='Sentimiento Negativo')
        
        # Etiquetas y título
        ax.set_xlabel('Fecha del Reporte', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sentiment Score\n(Positivo - Negativo)', fontsize=12, fontweight='bold')
        ax.set_title(f'Evolución del Sentimiento Financiero: {ticker.upper()}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Leyenda
        ax.legend(loc='best', framealpha=0.9)
        
        # Grid más sutil
        ax.grid(True, alpha=0.3)
        
        # Rotar etiquetas de fecha si hay muchas
        if len(df) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar
        output_path = os.path.join(self.output_dir, f"{ticker}_sentiment_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[+] Gráfico guardado en: {output_path}")
        return output_path


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
