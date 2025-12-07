#!/usr/bin/env python
"""
FinSentiment - Analizador de Sentimiento Financiero
Script principal que orquesta todo el pipeline de análisis.
"""

import argparse
import sys
import os
from datetime import datetime
import pandas as pd

# Importar módulos del proyecto
from src.ingestion import SECLoader
from src.preprocessing import TextPreprocessor
from src.model import FinBertModel, aggregate_sentiment
from src.visualization import SentimentVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='FinSentiment: Análisis de sentimiento de reportes financieros SEC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py AAPL --reports 2
  python main.py TSLA --reports 3 --output results/
        """
    )
    
    parser.add_argument('ticker', type=str, help='Ticker bursátil (ej. AAPL, TSLA)')
    parser.add_argument('--reports', type=int, default=2, 
                       help='Número de reportes a descargar (default: 2)')
    parser.add_argument('--output', type=str, default='results',
                       help='Directorio de salida (default: results/)')
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    num_reports = args.reports
    output_dir = args.output
    
    print("="*60)
    print(f"  FinSentiment: Análisis de {ticker}")
    print("="*60)
    
    # Crear directorios necesarios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # ============================================
    # FASE 1: INGESTA
    # ============================================
    print(f"\n[FASE 1/4] Ingesta de Datos")
    print("-" * 60)
    
    loader = SECLoader(data_dir="data")
    
    try:
        loader.download_filings(ticker, amount=num_reports)
        loader.process_local_filings(ticker)
    except Exception as e:
        print(f"[ERROR] Fallo en ingesta: {e}")
        return 1
    
    # ============================================
    # FASE 2: PREPROCESAMIENTO
    # ============================================
    print(f"\n[FASE 2/4] Preprocesamiento de Texto")
    print("-" * 60)
    
    preprocessor = TextPreprocessor()
    
    # Buscar archivos procesados
    processed_dir = os.path.join("data", "processed")
    processed_files = [f for f in os.listdir(processed_dir) 
                      if f.startswith(ticker) and f.endswith("_MDA.txt")]
    
    if not processed_files:
        print(f"[ERROR] No se encontraron archivos procesados para {ticker}")
        return 1
    
    print(f"   Encontrados {len(processed_files)} reportes procesados")
    
    # ============================================
    # FASE 3: ANÁLISIS DE SENTIMIENTO
    # ============================================
    print(f"\n[FASE 3/4] Análisis de Sentimiento (FinBERT)")
    print("-" * 60)
    print("   NOTA: La primera ejecución puede tardar varios minutos")
    print("   descargando el modelo (~440MB). Ejecuciones posteriores")
    print("   serán mucho más rápidas.")
    print()
    
    try:
        model = FinBertModel()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar FinBERT: {e}")
        print("\n[SOLUCIÓN] Ejecuta 'python test_model_load.py' para verificar la instalación")
        return 1
    
    results = []
    
    for file in processed_files:
        filepath = os.path.join(processed_dir, file)
        print(f"   Analizando: {file}")
        
        # Leer texto
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Limpiar y tokenizar
        cleaned = preprocessor.clean_text(text)
        sentences = preprocessor.split_sentences(cleaned)
        
        print(f"      - Oraciones detectadas: {len(sentences)}")
        
        if not sentences:
            print(f"      [!] No se encontraron oraciones válidas")
            continue
        
        # Analizar sentimiento
        df_sentiment = model.predict_batch(sentences)
        agg = aggregate_sentiment(df_sentiment)
        
        # Extraer información del nombre del archivo
        # Formato: TICKER_10-K_ACCESSION_MDA.txt
        parts = file.replace("_MDA.txt", "").split("_")
        doc_type = parts[1] if len(parts) > 1 else "10-K"
        
        # Guardar resultado
        results.append({
            'ticker': ticker,
            'document': file,
            'type': doc_type,
            'sentiment_score': agg['net_sentiment'],
            'dominant': agg['dominant_sentiment'],
            'sentences_analyzed': agg['sentence_count'],
            'avg_positive': agg['avg_positive'],
            'avg_negative': agg['avg_negative']
        })
        
        print(f"      - Sentimiento: {agg['dominant_sentiment'].upper()}")
        print(f"      - Score Neto: {agg['net_sentiment']:.3f}")
    
    # ============================================
    # FASE 4: VISUALIZACIÓN
    # ============================================
    print(f"\n[FASE 4/4] Generación de Visualización")
    print("-" * 60)
    
    if not results:
        print("[!] No hay resultados para visualizar")
        return 1
    
    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results)
    
    # Para la visualización, usar fecha actual como proxy
    # En producción, extraer fecha real del reporte
    df_results['date'] = pd.date_range(end=datetime.now(), periods=len(df_results), freq='3M')
    
    # Guardar CSV
    csv_path = os.path.join(output_dir, f"{ticker}_sentiment_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"   [+] Resultados guardados: {csv_path}")
    
    # Generar visualización
    viz = SentimentVisualizer(output_dir=output_dir)
    
    # Preparar datos para el gráfico
    df_viz = df_results[['date', 'sentiment_score']].copy()
    
    plot_path = viz.plot_sentiment_trend(df_viz, ticker=ticker)
    
    if plot_path:
        print(f"   [+] Gráfico generado: {plot_path}")
    
    # ============================================
    # RESUMEN FINAL
    # ============================================
    print("\n" + "="*60)
    print("  RESUMEN DEL ANÁLISIS")
    print("="*60)
    print(f"\nTicker: {ticker}")
    print(f"Reportes analizados: {len(results)}")
    print(f"\nSentimiento promedio: {df_results['sentiment_score'].mean():.3f}")
    print(f"Tendencia dominante: {df_results['dominant'].mode()[0].upper()}")
    
    print("\n" + "="*60)
    print("  ANÁLISIS COMPLETADO ✓")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
