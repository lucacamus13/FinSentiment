import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Importar módulos del proyecto
from src.ingestion import SECLoader
from src.preprocessing import TextPreprocessor
from src.model import FinBertModel, aggregate_sentiment
from src.visualization import SentimentVisualizer

# Configuración de página
st.set_page_config(
    page_title="FinSentiment | Análisis de Reportes SEC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    /* Estilizar las pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E2E8F0;
        border-bottom: 2px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_finbert():
    # Helper class to capture print statements / callbacks inside cache
    placeholder = st.empty()
    def update_text(msg):
        placeholder.info(f"⏳ Instalación Inicial: {msg}")
        
    model = FinBertModel(load_callback=update_text)
    placeholder.empty() # Clean up when done
    return model

@st.cache_resource
def load_components():
    return SECLoader(data_dir="data"), TextPreprocessor(), SentimentVisualizer(output_dir="results")

def run_analysis(ticker, num_reports):
    """
    Función principal que envuelve la lógica del Core Engine v2.1 para retornar los resultados a Streamlit
    """
    loader, preprocessor, viz = load_components()
    model = load_finbert()
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # ---------------------------------------------------------
    # FASE 1: INGESTA (v2.1)
    # ---------------------------------------------------------
    status_text.info(f"Fase 1/4: Descargando y extrayendo últimos {num_reports} reportes para {ticker} desde EDGAR...")
    loader.download_filings(ticker, amount=num_reports)
    raw_docs = loader.process_filings(ticker)
    
    if not raw_docs:
        st.error(f"No se procesaron reportes para {ticker}. Verifica si la empresa ha reportado recientemente.")
        return None, None
        
    progress_bar.progress(25)
    
    # ---------------------------------------------------------
    # FASE 2 & 3: NLP + FINBERT (v2.1 con Filtro Legal)
    # ---------------------------------------------------------
    all_results = []
    
    for i, doc in enumerate(raw_docs):
        status_text.info(f"Fases 2-3: Procesando documento {i+1}/{len(raw_docs)} (Fecha: {doc.get('date', 'N/A')})...")
        
        # Preprocesamiento v2.1 (Aplica filtro Legal Noise automáticamente en split)
        cleaned_text = preprocessor.clean_text(doc['text'])
        sentences = preprocessor.split_sentences(cleaned_text)
        
        if not sentences:
            continue
            
        sub_progress = st.progress(0, text=f"Inferencia FinBERT: procesando {len(sentences)} oraciones válidas...")
        
        def batch_callback(current, total):
            ratio = current / total
            sub_progress.progress(ratio, text=f"Inferencia FinBERT: {current}/{total} completadas (Lotes de 32)")
            
        df_sent = model.predict_batch(sentences, batch_size=32, progress_callback=batch_callback)
        sub_progress.empty()
        
        if not df_sent.empty:
            df_sent['date'] = doc.get('date', '2023-01-01') # Si no extrajo, proxy feo
            df_sent['accession'] = doc.get('accession', '')
            all_results.append(df_sent)
            
        current_progress = 25 + int((i + 1) / len(raw_docs) * 50)
        progress_bar.progress(current_progress)
        
    # ---------------------------------------------------------
    # FASE 4: RESULTADOS Y GRÁFICAS (Z-Score + YFinance)
    # ---------------------------------------------------------
    status_text.info("Fase 4/4: Calculando Z-Score y cruzando con YFinance...")
    
    if not all_results:
        st.warning("No se pudieron generar resultados a partir de los documentos.")
        status_text.empty()
        return None, None
        
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Generar Chart híbrido y resumen
    plot_path, summary_df, ys_success = viz.plot_sentiment_trend(final_df, ticker=ticker)
    
    # Guardar CSV crudo
    csv_path = os.path.join(output_dir, f"{ticker}_finsentiment_raw.csv")
    final_df.to_csv(csv_path, index=False)
    
    progress_bar.progress(100)
    status_text.success(f"¡Análisis completado para {ticker}!")
    
    if not ys_success:
        st.warning("⚠️ Yahoo Finance bloqueó la descarga del historial de precios (Típico en servidores Cloud gratuitos). El gráfico solo muestra la tendencia del Sentimiento Z-Score.")

    # Construir un mini dataframe para la vista "Cruda" resumiendo por fechas para no asfixiar a streamlit
    display_df = final_df[['date', 'sentence', 'sentiment_label', 'pos_val', 'neg_val']].copy()
    display_df['net_score'] = display_df['pos_val'] - display_df['neg_val']
    
    return display_df, plot_path

def main():
    st.markdown('<p class="main-header">FinSentiment 📊</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis Cuantitativo de Sentimiento Gerencial en Reportes SEC</p>', unsafe_allow_html=True)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        st.markdown("Configura los parámetros para la minería y análisis.")
        
        ticker_input = st.text_input("Ticker Bursátil", value="AAPL", help="Símbolo de la empresa en la bolsa (ej. MSFT, TSLA, AAPL)").upper()
        num_reports = st.slider("Cantidad de Reportes a Analizar", min_value=1, max_value=10, value=3, 
                                help="Cantidad histórica de reportes 10-K y 10-Q a descargar.")
        
        run_button = st.button("Ejecutar Análisis 🚀", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.markdown("""
        **¿Cómo funciona?**
        1. Descarga reportes oficiales de la SEC.
        2. Aísla la sección MD&A.
        3. Pasa el texto por **FinBERT**.
        4. Cuantifica el pesimismo/optimismo.
        """)

    # --- MAIN CONTENT ---
    if run_button:
        if not ticker_input:
            st.warning("Por favor ingresa un Ticker válido.")
            st.stop()
            
        resultados_df, grafico_path = run_analysis(ticker_input, num_reports)
        
        if resultados_df is not None:
            st.markdown("---")
            
            # --- TABS ---
            tab1, tab2 = st.tabs(["Resumen & Visualización", "Datos Crudos"])
            
            with tab1:
                # Métricas Clave
                col1, col2, col3 = st.columns(3)
                
                avg_sentiment = resultados_df['net_score'].mean()
                dominant_overall = resultados_df['sentiment_label'].mode()[0].capitalize()
                total_sentences = len(resultados_df)
                
                col1.metric("Sentimiento Z-Score Promedio", f"{avg_sentiment:.3f}", 
                            delta="Optimista" if avg_sentiment > 0.05 else ("Pesimista" if avg_sentiment < -0.05 else "Neutral"),
                            delta_color="normal" if avg_sentiment >= -0.05 else "inverse")
                col2.metric("Tono Predominante", dominant_overall)
                col3.metric("Oraciones Analizadas", f"{total_sentences:,}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Gráfico
                if grafico_path and os.path.exists(grafico_path):
                    st.image(grafico_path, caption=f"Evolución del Sentimiento ({ticker_input})", use_container_width=True)
                else:
                    st.info("No se generó el gráfico de evolución.")
                    
            with tab2:
                st.markdown("### Resultados por Documento")
                st.dataframe(
                    resultados_df.style.background_gradient(cmap='RdYlGn', subset=['net_score']),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("### Descargas")
                csv_data = resultados_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar datos como CSV",
                    data=csv_data,
                    file_name=f'{ticker_input}_finsentiment_data.csv',
                    mime='text/csv',
                )

    else:
        # Initial State Panel
        st.info("👈 Ingresa un Ticker en la barra lateral y haz clic en 'Ejecutar Análisis' para comenzar.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Sobre el Proyecto**\n
            Este dashboard aplica IA sobre la base de datos de la SEC para encontrar el valor prospectivo oculto en los reportes anuales y trimestrales.
            """)
        with col2:
            st.markdown("""
            **Métricas Explicadas**\n
            - **Sentimiento Neto**: `(Prob. Positiva - Prob. Negativa)`. Valores positivos indican optimismo gerencial.
            - **Tono Predominante**: La clase (Positivo, Negativo o Neutral) con mayor probabilidad.
            """)

if __name__ == "__main__":
    main()
