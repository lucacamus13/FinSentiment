import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time


def get_yahoo_prices(tickers, start_date, end_date):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_data = {}
    
    for ticker in tickers:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                'period1': int(start_date.timestamp()),
                'period2': int(end_date.timestamp()),
                'interval': '1d',
                'events': 'history'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                    result = data['chart']['result'][0]
                    if 'timestamp' in result and 'indicators' in result:
                        timestamps = result['timestamp']
                        close_prices = result['indicators']['quote'][0]['close']
                        
                        df = pd.DataFrame({
                            'Date': pd.to_datetime(timestamps, unit='s'),
                            ticker: close_prices
                        })
                        df.set_index('Date', inplace=True)
                        all_data[ticker] = df[ticker]
                        
            time.sleep(0.5)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    if all_data:
        return pd.DataFrame(all_data)
    return pd.DataFrame()

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

def run_sector_analysis(tickers_list, num_reports):
    loader, preprocessor, _ = load_components()
    model = load_finbert()
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    full_history_data = []
    
    for idx, ticker in enumerate(tickers_list):
        status_text.info(f"Procesando {ticker} ({idx+1}/{len(tickers_list)})...")
        
        loader.download_filings(ticker, amount=num_reports)
        raw_docs = loader.process_filings(ticker)
        
        if not raw_docs:
            st.warning(f"No se procesaron reportes para {ticker}")
            continue
            
        reports_to_scan = raw_docs[:num_reports]
        
        for doc in reports_to_scan:
            report_date = doc.get('date', 'Unknown')
            
            cleaned_text = preprocessor.clean_text(doc['text'])
            sentences = preprocessor.split_sentences(cleaned_text)
            
            if len(sentences) < 10:
                continue
                
            df_sent = model.predict_batch(sentences, batch_size=32)
            
            if not df_sent.empty:
                net_score = df_sent['pos_val'].mean() - df_sent['neg_val'].mean()
                full_history_data.append({
                    'ticker': ticker,
                    'date': report_date,
                    'net_score': net_score
                })
        
        progress_bar.progress((idx + 1) / len(tickers_list))
    
    if not full_history_data:
        st.warning("No se pudieron generar resultados para ningún ticker.")
        return None, None, None
    
    df_history = pd.DataFrame(full_history_data)
    df_history['date_obj'] = pd.to_datetime(df_history['date'], errors='coerce')
    df_history['year'] = df_history['date_obj'].dt.year
    
    global_mean = df_history['net_score'].mean()
    global_std = df_history['net_score'].std()
    if global_std == 0:
        global_std = 1
    df_history['z_score'] = (df_history['net_score'] - global_mean) / global_std
    
    pivot_table = df_history.pivot_table(index='ticker', columns='year', values='z_score', aggfunc='mean')
    
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, len(tickers_list) * 1.2 + 2))
    sns.heatmap(
        pivot_table, 
        cmap='RdYlGn', 
        center=0, 
        annot=True, 
        fmt=".2f", 
        linewidths=.5, 
        cbar_kws={'label': 'Z-Score'},
        ax=ax_heatmap
    )
    ax_heatmap.set_title('Evolución Histórica del Sentimiento Sectorial', fontsize=16, fontweight='bold')
    ax_heatmap.set_ylabel('Empresa')
    ax_heatmap.set_xlabel('Año Fiscal')
    
    latest_indices = df_history.groupby('ticker')['date_obj'].idxmax()
    df_latest = df_history.loc[latest_indices].copy()
    
    min_date = df_latest['date_obj'].min() - timedelta(days=10)
    max_date = datetime.now() + timedelta(days=1)
    
    try:
        with st.spinner("Descargando datos de mercado desde Yahoo Finance..."):
            close_prices = get_yahoo_prices(tickers_list, min_date, max_date)
        
        if close_prices.empty:
            st.warning("⚠️ Yahoo Finance bloquea descargas desde servidores cloud. El gráfico Alpha Hunter mostrará solo Z-Score.")
            df_latest['price_return_6m'] = 0.0
        else:
            if len(tickers_list) == 1:
                close_prices = pd.DataFrame({tickers_list[0]: close_prices[tickers_list[0]]})
            
            perf_list = []
            for _, row in df_latest.iterrows():
                t = row['ticker']
                d_start = row['date_obj']
                d_end = d_start + timedelta(days=180)
                if d_end > datetime.now():
                    d_end = datetime.now() - timedelta(days=1)
                
                if t in close_prices.columns:
                    ts = close_prices[t].dropna()
                    
                    prices_after_filing = ts[ts.index >= d_start]
                    prices_6m_later = ts[ts.index <= d_end]
                    
                    if not prices_after_filing.empty and not prices_6m_later.empty:
                        p_start = prices_after_filing.iloc[0]
                        p_end = prices_6m_later.iloc[-1]
                        ret = ((p_end - p_start) / p_start) * 100
                        perf_list.append(round(ret, 2))
                    else:
                        perf_list.append(0.0)
                else:
                    perf_list.append(0.0)
                    
            df_latest['price_return_6m'] = perf_list
    except Exception as e:
        st.warning(f"Error descargando datos de mercado: {e}")
        df_latest['price_return_6m'] = 0.0
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df_latest, x='z_score', y='price_return_6m',
        s=150, color='#2E86AB', edgecolor='black', alpha=0.8, ax=ax_scatter
    )
    for i in range(df_latest.shape[0]):
        ax_scatter.text(
            x=df_latest.z_score.iloc[i] + 0.02,
            y=df_latest.price_return_6m.iloc[i] + 0.5,
            s=df_latest.ticker.iloc[i],
            fontweight='bold', fontsize=10
        )
    ax_scatter.axvline(0, color='gray', linestyle='--', alpha=0.6)
    ax_scatter.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax_scatter.set_title('Alpha Hunter: IA Sentiment vs 6-Month Post-Filing Return', fontsize=16, fontweight='bold')
    ax_scatter.set_xlabel('Sentiment Z-Score (IA)')
    ax_scatter.set_ylabel('Post-Filing Price Return (%)')
    ax_scatter.grid(True, alpha=0.3)
    
    status_text.empty()
    progress_bar.empty()
    
    csv_path = os.path.join(output_dir, "sector_analysis.csv")
    df_history.to_csv(csv_path, index=False)
    
    return df_history, fig_heatmap, fig_scatter


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
        st.markdown("Selecciona el tipo de análisis que deseas realizar.")
        
        page_option = st.selectbox("Tipo de Análisis", ["Análisis Individual", "Análisis Sectorial"])
        
        st.markdown("---")
        st.markdown("""
        **¿Cómo funciona?**
        1. Descarga reportes oficiales de la SEC.
        2. Aísla la sección MD&A.
        3. Pasa el texto por **FinBERT**.
        4. Cuantifica el pesimismo/optimismo.
        """)

    # --- PÁGINAS PRINCIPALES ---
    tab_individual, tab_sector = st.tabs(["📈 Análisis Individual", "🏢 Análisis Sectorial"])
    
    # === PÁGINA ANÁLISIS INDIVIDUAL ===
    with tab_individual:
        st.markdown("### Configura tu análisis")
        
        col_config1, col_config2 = st.columns([1, 1])
        with col_config1:
            ticker_input = st.text_input("Ticker Bursátil", value="AAPL", 
                                        help="Símbolo de la empresa en la bolsa (ej. MSFT, TSLA, AAPL)")
            ticker_input = ticker_input.upper() if ticker_input else ""
        with col_config2:
            num_reports = st.slider("Cantidad de Reportes a Analizar", min_value=1, max_value=10, value=3, 
                                    help="Cantidad histórica de reportes 10-K y 10-Q a descargar.")
        
        run_individual = st.button("Ejecutar Análisis Individual 🚀", type="primary")
        
        if run_individual:
            if not ticker_input:
                st.warning("Por favor ingresa un Ticker válido.")
            else:
                resultados_df, grafico_path = run_analysis(ticker_input, num_reports)
                
                if resultados_df is not None:
                    st.markdown("---")
                    
                    tab1, tab2 = st.tabs(["Resumen & Visualización", "Datos Crudos"])
                    
                    with tab1:
                        avg_sentiment = resultados_df['net_score'].mean()
                        dominant_overall = resultados_df['sentiment_label'].mode()[0].capitalize()
                        total_sentences = len(resultados_df)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Sentimiento Z-Score Promedio", f"{avg_sentiment:.3f}", 
                                    delta="Optimista" if avg_sentiment > 0.05 else ("Pesimista" if avg_sentiment < -0.05 else "Neutral"),
                                    delta_color="normal" if avg_sentiment >= -0.05 else "inverse")
                        col2.metric("Tono Predominante", dominant_overall)
                        col3.metric("Oraciones Analizadas", f"{total_sentences:,}")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
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
            st.info("👆 Ingresa un Ticker y haz clic en 'Ejecutar Análisis Individual' para comenzar.")
    
    # === PÁGINA ANÁLISIS SECTORIAL ===
    with tab_sector:
        st.markdown("### Compara empresas o analiza una industria")
        
        col_config1, col_config2 = st.columns([2, 1])
        with col_config1:
            tickers_input = st.text_area("Tickers (separados por coma)", value="META, AAPL, MSFT, GOOGL, AMZN", 
                                         help="Ej: META, AAPL, MSFT, GOOGL, AMZN")
        with col_config2:
            num_reports_sector = st.slider("Reportes por Ticker", min_value=1, max_value=10, value=3)
        
        run_sector = st.button("Ejecutar Análisis Sectorial 🚀", type="primary")
        
        if run_sector:
            tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            if not tickers_list:
                st.warning("Por favor ingresa al menos un Ticker válido.")
            else:
                resultados_df, fig_heatmap, fig_scatter = run_sector_analysis(tickers_list, num_reports_sector)
                
                if resultados_df is not None:
                    st.markdown("---")
                    
                    tab1, tab2, tab3 = st.tabs(["🔥 Heatmap Sectorial", "🎯 Alpha Hunter", "📊 Datos"])
                    
                    with tab1:
                        st.pyplot(fig_heatmap)
                        
                    with tab2:
                        st.pyplot(fig_scatter)
                        
                    with tab3:
                        st.markdown("### Resultados por Empresa")
                        st.dataframe(resultados_df, use_container_width=True)
                        
                        csv_data = resultados_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar datos como CSV",
                            data=csv_data,
                            file_name='sector_analysis.csv',
                            mime='text/csv',
                        )
        else:
            st.info("👆 Ingresa varios Tickers separados por coma y haz clic en 'Ejecutar Análisis Sectorial' para comparar empresas.")

if __name__ == "__main__":
    main()
