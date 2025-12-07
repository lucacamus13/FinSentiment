# FinSentiment

Analizador de Sentimiento Financiero usando NLP (FinBERT) para reportes SEC.

## Descripción

FinSentiment es una herramienta modular que descarga reportes financieros (10-K, 10-Q) de la SEC, extrae la sección MD&A (Management's Discussion and Analysis), analiza el sentimiento usando el modelo FinBERT especializado en finanzas, y genera visualizaciones de tendencias.

## Requisitos

- Python 3.7+
- Conexión a internet (para descargar reportes y modelo)
- ~500MB de espacio libre (modelo FinBERT)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso Rápido

```bash
# Analizar Apple (2 reportes más recientes)
python main.py AAPL

# Analizar Tesla (3 reportes)
python main.py TSLA --reports 3

# Especificar directorio de salida
python main.py MSFT --reports 2 --output resultados/
```

## Estructura del Proyecto

```
FinSentiment/
├── src/
│   ├── ingestion.py      # Descarga de reportes SEC
│   ├── preprocessing.py  # Limpieza de texto
│   ├── model.py         # Análisis FinBERT
│   └── visualization.py # Gráficos
├── data/                 # Datos descargados
├── main.py              # Script principal
└── requirements.txt     # Dependencias
```

## Salida

Genera dos archivos en el directorio de salida:
- `{TICKER}_sentiment_results.csv` - Tabla con resultados detallados
- `{TICKER}_sentiment_trend.png` - Gráfico de evolución temporal

## Notas

- La primera ejecución tarda más (descarga del modelo FinBERT ~440MB)
- El análisis en CPU puede tardar 1-2 minutos por reporte
- Los datos se cachean localmente para ejecuciones futuras

## Autor

Proyecto desarrollado con fines educativos en econometría y NLP financiero.
