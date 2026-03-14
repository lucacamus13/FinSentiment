import pandas as pd
import numpy as np
from typing import List

class FinBertModel:
    def __init__(self, load_callback=None):
        if load_callback: load_callback("Cargando entorno base (PyTorch)...")
        import torch
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Cargando FinBERT en {self.device}...")
        
        if load_callback: load_callback("Cargando librerías NLP (Transformers)...")
        from transformers import BertTokenizer, BertForSequenceClassification
        
        if load_callback: load_callback("Descargando/Cargando Tokenizador de red FinBERT...")
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        
        if load_callback: load_callback("Descargando/Cargando Pesos del Modelo FinBERT (~400MB)...")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        self.labels = {0: 'positive', 1: 'negative', 2: 'neutral'}

    def predict_batch(self, sentences: List[str], batch_size: int = 32, progress_callback=None) -> pd.DataFrame:
        """
        Analiza una lista de oraciones en lotes (batches) y devuelve un DataFrame con los resultados.
        Permite usar un callback para reportar el progreso de la inferencia iterativamente.
        """
        if not sentences:
            return pd.DataFrame()

        import torch

        results = []
        total_sentences = len(sentences)
        
        for i in range(0, total_sentences, batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            # Tokenización del lote actual
            inputs = self.tokenizer(
                batch_sentences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Softmax para obtener probabilidades
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                
            # Construir resultados
            for j, sentence in enumerate(batch_sentences):
                idx = np.argmax(probs[j])
                row = {
                    "sentence": sentence,
                    "sentiment_label": self.labels[idx],
                    "pos_val": probs[j][0],
                    "neg_val": probs[j][1],
                    "neutral_val": probs[j][2],
                }
                results.append(row)
                
            if progress_callback:
                current_processed = min(i + batch_size, total_sentences)
                progress_callback(current_processed, total_sentences)
            
        return pd.DataFrame(results)

def aggregate_sentiment(df: pd.DataFrame):
    """
    Calcula métricas agregadas asumiendo las columnas pos_val y neg_val.
    """
    if df.empty:
        return {
            "net_sentiment": 0, "avg_positive": 0, "avg_negative": 0, 
            "avg_neutral": 0, "dominant_sentiment": "neutral", "sentence_count": 0
        }
        
    avg_pos = df["pos_val"].mean()
    avg_neg = df["neg_val"].mean()
    avg_neu = df["neutral_val"].mean()
    
    net_score = avg_pos - avg_neg
    
    dominant = "neutral"
    if net_score > 0.05: dominant = "positive"
    elif net_score < -0.05: dominant = "negative"
    
    return {
        "net_sentiment": net_score,
        "avg_positive": avg_pos,
        "avg_negative": avg_neg,
        "avg_neutral": avg_neu,
        "dominant_sentiment": dominant,
        "sentence_count": len(df)
    }

if __name__ == "__main__":
    # Test unitario
    model = FinBertModel()
    
    test_sentences = [
        "Revenue increased significantly due to higher sales volume.",
        "The company reported a net loss for the third quarter.",
        "We are maintaining our guidance for the fiscal year."
    ]
    
    print("\n--- Analizando Oraciones de Prueba ---")
    df_results = model.predict_batch(test_sentences)
    print(df_results[["sentence", "sentiment_label", "positive", "negative"]])
    
    agg = aggregate_sentiment(df_results)
    print("\n--- Resultado Agregado ---")
    print(agg)
