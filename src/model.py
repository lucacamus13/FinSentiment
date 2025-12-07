import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Dict

class FinBertModel:
    """
    Wrapper for ProsusAI/finbert model specialized in financial sentiment.
    """
    def __init__(self):
        print("[*] Cargando modelo FinBERT...")
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # Mapeo de IDs a etiquetas (check config del modelo para estar seguros)
        # ProsusAI/finbert usa: 0: positive, 1: negative, 2: neutral
        self.labels = {0: 'positive', 1: 'negative', 2: 'neutral'}
        print("[+] Modelo cargado correctamente.")

    def predict_batch(self, sentences: List[str]) -> pd.DataFrame:
        """
        Analiza una lista de oraciones y devuelve un DataFrame con los resultados.
        """
        if not sentences:
            return pd.DataFrame()

        # Tokenización masiva
        inputs = self.tokenizer(
            sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Softmax para obtener probabilidades
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convertir a numpy
        probs_np = probs.numpy()
        
        # Construir DataFrame
        results = []
        for i, sentence in enumerate(sentences):
            row = {
                "sentence": sentence,
                "positive": probs_np[i][0],
                "negative": probs_np[i][1],
                "neutral": probs_np[i][2]
            }
            # Etiqueta dominante
            max_idx = np.argmax(probs_np[i])
            row["sentiment_label"] = self.labels[max_idx]
            row["sentiment_score"] = probs_np[i][max_idx]
            results.append(row)
            
        return pd.DataFrame(results)

def aggregate_sentiment(df: pd.DataFrame) -> Dict:
    """
    Calcula métricas agregadas para todo el documento.
    Retorna un diccionario con el score global.
    """
    if df.empty:
        return {"net_sentiment": 0, "dominant_sentiment": "neutral"}
        
    # FinSentiment Score: (Positivo - Negativo) promedio
    avg_pos = df["positive"].mean()
    avg_neg = df["negative"].mean()
    
    net_score = avg_pos - avg_neg
    
    dominant = "neutral"
    if net_score > 0.05: dominant = "positive"
    elif net_score < -0.05: dominant = "negative"
    
    return {
        "net_sentiment": net_score,
        "avg_positive": avg_pos,
        "avg_negative": avg_neg,
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
