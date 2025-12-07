"""
Script de prueba simplificado para verificar que FinBERT se carga correctamente.
No ejecuta inferencia, solo confirma que el modelo y tokenizer están accesibles.
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification

print("[*] Iniciando prueba de carga de FinBERT...")
print("[*] Esto puede tardar unos minutos la primera vez (descarga ~440MB)")

try:
    model_name = "ProsusAI/finbert"
    
    print(f"[1/3] Cargando tokenizer de {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print("[+] Tokenizer cargado exitosamente")
    
    print(f"[2/3] Cargando modelo de {model_name}...")
    model = BertForSequenceClassification.from_pretrained(model_name)
    print("[+] Modelo cargado exitosamente")
    
    print("[3/3] Verificando configuracion del modelo...")
    print(f"    - Numero de etiquetas: {model.config.num_labels}")
    print(f"    - Etiquetas: {model.config.id2label}")
    
    print("\n" + "="*50)
    print("[SUCCESS] FinBERT esta listo para usar!")
    print("="*50)
    
except Exception as e:
    print(f"\n[ERROR] Fallo al cargar el modelo: {e}")
    import traceback
    traceback.print_exc()
