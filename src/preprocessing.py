import re
import os
from typing import List

class TextPreprocessor:
    """
    Módulo encargado de limpiar y preparar el texto financiero para el modelo NLP.
    """
    
    def __init__(self):
        # Patrones regex compilados para eficiencia
        self.whitespace_pattern = re.compile(r'\s+')
        # Patrón para detectar frases de "Safe Harbor" o "Forward-Looking Statements"
        # Estas secciones son legalmente obligatorias pero ruidosas para el análisis de sentimiento.
        self.safe_harbor_pattern = re.compile(
            r'forward[\s-]*looking\s+statements|safe\s+harbor\s+statement|cautionary\s+note', 
            re.IGNORECASE
        )

    def clean_text(self, text: str) -> str:
        """
        Limpieza general del texto crudo.
        """
        if not text:
            return ""
            
        # 1. Normalización de espacios (tabuladores, saltos de línea múltiples -> un espacio)
        text = self.whitespace_pattern.sub(' ', text)
        
        # 2. Eliminación de caracteres extraños pero manteniendo puntuación financiera (%, $, ., ,)
        # Mantenemos letras, números, puntuación básica y espacios.
        # Eliminamos caracteres de control o emojis si los hubiera.
        text = "".join(ch for ch in text if ch.isprintable())
        
        return text.strip()

    def remove_safe_harbor(self, text: str) -> str:
        """
        Intenta eliminar párrafos que contienen advertencias legales estándar.
        Estrategia: Si una oración/párrafo menciona explícitamente "forward-looking statements",
        a menudo es el comienzo de un bloque legal largo al principio o final.
        """
        # Nota: Eliminar bloques enteros es arriesgado sin una estructura clara.
        # Por ahora, implementamos una detección simple: Si encontramos la frase, 
        # podríamos intentar cortar si está al inicio o final.
        
        # Simple heurística: Si el texto empieza con una advertencia legal (común en press releases, menos en 10-K parsed),
        # cortamos hasta el final de esa frase.
        # En MD&A extraído, el riesgo es menor, así que por ahora solo loggeamos si se detecta.
        match = self.safe_harbor_pattern.search(text)
        if match:
            # TODO: Implementar lógica de corte inteligente si se detecta que es un bloque aislado.
            pass
            
        return text

        return valid_sentences

    def is_legal_noise(self, sentence: str) -> bool:
        """
        Detecta si una oración es ruido legal/disclaimer.
        """
        legal_keywords = [
            'forward-looking', 'safe harbor', 'uncertainty', 'may differ', 
            'subject to error', 'actual results', 'factors that could cause',
            'statements regarding', 'cautionary note', 'risk factors'
        ]
        s_lower = sentence.lower()
        return any(kw in s_lower for kw in legal_keywords)

    def split_sentences(self, text: str) -> List[str]:
        """
        Divide el texto en oraciones y filtra ruido legal.
        """
        # Regex para split por punto, interrogación o exclamación.
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            # Filtro 1: Longitud mínima
            if len(s) > 20 and len(s.split()) >= 4:
                # Filtro 2: Legal Noise (NUEVO)
                if not self.is_legal_noise(s):
                    valid_sentences.append(s)
                
        return valid_sentences

if __name__ == "__main__":
    # Test unitario rápido
    cleaner = TextPreprocessor()
    
    sample_text = """
    Item 7. Management's Discussion. 
    Net sales increased by 10% to $100 million. This was due to strong demand in the U.S. market!
    However, cost of sales also increased.   We expect inflation to continue.
    Cautionary Note Regarding Forward-Looking Statements: These results are not guaranteed.
    """
    
    print("--- Texto Original ---")
    print(sample_text)
    
    cleaned = cleaner.clean_text(sample_text)
    print("\n--- Texto Limpio ---")
    print(cleaned)
    
    sentences = cleaner.split_sentences(cleaned)
    print(f"\n--- Oraciones Detectadas ({len(sentences)}) ---")
    for i, s in enumerate(sentences):
        print(f"{i+1}. {s}")
