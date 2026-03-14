import re

class TextPreprocessor:
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return "".join(ch for ch in text if ch.isprintable()).strip()

    def is_legal_noise(self, sentence: str) -> bool:
        # Palabras clave de disclaimers y riesgo legal
        legal_keywords = [
            'forward-looking', 'safe harbor', 'uncertainty', 'may differ',
            'subject to error', 'actual results', 'factors that could cause',
            'statements regarding', 'cautionary note', 'risk factors',
            'include but are not limited to', 'assumptions'
        ]
        s_lower = sentence.lower()
        return any(kw in s_lower for kw in legal_keywords)

    def split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            # Filtro 1: Longitud
            if len(s) > 20 and len(s.split()) >= 4:
                # Filtro 2: Ruido Legal (Falso Negativo Commmon Source)
                if not self.is_legal_noise(s):
                    valid_sentences.append(s)
        return valid_sentences
