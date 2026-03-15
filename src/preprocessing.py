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

    def categorize_risk(self, sentence: str) -> str:
        s_lower = sentence.lower()
        categories = {
            'Macroeconomic & Geopolitics': ['inflation', 'interest rate', 'war', 'geopolitical', 'pandemic', 'exchange rate', 'tariff', 'macro', 'recession', 'currency'],
            'Operations & Supply Chain': ['supply chain', 'logistic', 'shortage', 'material', 'freight', 'raw material', 'disruption', 'weather', 'climate', 'inventory'],
            'Regulatory & Legal': ['regulation', 'compliance', 'lawsuit', 'sec', 'litigation', 'tax', 'legislation', 'government', 'antitrust', 'penalty'],
            'Competitive & Market': ['competition', 'market share', 'competitor', 'consumer demand', 'pricing pressure', 'disrupt', 'preference'],
            'Financial & Liquidity': ['liquidity', 'debt', 'credit', 'default', 'impairment', 'capital', 'borrow', 'interest expense'],
            'Technology & Cyber': ['cyber', 'security', 'breach', 'data privacy', 'ransomware', 'it system', 'artificial intelligence', 'ai ', 'hacker']
        }
        
        for category, keywords in categories.items():
            if any(kw in s_lower for kw in keywords):
                return category
                
        return 'General / Other'

    def split_sentences(self, text, filter_legal=True):
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            # Filtro 1: Longitud
            if len(s) > 20 and len(s.split()) >= 4:
                # Filtro 2: Ruido Legal
                if not filter_legal or not self.is_legal_noise(s):
                    valid_sentences.append(s)
        return valid_sentences
