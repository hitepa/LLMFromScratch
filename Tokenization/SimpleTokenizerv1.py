import re

class SimpleTokenizerv1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;!?_"()\']|--|\s)', text) 
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        preprocessed = [token if token in self.str_to_int else '<unk>' for token in preprocessed]   
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids   

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = re.sub(r'\s+([,.:;!?_"()\']|--)\s', r'\1', text)  # Remove spaces around punctuation
        return text
