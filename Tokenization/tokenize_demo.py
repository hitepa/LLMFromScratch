import urllib.request
import re
from SimpleTokenizerv1 import SimpleTokenizerv1


def download_sample_data(url, file_path):
    """Download the sample text file and return its local path."""
    urllib.request.urlretrieve(url, file_path)
    return file_path


def read_text(file_path):
    """Read and return the entire contents of a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def tokenize(raw_text):
    """Split text into tokens based on whitespace, punctuation and special characters."""
    preprocessed = re.split(r'([,.:;!?_"()\']|--|\s)', raw_text)
    # Remove whitespace tokens to save memory and reduce the number of tokens
    preprocessed = [token.strip() for token in preprocessed if token.strip()]
    return preprocessed


def build_vocab(tokens):
    """Build a vocabulary mapping each unique token to a unique integer id."""
    all_words = set(tokens)
    vocab = {word: idx for idx, word in enumerate(all_words)}
    return vocab

def addUnkAndEosTokens(vocab):
    """Add special tokens for unknown words and end of sequence to the vocabulary."""
    vocab['<unk>'] = len(vocab)  # Add <unk> token for unknown words
    vocab['<eos>'] = len(vocab)  # Add <eos> token for end of sequence
    return vocab


def main():
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"

    download_sample_data(url, file_path)
    raw_text = read_text(file_path)
    tokens = tokenize(raw_text)
    vocab = build_vocab(tokens)
    vocab = addUnkAndEosTokens(vocab)
    tokenizer = SimpleTokenizerv1(vocab)
    text = "Hello, world! This is a test of the SimpleTokenizerv1."
    print("Original text:", text)
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded text:", decoded_text)

if __name__ == "__main__":
    main()


