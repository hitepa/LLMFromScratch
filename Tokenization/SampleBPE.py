from importlib.metadata import version
import tiktoken


def main():
    print("tiktoken version:", version("tiktoken"))
    ##BPE preserves and tokenize space as well, so we can see the space token in the output
    enc = tiktoken.get_encoding("gpt2")
    text = "HelloAkwiwier."
    token_ids = enc.encode(text)
    print("Original text:", text)
    print("Token IDs:", token_ids)
    decoded_text = enc.decode(token_ids)
    print("Decoded text:", decoded_text)

if __name__ == "__main__":
    main()