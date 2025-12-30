import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

encoder = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 500
OVERLAP = 100


def token_len(text: str) -> int:
    return len(encoder.encode(text))


def chunk_text(text: str) -> list[str]:
    chunks = []
    current: list[str] = []
    current_len = 0

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        para_len = token_len(para)

        # Case 1: paragraph fits
        if para_len <= MAX_TOKENS:
            if current_len + para_len > MAX_TOKENS:
                chunks.append(" ".join(current))
                current = current[-OVERLAP:] if OVERLAP else []
                current_len = token_len(" ".join(current))
            current.append(para)
            current_len += para_len
        else:
            # Case 2: paragraph too large → split by sentence
            sentences = sent_tokenize(para)
            for sent in sentences:
                sent_len = token_len(sent)
                if current_len + sent_len > MAX_TOKENS:
                    chunks.append(" ".join(current))
                    current = current[-OVERLAP:] if OVERLAP else []
                    current_len = token_len(" ".join(current))
                current.append(sent)
                current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks
