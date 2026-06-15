import re

with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()


def tokenize(text):
    text = text.lower()
    # words and punctuation as separate tokens
    return re.findall(r"\w+|[^\w\s]", text)


tokens = tokenize(text)
vocab = sorted(set(tokens))
vocab_size = len(vocab)

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}


def encode(text):
    return [word_to_id[tok] for tok in tokenize(text)]


def decode(ids):
    return " ".join(id_to_word[i] for i in ids)


if __name__ == "__main__":
    print(f"vocab size: {vocab_size}")

    encoded = encode(text)
    decoded = decode(encoded)
    roundtrip_ok = decoded == " ".join(tokens)
    print(f"round-trip ok: {roundtrip_ok}")

    print("sample mappings:")
    for word in ["the", "tremendous", "believe", "."]:
        i = word_to_id[word]
        print(f"  {word!r} -> {i} -> {id_to_word[i]!r}")
