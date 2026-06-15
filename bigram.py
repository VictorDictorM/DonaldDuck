import torch

from tokenizer import vocab_size, word_to_id, id_to_word, encode, decode, text

ids = encode(text)

# counts[i][j] = number of times word j followed word i
counts = torch.zeros((vocab_size, vocab_size), dtype=torch.long)
for cur, nxt in zip(ids[:-1], ids[1:]):
    counts[cur, nxt] += 1

# +1 smoothing, then normalize each row to a probability distribution
probs = (counts + 1).float()
probs = probs / probs.sum(dim=1, keepdim=True)


def generate(start_id, length, probs):
    out = [start_id]
    for _ in range(length):
        current = out[-1]
        next_id = torch.multinomial(probs[current], num_samples=1).item()
        out.append(next_id)
    return out


if __name__ == "__main__":
    row_sums = probs.sum(dim=1)
    max_dev = (row_sums - 1).abs().max().item()
    print(f"max row-sum deviation from 1.0: {max_dev:.2e}")

    word = "great"
    word_id = word_to_id[word]
    top = torch.topk(probs[word_id], k=3)
    top_words = [(id_to_word[i.item()], p.item()) for p, i in zip(top.values, top.indices)]
    print(f"top-3 words after {word!r}: {top_words}")

    start_id = word_to_id["we"]
    generated = generate(start_id, length=25, probs=probs)
    print("generated:", decode(generated))
