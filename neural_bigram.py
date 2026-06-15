import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import vocab_size, word_to_id, id_to_word, encode, decode, text

ids = encode(text)

x = torch.tensor(ids[:-1], dtype=torch.long)
y = torch.tensor(ids[1:], dtype=torch.long)

model = nn.Embedding(vocab_size, vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


@torch.no_grad()
def generate(start_id, length):
    out = [start_id]
    for _ in range(length):
        current = torch.tensor([out[-1]])
        logits = model(current)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs[0], num_samples=1).item()
        out.append(next_id)
    return out


if __name__ == "__main__":
    for step in range(500):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    print(f"final loss: {loss.item():.4f}")

    word = "great"
    word_id = word_to_id[word]
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([word_id])), dim=-1)[0]
    top = torch.topk(probs, k=3)
    top_words = [(id_to_word[i.item()], p.item()) for p, i in zip(top.values, top.indices)]
    print(f"top-3 words after {word!r}: {top_words}")

    start_id = word_to_id["we"]
    generated = generate(start_id, length=25)
    print("generated:", decode(generated))
