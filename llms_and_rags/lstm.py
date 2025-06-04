from datasets import load_dataset
from collections import Counter
import torch
import re
import random
import torch.nn as nn

dataset = load_dataset("ag_news", split="train[:1000]")  # Use small dataset
text = " ".join(item["text"] for item in dataset)
text = text.lower()
tokens = re.findall(r'\b\w+\b', text)

# Build vocab
counter = Counter(tokens)
vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(5000))}
vocab["<unk>"] = 0
vocab["<pad>"] = 1
inv_vocab = {i: w for w, i in vocab.items()}

# Encode text
encoded = [vocab.get(token, vocab["<unk>"]) for token in tokens]






seq_len = 30

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        target = data[i+1:i+seq_len+1]
        sequences.append((torch.tensor(seq), torch.tensor(target)))
    return sequences

sequences = create_sequences(encoded, seq_len)
dataloader = torch.utils.data.DataLoader(sequences, batch_size=32, shuffle=True)

# class RNNLanguageModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x, hidden=None):
#         x = self.embed(x)
#         out, hidden = self.rnn(x, hidden)
#         out = self.fc(out)
#         return out, hidden

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        # LSTM returns (output, (h_n, c_n)) for hidden
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMLanguageModel(len(vocab), embed_dim=64, hidden_dim=128).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(5):
    total_loss = 0
    for x, y in dataloader:
        # x, y = x.cuda(), y.cuda()
        x, y = x.to(device), y.to(device) # Use .to(device)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out.view(-1, len(vocab)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# During generation, hidden will be a tuple (h, c)
def generate_text(start_text, max_len=50):
    model.eval()
    words = start_text.lower().split()
    input_ids = [vocab.get(w, vocab["<unk>"]) for w in words]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    hidden = None

    for _ in range(max_len):
        out, hidden = model(input_tensor[:, -seq_len:], hidden)
        probs = torch.softmax(out[0, -1], dim=0)
        next_id = torch.multinomial(probs, 1).item()
        words.append(inv_vocab.get(next_id, "<unk>"))
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]], dtype=torch.long).to(device)], dim=1)

    return " ".join(words)

print(generate_text("the government announced", max_len=5))
