import torch
import torch.nn as nn
import PyPDF2

# === 1. Extract text from PDF ===
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

text = extract_text_from_pdf("sophies.pdf")
text = text.strip()

# === 2. Character-level Tokenizer ===
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
block_size = 64

def get_batch(batch_size=32):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# === 3. Mini Transformer Model ===
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size, embed_size, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        head_size = embed_size // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, embed_size, block_size)
        self.ff = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        embed_size = 128
        num_heads = 4
        num_layers = 2

        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, block_size) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embed(pos)[None, :, :]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# === 4. Train the Model ===
model = MiniGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(10000):  # Increase steps for better results
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# === 5. Generate Text ===
print(stoi, itos)
context = torch.tensor([[stoi.get('T', 0)]], dtype=torch.long)  # You can change 'T' to another start char
print(context)
output = model.generate(context, max_new_tokens=200)
print("\nGenerated Text:\n")
print(decode(output[0].tolist()))
