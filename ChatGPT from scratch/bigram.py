import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
seq_len = 64 # how many tokens per sequence?
block_size = 8 # what is the max context length for prediction?
max_iters = 3000 
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters =200
n_embed = 32 # embedding dimension
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()
    
# here are all the unique characters that occur in this text
chars = list(set(text))
vocab_size = len(chars)
# Create mapping from characters tro integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]

# decoder: take a list of integers, output a string
decode = lambda l: ''.join([ itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype = torch.long)
# Let's now split up the data into train and
#    validation sets
# first 90% will be train, rest val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading 
def get_batch(split):
    # generate a small batch of data of input x and target y
    data = train_data if split == 'train' else val_data

    # len(data) - block_size -> we want last character that to predict in y
    ix = torch.randint(len(data) - block_size, (batch_size, ))

    # Stack up batch_size which each batch is row
    x = torch.stack([data[ i : i + block_size] for i in ix])
    y = torch.stack([data[ i + 1 : i + block_size + 1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
           X, Y = get_batch(split)
           logits, loss = model(X, Y)
           losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#  Super simple bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()

    # each token directly read off
    #   the logits for the next token from
    #   a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
  def forward(self, idx, targets = None):

    # idx and targets are both (B, T) tensor of integers
    # the next score of sequence prediction
    logits = self.token_embedding_table(idx) # (Batch, Time = block size, Channels = vocab size)

    if targets is None:
      loss = None
    else:
      # Reshape logits and targets
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view( B * T )

      # Evaluate by loss function: negative log likelihood
      loss = F.cross_entropy(logits, targets)


    return logits, loss


  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # Get the predictions
      logits, loss = self(idx)
      # Focus on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # Apply softmax to get prob
      probs = F.softmax(logits, dim = -1) # (B, C)
      # sample from distribution
      idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
      # Append sample index to the running samples
      # E.g: idx: we ate -> idx_next: rice
      #            := idx: we ate rice
      # We continue to generate up to max_block_size or max time step
      # We add idx_next into next time step by learn from all batch previous
      idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)

    return idx


model = BigramLanguageModel()
m = model.to(device)


# Create a Pytorch optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    
    # every once in a whil evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    

# Generate from the model
context = torch.zeros( (1, 1), dtype = torch.long, device = device)
print(decode(
    m.generate(context, max_new_tokens = 500)[0].tolist()
))
    