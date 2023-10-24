import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the max context length for prediction?
max_iters = 5000 
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384 # embedding dimension
n_head = 6
n_layer = 6
dropout = 0.2


#  -----------------
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

class Head(nn.Module):
    """ one head of self-attention"""
  
    def __init__(self, head_size): 
      super().__init__()
      self.key = nn.Linear(n_embed, head_size, bias = False)
      self.query = nn.Linear(n_embed, head_size, bias = False)
      self.value = nn.Linear(n_embed, head_size, bias = False)
      
      # Store tril matrix into buffer, cuz it's not parameter
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
      
      self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
      B, T, C = x.shape
      k = self.key(x) # (B, T, head_size)
      q = self.query(x) # (B, T, head_size)
      # compute attention score 'affinities'
      wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) ----> (B, T, T)
      # Features aren't commnunicate with the past node -> decoder block
      wei = wei.masked_fill(        # (B, T, T)
        self.tril[:T, :T] == 0,
        float('-inf')
      )
      
      wei = F.softmax(wei, dim = -1) # (B, T, T)
      wei = self.dropout(wei) # (B, T, T)
      # perform the weighted aggregation of the values
      v = self.value(x) # (B, T, head_size)
      out = wei @ v # (B, T, T) @ (B, T, head_size) ----> (B, T, head_size)
      return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim = -1)
        
        # Extract features from all heads and project to residual pathway
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by non-linearity """

  def __init__(self, n_embed):
    super().__init__()
    
    # In paper, input and output d = n_embed, so inner layer of FFWD d = 4 * n_embed -> growing by addition of residual pathway
    self.net = nn.Sequential(
      nn.Linear(
        in_features = n_embed, 
        out_features = 4 * n_embed
      ),
      nn.ReLU(),
      nn.Linear(
        in_features = 4 * n_embed, 
        out_features = n_embed
      ),
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computation"""
  def __init__(self, n_embed, n_head) -> None:
      super().__init__()
      head_size = n_embed // n_head
      # Communication
      self.sa = MultiHeadAttention(n_head, head_size) 
      # Computation
      self.ffwd = FeedForward(n_embed)
      self.ln1 = nn.LayerNorm(n_embed)
      self.ln2 = nn.LayerNorm(n_embed)
      
  def forward(self, x):
    x = x + self.sa(
                  self.ln1(x)
                )
    x = x + self.ffwd(
                  self.ln2(x)
                )
    
    return x



#  Super simple bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()

    # each token directly read off
    #   the logits for the next token from
    #   a lookup table
    # Identity of token
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 

    # Position encoding of token i to i + block_size - 1: block pass through attention
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    
    # # self attention
    # self.sa_heads = MultiHeadAttention(4, n_embed // 4) # i.e . 4 heads of 8-dimensional attention
    
    # # Feedforward step
    # self.ffwd = FeedForward(n_embed)
    
    # Blocks of transformer
    self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range(n_layer)] )
    
    self.ln_f = nn.LayerNorm(n_embed) # final layer norm
    
    self.lm_head = nn.Linear(n_embed, vocab_size)
    
  def forward(self, idx, targets = None):
      B, T = idx.shape
      
      # idx and targets are both (B, T) tensor of integers
      # the next score of sequence prediction
      tok_emb = self.token_embedding_table(idx) # (Batch, Time = block size, Channels = embedding size)
      pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (Time = block size, Channels = embedding size)
      
      x = tok_emb + pos_emb # (Batch, Time = block size, Channels = embedding size)
      
      # x = self.sa_heads(x) # apply one head of self-attention (Batch, Time = block size, Channels = embedding size)
      # x = self.ffwd(x) # apply feedforward (Batch, Time = block size, Channels = embedding size)
      
      # Apply n blocks of transformer -> try to communicate with past tokens
      x = self.blocks(x)  # (Batch, Time = block size, Channels = embedding size)
      x = self.ln_f(x) # (Batch, Time = block size, Channels = embedding size)
      
      # Apply decoder LM head 
      logits = self.lm_head(x) # (Batch, Time = block size, Channels = vocab size)

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
      # crop idx to the last block_size tokens (B, block_size) --> prevent out of scope of position
      idx_cond = idx[:, -block_size:]
      # Get the predictions
      logits, loss = self(idx_cond)
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
# pirnt the number of params in the mode
print(sum(p.numel() for p in m.parameters())/1e6, 'M params')

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
    