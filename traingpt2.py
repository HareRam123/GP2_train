from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # number of tokens :50,000 BPE Merges + 256 Byte Tokens + 1 <endoftext> token
    block_size: int = 1024 # maximum context length for predictions (e.g., how many tokens the model can look at when making a prediction)
    n_layer: int = 12 # number of transformer blocks (layers) in the model
    n_head: int = 12 # number of attention heads in each transformer block (multi-head attention allows the model to focus on different parts of the input simultaneously)
    n_embd: int = 768 # dimensionality of the token embeddings and the hidden states in the transformer blocks

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias but a mask to prevent attention to future tokens in the input
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()# batch size, sequence length, embedding dimensionality (n_embd)  
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh = number of heads, hs = head size (n_embd // n_head)
        # C = nh * hs ( C--> number of channels (n_embd)) 
        # e.g GPT2 small has n_embd=768, n_head=12, so hs = 768 // 12 = 64 , C = 12 * 64 = 768
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention: (B, nh, T, hs) @ (B, nh, hs, T) --> (B, nh, T, T)
        # attention (materialises a large tensor of shape (B, nh, T, T) and masks out (sets to -inf) the upper triangular part of the tensor, including the diagonal.)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)

        #replace the above 4 lines with Flash Attention ( Ref. Optimsation paper -2023 )
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        # In Transformer notation, attention block is followed by W_O (output projection). In your code, c_proj is exactly that W_O.
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm - GPT2 uses additional layer norm at the end of the model (before the head
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #weight sharing between the token embedding and the output projection layers
        self.transformer.wte.weight = self.lm_head.weight
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT') and module.NANOGPT_SCALE_INIT:
                std *=  ( 2* self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.transformer.wpe.weight.size(0), "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb # (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # final layer norm AFTER all blocks
        logits = self.lm_head(x) # (b, t, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

class DataLoaderLite:
    def __init__(self, B,T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            file_content = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(file_content) # encode the file content into tokens
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"DataLoaderLite initialized with {len(self.tokens)} tokens")
        print(f"DataLoaderLite will produce batches of size {B} and sequence length {T}")
        print(f"Total number of batches per epoch: {len(self.tokens) // (B*T)}")
        self.current_idx = 0

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_idx:self.current_idx + B*T +1]
        x = buf[:B*T].view(B,T)
        y = buf[1:B*T +1].view(B,T)
        self.current_idx += B*T    

        if self.current_idx + B*T +1 >= len(self.tokens):
            self.current_idx = 0 # reset for next epoch
        return x, y


# model = GPT.from_pretrained('gpt2') 
# print("Loaded the model with %e parameters" % sum(p.numel() for p in model.parameters())) 
# num_return_sequences = 5
# max_length = 30
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
# model.eval()
# model = model.to(device)

# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# prompt = "The meaning of life is"
# tokens = tokenizer.encode(prompt)
# x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_return_sequences, 1)
# x = x.to(device)

# torch.manual_seed(1337)
# with torch.no_grad():
#     for _ in range(max_length):
#         logits = model(x)
#         logits = logits[:, -1, :] # we only care about the last time step
#         probs = F.softmax(logits, dim=-1) # (B, vocab_size)
#         next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
#         x = torch.cat((x, next_token), dim=1) # append to the sequence and continue

#print the generated sequences
# for i in range(num_return_sequences):       
#     generated_tokens = x[i].tolist()
#     generated_text = tokenizer.decode(generated_tokens)
#     print(f"Generated text {i+1}: {generated_text}")

#load the file input.txt and generate text based on the content of the file
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
enc = tiktoken.get_encoding("gpt2")
with open('input.txt', 'r') as f:
    file_content = f.read() 


import time
tokens = enc.encode(file_content[:1024]) # encode the file content into tokens
B,T = 4,32
buf = torch.tensor(tokens[:B*T +1])
buf = buf.to(device)
x = buf[:B*T].view(B,T)
y = buf[1:B*T +1].view(B,T)
x = x.to(device)
y = y.to(device)

model = GPT(GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768))
model = model.to(device)
torch.compile(model)
logits , loss = model(x,y)

print(f"Device: {device}")



import time
B ,T = 8 ,1024
data_loader = DataLoaderLite(B, T)
torch.set_float32_matmul_precision('high')



optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
for epoch in range(100):
    t0 = time.time()
    optimiser.zero_grad()
    x, y = data_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits , loss = model(x,y)
        loss.backward()
    optimiser.step()
    #torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_sec = (B*T) / (t1 - t0)
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Time: {dt:.2f}, Tokens/sec: {tokens_per_sec:.2f}")