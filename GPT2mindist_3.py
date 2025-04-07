import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
 
torch.set_default_dtype(torch.float32)
global_dtype = torch.float32
global_np_dtype = np.float32

# hyperparameters
n_output = 36 # predict a class (minimum distance) 0, ... 35
n_input = 36
block_size = 36 # max length of input, for positional embedding
batchsize = 32

learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embd = 200 # the dimension of vector representation of code lines
n_head = 10
n_layer = 2 # number of transformer blocks
dropout = 0.2
# ------------

# for tensorboard logging
hparam_dict = {
    'n_output': n_output,
    'n_input': n_input,
    'block_size': block_size,
    'learning_rate': learning_rate,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout
}

# Custom Dataset
class MyDataset(Dataset):
  def __init__(self, codes, lengths, mindists):
    self.codes = codes
    self.lengths = lengths
    self.mindists = mindists

  def __len__(self):
    return len(self.codes)

  def __getitem__(self, index):
    return self.codes[index].to(dtype=global_dtype), self.lengths[index], self.mindists[index].long()

@torch.no_grad()
def estimate_loss(model, testset):
    model.eval()

    test_loss = torch.zeros(len(testset))
    for i, test_data in enumerate(testset):
      code, lengths, mindist = test_data
      code, lengths, mindist = code.to(device), lengths.to(device), mindist.to(device)

      output, loss = model(code, lengths, mindist)
      test_loss[i] = loss.item()
    test_loss = test_loss.mean()

    model.train()

    return test_loss

@torch.no_grad()
def estimate_mse_loss(model, testset):
    model.eval()
    # get the location of the model
    device = next(model.parameters()).device

    vector = torch.arange(36) # shape (n_output,)
    test_loss = torch.zeros(len(testset))
    for i, test_data in enumerate(testset):
      code, lengths, mindist = test_data
      code, lengths, mindist = code.to(device), lengths.to(device), mindist.to(device)

      probs, _ = model(code, lengths, mindist) # shape (B,n_output)

      expectations = (probs @ vector).sum(dim=1) # shape (B,)
      loss = torch.mean((mindist - expectations)**2)
      test_loss[i] = loss.item()
    test_loss = test_loss.mean()

    model.train()

    return test_loss

class Head(nn.Module):
    """ one head of self-attention - with proper variable sequence length handling"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        # input of size (batch, time-step, n_input)
        # output of size (batch, time-step, head size)

        # TODO check if input has no rubbish!
        
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)*(C, hs) -> (B,T,hs)
        q = self.query(x) # (B,T,C)*(C, hs) -> (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) * 1/sqrt(head_size) -> (B, T, T)
        wei = wei.masked_fill(padding_mask == 0, float('-inf')) # (B, T, T) - masking T in dim = 2
        wei = F.softmax(wei, dim=-1) # (B, T, T) - apply softmax to the last (T) dimension (normalize + get rig of ribbish from padding)
        padding_mask_transpose = padding_mask.transpose(-2,-1)
        wei = wei.masked_fill(padding_mask_transpose == 0, float('0'))# (B, T, T) - masking T in dim = 1
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)*(C, hs) -> (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

def create_padding_mask(seq, lengths, pad_token=0, transpose=False):
    """
    Creates a padding mask for variable-length sequences.

    Args:
        seq (torch.Tensor): Tensor of shape (batch_size, seq_len, input_size), containing the input sequences.
        lengths (torch.Tensor): Tensor of shape (batch_size), containing the lengths of the input sequences.
        pad_token (int, optional): Token used for padding. Defaults to 0.

    Returns:
        torch.Tensor: Padding mask of shape (batch_size, 1, seq_len).
    """
    # Generate a mask where padding tokens are 1 and other tokens are 0
    B = seq.size(0)
    seq_len = seq.size(1)
    mask = torch.arange(seq_len, device=lengths.device).expand(B, seq_len) < lengths.unsqueeze(1)
    if transpose:
        mask = mask.unsqueeze(2)
    else:
        mask = mask.unsqueeze(1)
    return mask.long()  # Shape: (batch_size, 1, seq_len) or (batch_size, seq_len, 1)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        padding_mask = create_padding_mask(x, lengths) # crate a padding mask for variable-length sequences
        out = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1) # (B, T, hs^num_heads) 
        out = self.proj(out) # (B, T, hs^num_heads) -> (B, T, n_embd)
        padding_mask_transpose = padding_mask.transpose(-2,-1)
        out = out.masked_fill(padding_mask_transpose == 0, float('0')) # masking along T dimension
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, lengths):
        # x size (batch, time-step, n_embd)
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # normalizing over the last dimension ONLY!
        self.ln2 = nn.LayerNorm(n_embd) # normalizing over the last dimension ONLY!

    def forward(self, x, lengths):
        x = x + self.sa(self.ln1(x), lengths)
        x = x + self.ffwd(self.ln2(x), lengths)
        return (x, lengths)

def expected_distance(probs, target_classes):
    """
    Equivalent to Wasserstein-1
    Compute the Expected Distance (equivalent to Wasserstein-1 distance).
    
    Args:
        probs (torch.Tensor): Probability distribution over classes (shape: [B, num_classes])
        class_labels (torch.Tensor): Array of class labels corresponding to probs (same shape)

    Returns:
        float: Expected Distance (Uncertainty measure)
    """
    B, num_classes = probs.shape
    class_labels = torch.arange(num_classes, device=probs.device).float()  # [0, 1, ..., 36]

    absolute_distances = torch.abs(class_labels.view(1, -1) - target_classes.view(-1, 1))  # [batch_size, num_classes]

    # Compute expected absolute distance
    loss = torch.sum(probs * absolute_distances, dim=1).mean()  # Mean over batch
    return loss

def wasserstein_2_loss(probs, target_class):
    """
    Compute Wasserstein-2 (Expected Squared Distance) loss.

    Args:
        probs (torch.Tensor): Probability distribution over classes (shape: [batch_size, num_classes])
        target_class (torch.Tensor): Ground truth class indices (shape: [batch_size])

    Returns:
        torch.Tensor: Wasserstein-2 loss
    """
    B, num_classes = probs.shape
    class_labels = torch.arange(num_classes, device=probs.device).float()  # [0, 1, ..., 36]

    # Compute squared distance from true class
    squared_distances = (class_labels.view(1, -1) - target_class.view(-1, 1)) ** 2 #shape (B, num_classes)

    # Compute expected squared distance
    loss = torch.sum(probs * squared_distances, dim=1).mean() #shape (B, num_classes) * (B, num_classes) -> (B,)
    return loss

class mySequential(nn.Sequential):
  def forward(self, *inputs):
    for module in self._modules.values():
      if type(inputs) == tuple:
        inputs = module(*inputs)
      else:
        inputs = module(inputs)
    return inputs

class GPT2mindist(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding = nn.Linear(n_input, n_embd) # just projects input to n_embd dim
        self.position_embedding = nn.Embedding(block_size, n_embd) # positional embedding
          
        self.blocks = mySequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, n_output)
        
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input, lengths, targets=None):
        # target shape (B,)
        B, T, _ = input.shape
        
        tok_emb = self.token_embedding(input) # (B,T, n_input) -> (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x, _ = self.blocks(x, lengths) # (B,T,C), second argument - lengths - ignored
        x = self.ln_f(x) # (B,T,C)
        padding_mask_transposed = create_padding_mask(x, lengths, transpose=True) # crate a padding mask for variable-length sequences
        x = x.masked_fill(padding_mask_transposed == 0, float('0')) # masking along T dimension to remove garbage
        x = torch.mean(x, dim=1) # (B,C)  - mean along time (want to predict probabilities)
        logits = self.lm_head(x)  # (B,n_output)
        probs = F.softmax(logits, dim=-1)

        if targets is None:
            loss = None
        else:
            loss = wasserstein_2_loss(probs, targets) #using wasserstein 2 loss -quadratic distance between distributions
        
        return probs, loss

    def save(self, hparams, save_path, optimizer, n_epochs):
        torch.save({
            'epoch': n_epochs,
            'model_parameters': hparams,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)

    def load(self, load_path, optimizer, cpu=False):
        if cpu:
            checkpoint = torch.load(load_path, weights_only=False, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(load_path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['model_parameters'], checkpoint['epoch']


def train_transformer():
    # Set up loggig to TensorBoard
    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter("transformer_attention_with_mean")

    model = GPT2mindist()
    model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # load data
    data_dir = '/kaggle/input/gigaset2/gigaset2_train.pt'
    data = torch.load(data_dir, weights_only=False)

    # loading from file changes dtype for some reason
    generators_data = data['generators']
    generator_lens_data = data['generator_lens']
    mindist_data = data['mindists']

    dataset_len = mindist_data.shape[0]
    permutation = torch.randperm(dataset_len)
    split_len = int(dataset_len * 0.9)

    trainset = MyDataset(generators_data[permutation[:split_len]], generator_lens_data[permutation[:split_len]], mindist_data[permutation[:split_len]])
    testset = MyDataset(generators_data[permutation[split_len:]], generator_lens_data[permutation[split_len:]], mindist_data[permutation[split_len:]])

    train_dataloader = DataLoader(trainset, batch_size = batchsize)
    test_dataloader = DataLoader(testset, batch_size = batchsize)

    # log model
    # code, lengths, mindist = next(iter(train_dataloader))
    # code, lengths, mindist = code.to(device), lengths.to(device), mindist.to(device)
    # writer.add_graph(model, [code, lengths, mindist], use_strict_trace=False)

    # test input
    B, T, C = code.shape
    code.requires_grad_()
    probs, loss = model(code, lengths, targets=mindist)

    # Check for cross-batch dependencies
    jacobian = torch.zeros(B, T, B, T, n_output)
    for b_out in range(B):
        for t_out in range(T):
            grad = torch.autograd.grad(probs[b_out, t_out], code, retain_graph=True)[0]
            jacobian[b_out, t_out] = grad  # Store gradient
    cross_batch_mix = jacobian.abs().sum(dim=(1,3,4))  # Sum over non-batch indices
    cross_batch_mix = cross_batch_mix - torch.diag(torch.diag(cross_batch_mix))  # Zero out diagonal
    print("Cross-batch gradient sum:", torch.sum(cross_batch_mix))

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', min_lr=0, eps=1e-08)
    last_lr = scheduler.get_last_lr()

    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss * (1. + self.min_delta)):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    # Set up scheduler
    early_stopper = EarlyStopper(patience=10, min_delta=0.05)

    eval_interval = 2000
    n_epochs = 50

    for epoch in range(n_epochs):
        print(f'epoch {epoch+1}')

        if epoch == 0:
            test_loss = estimate_loss(model, test_dataloader)
            print(f"Untrained model val loss {test_loss:.4f}")
            # writer.add_scalars('Losses', {'train':0, 'test':test_loss}, 0)

        if early_stopper.early_stop(test_loss):
            break 
        
        current_loss = 0
        for i, train_data in enumerate(train_dataloader):
            # sample a batch of data
            code, lengths, mindist = train_data
            code, lengths, mindist = code.to(device), lengths.to(device), mindist.to(device)
            
            # evaluate the loss
            output, loss = model(code, lengths, targets=mindist)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            current_loss += loss.item()
            optimizer.step()
            
            # every once in a while evaluate the loss on train and val sets
            if i % eval_interval == 0 and i != 0:
                train_loss = current_loss / eval_interval
                current_loss = 0
                
                test_loss = estimate_loss(model, test_dataloader)
                print(f"step {i}: train loss {train_loss:.4f}, val loss {test_loss:.4f}")
                
                # log losses
                # writer.add_scalars('Losses', {'train':train_loss, 'test':test_loss}, epoch*len(train_dataloader) + i)
                
                lr = scheduler.get_last_lr()[0]
                # writer.add_scalar('Learning Rate', lr, epoch*len(train_dataloader) + i)
                
                scheduler.step(test_loss)
                if lr != last_lr:
                    print('new lr: ', lr)
                    last_lr = lr

    metrics_dict = {
        'test_loss': test_loss
    }

    # writer.add_hparams(hparam_dict, metrics_dict)
    
    # writer.close()
    
    save_path = "transformer_3_attention_mean.pt"
    model.save(hparam_dict, save_path, optimizer, n_epochs)

 



