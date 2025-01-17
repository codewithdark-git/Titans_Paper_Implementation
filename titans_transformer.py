import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset

class TitansMemoryModule(nn.Module):
    def __init__(self, d_model, memory_size=512):
        super().__init__()
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.zeros(memory_size, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.forgetting_gate = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # Project input to keys and values
        keys = self.key_proj(x)  # [batch_size, seq_len, d_model]
        values = self.value_proj(x)  # [batch_size, seq_len, d_model]

        # Compute attention scores with memory
        attention_scores = torch.matmul(keys, self.memory.T)  # [batch_size, seq_len, memory_size]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Retrieve from memory
        retrieved_memory = torch.matmul(attention_weights, self.memory)  # [batch_size, seq_len, d_model]

        # Update memory based on surprise
        surprise = torch.norm(values - retrieved_memory, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        forgetting_weights = torch.sigmoid(self.forgetting_gate(values))  # [batch_size, seq_len, 1]

        # Update memory (during inference only)
        if not self.training:
            # Reduce batch and seq dimensions to match memory size
            avg_forgetting_weights = forgetting_weights.mean(dim=(0, 1))  # [1, d_model]
            avg_values = values.mean(dim=(0, 1))  # [1, d_model]

            # Expand or reshape to match memory shape
            avg_forgetting_weights = avg_forgetting_weights.unsqueeze(0).expand(self.memory.size(0), -1)  # [memory_size, d_model]
            avg_values = avg_values.unsqueeze(0).expand(self.memory.size(0), -1)  # [memory_size, d_model]

            # Update memory
            self.memory.data = avg_forgetting_weights * self.memory + (1 - avg_forgetting_weights) * avg_values

        return retrieved_memory


class TitansTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, memory_size=512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.titans_memory = TitansMemoryModule(d_model, memory_size)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Titans memory integration
        memory_output = self.titans_memory(src)
        src = src + self.dropout2(memory_output)
        src = self.norm2(src)

        # Feed-forward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        return src

class TitansTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, memory_size=512):
        super().__init__()

        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Create encoder layers with Titans memory
        self.layers = nn.ModuleList([
            TitansTransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, memory_size)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_tokens)

        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)

        src = self.norm(src)
        output = self.fc_out(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# # Training utilities
# def create_mask(size):
#     mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
#     return mask

# def train_epoch(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0

#     for batch in dataloader:
#         optimizer.zero_grad()

#         src = batch[:-1].to(device)
#         tgt = batch[1:].to(device)

#         mask = create_mask(src.size(1)).to(device)

#         output = model(src, src_mask=mask)
#         loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# Example usage:
def main():
    # Model parameters
    num_tokens = 50000  # Vocabulary size
    d_model = 512
    nhead = 8
    num_layers = 6
    memory_size = 512

    # Initialize model
    model = TitansTransformer(
        num_tokens=num_tokens,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        memory_size=memory_size
    )

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    print("Model initialized and ready for training")

if __name__ == "__main__":
    main()
