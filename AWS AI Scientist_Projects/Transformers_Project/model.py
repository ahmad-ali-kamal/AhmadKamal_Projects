import torch
import torch.nn as nn
import math

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer('casual_attention_mask', casual_attention_mask)

    def forward(self, x):
        batch_size, tokens_num, d_embed = x.shape
        Q = self.Q_weights(x)
        K = self.K_weights(x)
        V = self.V_weights(x)
        attention_scores = Q @ K.transpose(1,2)
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0, float('-inf')
        )
        attention_scores = attention_scores / math.sqrt(K.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config['heads_num'])])
        self.linear = nn.Linear(config['heads_num']*config['head_size'], config['d_embed'])
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, x):
        x_heads = [head(x) for head in self.heads]
        x = torch.cat(x_heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['d_embed'], 4*config['d_embed']),
            nn.GELU(),
            nn.Linear(4*config['d_embed'], config['d_embed']),
            nn.Dropout(config['dropout_rate'])
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config['d_embed'])
        self.ff = FeedForward(config)
        self.ln2 = nn.LayerNorm(config['d_embed'])

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class DemoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocabulary_size'], config['d_embed'])
        self.pos_embedding = nn.Embedding(config['context_size'], config['d_embed'])
        self.layers = nn.Sequential(*[Block(config) for _ in range(config['layers_num'])])
        self.ln = nn.LayerNorm(config['d_embed'])
        self.classifier = nn.Linear(config['d_embed'], config['num_classes'])

    def forward(self, token_ids):
        batch_size, tokens_num = token_ids.shape
        x = self.token_embedding(token_ids)
        positions = torch.arange(tokens_num, device=token_ids.device)
        pos_embed = self.pos_embedding(positions)
        x = x + pos_embed.unsqueeze(0)
        x = self.layers(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits
