import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, item_seq):
        # x:        [B, L, H]
        # item_seq: [B, L]

        B, L, H = x.shape

        Q = self.q_proj(x)  # [B, L, H]
        K = self.k_proj(x)  # [B, L, H]
        V = self.v_proj(x)  # [B, L, H]

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Q, K, V: [B, num_heads, L, head_dim]

        scores = Q @ K.transpose(-2, -1)
        # scores: [B, num_heads, L, L]

        scores = scores / (self.head_dim**0.5)

        padding_mask = item_seq != 0
        # [B, L]

        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        # [B, 1, 1, L]

        causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        # [L, L]

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # [1, 1, L, L]

        attention_mask = padding_mask & causal_mask
        # [B, 1, L, L]

        scores = scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        # [B, num_heads, L, L]

        attn_weights = self.attn_dropout(attn_weights)

        context = attn_weights @ V
        # [B, num_heads, L, head_dim]

        context = context.transpose(1, 2).contiguous()
        # [B, L, num_heads, head_dim]

        context = context.view(B, L, H)
        # [B, L, H]

        out = self.out_proj(context)
        out = self.out_dropout(out)

        query_mask = (item_seq != 0).unsqueeze(-1)
        # [B, L, 1]

        out = out * query_mask

        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()

        self.attn_norm = nn.LayerNorm(hidden_size)

        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
        )

        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, item_seq):
        # x:        [B, L, H]
        # item_seq: [B, L]

        attn_input = self.attn_norm(x)
        attn_output = self.attention(attn_input, item_seq)

        x = x + attn_output

        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)

        x = x + ffn_output

        query_mask = (item_seq != 0).unsqueeze(-1)
        x = x * query_mask

        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items,
        hidden_size=64,
        max_seq_len=50,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)

        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, item_seq):
        # item_seq: [B, L]

        B, L = item_seq.shape

        positions = torch.arange(L, device=item_seq.device)
        positions = positions.unsqueeze(0).expand(B, L)
        # positions: [B, L]

        item_emb = self.item_embedding(item_seq)
        # [B, L, H]

        pos_emb = self.position_embedding(positions)
        # [B, L, H]

        x = item_emb + pos_emb
        x = self.embedding_dropout(x)

        padding_mask = (item_seq != 0).unsqueeze(-1)
        x = x * padding_mask

        for block in self.blocks:
            x = block(x, item_seq)

        x = self.final_norm(x)
        x = x * padding_mask

        return x
        # [B, L, H]

    def calculate_loss(self, item_seq, pos_seq, neg_seq):
        # item_seq: [B, L]
        # pos_seq:  [B, L]
        # neg_seq:  [B, L]

        seq_output = self.forward(item_seq)
        # [B, L, H]

        pos_emb = self.item_embedding(pos_seq)
        neg_emb = self.item_embedding(neg_seq)

        # [B, L, H]

        pos_logits = (seq_output * pos_emb).sum(dim=-1)
        neg_logits = (seq_output * neg_emb).sum(dim=-1)

        # [B, L]

        valid_mask = pos_seq != 0
        # считаем loss только там, где есть настоящий positive item

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask])
        )

        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask])
        )

        loss = pos_loss + neg_loss

        return loss

    def predict(self, item_seq, candidate_items):
        # item_seq:         [B, L]
        # candidate_items:  [B, N]

        seq_output = self.forward(item_seq)
        # [B, L, H]

        last_hidden = seq_output[:, -1, :]
        # [B, H]

        candidate_emb = self.item_embedding(candidate_items)
        # [B, N, H]

        scores = torch.matmul(candidate_emb, last_hidden.unsqueeze(-1)).squeeze(-1)

        # scores: [B, N]

        return scores

    def full_sort_predict(self, item_seq):
        # item_seq: [B, L]

        seq_output = self.forward(item_seq)
        # [B, L, H]

        last_hidden = seq_output[:, -1, :]
        # [B, H]

        all_item_emb = self.item_embedding.weight
        # [num_items + 1, H]

        scores = last_hidden @ all_item_emb.T
        # [B, num_items + 1]

        scores[:, 0] = -1e9

        return scores


import copy


class StackedSASRec(SASRec):
    def __init__(
        self,
        item_num,
        maxlen,
        hidden_units,
        num_blocks,
        num_heads,
        dropout_rate,
    ):
        super().__init__(
            item_num=item_num,
            maxlen=maxlen,
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

    @property
    def num_transformer_blocks(self):
        return len(self.blocks)

    def double_blocks(self, max_blocks=None, mode="append"):
        old_num_blocks = len(self.blocks)

        if max_blocks is None:
            target_num_blocks = old_num_blocks * 2
        else:
            target_num_blocks = min(old_num_blocks * 2, max_blocks)

        if target_num_blocks <= old_num_blocks:
            return False

        old_blocks = list(self.blocks)
        new_blocks = torch.nn.ModuleList()

        if mode == "append":
            for block in old_blocks:
                new_blocks.append(block)

            idx = 0
            while len(new_blocks) < target_num_blocks:
                new_blocks.append(copy.deepcopy(old_blocks[idx]))
                idx += 1

        elif mode == "interleave":
            for block in old_blocks:
                if len(new_blocks) < target_num_blocks:
                    new_blocks.append(block)
                if len(new_blocks) < target_num_blocks:
                    new_blocks.append(copy.deepcopy(block))

        else:
            raise ValueError(f"Unknown stacking mode: {mode}")

        self.blocks = new_blocks

        print(f"Stacking: transformer blocks {old_num_blocks} -> {len(self.blocks)}")

        return True
