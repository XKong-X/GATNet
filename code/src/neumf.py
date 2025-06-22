import torch
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint
from torch import nn
from ggca import GGCA
from asp import ASP
from tbo import *

class NCF(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 hidden_dim=64,
                 output_dim=1,
                 sparsity=0.2,
                 ema_decay=0.95,
                 gate_threshold=0.3,
                 alpha=0.2,
                 dropout=0.2):
        super().__init__()

        # ---- Embedding ----
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        feat_dim = embedding_dim * 2          # 128

        # ---- 分支 1：GUP ----
        self.gup = ASP(input_dim=feat_dim,
                       embed_dim=hidden_dim,
                       sparsity=sparsity,
                       ema_decay=ema_decay)
        self.linear_gup = nn.Linear(hidden_dim, hidden_dim)

        # ---- 分支 2：SRM ----
        # 让 SRM 按 128 维初始化
        self.srm = GGCA(in_features=feat_dim,
                       gate_threshold=gate_threshold,
                       alpha=alpha)
        # 把 128 压到 64 方便后续拼接
        self.linear_srm = nn.Linear(feat_dim, hidden_dim)

        # ---- 拼接后的融合 ----
        self.final_linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_input, item_input, update_kalman=True):
        # Embedding
        u_emb = self.user_embedding(user_input)      # (B,64)
        i_emb = self.item_embedding(item_input)      # (B,64)
        x = torch.cat([u_emb, i_emb], dim=1)         # (B,128)

        # ---- 分支 1：GUP ----
        x_gup = self.gup(x, update_kalman=update_kalman)
        x_gup = F.relu(x_gup)
        x_gup = self.dropout(x_gup)
        x_gup = self.linear_gup(x_gup)               # (B,64)

        # ---- 分支 2：SRM ----
        x_srm = self.srm(x)                          # (B,128)
        x_srm = F.relu(x_srm)
        x_srm = self.dropout(x_srm)
        x_srm = self.linear_srm(x_srm)               # (B,64)

        # ---- 融合 ----
        out = self.final_linear(torch.cat([x_gup, x_srm], dim=1))  # (B,1)
        return out
