import torch
import torch.nn as nn
from layers.Invertible import  sample_norm
from layers.Embed import Patch_Embedding
import math
from einops import rearrange
import torch.nn.functional as F
import math
from layers.Transformer_EncDec import TSEncoder


# Temporal Structure Encoder
class TSEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_p, enc_in, k_structure, attn_dropout=0, dropout=0.3, e_layers=1):
        super(TSEncoder, self).__init__()
        self.d_model = d_model
        self.k_structure = k_structure
        self.enc_in = enc_in
        self.n_p = n_p
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.use_residual = (e_layers > 1)

        self.ts_structure = nn.Parameter(torch.randn(k_structure, n_p, n_p))
        nn.init.orthogonal_(self.ts_structure)

        self.channel_query = nn.Parameter(torch.randn(enc_in, k_structure))
        nn.init.normal_(self.channel_query, mean=0, std=1.0)

        self.instance_query = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),                       
            nn.Linear(d_model // 4, k_structure) 
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, n, d = x.shape
        residual = x
     
        x_flat = x.reshape(-1, d) # (B*C*n, d)
        z_flat = self.proj(x_flat) # (B*C*n, d)
        z = z_flat.view(B, C, n, d)
      
        static_coef = self.channel_query.unsqueeze(0).expand(B, -1, -1) # (B, C, k_structure)
        
        x_summary = x.mean(dim=2) 
        dynamic_coef = self.instance_query(x_summary) # (B, C, k_structure)

        final_coef = static_coef  +  dynamic_coef # (B, C, k_structure)

        norm_coef = torch.softmax(final_coef, dim=-1)

        norm_coef_flat = norm_coef.view(B*C, self.k_structure) # (B*C, k)
        ts_structure = self.ts_structure
        structure_flat = ts_structure.view(self.k_structure, -1) # (k, n*n)
        ts = torch.matmul(norm_coef_flat, structure_flat).view(B*C, n, n)

        ts = torch.softmax(ts, dim=-1) # (B*C, n, n)
        ts = self.attn_dropout(ts)


        z_in = z.reshape(B*C, n, d)        
        x_enc = torch.bmm(ts, z_in)  # (B*C, n, d)


        x_enc = x_enc.view(B, C, n, d)  
        
        if self.use_residual:
            x_enc = x_enc + residual

        x_enc = x_enc.reshape(-1, d)
        x_enc = self.mlp(x_enc) + x_enc

        x_enc = x_enc.view(B, C, n, d)
        
        return x_enc

# Parameter-efficient Channel Adaptation Head PCAHead
class PCAHead(nn.Module):   
    def __init__(self, d_model, n_p, pred_len, k_structure, r=4,  head_dropout=0.1):
        super().__init__()
        self.in_features = n_p * d_model
        self.out_features = pred_len
        self.r = r
        
        # Base Projection (Shared)
        self.base_head = nn.Linear(self.in_features, self.out_features, bias=False)

        # Low Rank Adaptations (K structure)
        self.lr_A = nn.Parameter(torch.randn(k_structure, self.in_features, r))
        self.lr_B = nn.Parameter(torch.randn(k_structure, r, self.out_features))
        self._reset_parameters()

        self.dropout = nn.Dropout(head_dropout)

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_head.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.lr_A, a=5**0.5)
        nn.init.zeros_(self.lr_B)  

    def forward(self, x_enc, channel_coeff):

        B, C, n, d = x_enc.shape
        x_flat = x_enc.reshape(B, C, -1)  # [B, C, d_in]
        out = self.base_head(x_flat)
        coefficients = torch.softmax(channel_coeff, dim=-1)
        coefficients = self.dropout(coefficients)
        K, In, r = self.lr_A.shape
        W_down = (coefficients @ self.lr_A.view(K, -1)).view(C, In, r)
        Out = self.out_features
        W_up = (coefficients @ self.lr_B.view(K, -1)).view(C, r, Out)
        x_perm = x_flat.permute(1, 0, 2) 
        delta = torch.bmm(x_perm, W_down)
        delta = torch.bmm(delta, W_up)
        delta = delta.permute(1, 0, 2)
        return out + delta


# TStruct
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.patch_number = configs.seq_len // configs.patch_size
        self.sample_norm = sample_norm(configs) 
        self.patch_size = configs.patch_size
        self.patch_embedding = Patch_Embedding(configs, num_p=self.patch_number, d_model=configs.d_model) 

        self.ts_encoder = nn.ModuleList(
                [TSEncoder(configs.d_model, configs.d_ff, n_p=self.patch_number, \
                    enc_in=self.enc_in, k_structure=configs.k_structure, \
                    attn_dropout=configs.attn_dropout, dropout=configs.dropout, e_layers=configs.e_layers)
                for i in range(configs.e_layers)]
        )

        self.head = PCAHead(
            d_model=configs.d_model,n_p=self.patch_number,pred_len=configs.pred_len,k_structure=configs.k_structure,\
            r=configs.rank,head_dropout=configs.head_dropout
        )


        self.channel_coeff = nn.Parameter(torch.randn(self.enc_in, configs.k_structure))
        nn.init.normal_(self.channel_coeff, mean=0, std=1.0)

    def forward(self, x, flag='train'):
        # x: [B, L, C]
        B,L,C = x.shape
        N = self.patch_number
        x = self.sample_norm(x, 'norm')

       
        x_emb = self.patch_embedding(x)  # (B, C, n, d)

        current_x = x_emb 

        for layer in self.ts_encoder:
            current_x = layer(current_x) # (B, C, n, d)
        
        y = self.head(current_x, self.channel_coeff) # [B, C, H]
        
        y = y.permute(0, 2, 1) # [B, H, C]
        y = self.sample_norm(y, 'denorm')

        return y

