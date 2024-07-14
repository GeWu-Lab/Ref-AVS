import torch
import torch.nn as nn
import torch.nn.functional as F

class REF_AVS_Transformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super(REF_AVS_Transformer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 256
        self.scaling = self.head_dim ** -0.5

        self.query_embedding = nn.Linear(embed_dim, embed_dim)
        self.key_embedding = nn.Linear(embed_dim, embed_dim)
        self.value_embedding = nn.Linear(embed_dim, embed_dim)

        self.out_projection = nn.Linear(embed_dim, embed_dim)

        self.beta_source_pool = nn.Parameter(torch.ones([1]))
        self.beta_source_attn = nn.Parameter(torch.ones([1]))

    def forward(self, target, source):
        seq_len_tgt, bsz, dim = target.size()

        _, seq_len_src, _ = source.size()
        seq_len_q = seq_len_tgt
        seq_len_kv = seq_len_src

        q = self.query_embedding(target.permute(1, 0, 2)) 
        k = self.key_embedding(source) 
        v = self.value_embedding(source) 

        q = q.view(bsz, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  
        k = k.view(bsz, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  
        v = v.view(bsz, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   

        scores = torch.matmul(q, k) * self.scaling 
        attention_weights = F.softmax(scores, dim=-1) 
        attended_values = torch.matmul(attention_weights, v)  
        attended_values = attended_values.permute(0, 2, 1, 3).reshape(bsz, seq_len_q, -1) 

        output = self.out_projection(attended_values)  
        
        output = nn.Sigmoid()(self.beta_source_attn) * output
        source_pool = nn.Sigmoid()(self.beta_source_pool) * torch.mean(source, dim=1).view(1, bsz, 256)
 
        return target + output.permute(1, 0, 2) + source_pool 