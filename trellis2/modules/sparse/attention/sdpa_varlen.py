import torch
import torch.nn.functional as F

def sdpa_varlen(q, k, v, q_seq_lens, kv_seq_lens):
    device = q.device
    H = q.shape[1]
    
    qs = torch.split(q, q_seq_lens)
    ks = torch.split(k, kv_seq_lens)
    vs = torch.split(v, kv_seq_lens)
    
    outs = []
    for q_i, k_i, v_i in zip(qs, ks, vs):
        # Add batch dimension and transpose to (batch, heads, seq_len, head_dim)
        q_i = q_i.unsqueeze(0).transpose(1, 2)
        k_i = k_i.unsqueeze(0).transpose(1, 2)
        v_i = v_i.unsqueeze(0).transpose(1, 2)
        
        # Use native SDPA without mask to trigger flash/efficient attention backends
        out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, attn_mask=None)
        
        # Transpose back and remove batch dimension
        out_i = out_i.transpose(1, 2).squeeze(0)
        outs.append(out_i)
        
    out = torch.cat(outs, dim=0)
    return out
