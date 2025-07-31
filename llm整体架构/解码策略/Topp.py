import torch
import torch.nn.functional as F

def top_p_sampling(logits, p=0.9):
    # 应用softmax得到概率分布
    probs = F.softmax(logits, dim=-1)
    # 对概率分布进行排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)    
    # 找到第一个使得累积概率大于p的位置
    cutoff_idx = torch.sum(cumulative_probs <= p, dim=-1)
    # 根据cutoff_idx截断概率分布
    masked_probs = torch.gather(probs, dim=-1, index=sorted_indices[:, :cutoff_idx + 1])    
    # 重新归一化剩余概率
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
    # 从候选词中随机选择一个词
    selected_idx = torch.multinomial(masked_probs, 1)    
    return sorted_indices.gather(-1, selected_idx)