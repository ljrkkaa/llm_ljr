import torch
import torch.nn.functional as F

def greedy_search(logits, max_len=20, eos_token_id=2):
    batch_size = logits.size(0)
    seq_len = logits.size(1)
    vocab_size = logits.size(2)
    
    # 初始化生成的序列，存储每个时间步选择的词索引
    generated_sequence = torch.zeros(batch_size, max_len, dtype=torch.long)
    # 记录当前生成的序列，初始化为开始 token（一般为0或者某个特殊的起始符号）
    current_input = torch.zeros(batch_size, dtype=torch.long)  # 起始token id  
    for t in range(max_len):
        # 获取当前步骤的 logits
        current_logits = logits[:, t, :]        
        # 计算每个词的概率分布（通过softmax）
        probs = F.softmax(current_logits, dim=-1)       
        # 选择概率最高的词索引（greedy）
        next_token = torch.argmax(probs, dim=-1)        
        # 将选择的词加入到生成的序列中
        generated_sequence[:, t] = next_token        
        # 检查是否遇到结束标记，若遇到则停止生成
        if (next_token == eos_token_id).all():
            break        
        # 更新当前输入（下一个时间步的输入）
        current_input = next_token
    return generated_sequence