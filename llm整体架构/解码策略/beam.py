import torch
import torch.nn.functional as F

def pred(input):
    batch,seq_len=input.shape
    generate=torch.randn(size=(batch,1,10))
    return generate

def beam_search(input_ids,max_length,num_beams):
    batch=input_ids.shape[0]

    #输入扩展：效果是把input_ids复制成 batch_size * beam_size的个数
    expand_size=num_beams
    # expanded_return_idx维度为(batch_size * beam_size,)
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    # torch.index_select 是一个用于按特定索引在某个维度上选择元素的操作 
    # input_ids维度为(batch_size * beam_size, seq_len)
    input_ids = input_ids.index_select(0, expanded_return_idx)
    print(input_ids)

    batch_beam_size,cur_len=input_ids.shape
    beam_scores=torch.zeros(size=(batch,num_beams),dtype=torch.float,device=input_ids.device)
    # 只有第一个 beam 的得分为 0，其它为负无穷
    beam_scores[:,1:]=-1e9

    beam_scores=beam_scores.view(size=(batch*num_beams,))
    next_tokens=torch.zeros(size=(batch,num_beams),dtype=torch.long,device=input_ids.device)
    next_indices=torch.zeros(size=(batch,num_beams),dtype=torch.long,device=input_ids.device)

    while cur_len<max_length:

        logits=pred(input_ids)    #batch,seq_len,vocab
        next_token_logits=logits[:,-1,:]  #当前时刻的输出

        #归一化
        next_token_scores=F.log_softmax(next_token_logits,dim=-1)   # (batch_size * num_beams, vocab_size)
        #求概率  beam_scores[:, None] 是把 (B,) 形状的向量变成 (B, 1) 的矩阵
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)  # 当前概率+先前概率

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch, num_beams * vocab_size)

        # 当前时刻的token 得分，  token_id
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = next_tokens // vocab_size  #对应的beam_id
        next_tokens = next_tokens % vocab_size    #对应的indices

        #集束搜索核心
        def process(input_ids,next_scores,next_tokens,next_indices):
            batch_size=3
            group_size=3
            next_beam_scores = torch.zeros((batch_size, num_beams), dtype=next_scores.dtype)
            next_beam_tokens = torch.zeros((batch_size, num_beams), dtype=next_tokens.dtype)
            next_beam_indices = torch.zeros((batch_size,num_beams), dtype=next_indices.dtype)

            for batch_idx in range(batch_size):
                beam_idx=0
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
                ):
                    batch_beam_idx=batch_idx*num_beams+next_index

                    next_beam_scores[batch_idx, beam_idx] = next_score      #当前路径得分
                    next_beam_tokens[batch_idx, beam_idx] = next_token      #当前时刻的token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx  #先前对应的id

                    beam_idx += 1

            return next_beam_scores.view(-1), next_beam_tokens.view(-1), next_beam_indices.view(-1)

        beam_scores, beam_next_tokens, beam_idx=process(input_ids,next_token_scores,next_tokens,next_indices)

        # 更新输入， 找到对应的beam_idx, 选择的tokens, 拼接为新的输入      #(batch*beam,seq_len)
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1
    #输出
    return input_ids,beam_scores


if __name__ == '__main__':
    input_ids=torch.randint(0,100,size=(3,1))
    print(input_ids)
    input_ids,beam_scores=beam_search(input_ids,max_length=10,num_beams=3)
    print(input_ids)
