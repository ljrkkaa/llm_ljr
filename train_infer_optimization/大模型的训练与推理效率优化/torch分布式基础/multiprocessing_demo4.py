"""
torch run启动! 

"""
import torch.multiprocessing as mp  # 用于多进程
import torch.distributed as dist     # PyTorch 分布式通信包
import torch
import os

#all-reduce 
def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.float64) + rank_id
    tensor = tensor.to(f"cuda:{rank_id}")

    print('--------------before reudce',' Rank ', rank_id, ' has data ', tensor,'\n')
    # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    print('--------------after reudce',' Rank ', rank_id, ' has data ', tensor,'\n')

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    dist.init_process_group(backend='nccl')
    run(rank_id=rank, size=0)

if __name__ == "__main__":
    main()