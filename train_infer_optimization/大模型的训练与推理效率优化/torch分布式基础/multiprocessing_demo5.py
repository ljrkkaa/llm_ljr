"""
多机模拟
"""
import torch.multiprocessing as mp  # 用于多进程
import torch.distributed as dist     # PyTorch 分布式通信包
import torch
import os

#all-reduce 
def run(local_rank, rank):
    print(f"local rank: {local_rank}, rank: {rank} \n")

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    dist.init_process_group(backend='nccl')
    run(local_rank=local_rank, rank=rank)

if __name__ == "__main__":
    main()

"""
$ bash run_demo05s1.sh 

local rank: 0, rank: 0 

local rank: 2, rank: 2 

local rank: 1, rank: 1 

local rank: 3, rank: 3 


$ bash run_demo05s2.sh 

local rank: 3, rank: 7 

local rank: 1, rank: 5 

local rank: 2, rank: 6 

local rank: 0, rank: 4 

"""