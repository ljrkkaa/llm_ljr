"""
加一个进程间数据通信 broadcast scatter等
"""
import torch.multiprocessing as mp  # 用于多进程
import torch.distributed as dist     # PyTorch 分布式通信包
import torch
import os


def init_process(rankid, size, func, backend='gloo') -> None:
    """
    初始化每个进程的分布式环境，并调用目标函数
    rankid：当前进程在分布式中的 rank 编号（0,1,...）
    size：总进程数（world size）
    func：每个进程要执行的函数
    backend：通信后端，'gloo' 支持 CPU 通信，'nccl' 用于 GPU
    """
    # 设置主节点的 IP 地址和通信端口
    # 所有进程必须保持一致，否则无法建立连接
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "29500"  # 合法端口号范围是 1024~65535

    # 初始化当前进程加入分布式通信组
    dist.init_process_group(
        backend=backend,
        rank=rankid,
        world_size=size
    )

    # 执行用户定义的逻辑
    func(rankid, size)

    # 销毁通信组，释放资源
    dist.destroy_process_group()

# broadcast
# def run(rank_id, size):
#     tensor = torch.tensor(rank_id)  # 每个进程都有一份张量，初始值为 0
#     print(f"before broadcast Rank {rank_id} has data {tensor}")
    
#     dist.broadcast(tensor=tensor,src=0)
#     print(f"after broadcast Rank {rank_id} has data {tensor}")

# scatter
# def run(rank_id, size):
#     tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
#     print('before scatter',' Rank ', rank_id, ' has data ', tensor)
#     if rank_id == 0:
#         scatter_list = [torch.tensor([0,0]), torch.tensor([1,1]), torch.tensor([2,2]), torch.tensor([3,3])]
#         print('scater list:', scatter_list)
#         dist.scatter(tensor, src = 0, scatter_list=scatter_list)
#     else:
#         dist.scatter(tensor, src = 0) # 其他进程只需要准备 tensor，接收数据
#     print('after scatter',' Rank ', rank_id, ' has data ', tensor)

# gather
# def run(rank_id, size):
#     # 每个进程各自创建一个张量 tensor
#     # 例如：rank 0 -> [1,2]， rank 1 -> [3,4]，rank 2 -> [5,6]，rank 3 -> [7,8]
#     # 用于模拟每个进程持有一份独立数据
#     tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
#     print('before gather', ' Rank ', rank_id, ' has data ', tensor)

#     # -------------------
#     # gather 操作：
#     # 每个进程都会执行 dist.gather()；
#     # 其中 dst=0 表示 rank 0 是“接收端”（收集所有进程的 tensor）；
#     # 其他 rank（1~size-1）是“发送端”，它们只需传入 tensor。
#     # -------------------
#     if rank_id == 0:
#         # rank 0 进程需要预先创建一个列表，用来接收所有进程的数据
#         # gather_list 的长度应等于进程数，每个元素的形状与发送端 tensor 一样
#         gather_list = [torch.zeros(2, dtype=torch.int64) for _ in range(4)]

#         # rank 0 调用 gather 时要多传入 gather_list 参数，用于接收数据
#         dist.gather(tensor, dst=0, gather_list=gather_list)

#         # gather 完成后：
#         # rank 0 的 gather_list[0] 会是它自己的 tensor，
#         # gather_list[1] 会是 rank 1 的 tensor，以此类推。
#         print('after gather', ' Rank ', rank_id, ' has data ', tensor)
#         print('gather_list:', gather_list)

#     else:
#         # 非 0 号进程调用 gather 时，只需传入自己的 tensor
#         # 它们的 tensor 会被发送到 rank 0 进程
#         dist.gather(tensor, dst=0)
#         print('after gather', ' Rank ', rank_id, ' has data ', tensor)

# reduce 需要注意这里有个副作用，就是rank 0、rank 1和rank 2的tensor也会被修改
# def run(rank_id, size):
#     tensor = torch.arange(2, dtype=torch.int64) + rank_id
#     print('before reudce',' Rank ', rank_id, ' has data ', tensor)
#     dist.reduce(tensor, dst = 3, op=dist.ReduceOp.SUM,)
#     print('after reudce',' Rank ', rank_id, ' has data ', tensor)


#all-reduce 
def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print('before reudce',' Rank ', rank_id, ' has data ', tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    print('after reudce',' Rank ', rank_id, ' has data ', tensor)

if __name__ == "__main__":
    size = 4  
    process_list = []

    # 创建并启动多个子进程
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        process_list.append(p)  # 必须把进程对象存下来，否则 join 不到

    # 等待所有子进程执行完毕
    for p in process_list:
        p.join()
