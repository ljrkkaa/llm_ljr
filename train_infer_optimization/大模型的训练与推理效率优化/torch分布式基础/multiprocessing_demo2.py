"""
加一个进程间数据通信（send / recv 示例）
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


def run(rank_id, size):
    """
    每个进程执行的任务：
    0 号进程先发送，再接收；
    1 号进程先接收，再发送；
    """
    tensor = torch.zeros(1)  # 每个进程都有一份张量，初始值为 0

    if rank_id == 0:
        tensor += 1
        # 发送 tensor 到进程 1
        dist.send(tensor=tensor, dst=1)
        print(f"[rank {rank_id}] after send -> {tensor.item()}", flush=True)

        # 从进程 1 接收 tensor
        dist.recv(tensor=tensor, src=1)
        print(f"[rank {rank_id}] after recv -> {tensor.item()}", flush=True)

    else:
        # 从进程 0 接收 tensor
        dist.recv(tensor=tensor, src=0)
        print(f"[rank {rank_id}] after recv -> {tensor.item()}", flush=True)

        # 修改 tensor 并发回 rank 0
        tensor += 1
        dist.send(tensor=tensor, dst=0)
        print(f"[rank {rank_id}] after send -> {tensor.item()}", flush=True)


if __name__ == "__main__":
    size = 2  # 两个进程（rank 0 和 rank 1）
    process_list = []

    # 创建并启动多个子进程
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        process_list.append(p)  # 必须把进程对象存下来，否则 join 不到

    # 等待所有子进程执行完毕
    for p in process_list:
        p.join()
