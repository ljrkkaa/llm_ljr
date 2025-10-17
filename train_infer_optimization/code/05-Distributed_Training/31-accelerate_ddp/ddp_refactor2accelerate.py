"""
跟着教程改为accelerate的过程
torchrun --nproc_per_node=3 ddp_refactor2accelerate.py
或者
accelerate launch ddp_refactor2accelerate.py


端口占用 换个端口或者杀掉端口都行
torchrun --nproc_per_node=3 --master_port=29501 ddp_refactor2accelerate.py
或者
让 Accelerate 自动选择可用端口
accelerate launch --main_process_port 0 ddp_refactor2accelerate.py

0 会让系统随机选择一个可用端口，避免冲突。
"""
import os
import torch
import pandas as pd
from torch.optim import Adam
from accelerate import Accelerator
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification


class MyDataset(Dataset): 

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def prepare_dataloader():

    dataset = MyDataset()

    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))

    tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs
    # DistributedSampler 是分布式训练用的采样器。它会根据当前进程（rank）只取该 GPU 对应的数据子集。
    # 每个 GPU 只处理自己负责的数据子集，避免重复训练相同样本。
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = BertForSequenceClassification.from_pretrained("hfl/rbt3")

    # if torch.cuda.is_available():
    #     model = model.to(int(os.environ["LOCAL_RANK"]))
    
    # 将模型包装成 DDP 模型，负责自动同步梯度。
    # model = DDP(model)

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)


def evaluate(model, validloader, accelerator:Accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            # if torch.cuda.is_available():
            #     batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            # 避免每批次数目都是batch 正确率大于1了
            # ep: 0, global_step: 0, loss: 0.9249660968780518
            # ep: 0, acc: 1.0747421979904175
            pred, refs = accelerator.gather_for_metrics((pred,batch["labels"]))
            accelerator.print(pred.shape)
            acc_num += (pred.long() == refs.long()).float().sum()
    # 验证阶段同步各 GPU 的准确率计数。
    # dist.all_reduce(acc_num)
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, accelerator:Accelerator, epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        # 保证每个 epoch shuffle 不同，避免每轮都取相同数据
        # trainloader.sampler.set_epoch(ep)
        for batch in trainloader:
            # if torch.cuda.is_available():
            #     batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                # 将所有 GPU 上的 loss 求平均，只打印一次（rank=0）。
                # dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                loss = accelerator.reduce(loss,"mean")
                accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate(model, validloader,accelerator)
        accelerator.print(f"ep: {ep}, acc: {acc}")


def main():

    # dist.init_process_group(backend="nccl") # 初始化分布式训练环境（创建通信组）
    accelerator = Accelerator()

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer,trainloader,validloader = accelerator.prepare(model, optimizer,trainloader,validloader)

    train(model, optimizer, trainloader, validloader,accelerator)


if __name__ == "__main__":
    main()