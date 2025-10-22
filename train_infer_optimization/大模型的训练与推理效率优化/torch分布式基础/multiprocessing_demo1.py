# from multiprocessing import Process
from torch.multiprocessing import Process
from typing import List
import torch
import os


# def func1(x:str)->str:
#     print(f"x value is:{x}")

def func1(x:torch.tensor)->None:
    x2 = x + 20
    print(f"pid : {os.getpid()} x2 shape is:{x2.shape}")

def main1():
    target_list = [f"value_{i}" for i in range(4)]
    result = [func1(x=i) for i in target_list]

    print(result)

def main2():
    target_list = [torch.randint(0,10,size=(i+1,1)) for i in range(6)]
    process_list = []

    for i in range(len(target_list)):
        p = Process(target=func1,args=(target_list[i],)) # 加,是一个 1 元组 不加是int
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()


if __name__ == "__main__":
    main2()