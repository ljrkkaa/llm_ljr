Data Parrallel 训练实战
实际效果

调用了多GPU进行训练，但是训练速度没有多大，甚至有可能下降(可能大家会观测到不同的现象)

Data Parrallel 的问颢

* 单进程，多线程，由于GIL锁的问题，不能充分发挥多卡的优势
* 由于Data Parrallel的训练策略问题，会存在一个主节点占用比其他节点高很多效率较低，每次训练开始都要重新同步模型，大模型的同步时间会较难接受
* 只适用于单机训练，无法支持真正的分布式多节点训练



DataParallel真的没有用吗

* 并非如此
* 对于并行推理，DataParallel可以派上用场!(其实也一般。。。)

DataParallel 并行推理验证:
--DataParallel.module.forward()

--DataParallel.forward()

--DataParallel.forward()改进版本
