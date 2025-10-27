"""
Accelerate实例
https://huggingface.co/docs/accelerate/quicktour

accelerate config 查看配置
accelerate test 测试

python -m accelerate.commands.launch train_infer_optimization/code/AccelerateDemo.py 运行
注意改好deepspeed的配置

"""

from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import DataLoader, TensorDataset


class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # .to("cuda:0")
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)  # .to("cuda:1")

    def forward(self, x):
        # x.to("cuda:0")
        x = torch.relu(self.fc1(x))
        # x.to("cuda:1")
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    batch_size = 64
    data_size = 10000

    input_data = torch.randn(data_size, input_dim)
    labels = torch.randn(data_size, output_dim)

    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(input_dim, hidden_dim, output_dim)

    # 这里是zaRO技术 优化状态、梯度与参数分割 Zero-3
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)

    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    optimization = torch.optim.Adam(model.parameters(), lr=0.00015)
    crition = torch.nn.MSELoss()

    # 主要改变了这里
    model, dataloader, optimization = accelerator.prepare(
        model, dataloader, optimization
    )

    for epoch in range(1000):
        model.train()
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = crition(outputs, labels)

            optimization.zero_grad()
            # 这里也改了
            accelerator.backward(loss)
            optimization.step()
        print(f"Epoch {epoch} loss: {loss.item()}")

    accelerator.save(model.state_dict(), "model.pth")
