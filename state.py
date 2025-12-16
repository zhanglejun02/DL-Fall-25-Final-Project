import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # 一个简单的线性层

    def forward(self, x):
        return self.fc(x)

# 初始化模型和优化器
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 进行一次前向传播和反向传播
x = torch.randn(10)  # 随机输入
output = model(x)
loss = output.mean()
loss.backward()

# 更新优化器的参数
optimizer.step()

# 查看优化器中的state
optimizer_state = optimizer.state_dict()

# 打印所有的 state 和 param_groups
print("Optimizer state:")
for p in model.parameters():
    print(optimizer.state[p])