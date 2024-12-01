from torch import nn
from spikingjelly.clock_driven import neuron, surrogate


class SNN(nn.Module):
    def __init__(self, input_dim, times):
        """
        参数
        input_dim: 输入图像的宽高
        times: 用于模拟脉冲神经网络的时间维度
        """
        super(SNN, self).__init__()
        self.times = times

        self.fc1 = nn.Linear(input_dim * input_dim, 800)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

        self.fc2 = nn.Linear(800, 10)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        # 定义全连接层和神经元层 使用Sigmoid替代梯度

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # 将输入的2D图像展平为1D向量

        spike_sum = 0
        # 累计所有时间步的脉冲结果

        for t in range(self.times):
            mem1 = self.fc1(x)
            spike1 = self.lif1(mem1)
            mem2 = self.fc2(spike1)
            spike2 = self.lif2(mem2)
            # 计算脉冲发放 如果膜电位超过阈值  发放脉冲并重置
            # 如果没输入或者未超过阈值则会按一个固定因子衰减

            spike_sum += spike2
        return spike_sum / self.times
        # 计算平均发放率
