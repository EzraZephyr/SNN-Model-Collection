from torch import nn
from spikingjelly.clock_driven import neuron, surrogate


class SNN(nn.Module):
    def __init__(self, input_dim, times):
        """
        Parameters
        input_dim: Width and height of the input image
        times: Time dimension used to simulate the spiking neural network
        """
        super(SNN, self).__init__()
        self.times = times

        self.fc1 = nn.Linear(input_dim * input_dim, 800)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

        self.fc2 = nn.Linear(800, 10)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        # Define fully connected layers and neuron layers, using Sigmoid as the surrogate gradient

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # Flatten the input 2D image into a 1D vector

        spike_sum = 0
        # Accumulate spike results across all time steps

        for t in range(self.times):
            mem1 = self.fc1(x)
            spike1 = self.lif1(mem1)
            mem2 = self.fc2(spike1)
            spike2 = self.lif2(mem2)
            # Compute spike firing: if the membrane potential exceeds the threshold,
            # fire a spike and reset; otherwise, the potential decays by a fixed factor

            spike_sum += spike2
        return spike_sum / self.times
        # Compute the average firing rate
