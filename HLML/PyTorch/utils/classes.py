import torch
import torch.nn as nn
import numpy as np

class Hyperparameter(nn.Parameter):
    def __new__(cls, data=None, search=True, initializer=None):
        """
        search : Include this hyperparameter in a hyperparameter search
        initializer : The initializer to use for initialization and
                      reinitialization
        """
        return nn.Parameter.__new__(cls, torch.FloatTensor([data]), False)

    def __init__(self, param, name, search=True, initializer=None):
        nn.Parameter.__init__(self)

        self.param_name = name
        self.search = search
        self.initializer = initializer

        self.initialize()

    def __deepcopy__(self, memo):
        result = super(nn.Parameter, self).__deepcopy__(memo)
        result.search = self.search

    def __repr__(self):
        return "Hyperparameter " + self.param_name + ": " + str(self.item())

    def initialize(self):
        if(self.initializer is not None):
            self.initializer(self)

class AMU(nn.Module):
    def __init__(self, input_units, block_units, num_blocks, output_units,
                 attn_dropout=0.1):
        """
        The attention memory unit, used to capture long term dependencies
        inputs should be shape (batch size, time length, input units)

        input_dim : The number of units in the input layer
        block_units : The number of units per memory block
        num_block : The number of blocks (rows) of memory
        output_units : The number of output units
        """
        nn.Module.__init__(self)

        self.input_units = input_units
        self.block_units = block_units
        self.num_blocks = num_blocks
        self.output_units = output_units

        self.memory = torch.zeros(num_blocks, block_units)

        self.query = nn.Linear(input_units, num_blocks + 1)
        self.key = nn.Linear(input_units, num_blocks + 1)
        self.mem_attn = nn.Linear(input_units, num_blocks + 1)
        self.mem_entry = nn.Sequential(
                            nn.Linear(input_units, block_units),
                            nn.ReLU()
                        )

        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(attn_dropout)

        self.linear = nn.Linear(block_units * (num_blocks + 1), output_units)

    def forward(self, inp, memory=None):
        """
        Computes the output of the unit and updates the memory. Uses internal
        memory is one is not given.
        """
        query = self.query(inp)
        key = self.key(inp)
        mem_attn = self.mem_attn(inp)
        mem_entry = self.mem_entry(inp).unsqueeze(2)

        attn = (torch.bmm(query, key.transpose(1, 2))
                / np.power(self.num_blocks + 1, 0.5))
        attn = torch.bmm(attn, mem_attn)
        softmax_weights = self.softmax(attn)
        attn = self.dropout(softmax_weights)
        attn = attn[:, -1, :].unsqueeze(-1)

        # Copy the memory for each batch
        if(memory is None):
            memory = self.memory.unsqueeze(0).repeat(inp.shape[0],
                                                     *[1 for _
                                                        in self.memory.shape])

        in_mems = []
        out_mem = []

        for i in range(inp.shape[1]):
            memory = torch.cat([memory, mem_entry[:, i, :, :]], dim=1)
            in_mems.append(memory)
            # Drop the lowest softmax weight
            weakest_memory = torch.argmin(softmax_weights[:, i, :], dim=-1)
            memory[:, weakest_memory, :] = memory[:, -1, :]
            memory = memory[:, :-1, :]

            out_mem.append(memory)

        # Set the new memory to the last timestep of the first batch
        out_mem = torch.stack(out_mem)[-1, :, :, :]
        self.memory = out_mem[0, :, :].detach()

        in_mems = torch.stack(in_mems)

        output = attn * in_mems
        output = output.view(inp.shape[0], -1, ((self.num_blocks + 1)
                                                 * self.block_units))
        output = self.linear(output)

        return output, out_mem

    def reset_memory(self):
        self.memory = torch.zeroes(block_units, num_blocks)

    def set_memory(self, mem):
        self.memory = mem