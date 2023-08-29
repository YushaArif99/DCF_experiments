import ivy
import control_flow_experimental.ivy_fx.fx as fx
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
import torch 

import pytest

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Define two stateful linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Apply the linear layers and a relu activation
        x = F.relu(self.linear1(x))
        # Apply the softmax layer
        x = F.softmax(self.linear2(x), dim=1)
        return x
  
@pytest.mark.parametrize("inp_size, hid_size, num_class", [(10,5,3), (16,32,10)])
def test_complile_linear( inp_size, hid_size, num_class):
    
    model = FeedforwardNN(inp_size, hid_size, num_class)
    ivy.set_torch_backend()
    _, ivy_graph = fx.symbolic_trace(model, args=[], stateful=model, generate_source=True) 
    x = torch.randn(1, inp_size)
    orig_ret = model(x).detach().numpy()
    graph_ret = ivy_graph(x).detach().numpy()
    assert np.allclose(orig_ret, graph_ret)

    

