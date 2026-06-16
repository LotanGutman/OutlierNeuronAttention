import torch
from mamba_ssm import Mamba2

dim = 256
model = Mamba2(
    d_model=dim, 
    d_state=64,  
    d_conv=4,    
    expand=2,    
).to("cuda")

x = torch.randn(2, 64, dim).to("cuda")
y = model(x)
assert y.shape == x.shape
print("Success! Shape match confirmed.")
