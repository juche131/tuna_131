'''
import torch
print(torch.dml.is_available())           # Windows设备看这里 → 应返回True
'''

import torch_directml
print(f"DirectML可用：{torch_directml.is_available()}")

device = torch_directml.device()
print(f"设备：{device}")

tensor = torch.tensor([1,2,3]).to(device)
print(tensor + 10)
