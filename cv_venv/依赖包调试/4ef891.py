import torch_directml

# 检查DirectML支持
print(f"DirectML可用: {torch_directml.is_available()}")  # 应输出True

# 测试Intel显卡调用
device = torch_directml.device()
tensor = torch.tensor([1, 2, 3]).to(device)
print(tensor * 2)  # 应输出tensor([2,4,6], device='privateuseone:0')