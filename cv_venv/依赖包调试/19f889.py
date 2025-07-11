import sys
import time
import numpy as np
import cv2
import torch
import torch_directml
import torchvision.transforms as transforms
from PIL import Image

def validate_core_packages():
    """验证核心包功能是否正常"""
    results = {}
    
    # 1. 基础环境验证
    results['Python'] = sys.version.split()[0]
    results['Platform'] = sys.platform
    
    # 2. NumPy验证
    try:
        arr = np.random.rand(3, 3)
        results['NumPy'] = f"OK | Shape: {arr.shape} | Sum: {arr.sum():.2f}"
    except Exception as e:
        results['NumPy'] = f"FAIL: {str(e)}"
    
    # 3. OpenCV验证
    try:
        # 生成虚拟图像
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results['OpenCV'] = f"OK | Gray shape: {gray.shape}"
    except Exception as e:
        results['OpenCV'] = f"FAIL: {str(e)}"
    
    # 4. Pillow验证
    try:
        pil_img = Image.fromarray(img)
        results['Pillow'] = f"OK | Mode: {pil_img.mode} | Size: {pil_img.size}"
    except Exception as e:
        results['Pillow'] = f"FAIL: {str(e)}"
    
    # 5. PyTorch+DirectML验证
    try:
        if torch_directml.is_available():
            device = torch_directml.device()
            # 直接使用设备对象
            results['DML_Device'] = f"OK | Device: {device}"
            
            # 测试实际计算性能
            start_time = time.time()
            a = torch.randn(5000, 5000, device=device)
            b = torch.randn(5000, 5000, device=device)
            c = torch.matmul(a, b)
            compute_time = time.time() - start_time
            results['DML_Performance'] = f"{compute_time:.2f}s (5000x5000 matmul)"
        else:
            results['DML_Device'] = "FAIL: DirectML not available"
    except Exception as e:
        results['PyTorch'] = f"FAIL: {str(e)}"
    
    # 6. Torchvision验证
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        tensor_img = transform(pil_img)
        results['Torchvision'] = f"OK | Tensor shape: {tensor_img.shape}"
    except Exception as e:
        results['Torchvision'] = f"FAIL: {str(e)}"
    
    # 打印结果
    print("\n" + "="*60)
    print("核心包功能验证报告 (兼容版)")
    print("="*60)
    for pkg, status in results.items():
        print(f"{pkg:>18}: {status}")
    print("="*60)

if __name__ == "__main__":
    validate_core_packages()
