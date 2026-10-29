# First CUDA (Python + CUDA 矩陣運算範例)

本專案使用 PyTorch 與 CUDA 13.x，在 NVIDIA RTX 5060 Ti GPU 上執行高維度矩陣運算（Matrix Multiplication），並提供 CPU vs GPU 效能對比基準測試。

## 執行方式

在專案目錄下，可以直接透過以下任一方式執行：

```bash
# 方式 1: 透過 main.py
uv run main.py

# 方式 2: 透過專案 CLI 指令
uv run first-cuda

# 方式 3: 直接透過虛擬環境 Python 執行
.venv/bin/python main.py
```

## 功能特點

- **自動硬體偵測**：檢查 CUDA 是否可用，並列印 GPU 型號、顯存容量與 CUDA 版本。
- **高維度矩陣相乘**：預設進行 $4096 \times 4096$ 的單精度（float32）矩陣相乘 $C = A \times B$。
- **精準計時**：使用 `torch.cuda.Event` 進行非同步 GPU 精確計時。
- **數值正確性比對**：透過 `torch.allclose` 自動比對 CPU 與 GPU 計算結果。
- **效能加速比**：即時輸出 GPU 相較於 CPU 的加速倍數（約 30x ~ 36x）。
