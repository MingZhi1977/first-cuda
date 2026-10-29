import time
import torch


def run_matrix_multiplication(size: int = 4096) -> None:
    print("=" * 60)
    print("⚡ CUDA 矩陣運算展示 (Matrix Multiplication Benchmark)")
    print("=" * 60)

    # 1. 檢查 CUDA 支援情況
    if not torch.cuda.is_available():
        print("❌ 未偵測到可用的 CUDA 裝置，將以 CPU 運行。")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        print(f"✅ 成功啟用 CUDA！")
        print(f"   - GPU 裝置名稱: {gpu_name}")
        print(f"   - 顯存大小: {total_mem:.2f} GB")
        print(f"   - PyTorch CUDA 版本: {cuda_ver}")

    print("-" * 60)
    print(f"📊 建立測試矩陣: {size} x {size} (單精度 float32)")

    # 建立隨機矩陣
    a_cpu = torch.randn(size, size, dtype=torch.float32)
    b_cpu = torch.randn(size, size, dtype=torch.float32)

    # 2. CPU 運算測試
    print("⏳ [CPU] 正在執行矩陣相乘 (A @ B)...")
    start_cpu = time.perf_counter()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    time_cpu = time.perf_counter() - start_cpu
    print(f"   CPU 耗時: {time_cpu * 1000:.2f} ms")

    # 3. GPU (CUDA) 運算測試
    if torch.cuda.is_available():
        print("⏳ [GPU] 將矩陣傳輸至 GPU 記憶體並執行 CUDA 矩陣相乘...")
        # 移至 GPU
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)

        # 預熱 (Warm-up)
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()

        # 計時 (使用 CUDA Event 精確測量)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        end_event.record()
        torch.cuda.synchronize()

        time_gpu_ms = start_event.elapsed_time(end_event)
        time_gpu = time_gpu_ms / 1000.0
        print(f"   GPU 耗時: {time_gpu_ms:.2f} ms")

        # 驗證計算正確性
        is_close = torch.allclose(c_cpu, c_gpu.cpu(), rtol=1e-3, atol=1e-3)
        print(f"🔍 結果正確性驗證 (CPU 與 GPU 計算結果一致): {'通過' if is_close else '有誤差'}")

        # 加速比
        speedup = time_cpu / time_gpu if time_gpu > 0 else float("inf")
        print("-" * 60)
        print(f"🚀 CUDA GPU 加速倍數: {speedup:.2f}x ！！")
        print("=" * 60)


def main() -> None:
    run_matrix_multiplication()


if __name__ == "__main__":
    main()
