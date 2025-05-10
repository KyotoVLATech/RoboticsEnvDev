import torch


def check_pytorch_cuda_cudnn() -> None:
    """
    PyTorch, CUDA, cuDNN の利用可能性とバージョン情報を確認する関数
    """
    print("--- PyTorch, CUDA, cuDNN 利用状況確認 ---")

    # PyTorchのバージョンの確認
    print(f"PyTorch Version: {torch.__version__}")

    # CUDAが利用可能かの確認
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # 利用可能なGPUの数の確認
        print(f"Number of GPUs Available: {torch.cuda.device_count()}")

        # 現在のGPUデバイスの確認
        current_device_index = torch.cuda.current_device()
        print(f"Current GPU Device Index: {current_device_index}")
        print(
            f"Current GPU Device Name: {torch.cuda.get_device_name(current_device_index)}"
        )

        # PyTorchがビルドされた際のCUDAバージョンの確認
        print(f"PyTorch built with CUDA Version: {torch.version.cuda}")

        # cuDNNが利用可能かの確認
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"cuDNN Available: {cudnn_available}")

        if cudnn_available:
            # cuDNNのバージョンの確認
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            print("cuDNN is not available or not enabled.")
    else:
        print("CUDA is not available. GPU acceleration is not possible with PyTorch.")

    print("-----------------------------------------")


if __name__ == "__main__":
    check_pytorch_cuda_cudnn()

    # 簡単なテンソル演算をGPUで実行してみるテスト (CUDAが利用可能な場合)
    if torch.cuda.is_available():
        print("\n--- GPUでの簡単なテンソル演算テスト ---")
        try:
            # GPU上にテンソルを作成
            tensor_gpu = torch.randn(3, 3).cuda()
            print(f"Tensor on GPU: \n{tensor_gpu}")
            print(f"Device of the tensor: {tensor_gpu.device}")
            print("GPUでのテンソル演算テスト成功！")
        except Exception as e:
            print(f"GPUでのテンソル演算テスト中にエラーが発生しました: {e}")
        print("-----------------------------------------")
