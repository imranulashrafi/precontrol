import torch


def eat_gpu_memory(gigabytes):
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot allocate GPU memory.")
        return

    try:
        bytes_to_allocate = int(gigabytes * 1024**3)
        num_elements = bytes_to_allocate // 4

        print(f"Attempting to allocate {gigabytes} GB of GPU memory...")

        dummy_tensor = torch.zeros(num_elements, dtype=torch.float32, device="cuda:0")
        dummy_tensor2 = torch.zeros(num_elements, dtype=torch.float32, device="cuda:1")

        print(f"Successfully allocated {gigabytes} GB of GPU memory.")
        print("Press Ctrl+C to release the memory.")

        while True:
            torch.cuda.synchronize()
            pass

    except torch.cuda.OutOfMemoryError:
        print(f"Failed to allocate {gigabytes} GB of GPU memory. Out of memory.")
    except KeyboardInterrupt:
        print("\nReleasing GPU memory...")
        del dummy_tensor
        torch.cuda.empty_cache()
        print("GPU memory released.")


if __name__ == "__main__":
    memory_to_eat_gb = 40

    eat_gpu_memory(memory_to_eat_gb)
