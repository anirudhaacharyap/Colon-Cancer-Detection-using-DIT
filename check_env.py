import sys

def check_environment():
    print("========== Environment Health Check ==========\n")
    
    # 1. Check Python Version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # 2. Check PyTorch & CUDA
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA Available: YES")
            print(f"GPU Device Count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
            print("Status: SUCCESS! The pipeline will run on GPU.")
        else:
            print("CUDA Available: NO")
            print("Status: WARNING! PyTorch cannot find a GPU. The pipeline will run on CPU, which is very slow.")
    except ImportError:
        print("PyTorch is NOT installed!")
        
    print("\n--- Checking Dependencies ---")
    dependencies = ['timm', 'sklearn', 'matplotlib', 'seaborn', 'pandas', 'tqdm']
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"[{dep}] Installed")
        except ImportError:
            print(f"[{dep}] MISSING")
            missing.append(dep)
            
    if len(missing) == 0:
        print("\nAll dependencies are successfully installed!")
    else:
        print(f"\nMissing Dependencies: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")

if __name__ == "__main__":
    check_environment()
