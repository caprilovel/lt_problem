import torch

def get_device(device):
    
    if device.startswith('cuda') or device.isdigit():
        if torch.cuda.is_available():
            try:
                if device == 'cuda':
                    torch.cuda.set_device(0)
                    device = 'cuda'
                elif device.startswith('cuda'):
                    torch.cuda.set_device(device)
                else:
                    torch.cuda.set_device(int(device))
                    device = f'cuda:{device}'
            except Exception:
                print(f"Invalid CUDA device: {device}. Falling back to CPU.")
                device = 'cpu'
        else:
            print("CUDA not available. Falling back to CPU.")
            device = 'cpu'

    
    elif device == 'mps':
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            print("MPS not available. Falling back to CPU.")
            device = 'cpu'

    
    elif device == 'cpu':
        device = 'cpu'
    else:
        print(f"Invalid device: {device}. Falling back to CPU.")
        device = 'cpu'

    print(f"Using device: {device}")
    return device
