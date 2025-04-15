import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the training script.")
    parser.add_argument('--dataset', type=str, default='credit', help='Dataset name')
    parser.add_argument('--model', type=str, default='logistic_regression', help='Model name')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='Loss function type')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for scaling gradients')
    parser.add_argument('--print_every', type=int, default=1, help='Print interval for logging')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs and models')
    parser.add_argument('--wandb_project', type=str, default='my_project', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name (optional)')
    parser.add_argument('--wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--log', action='store_true', help='Log to file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cpu or cuda)')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for training')

    args = parser.parse_args()
    
    return args