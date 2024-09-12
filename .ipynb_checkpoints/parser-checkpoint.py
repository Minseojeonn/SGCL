import argparse

def parsing():
    parser = argparse.ArgumentParser(description='Parser example')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--dataset_name', type=str, default='bonanza', help='Dataset name')
    parser.add_argument('--input_dim', type=int, default=16, help='Input dimension')    
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')  
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--tao', type=float, default=0.5, help='Temperature')
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha')
    parser.add_argument('--beta', type=float, default=1e-4, help='Beta')
    parser.add_argument('--mask_ratio', type=float, default=0.1, help='Mask ratio')
    parser.add_argument('--augment', type=str, default='delete', help='Augment')
    args = parser.parse_args()
    return args
