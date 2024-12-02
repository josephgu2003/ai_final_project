import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--base_dir', type=str, default='./log')
    parser.add_argument('--exp_name', type=str, default='an_experiment')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--dropout_samples', type=int, default=8)
    parser.add_argument('--vis_interval', type=int, action="append", default=[0, 19, 49, 99, 149])

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
