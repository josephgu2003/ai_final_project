import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=16)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()