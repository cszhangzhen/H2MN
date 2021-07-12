import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run the code.")

    parser.add_argument('--seed',
                        type=int,
                        default=2020,
                        help='Random seed')

    parser.add_argument("--dataset",
                        nargs="?",
                        default="AIDS700nef",
                        help="Dataset name. reg: AIDS700nef/LINUX/IMDBMulti and cls: ffmpeg_min3/20/50, openssl_min3/20/50")

    parser.add_argument("--epochs",
                        type=int,
                        default=10000,
                        help="Number of training epochs. Default is 10000.")

    parser.add_argument("--nhid",
                        type=int,
                        default=100,
                        help="Hidden dimension in convolution. Default is 64.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=512,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.0,
                        help="Dropout probability. Default is 0.0.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.0001,
                        help="Learning rate. Default is 0.0001.")

    parser.add_argument("--ratio1",
                        type=float,
                        default=1.0,
                        help="Pooling rate. Default is 0.8.")

    parser.add_argument("--ratio2",
                        type=float,
                        default=1.0,
                        help="Pooling rate. Default is 0.8.")

    parser.add_argument("--ratio3",
                        type=float,
                        default=0.8,
                        help="Pooling rate. Default is 0.8.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5e-4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Specify cuda devices')
    
    parser.add_argument('--mode',
                        type=str,
                        default='RW',
                        help='Specify hypergraph construction mode NEighbor(NE)/RandomWalk(RW)')

    parser.add_argument('--patience',
                        type=int,
                        default=100,
                        help='Patience for early stopping')
    
    parser.add_argument('--k',
                        type=int,
                        default=5,
                        help='Hyperparameter for construction hyperedge')

    return parser.parse_args()
