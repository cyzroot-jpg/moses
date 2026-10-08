import argparse
import json
import random
import re
import numpy as np
import pandas as pd
import torch


def torch_device(arg):
    """argparse validator accepting cpu / cuda:<n> / sdaa:<n> / auto."""
    if re.match('^(cuda(:[0-9]+)?|sdaa(:[0-9]+)?|cpu|auto)$', arg) is None:
        raise argparse.ArgumentTypeError(
            'Wrong device format: {}'.format(arg)
        )

    if arg in ('cpu', 'auto'):
        return arg

    from moses.device_utils import sdaa_available
    splited_device = arg.split(':')
    n = int(splited_device[1]) if len(splited_device) > 1 else 0

    if splited_device[0] == 'sdaa':
        if not sdaa_available():
            raise argparse.ArgumentTypeError(
                'Wrong device: {} is not available'.format(arg)
            )
    else:  # cuda
        if (not torch.cuda.is_available()) or \
                (n >= torch.cuda.device_count()):
            raise argparse.ArgumentTypeError(
                'Wrong device: {} is not available'.format(arg)
            )

    return arg


def add_common_arg(parser):
    # Base
    parser.add_argument('--device',
                        type=torch_device, default=None,
                        help='Device to run: "cpu", "cuda:<n>" or "sdaa:<n>" '
                             '(default: auto-detect sdaa > cuda > cpu)')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Seed')

    return parser


def add_train_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--train_load',
                            type=str,
                            help='Input data in csv format to train')
    common_arg.add_argument('--val_load', type=str,
                            help="Input data in csv format to validation")
    common_arg.add_argument('--model_save',
                            type=str, required=True, default='model.pt',
                            help='Where to save the model')
    common_arg.add_argument('--save_frequency',
                            type=int, default=20,
                            help='How often to save the model')
    common_arg.add_argument('--log_file',
                            type=str, required=False,
                            help='Where to save the log')
    common_arg.add_argument('--config_save',
                            type=str, required=True,
                            help='Where to save the config')
    common_arg.add_argument('--vocab_save',
                            type=str,
                            help='Where to save the vocab')
    common_arg.add_argument('--vocab_load',
                            type=str,
                            help='Where to load the vocab; '
                                 'otherwise it will be evaluated')

    return parser


def add_sample_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")
    common_arg.add_argument("--time_path",
                            type=str, default=None,
                            help="Optional file to append JSON timing records")

    return parser


# def read_smiles_csv(path):
#     return pd.read_csv(path,
#                        usecols=['SMILES'],
#                        squeeze=True).astype(str).tolist()


def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'])['SMILES'].astype(str).tolist()


def record_timing(time_path, stage, seconds, **extra):
    """Report one timing measurement.

    Always prints a single line to stdout. When ``time_path`` is given, also
    appends a JSON object (one per line) so several stages/models can be
    aggregated later. Extra keyword arguments are stored alongside the value.
    """
    suffix = ''.join(' {}={}'.format(k, v) for k, v in extra.items())
    print('[time] {}: {:.3f} s{}'.format(stage, seconds, suffix), flush=True)
    if time_path:
        record = {'stage': stage, 'seconds': seconds}
        record.update(extra)
        with open(time_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


def set_seed(seed):
    # Backend agnostic: seeds python/numpy and whichever of cuda/sdaa exists.
    from moses.device_utils import seed_everything
    seed_everything(seed)
