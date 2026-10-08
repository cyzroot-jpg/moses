import argparse
import numpy as np
import rdkit

from moses.metrics.metrics import get_all_metrics
from moses.script_utils import read_smiles_csv, record_timing
from moses.device_utils import resolve_device, warmup_device, DeviceTimer

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def main(config, print_metrics=True):
    # Normalise cpu / cuda:<n> / sdaa:<n> for the metric backend (ChemNet/FCD).
    config.device = resolve_device(config.device)
    test = None
    test_scaffolds = None
    ptest = None
    ptest_scaffolds = None
    train = None
    if config.test_path:
        test = read_smiles_csv(config.test_path)
    if config.test_scaffolds_path is not None:
        test_scaffolds = read_smiles_csv(config.test_scaffolds_path)
    if config.train_path is not None:
        train = read_smiles_csv(config.train_path)
    if config.ptest_path is not None:
        ptest = np.load(
            config.ptest_path,
            allow_pickle=True)['stats'].item()
    if config.ptest_scaffolds_path is not None:
        ptest_scaffolds = np.load(
            config.ptest_scaffolds_path,
            allow_pickle=True)['stats'].item()
    gen = read_smiles_csv(config.gen_path)

    # Exclude one-off SDAA/CUDA start-up (context creation, module load) from
    # the measured evaluation time.
    warmup_device(config.device)

    with DeviceTimer(config.device, 'eval') as timer:
        metrics = get_all_metrics(gen=gen, k=config.ks, n_jobs=config.n_jobs,
                                  device=config.device,
                                  test_scaffolds=test_scaffolds,
                                  ptest=ptest, ptest_scaffolds=ptest_scaffolds,
                                  test=test, train=train)

    record_timing(getattr(config, 'time_path', None), 'eval', timer.elapsed,
                  n_gen=len(gen), device=config.device)

    if print_metrics:
        for name, value in metrics.items():
            print('{},{}'.format(name, value))
    else:
        return metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',
                        type=str, required=False,
                        help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path',
                        type=str, required=False,
                        help='Path to scaffold test molecules csv')
    parser.add_argument('--train_path',
                        type=str, required=False,
                        help='Path to train molecules csv')
    parser.add_argument('--ptest_path',
                        type=str, required=False,
                        help='Path to precalculated test npz')
    parser.add_argument('--ptest_scaffolds_path',
                        type=str, required=False,
                        help='Path to precalculated scaffold test npz')
    parser.add_argument('--gen_path',
                        type=str, required=True,
                        help='Path to generated molecules csv')
    parser.add_argument('--ks', '--unique_k',
                        nargs='+', default=[1000, 10000],
                        type=int,
                        help='Number of molecules to calculate uniqueness at.'
                             'Multiple values are possible. Defaults to '
                             '--unique_k 1000 10000')
    parser.add_argument('--n_jobs',
                        type=int, default=1,
                        help='Number of processes to run metrics')
    parser.add_argument('--device',
                        type=str, default='sdaa',
                        help='Metric device: `cpu`, `cuda:n` or `sdaa:n`')
    parser.add_argument('--time_path',
                        type=str, default=None,
                        help='Optional file to append JSON timing records')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
