import argparse
import sys
import torch
import rdkit
import pandas as pd
from tqdm.auto import tqdm
from moses.models_storage import ModelsStorage
from moses.script_utils import add_sample_args, set_seed, record_timing
from moses.device_utils import build_device, warmup_device, DeviceTimer

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models sampler script', description='available models')
    for model in MODELS.get_model_names():
        add_sample_args(subparsers.add_parser(model))
    return parser


def main(model, config):
    model_name = model
    set_seed(config.seed)
    # Builds a torch.device for cpu / cuda:<n> / sdaa:<n> and selects the card.
    device = build_device(config.device)
    config.device = str(device)

    # weights_only=False: config/vocab are pickled python objects (trusted).
    model_config = torch.load(config.config_load, weights_only=False)
    model_vocab = torch.load(config.vocab_load, weights_only=False)
    model_state = torch.load(config.model_load, weights_only=False)

    model = MODELS.get_model_class(model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # Pay the one-off SDAA/CUDA start-up cost before the clock starts so the
    # reported sampling time only covers steady-state generation.
    warmup_device(device)

    with DeviceTimer(device, 'sample') as timer:
        samples = []
        n = config.n_samples
        with tqdm(total=config.n_samples, desc='Generating samples') as T:
            while n > 0:
                current_samples = model.sample(
                    min(n, config.n_batch), config.max_len
                )
                samples.extend(current_samples)

                n -= len(current_samples)
                T.update(len(current_samples))

    record_timing(getattr(config, 'time_path', None), 'sample', timer.elapsed,
                  model=model_name, n_samples=config.n_samples,
                  device=config.device)

    samples = pd.DataFrame(samples, columns=['SMILES'])
    samples.to_csv(config.gen_save, index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
