#!/usr/bin/env python
"""AAE inference verification and cross-device benchmark.

Runs the MOSES AAE decoder on one or more backends (``cpu``, ``cuda:<n>``,
``sdaa:<n>``) and reports:

* numerical parity of the encoder / decoder against CPU (same weights, eval
  mode, deterministic ops), and
* end-to-end sampling cost: wall time, throughput, validity and uniqueness.

Nothing here is SDAA specific: the *same* script runs on a CUDA-only box.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import rdkit
from rdkit import Chem

from moses.aae import AAE
from moses.device_utils import build_device, resolve_device
from moses.script_utils import set_seed

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def load_case(model_load, config_load, vocab_load):
    config = torch.load(config_load, map_location='cpu', weights_only=False)
    vocab = torch.load(vocab_load, map_location='cpu', weights_only=False)
    state = torch.load(model_load, map_location='cpu', weights_only=False)
    return config, vocab, state


def make_model(vocab, config, state, device):
    model = AAE(vocab, config)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def _batch_encoder(model, strings, device):
    tensors = [model.string2tensor(s, device='cpu') for s in strings]
    # pack_padded_sequence needs decreasing lengths.
    order = sorted(range(len(tensors)), key=lambda i: tensors[i].numel(),
                   reverse=True)
    tensors = [tensors[i] for i in order]
    lengths = torch.tensor([t.numel() - 2 for t in tensors], dtype=torch.long)
    inputs = pad_sequence(tensors, batch_first=True,
                          padding_value=model.vocabulary.pad).to(device)
    with torch.no_grad():
        return model.encoder_forward(inputs, lengths).cpu()


def _batch_decoder_step(model, device, n_batch):
    """One decoder step with fixed latent states and tokens (parity probe)."""
    latent = torch.linspace(-1.0, 1.0, n_batch * model.latent_size,
                            dtype=torch.float).view(n_batch, model.latent_size)
    latent = latent.to(device)
    prevs = torch.zeros(n_batch, 1, dtype=torch.long, device=device)
    prevs.fill_(model.vocabulary.bos)
    one_lens = torch.ones(n_batch, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model.decoder(prevs, one_lens, latent,
                                     is_latent_states=True)
    return logits.cpu()


def parity_check(ref_model, other_model, strings, device, n_batch=32):
    ref_latent = _batch_encoder(ref_model, strings[:n_batch], 'cpu')
    oth_latent = _batch_encoder(other_model, strings[:n_batch], device)

    ref_logits = _batch_decoder_step(ref_model, 'cpu', n_batch)
    oth_logits = _batch_decoder_step(other_model, device, n_batch)

    enc_diff = (ref_latent - oth_latent).abs().max().item()
    dec_diff = (ref_logits - oth_logits).abs().max().item()
    return {
        'encoder_max_abs_diff': enc_diff,
        'decoder_max_abs_diff': dec_diff,
        'encoder_allclose_1e-3': bool(
            torch.allclose(ref_latent, oth_latent, atol=1e-3, rtol=1e-3)),
        'decoder_allclose_1e-3': bool(
            torch.allclose(ref_logits, oth_logits, atol=1e-3, rtol=1e-3)),
    }


def _valid_unique(samples):
    valid, seen = 0, set()
    for s in samples:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            valid += 1
            seen.add(Chem.MolToSmiles(m))
    n = len(samples)
    return {
        'n_samples': n,
        'valid': valid,
        'validity': valid / n if n else 0.0,
        'unique_valid': len(seen),
        'unique_valid_ratio': len(seen) / n if n else 0.0,
    }


def benchmark(model, device, n_samples, n_batch, max_len, seed,
              save_csv=None):
    set_seed(seed)
    # warmup (excluded from timing)
    model.sample(min(n_batch, 8), max_len)

    samples = []
    n = n_samples
    t0 = time.perf_counter()
    while n > 0:
        cur = model.sample(min(n, n_batch), max_len)
        samples.extend(cur)
        n -= len(cur)
    elapsed = time.perf_counter() - t0

    stats = _valid_unique(samples)
    stats.update({
        'device': str(device),
        'wall_time_s': round(elapsed, 3),
        'throughput_smiles_per_s': round(len(samples) / elapsed, 2)
        if elapsed > 0 else None,
        'n_batch': n_batch,
        'max_len': max_len,
        'seed': seed,
    })
    if save_csv is not None:
        import pandas as pd
        pd.DataFrame(samples, columns=['SMILES']).to_csv(save_csv, index=False)
        stats['gen_csv'] = save_csv
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model_load', required=True)
    ap.add_argument('--config_load', required=True)
    ap.add_argument('--vocab_load', required=True)
    ap.add_argument('--devices', nargs='+', default=['cpu'],
                    help='e.g. "cpu sdaa:0" or "cpu cuda:0"')
    ap.add_argument('--n_samples', type=int, default=1024)
    ap.add_argument('--n_batch', type=int, default=512)
    ap.add_argument('--max_len', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default=None,
                    help='Directory for generated csv + results json')
    ap.add_argument('--skip_parity', action='store_true')
    args = ap.parse_args()

    devices = [resolve_device(d) for d in args.devices]
    outdir = args.outdir or '.'
    os.makedirs(outdir, exist_ok=True)

    config, vocab, state = load_case(
        args.model_load, args.config_load, args.vocab_load)

    results = {'args': vars(args), 'devices': devices, 'benchmarks': {},
               'parity': {}}

    # CPU reference (always built once)
    cpu_dev = torch.device('cpu')
    cpu_model = make_model(vocab, config, state, cpu_dev)

    strings = None
    if not args.skip_parity:
        from moses.dataset import get_dataset
        strings = get_dataset('test')[:512]

    for dev in devices:
        print('=' * 70)
        print('device:', dev)
        device = build_device(dev)
        model = make_model(vocab, config, state, device)

        if not args.skip_parity:
            if dev.startswith('cpu'):
                results['parity'][dev] = {
                    'encoder_max_abs_diff': 0.0,
                    'decoder_max_abs_diff': 0.0,
                    'encoder_allclose_1e-3': True,
                    'decoder_allclose_1e-3': True,
                    'note': 'reference',
                }
            else:
                par = parity_check(cpu_model, model, strings, device)
                results['parity'][dev] = par
                print('parity:', par)

        tag = dev.replace(':', '_')
        csv_path = os.path.join(outdir, 'aae_gen_{}.csv'.format(tag))
        bench = benchmark(model, device, args.n_samples, args.n_batch,
                          args.max_len, args.seed, save_csv=csv_path)
        results['benchmarks'][dev] = bench
        print('benchmark:', json.dumps(bench, ensure_ascii=False))

        del model
        if device.type != 'cpu':
            try:
                getattr(torch, device.type if device.type != 'cuda'
                        else 'cuda').empty_cache()
            except Exception:
                pass

    json_path = os.path.join(outdir, 'aae_device_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('=' * 70)
    print('saved:', json_path)
    return results


if __name__ == '__main__':
    main()