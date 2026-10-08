#!/usr/bin/env python
"""Zero-intrusion FCD debugger: does NOT modify fcd_torch/moses."""
import argparse
import numpy as np
import pandas as pd


def read_gen(path):
    # 读取生成的 SMILES（等价于 moses.script_utils.read_smiles_csv）
    smiles = pd.read_csv(path, usecols=['SMILES'])['SMILES'].astype(str).tolist()
    print(f"[DBG][read] {path}: rows={len(smiles)} first={smiles[0][:60]!r}")
    return smiles


def canon_valid(smiles):
    # 复现 moses remove_invalid：过滤非法 SMILES 并规范化，统计有效/唯一数
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    valid = []
    invalid = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            invalid += 1
        else:
            valid.append(Chem.MolToSmiles(m))
    uniq = sorted(set(valid))
    print(f"[DBG][valid] total={len(smiles)} invalid={invalid} "
          f"valid={len(valid)} valid_unique={len(uniq)}")
    return uniq


def unknown_rate(smiles):
    # 复现 fcd_torch.utils.get_one_hot 的分词，统计不在词表里的字符比例
    #（词表 utils.py:19-24 不含 '*'，用于验证未知 token 是否导致特征退化）
    import fcd_torch.utils as U
    vocab = set(getattr(U, '__vocab'))
    two = getattr(U, '__two_letters')
    tot = unk = 0
    for s in smiles:
        x = s + '.'
        src = 0
        while x[src] != '.':
            if src + 1 < len(x) and x[src + 1] in two:
                sym, src = x[src:src + 2], src + 2
            else:
                sym, src = x[src], src + 1
            tot += 1
            if sym not in vocab:
                unk += 1
    print(f"[DBG][token] total_chars={tot} unknown={unk} "
          f"unknown_rate={unk / max(tot, 1):.3f}  (vocab has no '*')")


def describe(tag, S):
    # 核心：打印协方差的秩/条件数/特征值（判断是否秩亏）
    ev = np.linalg.eigvalsh(S)
    print(f"[DBG][cov] {tag}: shape={S.shape} rank={np.linalg.matrix_rank(S)} "
          f"cond={np.linalg.cond(S):.3e} min_eig={ev.min():.3e} "
          f"max_eig={ev.max():.3e} neg={int((ev < 0).sum())}")


def dbg_frechet(mu1, sigma1, mu2, sigma2, tag1="sigma_test(ref)",
                tag2="sigma_gen", eps=1e-6):
    # 复现 fcd_torch.utils.calculate_frechet_distance (utils.py:142-173)，并打印中间量
    from scipy import linalg
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    describe(tag1, sigma1)
    describe(tag2, sigma2)
    P = sigma1.dot(sigma2)
    pe = np.linalg.eigvals(P)
    print(f"[DBG][prod] eig(sigma1@sigma2): min_real={pe.real.min():.3e} "
          f"neg={int((pe.real < 0).sum())}")
    covmean, _ = linalg.sqrtm(P, disp=False)
    imag = float(np.max(np.abs(covmean.imag))) if np.iscomplexobj(covmean) else 0.0
    print(f"[DBG][sqrtm] imag_max={imag:.6e}  (library raises if diag imag > 1e-3)")
    print(f"[DBG][fcd] ||mu1-mu2||^2={float(diff.dot(diff)):.4e}")
    print("[DBG][fcd] -> " + ("WOULD RAISE 'Imaginary component'"
          if (np.iscomplexobj(covmean) and not np.allclose(
              np.diagonal(covmean).imag, 0, atol=1e-3)) else "sqrtm OK, no raise"))


def load_ref(ptest_path):
    # 加载参照统计量：显式 .npz 或 moses 内置默认 test 统计量
    if ptest_path:
        return np.load(ptest_path, allow_pickle=True)['stats'].item()['FCD']
    from moses.dataset import get_statistics
    print("[DBG][ref] using built-in default test statistics")
    return get_statistics('test')['FCD']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_path', required=True)
    ap.add_argument('--ptest_path', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--n_jobs', type=int, default=0)   # 0/1 避免多进程吞日志
    args = ap.parse_args()

    gen = read_gen(args.gen_path)
    valid_unique = canon_valid(gen)
    if len(valid_unique) < 2:
        print("[DBG] too few valid molecules, stop")
        return
    unknown_rate(valid_unique)

    from fcd_torch import FCD
    fcd = FCD(device=args.device, n_jobs=args.n_jobs, batch_size=args.batch_size)
    feat = fcd.get_predictions(valid_unique)   # (N, 512) ChemNet 特征
    print(f"[DBG][pred] feat shape={feat.shape} nan={int(np.isnan(feat).sum())} "
          f"std={feat.std():.4e} "
          f"uniq_rows={np.unique(feat.round(6), axis=0).shape[0]}")
    mu_gen, sigma_gen = feat.mean(0), np.cov(feat.T)

    ref = load_ref(args.ptest_path)
    mu_ref, sigma_ref = np.asarray(ref['mu']), np.asarray(ref['sigma'])
    print(f"[DBG][ref] mu={mu_ref.shape} sigma={sigma_ref.shape}")

    # 与库调用顺序一致：calculate_frechet_distance(ref, gen) 见 fcd.py:91-93
    dbg_frechet(mu_ref, sigma_ref, mu_gen, sigma_gen)

    from fcd_torch.utils import calculate_frechet_distance
    try:
        d = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
        print(f"[DBG][real] FCD={d}")
    except ValueError as e:
        print(f"[DBG][real] raised: {e}")


if __name__ == '__main__':
    main()
