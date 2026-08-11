#!/usr/bin/env python3
"""
diagnose_gate.py — Check whether the MoE gate and the shock expert are actually used.

Reports, over a spread of test simulations:
  1. Shock indicator calibration (p_s vs the true shock label)
  2. Gate usage per expert, split by shock / non-shock nodes, plus gate entropy
  3. Magnitude of the additive shock residual  p_s * shock_expert(h)
  4. Cp error on shock vs non-shock nodes
  5. Ablation: Cp MAE on shock nodes with and without the shock residual

A healthy MoE has gate entropy well above 0 (max = log num_experts) and
different expert usage between shock and non-shock nodes. Entropy near 0 means
the gate has collapsed onto a single expert and the other experts are dead.

Usage:
  python diagnose_gate.py            # 8 sims
  python diagnose_gate.py 20         # 20 sims
  python diagnose_gate.py 8 --ckpt path/to/model.pt
"""
import os; os.environ['PAPER_NUM_WORKERS'] = '0'
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, CACHE_DIR, MODEL_DIR, RESULT_DIR, MODEL_CONFIG, KNN_K
from src.preprocessing import CFDPreprocessor
from src.models_v2 import AeroSurrogatev2
from src.dataset import SimulationDataset, load_sim_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_sims', nargs='?', type=int, default=8)
    parser.add_argument('--ckpt', default=None)
    args = parser.parse_args()

    device    = torch.device('cpu')
    ckpt_path = Path(args.ckpt) if args.ckpt else MODEL_DIR / 'surrogate_v2_best.pt'

    scaler        = np.load(str(MODEL_DIR / 'scaler_v2.npy'),        allow_pickle=True).item()
    spatial_stats = np.load(str(MODEL_DIR / 'spatial_stats_v2.npy'), allow_pickle=True).item()
    preprocessor  = CFDPreprocessor(spatial_stats=spatial_stats)
    test_weights  = load_sim_weights(DATA_DIR / 'dataset.csv', 'test')
    test_ds = SimulationDataset('test', CACHE_DIR, DATA_DIR,
                                test_weights, preprocessor, scaler, k=KNN_K)

    model = AeroSurrogatev2(MODEL_CONFIG).to(device).eval()
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)

    Y_std, Y_mean   = np.array(scaler['Y_std'], np.float32), np.array(scaler['Y_mean'], np.float32)
    cp_std, cp_mean = float(Y_std[0]), float(Y_mean[0])
    n_experts       = MODEL_CONFIG['moe']['num_experts']

    idxs = np.linspace(0, len(test_ds) - 1, args.n_sims).astype(int)
    P, G, S, SMOOTH, RESID, CPT, CPP = [], [], [], [], [], [], []

    print(f"Checkpoint: {ckpt_path}")
    with torch.no_grad():
        for j, idx in enumerate(idxs):
            b = test_ds[int(idx)]
            h = model.backbone(b['x'], b['edge_index'], b['edge_attr'])
            _, p_s = model.shock_indicator(h)
            tau    = float(model.moe.tau)
            gates  = torch.softmax(
                model.moe.gate(torch.cat([h, p_s], dim=1)) / tau, dim=-1)
            expert_stack = torch.stack([e(h) for e in model.moe.experts], dim=1)
            smooth = (gates.unsqueeze(-1) * expert_stack).sum(dim=1)
            resid  = p_s * model.moe.shock_expert(h)

            P.append(p_s.squeeze(-1).numpy())
            G.append(gates.numpy())
            S.append(b['y_shock'].numpy())
            SMOOTH.append(smooth.squeeze(-1).numpy() * cp_std)
            RESID.append(resid.squeeze(-1).numpy() * cp_std)
            CPT.append(b['y_phys'].numpy()[:, 0])
            CPP.append((smooth + resid).squeeze(-1).numpy() * cp_std + cp_mean)
            print(f'  sim {idx}  ({j+1}/{len(idxs)})', flush=True)

    p_s    = np.concatenate(P)
    gates  = np.concatenate(G)
    shock  = np.concatenate(S).astype(bool)
    smooth = np.concatenate(SMOOTH)
    resid  = np.concatenate(RESID)
    cp_t   = np.concatenate(CPT)
    cp_p   = np.concatenate(CPP)

    tau = float(model.moe.tau)
    print(f'\n{"="*78}')
    print(f'GATE / SHOCK-EXPERT DIAGNOSTIC — {len(p_s):,} nodes, {len(idxs)} sims')
    print(f'{"="*78}')
    print(f'tau = {tau:.4f}   |   true shock nodes: {shock.sum():,} ({100*shock.mean():.1f}%)')

    print(f'\n[1] SHOCK INDICATOR  p_s')
    print(f'  mean p_s | shock=1 : {p_s[shock].mean():.4f}')
    print(f'  mean p_s | shock=0 : {p_s[~shock].mean():.4f}')
    pred_pos = p_s > 0.5
    tp = (pred_pos & shock).sum(); fp = (pred_pos & ~shock).sum(); fn = (~pred_pos & shock).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f'  precision={prec:.4f}  recall={rec:.4f}  F1={2*prec*rec/max(prec+rec,1e-9):.4f}')

    print(f'\n[2] GATE — mean weight per expert')
    print('  ' + f'{"":12}' + ''.join(f'{f"E{i}":>10}' for i in range(n_experts)))
    print('  ' + f'{"shock=1":12}' + ''.join(f'{gates[shock,i].mean():10.4f}'  for i in range(n_experts)))
    print('  ' + f'{"shock=0":12}' + ''.join(f'{gates[~shock,i].mean():10.4f}' for i in range(n_experts)))
    hmax = np.log(n_experts)
    # Two distinct quantities, both needed to judge the MoE:
    #   - load balance: entropy of the MEAN gate. This is what the L_lb term in
    #     the loss maximises. Near hmax = all experts carry a similar share.
    #   - routing sharpness: MEAN of the per-node entropies. Near 0 = each node
    #     commits to a single expert, which is healthy hard routing, NOT collapse.
    # Collapse is load balance near 0; sharp routing with balanced load is fine.
    mean_gate = gates.mean(0)
    h_load    = -(mean_gate * np.log(mean_gate + 1e-12)).sum()
    h_node    = -(gates * np.log(gates + 1e-12)).sum(1).mean()
    print(f'  load balance   (entropy of mean gate) = {h_load:.4f} / {hmax:.4f}'
          f'   -> {100*h_load/hmax:.1f}% of uniform')
    print(f'  routing sharpness (mean per-node entropy) = {h_node:.4f} / {hmax:.4f}'
          f'   (near 0 = each node picks one expert)')
    dead = [i for i in range(n_experts) if gates[:, i].mean() < 0.01]
    print(f'  experts with <1% mean usage: {dead if dead else "none"}')
    # Specialisation: does the gate route shock nodes differently at all?
    tv = 0.5 * np.abs(gates[shock].mean(0) - gates[~shock].mean(0)).sum()
    print(f'  shock specialisation (total variation shock vs non-shock) = {tv:.4f}'
          f'   (0 = gate ignores shock, 1 = fully disjoint experts)')

    print(f'\n[3] SHOCK RESIDUAL  p_s * shock_expert(h)   [Cp units]')
    print(f'  mean |resid| | shock=1 : {np.abs(resid[shock]).mean():.6f}')
    print(f'  mean |resid| | shock=0 : {np.abs(resid[~shock]).mean():.6f}')
    print(f'  ratio |resid| / |smooth| on shock nodes: '
          f'{np.abs(resid[shock]).mean() / max(np.abs(smooth[shock]).mean(), 1e-12):.4f}')

    err = cp_p - cp_t
    print(f'\n[4] Cp ERROR')
    print(f'  MAE | shock=1 : {np.abs(err[shock]).mean():.6f}   bias={err[shock].mean():+.6f}')
    print(f'  MAE | shock=0 : {np.abs(err[~shock]).mean():.6f}   bias={err[~shock].mean():+.6f}')

    mae_with    = np.abs(err[shock]).mean()
    mae_without = np.abs((smooth[shock] + cp_mean) - cp_t[shock]).mean()
    print(f'\n[5] ABLATION of the shock residual (shock nodes only)')
    print(f'  MAE with residual    : {mae_with:.6f}')
    print(f'  MAE without residual : {mae_without:.6f}')
    print(f'  contribution         : {mae_without - mae_with:+.6f} '
          f'({100*(mae_without-mae_with)/max(mae_without,1e-12):+.2f}%)')


if __name__ == '__main__':
    main()
