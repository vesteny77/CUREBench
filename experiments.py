#!/usr/bin/env python3
"""
Simple hyperparameter sweep runner for CUREBench.

Usage:
  python -m CUREBench.experiments --config CUREBench/config-examples/metadata_config_val_hybrid.json --subset-size 100

Expands the `runtime` settings by arrays in `experiments` section of the config, e.g.:
{
  "runtime": { "model_type": "multiagent", "temperature": 0.3, "step_budget": 6, "self_consistency_k": 3 },
  "experiments": { "temperature": [0.2,0.4], "step_budget": [6,10], "self_consistency_k": [1,3] }
}
"""

import argparse
import copy
import itertools
import json
import os
import time
from typing import Dict, List

from .eval_framework import CompetitionKit


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def combos_from_experiments(exp_cfg: Dict) -> List[Dict]:
    if not exp_cfg:
        return [{}]
    keys = list(exp_cfg.keys())
    values = [exp_cfg[k] if isinstance(exp_cfg[k], list) else [exp_cfg[k]] for k in keys]
    out = []
    for vals in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, vals)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Base config JSON path')
    ap.add_argument('--subset-size', type=int, default=None, help='Optional subset size for quick iteration')
    args = ap.parse_args()

    base = load_json(args.config)
    runtime = base.get('runtime', {})
    exp_cfg = base.get('experiments', {})
    combos = combos_from_experiments(exp_cfg)

    output_dir = base.get('output_dir', 'competition_results')
    sweep_dir = os.path.join(output_dir, 'sweeps')
    ensure_dir(sweep_dir)

    summary_rows = []
    for i, hps in enumerate(combos):
        cfg = copy.deepcopy(base)
        cfg.setdefault('runtime', {})
        cfg['runtime'].update(hps)

        # Unique naming
        stamp = int(time.time())
        slug = "_".join([f"{k}{str(v).replace('.', '')}" for k, v in sorted(hps.items())]) or f"trial{i}"
        trial_dir = os.path.join(output_dir, f"trial_{slug}")
        ensure_dir(trial_dir)

        cfg['output_dir'] = trial_dir
        cfg['output_file'] = f"submission_{slug}.csv"

        # Save trial config
        trial_cfg_path = os.path.join(trial_dir, f"config_{slug}.json")
        with open(trial_cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)

        # Run
        kit = CompetitionKit(config_path=trial_cfg_path)
        model_type = cfg.get('runtime', {}).get('model_type', 'auto')
        model_name = cfg.get('metadata', {}).get('model_name')
        kit.load_model(model_name, model_type)

        dataset_name = cfg.get('dataset', {}).get('dataset_name')
        res = kit.evaluate(dataset_name, subset_size=args.subset_size)
        zip_path = kit.save_submission_with_metadata([res], filename=cfg['output_file'], config_path=trial_cfg_path)

        summary_rows.append({
            'slug': slug,
            **{k: v for k, v in hps.items()},
            'accuracy': res.accuracy,
            'correct': res.correct_predictions,
            'total': res.total_examples,
            'zip_path': zip_path,
        })

    # Write summary
    summary_path = os.path.join(sweep_dir, 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Wrote sweep summary: {summary_path}")


if __name__ == '__main__':
    main()

