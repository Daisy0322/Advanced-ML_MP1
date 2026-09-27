"""将若干同结构 checkpoint 的参数做均匀平均，保存为可直接评测的一份权重。"""
import argparse
from pathlib import Path
import torch
from common import PROTOCOL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    packs = [torch.load(path, map_location='cpu', weights_only=True) for path in args.inputs]
    keys = packs[0]['model'].keys()
    for pack in packs[1:]:
        if pack['config'] != packs[0]['config']:
            raise ValueError('只能平均结构相同的 checkpoint。')
        if pack['model'].keys() != keys:
            raise ValueError('权重键不一致，无法平均。')
    averaged = {}
    scale = 1.0 / len(packs)
    for key in keys:
        averaged[key] = sum(pack['model'][key].float() for pack in packs) * scale
        if packs[0]['model'][key].dtype != averaged[key].dtype:
            averaged[key] = averaged[key].to(dtype=packs[0]['model'][key].dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'protocol': PROTOCOL,
        'implementation': packs[0]['implementation'],
        'config': packs[0]['config'],
        'model': averaged,
        'seed': packs[0].get('seed'),
        'train_tokens': sum(pack.get('train_tokens', 0) for pack in packs),
        'averaged_from': [str(path) for path in args.inputs],
    }, args.output)
    print(f'wrote {args.output} from {len(packs)} checkpoints')


if __name__ == '__main__':
    main()
