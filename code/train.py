"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from collections import deque
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def copy_state(source):
    return {key: value.detach().cpu().clone() for key, value in source.items()}


def average_states(states):
    blended = {key: torch.zeros_like(states[0][key]) for key in states[0]}
    scale = 1. / len(states)
    for state in states:
        for key, value in state.items():
            blended[key].add_(value, alpha=scale)
    return blended


def window_starts(length, batch_size, generator, unique):
    limit = length - 257
    if not unique:
        while True:
            yield torch.randint(limit, (batch_size,), generator=generator)
    order = torch.randperm(limit, generator=generator)
    cursor = 0
    while True:
        if cursor + batch_size > len(order):
            order = torch.randperm(limit, generator=generator)
            cursor = 0
        yield order[cursor:cursor + batch_size]
        cursor += batch_size


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--lr', type=float, default=.001, help='Peak AdamW learning rate after warmup.')
    p.add_argument('--ema', type=float, default=0.,
                   help='If >0, keep an EMA of weights and use it for validation/checkpoint.')
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--swa', type=int, default=0,
                   help='If >0, average the last N validation snapshots and keep it if better.')
    p.add_argument('--teacher', type=Path,
                   help='Optional frozen teacher checkpoint for logit distillation.')
    p.add_argument('--distill', type=float, default=0.,
                   help='Weight on KL(student || teacher); 0 disables distillation.')
    p.add_argument('--distill-temp', type=float, default=2.)
    p.add_argument('--unique-windows', action='store_true',
                   help='Cover training windows without replacement before reshuffling.')
    args = p.parse_args()
    if args.ema < 0 or args.ema >= 1:
        p.error('EMA decay must be 0 (disabled) or in [0, 1).')
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.swa < 0:
        p.error('SWA count must be non-negative.')
    if args.teacher is None and args.distill != 0:
        p.error('Distillation weight requires --teacher.')
    if args.teacher is not None and not (0. <= args.distill <= 1.):
        p.error('Distillation weight must be in [0, 1].')
    if args.distill_temp <= 0:
        p.error('Distillation temperature must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 同一条训练轨迹上的权重滑动平均，评测时仍是一份模型
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()} if args.ema > 0 else None
    teacher = None
    if args.teacher is not None:
        teacher_ckpt = torch.load(args.teacher, map_location='cpu', weights_only=True)
        if teacher_ckpt['protocol'] != PROTOCOL:
            raise ValueError('Teacher checkpoint belongs to a different course protocol.')
        teacher, _ = make_model(teacher_ckpt['implementation'], teacher_ckpt['config'], device)
        teacher.load_state_dict(teacher_ckpt['model'])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    batches = window_starts(len(tokens), args.batch_size, rng, args.unique_windows)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_bpb = float('inf')
    best_state = None
    best_step = None
    recent_states = deque(maxlen=args.swa) if args.swa > 0 else None
    for step in range(args.steps):
        starts = next(batches).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.lr * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:,:-1])
            loss = F.cross_entropy(logits.flatten(0,1).float(), batch[:,1:].flatten())
            if teacher is not None and args.distill > 0:
                with torch.no_grad():
                    teacher_logits = teacher(batch[:,:-1])
                temperature = args.distill_temp
                student_log = F.log_softmax(logits.float() / temperature, dim=-1)
                teacher_prob = F.softmax(teacher_logits.float() / temperature, dim=-1)
                distill = F.kl_div(student_log.flatten(0, 1), teacher_prob.flatten(0, 1), reduction='batchmean')
                loss = (1. - args.distill) * loss + args.distill * temperature * temperature * distill
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    ema[key].mul_(args.ema).add_(value.detach(), alpha=1. - args.ema)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            raw = None
            if ema is not None:
                raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(ema)
            intermediate = score(model,*data['validation'],device,'fp32')
            if raw is not None:
                model.load_state_dict(raw)
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            # 只保留验证 BPB 最好的权重（若启用 EMA 则存平均后的权重）
            source = ema if ema is not None else model.state_dict()
            snapshot = copy_state(source)
            if recent_states is not None:
                recent_states.append(snapshot)
            if intermediate['bpb'] < best_bpb:
                best_bpb = intermediate['bpb']
                best_step = step+1
                best_state = snapshot
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    if recent_states and len(recent_states) >= 2:
        swa_state = average_states(list(recent_states))
        model.load_state_dict(swa_state)
        model.to(device)
        swa_validation = score(model, *data['validation'], device, 'fp32')
        swa_validation.pop('window_nll_nats')
        intermediate_validation_seconds += swa_validation['seconds']
        if swa_validation['bpb'] < best_bpb:
            best_bpb = swa_validation['bpb']
            best_step = f'swa-{len(recent_states)}'
            best_state = swa_state
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    if best_step is not None:
        validation = {**validation, 'selected_step': best_step, 'selected_bpb': best_bpb}
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
