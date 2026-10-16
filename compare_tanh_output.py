"""Isolated tanh-output ablation; never replaces the manuscript's frozen policy.

Run with the same Python environment as the reproduction scripts. Checkpoint
selection uses action-validation data only; all specified road tests are kept.
"""
import bootstrap
from bootstrap import ROOT
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
import core
from mass_controller import MassConditionedPolicy, MassEstimator, DEFAULT_POLICY
from mass_optimization import MASSES, WINDOWS, savecsv
from train_paired_mass import GRID, NOM
from train_mass_heads import forward

OUT = ROOT / 'results/tanh_output_comparison'
DATA = ROOT / 'results/mass_optimization/paired_dense/paired_dataset.npz'


class TanhPolicy(MassConditionedPolicy):
    def heads(self, xx):
        h = (xx-self.mean)/self.scale
        for w, b in zip(self.w[:-1], self.b[:-1]):
            h = np.maximum(h@w+b, 0)
        return 500*np.tanh(h@self.w[-1]+self.b[-1])


def loss_gradient(x, y, w, b, activation):
    logits, a = forward(x, w, b)
    raw = np.tanh(logits) if activation == 'tanh' else logits
    applied = np.clip(raw, -1, 1)
    delta = (applied-applied[:, NOM, None])-(y-y[:, NOM, None])
    gd = 50*delta/raw.size
    gd[:, NOM] -= gd.sum(1)
    g = 2*(raw-y)/raw.size + gd*((raw > -1) & (raw < 1))
    if activation == 'tanh':
        g *= 1-raw*raw
    gw, gb = [None]*3, [None]*3
    for j in [2, 1, 0]:
        gw[j] = a[j].T@g+1e-7*w[j]
        gb[j] = g.sum(0)
        if j:
            g = (g@w[j].T)*(a[j] > 0)
    loss = np.mean((raw-y)**2)+25*np.mean(delta**2)
    loss += .5e-7*sum(np.sum(v*v) for v in w)
    return float(loss), gw+gb


def score(x, y, w, b, activation):
    logits = forward(x, w, b)[0]
    raw = np.tanh(logits) if activation == 'tanh' else logits
    p = np.clip(raw, -1, 1)
    delta = (p-p[:, NOM, None])-(y-y[:, NOM, None])
    sat = abs(y) >= .9
    return dict(action_rmse_N=float(500*np.sqrt(np.mean((p-y)**2))),
                nominal_rmse_N=float(500*np.sqrt(np.mean((p[:, NOM]-y[:, NOM])**2))),
                mass_delta_rmse_N=float(500*np.sqrt(np.mean(delta[:, np.arange(len(GRID)) != NOM]**2))),
                saturation_mae_N=float(500*np.mean(abs(p[sat]-y[sat]))),
                raw_over_pct=float(100*np.mean(abs(raw) > 1+1e-6/500)),
                applied_over_pct=float(100*np.mean(abs(p) > 1+1e-6/500)),
                loss=float(np.mean((raw-y)**2)+25*np.mean(delta**2)))


def gradient_check():
    rng = np.random.default_rng(12)
    w = [rng.normal(0, .2, (a, b)) for a, b in [(3, 5), (5, 6), (6, 8)]]
    b = [np.full(n, .4) for n in [5, 6, 8]]
    x, y = rng.normal(size=(7, 3)), rng.uniform(-1, 1, (7, 8))
    errors = []
    for activation in ['linear', 'tanh']:
        _, grads = loss_gradient(x, y, w, b, activation)
        for p, g in zip(w+b, grads):
            for idx in [tuple(0 for _ in p.shape), tuple(n-1 for n in p.shape)]:
                v, eps = p[idx], 1e-6
                p[idx] = v+eps
                plus = loss_gradient(x, y, w, b, activation)[0]
                p[idx] = v-eps
                minus = loss_gradient(x, y, w, b, activation)[0]
                p[idx] = v
                errors.append(abs((plus-minus)/(2*eps)-g[idx]))
    assert max(errors) < 1e-6, max(errors)
    return max(errors)


def train(activation, epochs):
    folder = OUT / activation
    folder.mkdir(exist_ok=True)
    z = np.load(DATA)
    base = z['train_x'][:, 0, :-1]
    mean, scale = base.mean(0), np.maximum(base.std(0), 1e-6)
    x = (base-mean)/scale
    val = (z['validation_x'][:, 0, :-1]-mean)/scale
    y = z['train_y']
    rng = np.random.default_rng(17001)
    # Consume the same initialization draws as train_mass_heads.py.
    for a, bdim in [(34, 128), (128, 128), (128, 8)]:
        rng.normal(0, np.sqrt(2/a), (a, bdim))
    old = np.load(ROOT/'results/models/policy.npz')
    w = [old['w0'][:-1].copy()*(scale/old['scale'][:-1])[:, None],
         old['w1'].copy(), np.repeat(old['w2'], len(GRID), axis=1)]
    b = [old['b0'].copy()+(mean-old['mean'][:-1])/old['scale'][:-1]@old['w0'][:-1]
         +(300-old['mean'][-1])/old['scale'][-1]*old['w0'][-1],
         old['b1'].copy(), np.repeat(old['b2'], len(GRID))]
    params = w+b
    mom, vel = [[np.zeros_like(p) for p in params] for _ in range(2)]
    step, best, history = 0, np.inf, []
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(x))
        lr = .001*(.3 if epoch > 500 else 1)*(.3 if epoch > 1000 else 1)*(.3 if epoch > 1500 else 1)
        for offset in range(0, len(x), 128):
            ids = order[offset:offset+128]
            _, grads = loss_gradient(x[ids], y[ids], w, b, activation)
            step += 1
            for p, m, v, g in zip(params, mom, vel, grads):
                m *= .9
                m += .1*g
                v *= .999
                v += .001*g*g
                p -= lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        sc = score(val, z['validation_y'], w, b, activation)
        history.append(dict(epoch=epoch, **sc))
        if sc['loss'] < best-1e-10:
            best, best_epoch = sc['loss'], epoch
            best_w, best_b = [p.copy() for p in w], [p.copy() for p in b]
        if epoch % 100 == 0:
            print(activation, epoch, 'validation RMSE', sc['action_rmse_N'],
                  'elapsed', round(time.perf_counter()-start, 1), flush=True)
    w, b = best_w, best_b
    np.savez(folder/'policy.npz', mean=mean, scale=scale, grid=GRID,
             activation=activation, **{f'w{i}':p for i,p in enumerate(w)},
             **{f'b{i}':p for i,p in enumerate(b)})
    report = dict(activation=activation, epochs=epochs, best_epoch=best_epoch,
                  seconds=time.perf_counter()-start,
                  validation=score(val, z['validation_y'], w, b, activation))
    (folder/'training.json').write_text(json.dumps(report, indent=2))
    savecsv(folder/'training_history.csv', history)


def measured_rollout(policy, road):
    count = 800
    x = np.array([road[0], 0., road[0], 0.])
    ad, bd, ed = core.discrete()
    ys, us, raws = [], [], []
    for k in range(count):
        u, raw = policy.action(x, road[k:k+30], 300.)
        ys.append(core.outputs(x, u, road[k]))
        us.append(u)
        raws.append(raw)
        x = ad@x+bd*u+ed*road[k]
    return dict(t=np.arange(count)*core.DT, y=np.array(ys), u=np.array(us),
                raw=np.array(raws), mass=np.full(count, 300.))


def evaluate():
    policies = {'current_linear': MassConditionedPolicy(),
                'matched_linear': MassConditionedPolicy(OUT/'linear/policy.npz'),
                'tanh': TanhPolicy(OUT/'tanh/policy.npz')}
    z = np.load(DATA)
    action = []
    for name, pol in policies.items():
        x = (z['test_x'][:, 0, :-1]-pol.mean)/pol.scale
        action.append(dict(model=name, **score(x, z['test_y'], pol.w, pol.b,
                                              'tanh' if name == 'tanh' else 'linear')))
    savecsv(OUT/'action_test.csv', action)
    core.MassRLS = MassEstimator
    cases = [('in_domain', 'sine', 300, 0, 0), ('unseen_sine', 'unseen_sine', 300, 0, 0),
             ('unseen_mass', 'multisine', 330, 0, 0), ('bump', 'bump', 300, 0, 0),
             ('preview_noise', 'multisine', 300, .002, 0),
             ('preview_delay', 'multisine', 300, 0, -2)]
    rows = []
    for name, pol in policies.items():
        for ci, (group, kind, mass, noise, shift) in enumerate(cases):
            for rep in range(8):
                extra = dict(frequency=3.4+.1*rep, amplitude=.018) if kind == 'unseen_sine' else {}
                tr = core.rollout(kind, 4000+100*ci+rep, 'NN', mass=mass, policy=pol,
                                  preview_noise=noise, preview_shift=shift, **extra)
                rows.append(dict(model=name, group=group, road=rep, **core.metrics(tr)))
        for i, profile in enumerate(np.load(ROOT/'results/road_halfcar/road_inputs.npz')['profiles']):
            tr = measured_rollout(pol, profile[:, 0])
            rows.append(dict(model=name, group='measured_road', road=i, **core.metrics(tr, start=2)))
        print('generalization finished', name, flush=True)
    savecsv(OUT/'road_test.csv', rows)
    protocol = json.loads((ROOT/'results/mass_optimization/protocol.json').read_text())
    rows = []
    # Re-run all three policies to avoid relying on cached numerical results.
    for name, pol in policies.items():
        for kind, seeds in protocol['heldout_seeds'].items():
            for seed in seeds:
                for mass in MASSES:
                    for mode in ['NN_fixed', 'NN_adaptive', 'NN_oracle']:
                        tr = core.rollout(kind, seed, mode, duration=24, policy=pol,
                                          mass_schedule=lambda t: mass if 4 <= t < 20 else 300., acc_noise=.05)
                        for window, (a, b) in WINDOWS.items():
                            mask = (tr['t'] >= a) & (tr['t'] < b)
                            sub = {k:v[mask] for k,v in tr.items() if len(v) == len(mask)}
                            rows.append(dict(model=name, kind=kind, road=seed, mass=mass,
                                             mode=mode, window=window, **core.metrics(sub)))
                print('mass test', name, kind, seed, flush=True)
                savecsv(OUT/'mass_test.csv', rows)


def summarize():
    def read(name):
        return list(csv.DictReader((OUT/name).open()))

    def mean(rows, key):
        return float(np.mean([float(r[key]) for r in rows]))

    road, mass, actions = read('road_test.csv'), read('mass_test.csv'), read('action_test.csv')
    assert len(road) == 3*56 and len(mass) == 3*96*3*5
    assert all(float(r['violation_pct']) == 0 for r in road+mass)
    assert all(float(r['raw_violation_pct']) == 0 for r in road+mass if r['model'] == 'tanh')
    for r in road+mass:
        for key in ['acc_rms', 'travel_rms_mm', 'tire_rms_N', 'min_contact_ratio']:
            assert np.isfinite(float(r[key]))
    frozen = list(csv.DictReader((DEFAULT_POLICY.parent/'heldout.csv').open()))
    reference = {(r['kind'], int(r['seed']), float(r['mass']), r['mode'], r['window']):r for r in frozen}
    differences = []
    for r in mass:
        if r['model'] == 'current_linear':
            ref = reference[(r['kind'], int(r['road']), float(r['mass']), r['mode'], r['window'])]
            differences.append(abs(float(r['acc_rms'])-float(ref['acc_rms'])))
    assert max(differences) < 1e-10
    settings = json.loads((OUT/'protocol.json').read_text())
    assert settings['baseline_sha256'] == hashlib.sha256(DEFAULT_POLICY.read_bytes()).hexdigest()
    validation = dict(road_trajectories=len(road), mass_trajectories=len(mass)//5,
                      all_metrics_finite=True, all_applied_force_violations_zero=True,
                      all_tanh_raw_force_violations_zero=True,
                      current_model_rms_max_difference_from_frozen_results=max(differences),
                      production_policy_unchanged=True)
    (OUT/'validation.json').write_text(json.dumps(validation, indent=2))
    models = ['current_linear', 'matched_linear', 'tanh']
    groups = list(dict.fromkeys(r['group'] for r in road))
    road_summary, mass_summary = [], []
    for group in groups:
        for model in models:
            rr = [r for r in road if r['group'] == group and r['model'] == model]
            road_summary.append(dict(group=group, model=model, **{key:mean(rr, key) for key in
                ['acc_rms', 'travel_rms_mm', 'tire_rms_N', 'raw_violation_pct', 'violation_pct']}))
    for kind in ['sine', 'multisine']:
        for load in MASSES:
            for window in WINDOWS:
                for model in models:
                    rr = [r for r in mass if r['model'] == model and r['kind'] == kind
                          and float(r['mass']) == load and r['window'] == window]
                    vals = {mode:mean([r for r in rr if r['mode'] == mode], 'acc_rms')
                            for mode in ['NN_fixed', 'NN_adaptive', 'NN_oracle']}
                    mass_summary.append(dict(kind=kind, mass=load, window=window, model=model,
                        **vals, adaptive_gain_pct=100*(1-vals['NN_adaptive']/vals['NN_fixed']),
                        oracle_gain_pct=100*(1-vals['NN_oracle']/vals['NN_fixed'])))
    savecsv(OUT/'road_summary.csv', road_summary)
    savecsv(OUT/'mass_summary.csv', mass_summary)
    lines = ['# Tanh 输出层对照实验', '',
        '本实验未替换论文模型。tanh 放在八个质量输出头上，再按质量做凸组合插值，因此输出幅值不超过 500 N。', '',
        '## 方法', '',
        '复用 160,000 个训练 MPC 标签、32,000 个验证标签和 32,000 个测试标签；网络为 34–128–128–8，质量节点为 200–450 kg。'
        f"线性输出和 tanh 输出均训练 {settings['epochs']:,} 轮，初始化权重、批次顺序、学习率和损失权重一致；各自只按验证损失选取权重。"
        '初始化使用相同权重，经过 tanh 后的初始控制映射不再相同。额外重训线性模型用于区分训练数值差异与激活函数影响。', '',
        '质量实验保持现有 RLS 参数、0.05 m/s² 加速度噪声、4 s 加载和 20 s 恢复不变；'
        '24 条道路 × 4 个质量 × 3 种质量输入，逐项重算三个模型。一般道路为六组各八条，另复核八段实测道路。'
        '这些是既有测试工况的配对回归比较，不是另行采集的独立新数据。所有工况均保留。', '',
        '## MPC 动作拟合', '',
        '| 模型 | 力 RMSE (N) | 饱和区 MAE (N) | 质量动作差 RMSE (N) | 原始超限 (%) |',
        '|---|---:|---:|---:|---:|']
    for r in actions:
        lines.append(f"| {r['model']} | {float(r['action_rmse_N']):.3f} | {float(r['saturation_mae_N']):.3f} | {float(r['mass_delta_rmse_N']):.3f} | {float(r['raw_over_pct']):.3f} |")
    lines += ['', '饱和区定义为 MPC 标签的绝对值不低于 450 N；力 RMSE 使用实际施加的有界输出。', '',
              '## 道路闭环加速度 RMS (m/s²)', '',
              '| 工况 | 当前线性 | 同设置重训线性 | tanh | tanh 相对当前降低 (%) |', '|---|---:|---:|---:|---:|']
    for group in groups:
        vals = {r['model']:r['acc_rms'] for r in road_summary if r['group'] == group}
        lines.append(f"| {group} | {vals['current_linear']:.6f} | {vals['matched_linear']:.6f} | {vals['tanh']:.6f} | {100*(1-vals['tanh']/vals['current_linear']):.2f} |")
    lines += ['', '正数为改善，负数为退化。每项是八条道路各自 RMS 的平均值；实测道路评价 2–8 s，其余评价完整 0–6 s。', '',
              '## 质量变化后 4–20 s 的表现', '',
              '| 道路 | 质量 (kg) | 当前 adaptive RMS | tanh adaptive RMS | 当前辨识收益 (%) | tanh 辨识收益 (%) |',
              '|---|---:|---:|---:|---:|---:|']
    for kind in ['sine', 'multisine']:
        for load in MASSES:
            rr = {r['model']:r for r in mass_summary if r['kind'] == kind and r['mass'] == load and r['window'] == 'loaded'}
            a, b = rr['current_linear'], rr['tanh']
            lines.append(f"| {kind} | {load:.0f} | {a['NN_adaptive']:.6f} | {b['NN_adaptive']:.6f} | {a['adaptive_gain_pct']:.2f} | {b['adaptive_gain_pct']:.2f} |")
    lines += ['', '辨识收益 = 100 × (1 − 平均 adaptive RMS / 同一模型的平均 fixed RMS)。'
              '不能仅凭辨识收益变大判断控制改善，必须同时比较 adaptive RMS 的绝对值。'
              '完整 fixed/adaptive/oracle 结果以及早期、稳定段、恢复段结果见 mass_summary.csv；行级数据见 mass_test.csv。', '']
    lookup = {(r['group'], r['model']):r['acc_rms'] for r in road_summary}
    relative = {group:100*(1-lookup[group, 'tanh']/lookup[group, 'matched_linear']) for group in groups}
    lines += ['## 判断', '',
              'tanh 实现了结构性的力幅值约束，但本轮测试没有显示全面的性能优势。'
              '动作拟合误差高于两个线性基准，常规道路的闭环变化较小；质量辨识收益没有增强。', '',
              f"相对同设置重训线性网络，tanh 在预瞄噪声和延迟工况下分别降低加速度 RMS {relative['preview_noise']:.2f}% 和 {relative['preview_delay']:.2f}%。"
              f"实测道路则升高 {-relative['measured_road']:.2f}%；相对旧冻结模型的明显改善也出现在重训线性模型中，不能归因于 tanh。", '',
              '因此暂不替换论文网络。若主要目标是省去显式力投影，tanh 是可行方案；'
              '若目标是提高减振性能或突出质量辨识作用，本次直接替换并不足以支持采用。'
              '这里的结论针对当前训练配置的一组配对实验，不代表 tanh 在其他训练方案下的最优性能。'
              '输出有界也不等于悬架行程、轮胎接地或闭环稳定性获得保证。', '',
              '## 复现', '', '`python compare_tanh_output.py` 完成训练与评估；'
              '`--stage evaluate` 用已保存权重重算闭环；`--stage summarize` 重建此报告。'
              '输入文件哈希、梯度检验和预先固定的协议见 protocol.json，权重与每轮验证记录在 linear/ 和 tanh/。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['train', 'evaluate', 'summarize', 'all'], default='all')
    ap.add_argument('--epochs', type=int, default=1100)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage in ['train', 'all']:
        settings = dict(epochs=args.epochs, seed=17001, batch_size=128, force_bound_N=500,
                        architecture=[34,128,128,8], checkpoint_selection='minimum validation loss only',
                        loss='raw action MSE + 25 * applied mass-difference MSE',
                        initialization='same original nominal policy weights and same minibatch order',
                        tanh_position='each mass head, before convex mass interpolation',
                        dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                        baseline_sha256=hashlib.sha256(DEFAULT_POLICY.read_bytes()).hexdigest(),
                        gradient_check_max_absolute_error=gradient_check(),
                        evaluation='all 96 existing mass scenarios, all 48 synthetic road cases, 8 measured road segments; no test-based checkpoint selection')
        (OUT/'protocol.json').write_text(json.dumps(settings, indent=2))
        for activation in ['linear', 'tanh']:
            train(activation, args.epochs)
    if args.stage in ['evaluate', 'all']:
        evaluate()
    if args.stage in ['evaluate', 'summarize', 'all']:
        summarize()


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
