"""Package the audited fixed-configuration, single-seed full-data comparison."""
import _bootstrap
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import zipfile
import pandas as pd
from src.continual.full7_protocol import METHODS, sha
from src.utils.config import load_config
from src.utils.logging import save_json


LABELS = {'A_direct': 'Direct / Sequential FT', 'B_online_ewc': 'Online EWC', 'E_full_sgcr': 'SGCR (ours)'}


def main(cfg):
    root = Path(cfg['sgcr']['output_root'])
    audit = json.loads((root / 'audit.json').read_text()); assert audit['status'] == 'passed'
    results = pd.read_csv(root / 'final_results.csv')
    assert results.method.tolist() == METHODS
    domains = pd.read_csv(root / 'domain_summary.csv').sort_values('stage_id')
    split = json.loads((root / 'split_protocol.json').read_text())
    stage1 = json.loads((Path(cfg['experiment']['shift_output']) / 'stage1_status.json').read_text())
    test_count = json.loads((root / 'tests_summary.json').read_text())['passed']
    per_domain = []; continual = []
    for method in METHODS:
        path = Path(cfg['experiment']['run_root']) / method
        samples = pd.read_csv(path / 'sample_predictions.csv')
        for s in range(1, 8):
            table = samples[(samples.stage_id == s) & (samples.split == 'test')]
            for d in domains[domains.stage_id <= s].itertuples():
                part = table[table.domain_id == d.domain_id]
                row = dict(method=method, stage_id=s, domain_id=d.domain_id, domain_name=d.domain_name,
                    domain_stage=d.stage_id, test_count=len(part), mae=float(part.absolute_error.mean()))
                continual.append(row)
                if s == 7: per_domain.append(row)
        last = samples[(samples.stage_id == 7) & (samples.split == 'test')]
        results.loc[results.method == method, 'pooled_test_mae'] = last.absolute_error.mean()
        results.loc[results.method == method, 'label'] = LABELS[method]
    pd.DataFrame(per_domain).to_csv(root / 'per_domain_final_mae.csv', index=False)
    pd.DataFrame(continual).to_csv(root / 'continual_matrix_long.csv', index=False)
    results.to_csv(root / 'performance_comparison.csv', index=False)
    active = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: command = path.read_bytes()
        except (OSError, ProcessLookupError): continue
        if b'80_prepare_full7.py' in command or b'81_run_full7.py' in command:
            active.append(int(path.parent.name))
    assert not active, active
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    save_json(root / 'training_stop_verification.json', dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        training_stopped=True, training_processes=active, gpu_processes=gpu, seed=42,
        additional_experiments_scheduled=False, total_new_optimizer_steps=audit['total_new_optimizer_steps']))
    text = ['# 全量七域：Direct、Online EWC 与 SGCR 初步对照', '',
        '**固定配置、seed 42 的单次全量实验。没有运行额外随机种子；不能据此判断差异的统计显著性。**', '',
        '形成能数据共 132,752 条，全部保留；训练顺序为三斜 → 单斜 → 正交 → 四方 → 三方 → 六方 → 立方。',
        '各域按约 80% / 10% / 10% 分为训练、验证、测试。结构匹配组整体分配到同一集合，先前参与开发的 1 万条及其同组结构不进入新测试集。',
        f'实际划分：训练 {split["split_counts"]["train"]}，验证 {split["split_counts"]["val"]}，测试 {split["split_counts"]["test"]}。',
        '结构分组使用 pymatgen StructureMatcher，ltol=0.001、stol=0.001、angle_tol=0.1°、scale=False、primitive_cell=True、symmetric=True；这是给定容差下的结构分组，不是按化学家族留出。', '',
        '| 方法 | 最终七域平均 MAE ↓ | 全测试样本 MAE ↓ | 最终遗忘 ↓ | 阶段最大遗忘 ↓ | 后续阶段训练时间/min |',
        '|---|---:|---:|---:|---:|---:|']
    for r in results.itertuples():
        text.append(f'| {r.label} | {r.final_avg_mae:.6f} | {r.pooled_test_mae:.6f} | {r.forgetting:.6f} | {r.peak_stage_forgetting:.6f} | {r.wall_clock_seconds / 60:.2f} |')
    indexed = results.set_index('method'); ours = indexed.loc['E_full_sgcr']
    text += ['', f'SGCR 相对 Direct 的最终平均 MAE 变化：{(ours.final_avg_mae / indexed.loc["A_direct", "final_avg_mae"] - 1) * 100:+.2f}%。',
        f'SGCR 相对 Online EWC 的最终平均 MAE 变化：{(ours.final_avg_mae / indexed.loc["B_online_ewc", "final_avg_mae"] - 1) * 100:+.2f}%。', '',
        '## 各晶系数据量与最终误差', '',
        '| 顺序 | 晶系 | 总数 | 训练 | 验证 | 测试 | Direct MAE | EWC MAE | SGCR MAE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    pdomain = pd.DataFrame(per_domain)
    for d in domains.itertuples():
        values = [float(pdomain[(pdomain.method == m) & (pdomain.domain_id == d.domain_id)].mae.iloc[0]) for m in METHODS]
        text.append(f'| D{d.stage_id} | {d.domain_name} | {d.samples} | {d.train} | {d.val} | {d.test} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |')
    text += ['', '## 配置与比较口径', '',
        f'- 三组共用仅以 D1 训练和验证得到的首域检查点，首域训练 {stage1["epochs_run"]} 个 epoch，共用首域训练耗时 {stage1["elapsed_seconds"] / 60:.2f} 分钟；后续各域均训练 10 个 epoch，保存最后一步。',
        '- 后续训练统一使用 AdamW、学习率 0.0005、weight decay 0.0001、当前 batch size 32、Huber 监督损失和梯度裁剪 5。三组当前样本顺序和优化更新数一致。',
        '- Online EWC 使用现有默认 λ=10、衰减系数=1；每域从当前训练集固定抽取 128 条估计单位方差高斯回归的经验 Fisher，之后仅保存参数锚点和 Fisher，不保存历史样本回放。Fisher 计算时间已计入训练时间。',
        '- SGCR 沿用此前 1 万条试验选出的参数：β=0.1、局部检索比例=0.25、冲突/结构优先级权重=0.1/0.9。全局记忆上限 500，其中每阶段最多 400 条可参与梯度回放、100 条预留；每步回放数量与当前样本数量一致。',
        '- SGCR 的超参数来自已有试跑，EWC 本轮没有同等规模的超参数搜索；这是固定配置的初步对照，不是充分且等预算调参后的方法排名。全量新测试集没有用于参数或检查点选择。',
        '- SGCR 有额外历史样本前向/反向及检索计算，三组并非相同计算量。表中时间不含共同 D1、数据/图准备和结果审计；SGCR 的一次性冻结特征提取日志单独保存在 features.log。',
        '- 主 MAE 为学完 D7 后七个域测试 MAE 的等权平均；全样本 MAE 按测试样本数加权。最终遗忘为前六域最终 MAE 相对各自引入后的最低 MAE 的回升量平均，最低值包含最终阶段。',
        f'- {test_count} 项测试通过；21 个阶段检查点重新计算测试预测并核对指标。审计还复算每域 EWC Fisher 累积、SGCR 记忆选择、检索边界、投影公式与当前样本顺序。',
        '- 个别结构在固定 8 Å 半径内没有邻居，保留上游生成的空边图，未改变截断半径或删除样本；空边图的前向、反向和评估有限性经过验证，具体样本记录在 graph_preparation.json。',
        '- 化学式 NaN 按氮化钠的字符串读取，相关 3 条材料已保留并完成结构分组，详见 grouping_resume.json。',
        '- 旧 5k/10k 代码、配置和结果保留，完整权重保存在服务器，本结果包不含权重和图缓存。', '',
        f'服务器结果目录：`{root.resolve()}`。',
        '逐样本预测与每阶段记录位于 runs/；完整参数见 resolved_config.yaml；审计见 audit.json。']
    (root / 'FINAL_REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    weights = {str(p.resolve()): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in root.rglob('*.pt')}
    save_json(root / 'remote_weights_manifest.json', weights)
    for name in ('protected_before_run.json', 'training_source_manifest.json'):
        manifest = json.loads((root / name).read_text()); assert all(sha(p) == digest for p, digest in manifest.items())
    save_json(root / 'completion.json', dict(status='passed', audited=True, seed=42, methods=METHODS, training_stopped=True))
    files = {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()
        and p.suffix not in ('.pt', '.npz', '.zip', '.pid') and '__pycache__' not in p.parts
        and p.name not in ('packaging.log', 'pipeline.log', 'pipeline_status.json', 'package_verification.json')}
    for folder in ('src', 'scripts', 'configs', 'tests'):
        for p in Path(folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.yaml', '.md') and '__pycache__' not in p.parts:
                files['source/' + p.as_posix()] = p
    for filename in ('requirements.txt', 'README.md', 'third_party/ALIGNN_COMMIT.txt'):
        p = Path(filename)
        if p.exists(): files['source/' + filename] = p
    for filename in (cfg['domains']['assignments'], cfg['domains']['splits'], cfg['domains']['split_audit'],
                     str(Path(cfg['data']['processed']) / 'metadata.csv'), str(Path(cfg['data']['processed']) / 'manifest.json')):
        p = Path(filename); files['data_snapshot/' + p.name] = p
    files['data_snapshot/legacy_pilot_splits_seed42.json'] = Path('data/domains/crystal4_n10000_splits_seed42.json')
    manifest = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(files.items())}
    destination = root.parent / 'Full7_Direct_OnlineEWC_SGCR_seed42_results.zip'
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, p in sorted(files.items()): archive.write(p, name)
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        import hashlib
        for name, item in manifest.items():
            digest = hashlib.sha256(); count = 0
            with archive.open(name) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(block); count += len(block)
            assert digest.hexdigest() == item['sha256'] and count == item['bytes']
    save_json(root / 'package_verification.json', dict(status='passed', archive=str(destination),
        bytes=destination.stat().st_size, sha256=sha(destination), verified_files=len(manifest), weights_included=False))
    print(results[['label', 'final_avg_mae', 'forgetting', 'wall_clock_seconds']].to_string(index=False), flush=True)
    print('FULL7_PACKAGE_COMPLETE', str(destination), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    main(load_config(p.parse_args().config))
