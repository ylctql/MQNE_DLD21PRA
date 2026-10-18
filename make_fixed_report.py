"""Generate a report from completed experiments only, without promoting pilots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mqne import save_json


EXPECTED=[
    'cancer_fixed_paper_seed21','cancer_fixed_minmax_seed21','cancer_yao_minmax_seed21',
    'spt_fixed_fused_seed21','spt_fixed_seed22','spt_legacy_control_fixed_seed21',
    'mnist_fixed_fused_seed21','mnist_yao_search_seed21','mnist_paper_s11_seed21',
    'mnist_paper_s11_yao_seed21','mnist_paper_s11_yao_seed22','mnist_paper_s11_yao_seed23',
    'mnist_brickwork10_yao_seed21',
]


def draw_circuit(circuit,output):
    blocks=circuit['blocks']; n=1+max(q for b in blocks for q in list(b['rotations'])+[q for p in b['pairs'] for q in p])
    fig,ax=plt.subplots(figsize=(max(8,len(blocks)*.6),4.8))
    for q in range(n):
        ax.hlines(n-1-q,-.6,len(blocks)-.3,color='#b8b8b8',linewidth=1)
        ax.text(-.65,n-1-q,f'q{q}',ha='right',va='center',fontsize=10)
    for i,b in enumerate(blocks):
        for q in b['rotations']:
            ax.text(i,n-1-q,'R',ha='center',va='center',fontsize=9,
                    bbox=dict(boxstyle='square,pad=.22',facecolor='#d8eafa',edgecolor='#3274a1'))
        for control,target in b['pairs']:
            ax.plot([i,i],[n-1-control,n-1-target],color='#c06b22',lw=1.5)
            ax.plot(i,n-1-control,'o',color='#c06b22',ms=4)
            ax.text(i,n-1-target,'Rx',ha='center',va='center',fontsize=8,
                    bbox=dict(boxstyle='square,pad=.15',facecolor='#ffe9cd',edgecolor='#c06b22'))
    ax.set(xlim=(-1,len(blocks)),ylim=(-.7,n-.4),xticks=range(len(blocks)),
           xticklabels=range(1,len(blocks)+1),yticks=[],xlabel='Gate block (R = Rz-Rx-Rz)')
    ax.set_title(f"Readout q{n-1}; {circuit['record']['parameters']} parameters")
    for spine in ax.spines.values():spine.set_visible(False)
    fig.tight_layout(); fig.savefig(output/'circuit.png',dpi=180); plt.close(fig)


def main():
    root=Path(__file__).resolve().parent; results={}
    parser=argparse.ArgumentParser()
    parser.add_argument('--display-root',type=Path,default=root,help='Absolute delivery directory for Markdown image links')
    display_root=parser.parse_args().display_root.as_posix()
    for name in EXPECTED:
        directory=root/'results'/name
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        verified=json.loads((directory/'restored_verification.json').read_text(encoding='utf-8'))
        results[name]=dict(summary=summary,config=config,restored=verified)
        draw_circuit(json.loads((directory/'best_circuit.json').read_text(encoding='utf-8')),directory)
    searches=[r for r in results.values() if r['summary']['run_type']=='MQNE_search']
    if any(r['summary']['circuits_trained']!=60 for r in searches):raise ValueError('Missing full search')
    total=sum(r['summary']['circuits_trained'] for r in results.values())
    updates=sum(r['summary']['optimizer_updates'] for r in results.values())
    mn=results['mnist_yao_search_seed21']['summary']; sp=results['spt_fixed_seed22']['summary']
    msb=results['mnist_fixed_fused_seed21']['summary']
    ca=results['cancer_fixed_minmax_seed21']['summary']
    search_note=(f"MNIST MSB 编码的自主完整搜索：适应度 {100*msb['best']['fitness']:.2f}%、"
                 f"验证 {100*msb['best']['final']['validation']['accuracy']:.2f}%、测试 {100*msb['independent_test']['accuracy']:.2f}%、"
                 f"{msb['best']['parameters']} 参数、{msb['best']['block_depth']} 块，第 {msb['best']['generation']} 代选中，达到论文约 97% 的水平；"
                 f"Yao 编码自主完整搜索：最优适应度 {100*mn['best']['fitness']:.2f}%、"
                 f"验证 {100*mn['best']['final']['validation']['accuracy']:.2f}%、测试 {100*mn['independent_test']['accuracy']:.2f}%、"
                 f"{mn['best']['parameters']} 参数、{mn['best']['block_depth']} 块；"
                 f"SPT 种子 22：适应度 {100*sp['best']['fitness']:.2f}%、验证 {100*sp['best']['final']['validation']['accuracy']:.2f}%、"
                 f"测试 {100*sp['independent_test']['accuracy']:.2f}%，{sp['best']['parameters']} 参数、第 {sp['best']['generation']} 代、{sp['best']['block_depth']} 块；"
                 f"Cancer minmax：适应度 {100*ca['best']['fitness']:.2f}%、验证 {100*ca['best']['final']['validation']['accuracy']:.2f}%。")
    rows=[]
    for name,r in results.items():
        s=r['summary']; b=s['best']; test=s['independent_test']
        rows.append('| '+ ' | '.join([name,s['run_type'],str(s['circuits_trained']),
            f"{100*b['fitness']:.2f}%",f"{100*b['final']['validation']['accuracy']:.2f}%",
            f"{100*test['accuracy']:.2f}%" if test else '无',str(b['parameters']),str(b['block_depth'])])+' |')
    names=['mnist_paper_s11_yao_seed'+str(i) for i in (21,22,23)]
    fitness=np.array([results[n]['summary']['best']['fitness'] for n in names])*100
    validation=np.array([results[n]['summary']['best']['final']['validation']['accuracy'] for n in names])*100
    tests=np.array([results[n]['summary']['independent_test']['accuracy'] for n in names])*100
    mean_note=(f'Fig. S11 的三个训练种子 21/22/23，适应度平均 {fitness.mean():.2f}%'
               f'（范围 {fitness.min():.2f}%-{fitness.max():.2f}%），验证平均 {validation.mean():.2f}%，'
               f'测试平均 {tests.mean():.2f}%（范围 {tests.min():.2f}%-{tests.max():.2f}%）。'
               '它们共用固定数据划分，只反映参数初始化差异，不是三个独立数据实验；未把最好的种子冒充平均表现。')
    figures=root/'figures'; figures.mkdir(exist_ok=True)
    spt_dir=root/'results'/'spt_fixed_seed22'
    manifest=json.loads((spt_dir/'data_manifest.json').read_text(encoding='utf-8'))
    values=np.loadtxt(spt_dir/'test_predictions.csv',delimiter=',',skiprows=1)
    lambdas=np.array(manifest['lambdas'])[manifest['test_indices']]
    wrong=values[:,0]!=values[:,2]
    errors=[dict(lambda_value=float(lambdas[i]),label=int(values[i,0]),probability_class1=float(values[i,1]),prediction=int(values[i,2])) for i in np.flatnonzero(wrong)]
    save_json(figures/'spt_test_errors.json',errors)
    phase_note=(f'SPT 额外 500 样本测试中有 {len(errors)} 个误分类；'
                f"其 lambda 范围 {lambdas[wrong].min():.6f}-{lambdas[wrong].max():.6f}。" if errors else 'SPT 额外测试无误分类。')
    fig,ax=plt.subplots(figsize=(8,4.6))
    for label,color in ((0,'#3274a1'),(1,'#d55e00')):
        keep=values[:,0]==label
        ax.scatter(lambdas[keep],values[keep,1],s=12,alpha=.55,color=color,label=f'True phase {label}')
    if errors:ax.scatter(lambdas[wrong],values[wrong,1],marker='x',s=55,color='red',label='Misclassified')
    ax.axhline(.5,color='black',ls='--',lw=1,label='Readout threshold')
    ax.axvline(1,color='gray',ls=':',lw=1,label='True phase boundary')
    ax.set(xlabel='lambda',ylabel='P(class 1)',xlim=(0,2),ylim=(0,1),title=f"SPT MQNE seed 22: 500 extra test states, {100*sp['independent_test']['accuracy']:.1f}% accuracy")
    ax.legend(fontsize=8,loc='lower right'); ax.grid(alpha=.15); fig.tight_layout()
    fig.savefig(figures/'spt_phase_predictions.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
    selected=[('Cancer (minmax; seed 21)','cancer_fixed_minmax_seed21',.946),('SPT (seed 22)','spt_fixed_seed22',1),('MNIST (MSB bits; seed 21)','mnist_fixed_fused_seed21',.97)]
    for ax,(title,name,target) in zip(axes,selected):
        records=json.loads((root/'results'/name/'candidates.json').read_text(encoding='utf-8'))
        for r in records:ax.scatter(r['generation'],r['fitness'],color='#3274a1',alpha=.45,s=18)
        generations=sorted(set(r['generation'] for r in records))
        ax.plot(generations,[max(r['fitness'] for r in records if r['generation']==g) for g in generations],'o-',color='#d55e00',label='Generation maximum')
        ax.axhline(target,ls='--',color='black',lw=1,label='Paper reported fitness')
        ax.set(title=title,xlabel='Generation',xticks=generations,ylim=(.3,1.03)); ax.grid(alpha=.15)
    axes[0].set_ylabel('Training + validation fitness'); axes[-1].legend(fontsize=8,loc='lower right')
    fig.tight_layout()
    fig.savefig(figures/'fixed_search_comparison.png',dpi=180); plt.close(fig)
    # All seeds displayed, so resource/accuracy comparison does not select the best seed.
    fig,ax=plt.subplots(figsize=(8,4.6))
    for i,n in enumerate(names):
        s=results[n]['summary']; ax.scatter(s['best']['parameters'],100*s['best']['final']['validation']['accuracy'],s=65,color='#3274a1',label='Paper Fig. S11 (3 seeds)' if i==0 else None)
    for name,label,color in [('mnist_yao_search_seed21','MQNE (Yao bits)','#d55e00'),('mnist_fixed_fused_seed21','MQNE (MSB bits)','#9063a1'),('mnist_brickwork10_yao_seed21','Fixed 10-layer control (Yao bits)','#438b46')]:
        s=results[name]['summary']; x=s['best']['parameters']; y=100*s['best']['final']['validation']['accuracy']
        ax.scatter(x,y,s=90,color=color,marker='s',label=label)
    ax.set(xlabel='Trainable parameters',ylabel='Selection-validation accuracy (%)',title='MNIST: explicit paper circuit (blue, seeds 21/22/23) and controls',xlim=(20,390),ylim=(85,100))
    ax.grid(alpha=.2); ax.legend(loc='lower right'); fig.tight_layout(); fig.savefig(figures/'mnist_resources_accuracy.png',dpi=180); plt.close(fig)
    text=f'''# 审查修复及完整规模复现核验

日期：2026-10-13。目标：Lu、Shen、Deng，Physical Review Applied 16, 044039 (2021)。

## 结论如何判定

三个审查问题已修复，并通过回归和参数恢复核验。修复后的代码可以训练高精度、较小的 MNIST 线路，但**不能据此宣布已完整复现论文所有结论**。应分别考察：给定论文线路的能力、MQNE 自行搜索的结果、在同等精度下的资源比较，以及 Cancer/SPT 的具体精度与代数。

{search_note}

论文 MNIST Fig. S11 的明确门结构已逐门转录，106 参数、28 个 R、22 个 CRx、15 块均吻合；Yao 位顺序下可以达到约 97% 的适应度。{mean_note}

七次完整搜索均完成 10 候选×6 代，合计 420 条线路；另有六次明确线路/固定结构对照，总计 {total} 条已完成线路，{updates:,} 次 Adam 更新。不是把配置准备、pilot 或单线路再训练算作完整搜索。最终指标在下表，包含失败结果；没有只报告最优编码/种子。

| 结果目录（results/ 下） | 类型 | 已训练线路 | 最优适应度 | 选型验证 | 额外测试 | 参数 | 块深度 |
|---|---|---|---|---|---|---|---|
'''+'\n'.join(rows)+f'''

适应度按训练+验证样本数加权，与论文报告的 fitness 对齐。表中最优线路按最终适应度选取，未用测试集选架构或选 epoch；最高历史验证值也未替代最终值。论文 MNIST/Cancer/SPT 目标分别为约 97%/94.6%/100%；SPT 的更严格目标是第三代、46 参数达到 100%，不能拿更深线路的近似高分代替它。

{phase_note}完整概率曲线与逐样本错误见下图及 `figures/spt_test_errors.json`，未据这些测试错误再次调参。

## 三项修复及核验

1. **固定数据身份**：每个任务保存固定训练池和验证池，pilot 取对应前缀，搜索种子独立于 split_seed=20261013。旧审查反例中的 SPT 30、MNIST 37 个训练→验证交叉，在新协议中均为 0；反方向交叉也为 0，见 `FIXED_SPLIT_AUDIT.json`。保存实际索引与来源指纹。后续线路训练核对来源、预处理、划分及训练/验证/测试交叉。
2. **可恢复下载**：MNIST 临时下载、gzip/IDX/标签/尺寸/既有 SHA 校验、原子替换、损坏缓存重新下载。模拟中断不留下最终文件或临时文件，污染缓存后仅重新获取对应文件并恢复正确内容，见回归日志。
3. **优化模式仍核验**：损失、准确率及有限值用显式异常检查。正常及 `python -O` 对被改为 0 的验证准确率都以 ValueError 非零退出，不生成预测/核验文件，见 `fix_evidence_20261013/corruption_regression.json`。所有完成运行另由 PyTorch 恢复参数，优化模式下核对训练、验证、测试指标，见 `finished_result_checks.json`。

12 项物理/协议/后端测试通过；5 项协议检查也在优化模式下通过。Yao 约定另与独立小端 Kronecker 稠密矩阵对照，输出辅助位位于最高编号。Numba 后端的概率、损失、梯度、相同 batch 流及 Adam 更新，与原 PyTorch 自动微分后端核对。它融合 R、逆向重构中间态以降低执行成本，使用 float64/complex128，未缩减样本数、epoch、候选或代数。

## 位顺序与资源比较

原实现 wire 0 为 MSB，Yao 低编号量子位为 LSB。在相同最近邻门结构及末端输出位下，直接复用振幅向量会把输入特征分配给不同物理线。新增显式 `input_bit_order="yao_lsb"` 映射并写入数据清单。相同 Fig. S11 线路、种子 21、样本、预算，原 MSB 验证 82.8%，Yao 映射后 96.2%，说明该约定是实际影响因素。该修正根据 [Yao 官方文档](https://docs.yaoquantum.org/v0.3/man/registers/) 推导，不能证明作者使用了与这里完全相同的缩放、展开和预处理。SPT 全部 2500 个生成基态在位反转下只有全局相位差，最低反射保真度大于 1-1e-13，故此改动不改变其分类问题。

额外的 10 层固定结构对照为 90 个 R、80 个 CRx、350 参数、30 块，其验证 98.0%、测试 99.0%。论文明确线路用 106 参数、15 块，参数少约 69.7%，块深度少 50%，验证水平接近但测试仍有差距。这是本实现中的结构对照，不是作者原始基线的精确复现；未以低精度短线路证明资源优势。

## 统计与复现边界

- 验证集参与架构搜索，因此不能作为独立盲测。Cancer 400+169 用满 569 个样本，无额外测试；SPT 额外生成 500 个新 lambda 的测试态；MNIST 额外测试来自官方测试池 500 个样本。不同种子/编码实验读取同一测试池后，表中比较应视为本轮探索结果，不能在后续调参后继续称为全新盲测。
- Cancer 的原始特征、minmax、Yao-minmax 是三个公开列出的预处理诊断；均仅用训练集拟合变换。作者没有指定完整预处理，本轮不宣称这些就是其设定，也不能把本轮未达标当作论文结论被推翻。
- SPT 种子 21 和 22 的完整搜索均列出，不能把其中更好的一次当成算法稳定保证。历史 SPT 架构控制在新 lambda 池重训，旧池与新池无同值 lambda；其验证 98.0%、测试 97.6%、85 参数、11 块，只诊断表达能力，不等于新搜索或第三代 46 参数的论文结果。
- 默认训练角度和 CRx 都随机；重新初始化候选参数、不继承父权重。Training step 按 epoch 解释，SPT 有限尺寸简并基态、图像插值、Cancer 编码及论文门块编号仍存在未确认约定。随机/固定初始化的完整多种子对照、遗传算法、受限门库、硬件、噪声和量子加速尚未复现。
- `mnist_fixed_paper_seed21`（6 候选）和 `spt_fixed_paper_seed21`（13 候选）是加速前中止并替换的运行，原日志和记录分别归档于 `archive/logs_interrupted/` 和 `archive/results_interrupted/`，未纳入完整搜索表。Cancer 原始特征运行使用保存的 `archive/code_intermediate_20261013/native_backend_primitive.py` 旧版本；其结果已由独立 PyTorch 恢复核验。

![完整搜索逐代适应度]({display_root}/figures/fixed_search_comparison.png)

![MNIST 参数与验证精度]({display_root}/figures/mnist_resources_accuracy.png)

![SPT 额外测试状态的预测概率]({display_root}/figures/spt_phase_predictions.png)

## 交付与复查

源代码、配置、数据划分、最优明确门结构、参数、逐候选历史、逐样本预测、混淆矩阵、恢复核验、图和日志均保存在项目目录。原实现备份位于 `archive/code_before_fix_20261013/`，历史结果和原 PDF 保留。新的运行说明见 `README.md`，代码哈希及环境在 `fix_evidence_20261013/` 和 `environment_fixed_v2.json`。`verify_finished.py` 可重新核验所有完成结果，`make_fixed_report.py` 从实际 JSON 重生成本报告。

主文 [DOI](https://doi.org/10.1103/PhysRevApplied.16.044039)；本地 SI 第 IV-VI 节及 Fig. S11；作者门块库 [公开仓库](https://github.com/peixinshen/PhD-Thesis-Source-Code)。本报告是本地代码检查和实测，当前 Code Review 插件仅提供 PR CI 读取接口，未把它描述为插件自动审查。
'''
    (root/'REPRODUCTION_FIXED_20261013.md').write_text(text,encoding='utf-8')
    save_json(root/'results'/'fixed_run_index.json',results)
    print('Generated report for',total,'finished circuits')


if __name__=='__main__':main()
