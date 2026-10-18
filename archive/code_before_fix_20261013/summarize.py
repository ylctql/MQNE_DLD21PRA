"""Create reviewable plots and a Chinese report from finished experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mqne import Block, Library, save_json


def draw_circuit(circuit, output):
    blocks=circuit['blocks']; n=1+max(q for b in blocks for q in
                                     list(b['rotations'])+[x for p in b['pairs'] for x in p])
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
    ax.set_title(f"Readout: q{n-1}; {circuit['record']['parameters']} trainable parameters")
    for spine in ax.spines.values():spine.set_visible(False)
    fig.tight_layout();fig.savefig(output/'circuit.png',dpi=180);plt.close(fig)


def main():
    results=[]
    for path in sorted(Path('results').glob('*/summary.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        config=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        circuit=json.loads((path.parent/'best_circuit.json').read_text(encoding='utf-8'))
        draw_circuit(circuit,path.parent)
        candidates=json.loads((path.parent/'candidates.json').read_text(encoding='utf-8'))
        library=Library({'cancer':7,'spt':9,'mnist':9}[result['task']])
        for r in candidates:
            indexes=[i-1 for i in r['path_1based']]
            assert all(library.allowed(library.blocks[a],library.blocks[b]) for a,b in zip(indexes,indexes[1:]))
        for b,i in zip(circuit['blocks'],circuit['record']['path_1based']):
            assert library.blocks[i-1]==Block(tuple(tuple(p) for p in b['pairs']),tuple(b['rotations']))
        results.append((path.parent.name,result,config))
    rows=[]
    highlights=[]
    for name,r,c in results:
        b=r['best']; te=r['independent_test']
        if name=='spt_retrain_full_data_seed21':
            highlights.append(f"SPT 在论文规模的 1600/400 样本上，已选线路重新训练 50 epoch，验证准确率为 {100*b['final']['validation']['accuracy']:.2f}%；没有独立测试集。")
        if name=='mnist_retrain_full_data_seed21':
            highlights.append(f"MNIST 在论文规模的 2000/500 训练/验证样本上，已选线路重新训练 200 epoch，验证准确率 {100*b['final']['validation']['accuracy']:.2f}%，额外官方测试子集（500 样本）准确率 {100*te['accuracy']:.2f}%。")
        actual_generations=max(x['generation'] for x in json.loads((Path('results')/name/'candidates.json').read_text()))
        rows.append('| '+ ' | '.join([name,c['crx_initialization'],str(actual_generations),
                     str(r['circuits_trained']),str(c['epochs']),
                     f"{100*b['final']['train']['accuracy']:.2f}%",
                     f"{100*b['final']['validation']['accuracy']:.2f}%",
                     f"{100*te['accuracy']:.2f}%" if te else '无',
                     str(b['block_depth']),str(b['parameters']),f"{r['seconds']:.1f}"])+' |')
    text='''# 初步复现报告

日期：2026-10-12。目标是 2021 年 Physical Review Applied 16, 044039 的 MQNE 数值模拟实验。

## 当前结论

已建立可运行的三任务实现，并完成下表中列出的实际运行。门块标度、精确态矢模拟和自动微分有独立数值校验。**当前仍是初步复现，不代表已复现论文的 97% / 94.6% / 100% 精度。**完整 10 候选×6 代和两种单比特初始化的多种子对照尚未全部运行。

## 实际结果

| 运行目录 | CRx 初始化 | 代数 | 已训练线路 | 每条 epoch | 最优训练准确率 | 选型验证准确率 | 独立测试准确率 | 最优块数 | 参数 | 训练及绘图秒数 |
|---|---|---|---|---|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'\n\n'+'\n\n'.join(highlights)+'''

上述单线路再训练没有重新进行完整架构搜索；小规模搜索与后续全数据训练使用相同种子但划分样本数不同。Cancer 的 `standard` 变体和 `depth13` 直接结构采样均为诊断探索，尚未复现原文 94.6%。最优精度应连同样本数、训练预算和选型方式一起解读。

“最优”按训练+验证加权准确率选择，并非仅按验证集。表中验证集参与架构选型，不能作为盲测。Cancer 总共 569 个样本，按 400/169 划分，没有额外独立测试集；SPT 留出集同样参与选型。MNIST 独立测试来自官方测试集 500 个样本。

首轮 `*_pilot_seed21` 中 cancer/SPT 的 CRx 初始化为 0，均停留在多数类/随机水平。已保留原始输出，并进行了随机 CRx 初始化的后续搜索。不同运行的样本数、epoch 和代数并不完全相同，不能将差异直接归因于初始化；它们是排查和试跑，不是受控消融结论。

## 验证与可追溯性

- 六项检查全部通过：7/9 比特门块数和随机路径合法性；向量化后继与逐门规则一致；量子门与独立 Kronecker 稠密模拟一致并保持范数；自动微分对有限差分一致；周期 cluster–Ising 哈密顿量对独立稠密矩阵一致、λ=0 基态能量为 -8；零角扩展保持父线路输出。
- `results/library_audit/scaling.json` 和 `scaling.png` 完成 2 至 10 比特的门块库与出度标度。7/9 比特含恒等块计数分别为 896/6688，排除恒等块为 895/6687，与正文公式和公开 notebook 的约定相符。
- 各运行记录了真实 optimizer 更新数、耗时、数据索引、随机种子、最优明确门结构、数值参数和软件版本。`candidates.json` 保留所有候选的最终指标与学习曲线，`circuit.png` 显示实际选中的线路。
- `results/library_audit/published_path_numbering_audit.json` 表明，本实现的字典序编号不能直接解释正文/SI 的全部指定路径。因此没有把这些整数路径训练结果冒充作者指定线路。原始训练代码与编号表仍需获取。
- 精确态矢、无 shots 和噪声；候选串行训练。此次未复现量子硬件实验或并行速度主张。

## 下一步

1. 核实原始预处理、门块编号、CRx 初始化以及 “150 training steps” 是 150 mini-batch 还是 150 个 epoch。
2. 在确认约定后运行 `configs/*_paper.json` 的完整搜索，再做随机/固定初始化与至少三个随机种子的重复。
3. 采用独立测试数据报告最终泛化，并对齐 Fig. 2、Fig. 3、Fig. S1。补充遗传算法与受限门块库对照。

具体启动方式和所有实现假设见 `README.md`。参考论文和补充材料仍在同目录，未修改。
'''
    Path('REPRODUCTION_REPORT.md').write_text(text,encoding='utf-8')
    save_json(Path('results/run_index.json'),[dict(name=name,summary=r,config=c) for name,r,c in results])
    print(text)


if __name__=='__main__':main()
