# MQNE：2021 年论文实验的初步复现

目标论文：Zhide Lu, Pei-Xin Shen, Dong-Ling Deng, *Markovian Quantum Neuroevolution for Machine Learning*, **Physical Review Applied 16, 044039 (2021)**，DOI：<https://doi.org/10.1103/PhysRevApplied.16.044039>。

这是从正文和补充材料独立实现的 Python/PyTorch 版本；原论文使用 Julia/Yao.jl。目录中的两份原始 PDF 保持不变。当前实际运行结果见 `REPRODUCTION_REPORT.md`；不能把配置文件已准备好等同于全规模复现已完成。

## 文件

- `mqne.py`：门块库、合法连接、均匀随机游走、逐代扩展、可微精确态矢模拟、Adam、三类数据集。
- `configs/*_paper.json`：论文给出的规模与超参数，加上明确列出的实现假设。
- `configs/*_pilot.json`：缩小候选数和训练预算的初步实验。
- `test_mqne.py`：独立稠密矩阵、有限差分、谱、范数和扩展一致性检查。
- `audit.py`：补充材料 Fig. S2 对应的门块库/出度标度，以及论文路径编号的兼容性审计。
- `predict.py`：从保存的明确门结构和参数恢复预测，交叉核对指标并输出混淆矩阵及逐样本预测。
- `results/`：实际指标、每个候选的学习曲线、明确的线路门结构、参数、数据划分与环境记录。
- `data/`：MNIST 原始数据及 SHA256、生成的 SPT 态缓存。

## 运行

在本目录打开 PowerShell。Python 3.11 或更新版本可用；本次运行使用已有的 Python 3.13/PyTorch 2.11 环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_mqne
.\run.ps1 -Task cancer -Scale pilot -Python .\.venv\Scripts\python.exe
.\run.ps1 -Task spt -Scale pilot -Python .\.venv\Scripts\python.exe
.\run.ps1 -Task mnist -Scale pilot -Python .\.venv\Scripts\python.exe
```

已安装依赖时直接使用 `python`。运行全规模搜索：

```powershell
.\run.ps1 -Task cancer -Scale paper
.\run.ps1 -Task spt -Scale paper
.\run.ps1 -Task mnist -Scale paper
python audit.py
```

`-Device cuda` 可切换 GPU，需对应 CUDA 版本的 PyTorch。小批次小线路不一定比 CPU 快。当前按候选串行训练；原论文设想并行训练，本实现尚未复现其并行时间主张。完整搜索每个任务训练 60 条线路，SPT/MNIST 每个样本的态矢有 512 个复振幅；完整训练可能明显耗时。

重新训练一条已保存的明确线路（重新初始化参数，不载入搜索中的训练权重）：

```powershell
python mqne.py --config configs/cancer_paper.json --output results/cancer_retrain_new --circuit results/cancer_pilot_seed21/best_circuit.json
python predict.py --result results/spt_random6_seed21
```

输出目录要求不存在，以避免覆盖实验。每条候选结束后写入 `candidates.json` 和当前最优线路；最终写入 `summary.json` 和图片。中途中断可保留已完成候选，但当前不支持自动恢复未完成的搜索。

## 对齐论文的部分

| 任务 | 输入/总量子比特 | 首代长度 / 每代扩展 | 每代候选 / 保留数 | 训练 / 验证样本 | Adam 学习率 | batch | 训练长度 |
|---|---|---|---|---|---|---|---|
| cancer | 6 / 7 | 3 / 2 | 10 / 1 | 400 / 169 | 0.0015 | 20 | 150 |
| MNIST 1 vs 9 | 8 / 9 | 5 / 2 | 10 / 1 | 2000 / 500 | 0.0015 | 30 | 200 |
| SPT | 8 / 9 | 3 / 2 | 10 / 1 | 1600 / 400 | 0.0015 | 20 | 150 |

SPT 首代长度 3、每代扩展 2 根据补充材料第三代长度 7 的路径与实验描述推定。门集为 Rz-Rx-Rz 与双向最近邻 CRx；门块中的门互不重叠。新块每个门必须与前块发生重叠，禁止可合并的相邻 R 与同向 CRx。每一步在合法后继中均匀抽样，并不声称在所有完整路径之间均匀抽样。

首块固定为所有量子比特上的 R。首代长度包含首块；因此 MNIST 第六代为 5+5×2=15 块。门块深度是把一个 R 视为整体的逻辑深度；物理分解后 R 有三个单比特门，另行统计。

## 尚需验证的假设和限制

1. **门块计数/编号**：正文计数含全恒等块，7/9 比特是 896/6688；作者公开 notebook 删掉全恒等块，得到 895/6687。本实现采用后者并固定自己的字典序。公开 notebook 的排列方式和论文所列线路编号尚未完全对齐。`PAPER_PATHS` 仅供编号审计，不用于声称训练了作者指定线路。保存实际门结构比单独保存整数路径更可靠。
2. **数据预处理**：正文没有完整指定 cancer 的特征变换。本实现用 sklearn 内置 WDBC 原始 30 维特征，补零至 64 维，逐样本 L2 归一化；没有全数据标准化。MNIST 使用 Pillow 双线性缩至 16×16，再按行展开及归一化；原文没有指定缩放插值。随机划分采用分层抽样，种子 21。
3. **SPT 数据**：实现严格使用周期边界 `H=-Σ X[j-1]Z[j]X[j+1]+λΣ Y[j]Y[j+1]`，λ 在 [0,2) 随机均匀采样，以 λ>1 为类 1。稀疏求解器取最低能量本征态，固定初始向量并记录残差；有限尺寸简并时具体基态选取可能与作者不同。
4. **训练长度歧义**：补充材料用词为 “training step”，图横轴为 “epochs”。默认把 150/200 解释为完整数据集的 epoch；设 `training_unit="step"` 可改成同样数量的随机 mini-batch 更新，成本和结果会不同。报告保存真实更新次数。
5. **初始化**：单比特角度均匀采样 [0,2π)，CRx 默认也随机初始化；设 `crx_initialization="zero"` 可将 CRx 初始角度置 0。首轮试跑采用过零初始化，原始失败结果保留在 results。原文未完整指定 CRx 的初始化。`initialization="fixed"` 在所有候选重用同一参数随机种子，`random` 每条候选改变种子。两者均重新训练；不继承父代训练权重。结构的包含关系本身不保证有限次 Adam 的实际精度单调提升。
6. **适应度**：按 SI “averaged on ... training ... and ... validation” 采用按样本数加权的训练+验证准确率。设 `fitness="validation"` 可仅按验证准确率选型。搜索会使用验证集，因此它不能作为独立测试精度。Cancer 的 400+169 已用完 569 个样本，没有另造独立测试集；SPT 的 400 也是参与选型的留出集。MNIST 另外使用官方测试集的 500 个样本，仅在最终选型完成后评估。
7. **停止条件**：默认完成 6 代（阈值 1.01），避免得到偶然高分时提前终止影响代数对比。可将 `fitness_threshold` 设为实际期望阈值。
8. **模拟边界**：理想精确态矢概率，无有限 shots、噪声或真实量子硬件；这里没有证明量子加速或复现原文平行运行时间。尚未实现 SI 的遗传算法对照及受限门块库 Fig. S3。

附加配置 `cancer_standard_*` 是预处理探索：仅以训练集均值和标准差对 30 维特征做标准化，再补零并进行幅度编码。`cancer_standard_depth13.json` 直接搜索长度 13 的线路（3 候选、1 代、150 epoch），是诊断性结构探索，不是论文逐代演化过程。`spt_full_data50.json` 用论文样本数训练 50 epoch；它与 MNIST 的单线路再训练只验证已发现架构，不能代替完整搜索。

下一阶段应先核对作者训练代码、原始门块编号和预处理，再进行完整 6 代搜索、随机/固定初始化对照和多随机种子统计。仅一次缩小规模运行不能确认论文报告的 97%、94.6%、100%。

## 参考

- 正文：本目录 `MarkovianQuantumNeuroevolution_DLD2110PRA.pdf`，尤其 II-IV 节、Table II 和 Fig. 2-3。
- 补充材料：本目录 `SI_MQNE.pdf`，Algorithm S1、IV-VI 节、Fig. S1-S2。
- 作者公开代码：<https://github.com/peixinshen/PhD-Thesis-Source-Code>，`GateBlockLibraryScaling.ipynb`。该仓库介绍训练结果，但当前公开文件不提供完整的 MQNE 训练流水线。
- MNIST 原始下载来源：<https://storage.googleapis.com/cvdf-datasets/mnist/>。

