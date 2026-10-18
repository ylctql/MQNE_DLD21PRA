# MQNE：2021 Physical Review Applied 数值实验复现

论文：Zhide Lu, Pei-Xin Shen, Dong-Ling Deng，*Markovian Quantum Neuroevolution for Machine Learning*，**Physical Review Applied 16, 044039 (2021)**，<https://doi.org/10.1103/PhysRevApplied.16.044039>。这里的 PRA 实际是 Physical Review Applied。

本目录是根据正文、补充材料和作者公开门块库独立实现的 Python 版本。最新修复与实测结果见 **REPRODUCTION_FIXED_20261013.md**。旧报告 `archive/reports/REPRODUCTION_REPORT.md` 和旧结果集中存放于 `archive/`，作为历史探索记录；旧 SPT 96.75% 有跨运行训练/验证样本交叉，不能作为修复后的结论。两份论文 PDF 保持不变。

## 文档导航

- [代码架构与关键功能实现](docs/架构与实现说明.md)：整体流程、模块职责、数据划分、门库、计算后端和模型恢复。
- [使用与结果解读](docs/使用与结果解读.md)：环境准备、运行命令、配置参数、输出文件和常见问题。

## 审查问题的修复

- `datasets_v2.py` 保存固定训练池和验证池，小规模运行只取各池前缀。划分种子与架构搜索种子独立；保存数据指纹、母划分指纹和实际索引。SPT 另生成 500 个独立测试样本，MNIST 使用官方测试池。Cancer 569 个样本按 400/169 分配，没有独立测试池。
- MNIST 下载先写唯一临时文件，校验 gzip、IDX 头、尺寸、标签和既有 SHA256 后原子改名；损坏缓存会重新获取，中断会清理临时文件。首次下载的 SHA256 是校验后建立的本地基准，并非第三方签名。
- `predict.py` 用显式 `ValueError` 核验损失、准确率及有限值，在 `python -O` 下仍有效；读取保存的实际数据索引，全部分区核验通过后才写预测文件。
- 继续训练旧线路时检查来源、划分、预处理和数据角色。没有稳定划分记录的旧架构会被拒绝，建议重新搜索。

## 运行

在本目录打开 PowerShell，Python 3.11 或更新版本。本次实际使用 Python 3.13.6，环境明细见 `environment_fixed_v2.json`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest test_mqne test_fixes
.\.venv\Scripts\python.exe -O -m unittest test_fixes.ProtocolTests
.\run.ps1 -Task cancer -Scale fixed -Python .\.venv\Scripts\python.exe
.\run.ps1 -Task spt -Scale fixed -Python .\.venv\Scripts\python.exe
.\run.ps1 -Task mnist -Scale fixed -Python .\.venv\Scripts\python.exe
```

`-Scale paper` 与 `fixed` 选同一组完整规模配置，`pilot` 选旧小规模配置但使用新的固定划分。完整配置默认使用 CPU 的 Numba 精确伴随梯度后端；`-Device cuda` 显式改用 PyTorch 后端。首次 Numba 编译需额外等待。本实现没有验证 GPU 实验结果或并行时间主张。

直接运行及恢复预测：

```powershell
python mqne.py --config configs/mnist_yao_paper.json --output results/mnist_new
python mqne.py --config configs/mnist_fixed_paper.json --output results/mnist_msb_new
python run_reference.py --config configs/mnist_yao_paper.json --output results/s11_new --architecture paper_s11 --seed 21
python predict.py --result results/mnist_paper_s11_yao_seed21
python -O predict.py --result results/mnist_paper_s11_yao_seed21
```

输出目录必须不存在。每个候选完成后保存记录和当前最优线路，搜索结束才生成 `summary.json`。无 summary 的目录是未完成运行，不应计入完整复现。当前不支持中途恢复搜索。

## 实验与实现约定

| 任务 | 输入/总量子比特 | 首代块数/扩展块数 | 每代候选/保留 | 训练/验证 | Adam 学习率 | batch | epoch/候选 |
|---|---|---|---|---|---|---|---|
| Cancer | 6/7 | 3/2 | 10/1 | 400/169 | 0.0015 | 20 | 150 |
| MNIST 1 vs 9 | 8/9 | 5/2 | 10/1 | 2000/500 | 0.0015 | 30 | 200 |
| SPT | 8/9 | 3/2 | 10/1 | 1600/400 | 0.0015 | 20 | 150 |

完整搜索执行 6 代，每个任务 60 条候选。SPT 首代长度 3 由补充材料第三代线路长度 7 推定。训练长度按图示解释为 epoch；原文 training step 的用词仍有歧义。设 `training_unit="step"` 可选择相应数量的随机 batch 更新，结果不能与 epoch 预算混同。

门集为 Rz-Rx-Rz 和双向最近邻 CRx，门块中门不重叠。每个新门必须与上一块重叠，禁止可合并的相邻 R 和同向 CRx。首块是所有比特上的 R，在合法后继中均匀抽样。7/9 比特排除全恒等块后为 895/6687，与作者公开 notebook 相符。自己的编号尚未完全对齐论文路径，所以保存明确门结构；`run_reference.py` 从 Fig. S11 逐门转录 MNIST 线路，28 个 R、22 个 CRx、106 参数、15 块均核对通过。

`mqne.py` 内部 wire 0 是最高有效位，最后一线是 |0> 输出辅助位。Yao 的低编号量子位是最低有效位，因此 `input_bit_order="yao_lsb"` 对输入振幅做位反转映射；保存的 manifest 记录这一约定。`mnist_yao_paper.json` 为此约定，`mnist_fixed_paper.json` 保留原 MSB 约定作为对照。仅编号的映射有文档依据，不代表已经确定作者的图像展开方式。MNIST 仍采用 Pillow 双线性 16×16、按行展开、L2 归一化，论文没有指定完整插值和布局细节。SPT 所生成状态在位反转下仅差全局相位，两个编码预测等价。

Cancer 默认原始 30 维 WDBC 特征补零至 64，再 L2 归一化。minmax/standard 配置仅为预处理诊断，统计量只由训练样本计算，不是已确认的作者预处理。SPT 使用周期边界 `H=-sum(X[j-1] Z[j] X[j+1])+lambda sum(Y[j]Y[j+1])`、8 个自旋、lambda 均匀取自 [0,2)、lambda>1 标为 1；记录本征态残差。有限尺寸简并下的基态选取可能与原实验不同。

角度均匀初始化于 [0,2π)，CRx 默认随机；候选参数重新训练、不继承父参数。适应度为训练和验证准确率按样本数加权的平均值。验证集参与架构选择，因此验证准确率不能当作盲测。最终测试在架构选完后评估；多种编码、种子实验已读取的同一测试子集只能作为本轮比较数据，不能称为后续调参的全新盲测。

## 核验与边界

`test_mqne.py` 校验门、有限差分、独立哈密顿量、库计数、后继和零角扩展；`test_fixes.py` 校验三个修复、数据恢复、Yao 独立稠密矩阵及 Numba/PyTorch 的概率、梯度、Adam 一致性。Numba 使用 float64/complex128，融合 R 并逆向重构中间态，改变执行成本，不改变线路和训练预算。已保存后端对照日志；最终模型由 PyTorch 另行恢复核验。

资源节约需要在相近精度下比较。找到少参数但低精度的线路不能验证论文的资源优势。本轮未复现完整遗传算法、受限门库、量子硬件、噪声、shots 或量子加速结论。

参考：正文及 SI 原 PDF；作者公开库 <https://github.com/peixinshen/PhD-Thesis-Source-Code>；Yao 位顺序 <https://docs.yaoquantum.org/v0.3/man/registers/>；MNIST 原数据 <https://storage.googleapis.com/cvdf-datasets/mnist/>。

## 目录整理

- `results/`：当前 13 组完整实验结果；本次整理不改动实验记录或数值。
- `archive/`：旧代码、早期探索结果、中止运行、旧配置、历史报告和审查材料，详细位置见 [归档说明](archive/README.md)。
- `configs/`：当前完整规模配置及仍可运行的三份 pilot 配置。
- `data/`：运行及恢复预测所需数据。两份旧 SPT 缓存继续保留，供恢复归档结果使用。

恢复旧结果时使用当前 `predict.py` 并指定归档路径，例如：

```powershell
python predict.py --result archive/results_legacy/spt_random6_seed21
```
