# 历史归档（2026-10-13）

本目录集中保存已被替代的代码和实验材料。历史数值及文件内容未改写，不应作为最新修复后的复现结论。

| 目录 | 内容 |
|---|---|
| code_before_fix_20261013 | 修复前的原实现及原有备份哈希 |
| code_intermediate_20261013 | 旧计算后端、数据加载器中间版本 |
| results_legacy | 早期探索、旧划分实验和旧门库核验 |
| results_interrupted | 加速前中止的 MNIST、SPT 运行 |
| configs_legacy | 被替代或仅用于早期探索的配置 |
| reports | 旧复现报告、Code Review 报告、旧环境信息 |
| review_evidence_20261013 | 历史代码审查证据 |
| logs_interrupted | 两个中止运行的原日志 |

共归档 135 个文件。file_integrity.json 保存每个文件的原路径、新路径及 SHA256；path_mapping.json 保存目录映射。本次移动逐文件校验一致，当前 13 组结果的 166 个文件也逐文件确认未改变。

最新实现位于项目主目录，最新结论见 ../REPRODUCTION_FIXED_20261013.md。归档代码作为历史快照保存；恢复历史模型请使用主目录的 predict.py，并传入 archive/results_legacy/运行名。共享数据依然位于主目录 data/，避免重复存放。

delivery_sha256_before_organization.json 和 source_sha256_before_organization.json 是整理前的交付清单；原路径已迁移，当前清单位于 ../fix_evidence_20261013/。