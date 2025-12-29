# Nucleotide Transformer 回归模型微调

基于 Nucleotide Transformer 预训练模型的回归任务微调框架，用于预测 CRISPR sgRNA 活性等生物序列相关数值。

## 特性

- 🧬 支持多种 Nucleotide Transformer 变体 (NT, ESM, DNABERT 等)
- 📊 自动处理序列数据和数值特征
- 🔄 灵活的池化策略 (Mean Pooling)
- 📈 支持多种回归评估指标 (MSE, MAE, R², Pearson, Spearman)
- 🚀 可配置的训练参数 (批量大小、学习率、序列长度等)
- 💾 自动保存最佳模型检查点
- 🎯 适用于 sgRNA 效率预测、蛋白质表达量预测等任务

## 快速开始

### 安装依赖

```bash
pip install torch transformers pandas numpy scipy scikit-learn tqdm
```

### 数据准备

#### 输入文件格式
模型需要三个 CSV 文件：训练集、验证集、测试集。CSV 文件应包含以下列：

**必需列：**
- `sequence` 或 `seq`：DNA/RNA 序列字符串（如："ATCGATCGAT"）
- **目标列**：包含要预测的数值标签（如："CRISPRscan"、"Doench2016" 等）

**可选列：**
- 任何数值列（如："GC_content"、"length"、"melting_temp" 等）将自动作为辅助特征使用

#### 示例 CSV 文件
```
ATCGATCGAT,0.5,10,0.85
GCTAGCTAGC,0.6,10,0.92
TTTTAAAAAA,0.2,10,0.31
...
```

### 运行训练

```bash
python finetune_nt_pytorch_multifeature.py /
--model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species /
--train_csv data/train.csv /
--dev_csv data/dev.csv /
--test_csv data/test.csv /
--batch_size 32 /
--epochs 120 /
--lr 0.1 /
--max_length 64 /
--freeze_backbone /
--ckpt_dir lr_0.1 /
```

### 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name` | str | **必需** | HuggingFace 模型名称或本地路径 |
| `--train_csv` | str | **必需** | 训练集 CSV 文件路径 |
| `--dev_csv` | str | **必需** | 验证集 CSV 文件路径 |
| `--test_csv` | str | **必需** | 测试集 CSV 文件路径 |
| `--batch_size` | int | 16 | 训练批量大小 |
| `--epochs` | int | 10 | 训练轮数 |
| `--lr` | float | 5e-5 | 学习率 |
| `--max_length` | int | 100 | 序列最大长度（自动填充/截断） |
| `--ckpt_dir` | str | "checkpoints" | 模型保存目录 |
| `--freeze_backbone` | flag | False | 冻结预训练模型参数 |

### 支持的预训练模型

- `InstaDeepAI/nucleotide-transformer-2.5b-multi-species`

## 输出文件

### 1. 模型检查点
训练过程中会在指定目录（默认为 `checkpoints/`）保存最佳模型：

```
checkpoints/
└── best_model.pth    # PyTorch 模型权重文件
```

### 2. 终端输出
训练和评估过程中会显示详细指标：

```
最终测试结果 (目标列: CRISPRscan):
  MSE Loss:    0.0245  (越低越好)
  MAE Loss:    0.1256  (越低越好)
  R2 Score:    0.8732   (越接近1越好)
  ------------------------------
  Pearson R:   0.9356 (线性相关性)
  Spearman R:  0.9214 (排名相关性)
```

### 3. 训练日志
每个 epoch 的训练和验证结果：

```
Epoch 5/10
  [Train] Loss: 0.0321
  [Val]   MSE: 0.0356 | MAE: 0.1521 | R2: 0.8523
          Pearson: 0.9234 | Spearman: 0.9125
  >>> 新的最佳模型已保存 (Pearson: 0.9234)
```

## 输出指标解释

| 指标 | 范围 | 解释 | 适用场景 |
|------|------|------|----------|
| **MSE** | [0, +∞) | 均方误差，惩罚大误差 | 数值精确度要求高 |
| **MAE** | [0, +∞) | 平均绝对误差，直观误差大小 | 稳健性要求高 |
| **R²** | (-∞, 1] | 决定系数，模型解释力 | 模型拟合优度 |
| **Pearson R** | [-1, 1] | 线性相关系数 | 线性趋势预测 |
| **Spearman R** | [-1, 1] | 等级相关系数 | 排序/排名预测 |

## 进阶使用

### 仅使用序列（无额外特征）
如果 CSV 文件只有序列和目标列，模型会自动仅使用序列信息。

### 冻结骨干网络
对于小数据集，建议冻结预训练模型：

```bash
python train_nt_regression.py \
  --model_name "InstaDeepAI/nucleotide-transformer-500m-multi-species" \
  --train_csv ./data/train.csv \
  --dev_csv ./data/dev.csv \
  --test_csv ./data/test.csv \
  --target_col Doench2016 \
  --freeze_backbone \
  --lr 1e-4  # 冻结时可用稍大的学习率
```

### 自定义序列长度
根据任务调整序列最大长度：

```bash
--max_length 200  # 对于较长的 DNA 片段
```

## 故障排除

### 常见问题
1. **目标列不存在**：检查 `--target_col` 参数与 CSV 文件列名是否一致
2. **CUDA 内存不足**：减小 `--batch_size` 或 `--max_length`
3. **序列列未识别**：确保列名为 "sequence" 或 "seq"，或修改代码中的列名检测逻辑

### 性能调优建议
- **大数据集**：可尝试全量微调（默认）
- **小数据集**：建议使用 `--freeze_backbone`
- **预测准确性要求高**：关注 MSE、MAE、R² 指标
- **排序能力要求高**：关注 Pearson、Spearman 指标

## 引用

如使用本代码，请引用相关预训练模型和本框架：

```bibtex
@article{dalla2023nucleotide,
  title={Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza Revilla, Javier and Lopez Carranza, Nicolas and Henryk Grywaczewski, Adam and Oteri, Francesco and Dallago, Christian and Trop, Evan and Sirelkhatim, Hassan and Richard, Guillaume and others},
  journal={bioRxiv},
  pages={2023--01},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

## 许可证

MIT License
