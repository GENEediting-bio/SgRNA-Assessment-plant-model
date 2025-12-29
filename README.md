# Nucleotide Transformer 回归模型微调与评估框架

基于 Nucleotide Transformer 预训练模型的回归任务微调与评估框架，用于预测 CRISPR sgRNA 活性等生物序列相关数值。本框架支持完整的训练流程和专业的科学绘图输出。

## 特性

### 🧬 模型支持
- 支持多种 Nucleotide Transformer 变体 (NT, ESM, DNABERT 等)
- 灵活的模型架构，可扩展附加特征
- 支持冻结预训练骨干网络

### 📊 数据处理
- 自动检测序列列和目标列
- 智能处理数值特征
- 自动处理缺失值和异常值
- 支持多种输入格式

### 🔄 训练流程
- 完整的训练-验证-测试流程
- 学习率调度器支持
- 早停机制
- 自动保存最佳模型检查点

### 📈 评估与可视化
- 全面的回归评估指标
- 专业科学绘图（PDF格式）
- 多种可视化分析
- 结果可重复性保证

## 项目结构

```
nucleotide-transformer-regression/
├── train_nt_regression.py      # 主训练脚本
├── evaluate_nt_regression.py   # 评估与可视化脚本
├── requirements.txt            # 依赖包列表
├── README.md                   # 本文档
├── checkpoints/               # 模型保存目录
├── sci_plots_pdf/             # 可视化输出目录
└── data/                      # 数据目录（示例）
    ├── train.csv
    ├── dev.csv
    └── test.csv
```

## 快速开始

### 安装依赖

```bash
# 安装基础依赖
pip install torch transformers pandas numpy scipy scikit-learn tqdm matplotlib seaborn
```

或者使用提供的 requirements.txt：

```bash
pip install -r requirements.txt
```

### 数据准备

#### 输入文件格式

模型需要三个 CSV 文件：**训练集、验证集、测试集**。CSV 文件应包含以下列：

**必需列：**
- `sequence` 或 `seq`：DNA/RNA 序列字符串（如："ATCGATCGAT"）
- **目标列**：包含要预测的数值标签（如："CRISPRscan"、"Doench2016_RuleSet2" 等）

**可选列：**
- 任何数值列用户可根据数据特征自行计算，计算结果将自动作为辅助特征使用

#### 示例 CSV 格式

以您提供的 `test.csv` 为例：
|----------|------------|
| AGTTGGTGATTATCTGTAGG | 6 |
| GAGCATGTGTGCTACGTGCA | 7 |
| GTTGAACTTGGAGCAATGAT | 0 |

在这个例子中：
- `sequence`：序列列（必需）
- `CRISPRscan`：目标列（您要预测的值）
- 其他数值列（`EPI`, `Doench2016_RuleSet2`, `E-CRISP`, `DeepCRISPR_Approx`, `CRISPOR_Specificity`）：将作为辅助特征

### 训练模型

```bash
python train_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --train_csv ./data/train.csv \
  --dev_csv ./data/dev.csv \
  --test_csv ./data/test.csv \
  --target_col CRISPRscan \
  --epochs 10 \
  --batch_size 16 \
  --lr 5e-5 \
  --max_length 100 \
  --ckpt_dir ./checkpoints
```

### 评估和可视化

```bash
python evaluate_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --test_csv ./data/test.csv \
  --ckpt_path ./checkpoints/best_model.pth \
  --target_col CRISPRscan \
  --output_dir ./sci_plots_pdf
```

## 详细参数说明

### 训练脚本参数 (`train_nt_regression.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name` | str | **必需** | HuggingFace 模型名称或本地路径 |
| `--train_csv` | str | **必需** | 训练集 CSV 文件路径 |
| `--dev_csv` | str | **必需** | 验证集 CSV 文件路径 |
| `--test_csv` | str | **必需** | 测试集 CSV 文件路径 |
| `--target_col` | str | "CRISPRscan" | 目标列名称 |
| `--batch_size` | int | 16 | 训练批量大小 |
| `--epochs` | int | 10 | 训练轮数 |
| `--lr` | float | 5e-5 | 学习率 |
| `--max_length` | int | 100 | 序列最大长度 |
| `--ckpt_dir` | str | "checkpoints" | 模型保存目录 |
| `--freeze_backbone` | flag | False | 冻结预训练模型参数 |

### 评估脚本参数 (`evaluate_nt_regression.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name` | str | **必需** | HuggingFace 模型名称或本地路径 |
| `--test_csv` | str | **必需** | 测试集 CSV 文件路径 |
| `--ckpt_path` | str | **必需** | 模型权重文件路径 |
| `--target_col` | str | "CRISPRscan" | 目标列名称 |
| `--max_length` | int | 100 | 序列最大长度 |
| `--output_dir` | str | "sci_plots_pdf" | 输出目录 |

## 输出文件说明

### 训练过程输出

#### 1. 终端输出
```
Epoch 5/10
  [Train] Loss: 0.0321
  [Val]   MSE: 0.0356 | MAE: 0.1521 | R2: 0.8523
          Pearson: 0.9234 | Spearman: 0.9125
  >>> 新的最佳模型已保存 (Pearson: 0.9234)
```

#### 2. 模型检查点
```
checkpoints/
└── best_model.pth    # PyTorch 模型权重文件（最佳模型）
```

### 评估过程输出

#### 1. 预测结果文件
```
sci_plots_pdf/
└── prediction_results.csv    # 详细的预测结果
```

**prediction_results.csv 示例：**
| sequence | true_value | predicted_value |
|----------|------------|-----------------|
| AGTTGGTGATTATCTGTAGG | 0.83 | 0.812 |
| GAGCATGTGTGCTACGTGCA | 1.00 | 0.956 |
| GTTGAACTTGGAGCAATGAT | 0.35 | 0.324 |

#### 2. 科学可视化图表（PDF格式）

| 文件名 | 图表类型 | 说明 |
|--------|----------|------|
| **Fig1_DensityScatter.pdf** | 密度散点图 | 预测值与真实值的散点图，包含回归线和主要指标 |
| **Fig2_Residuals.pdf** | 残差图 | 残差分析，检查模型偏差 |
| **Fig3_Distribution.pdf** | 分布对比图 | 预测值与真实值分布的核密度估计 |
| **Fig4_QuartileBoxplot.pdf** | 四分位箱线图 | 按真实值四分位数分组的预测性能 |
| **Fig5_MetricsBar.pdf** | 指标柱状图 | 主要评估指标的柱状图展示 |


## 评估指标解释

| 指标 | 范围 | 解释 | 适用场景 |
|------|------|------|----------|
| **MSE/RMSE** | [0, +∞) | 均方误差/均方根误差，惩罚大误差 | 数值精确度要求高 |
| **MAE** | [0, +∞) | 平均绝对误差，直观误差大小 | 稳健性要求高 |
| **R²** | (-∞, 1] | 决定系数，模型解释力 | 模型拟合优度 |
| **Pearson R** | [-1, 1] | 线性相关系数 | 线性趋势预测 |
| **Spearman R** | [-1, 1] | 等级相关系数 | 排序/排名预测 |

## 使用示例

### 示例1：预测 CRISPRscan 分数

```bash
# 1. 训练模型
python train_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --train_csv ./data/train.csv \
  --dev_csv ./data/dev.csv \
  --test_csv ./data/test.csv \
  --target_col CRISPRscan \
  --epochs 15 \
  --batch_size 32

# 2. 评估模型
python evaluate_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --test_csv ./data/test.csv \
  --ckpt_path ./checkpoints/best_model.pth \
  --target_col CRISPRscan \
  --output_dir ./results_CRISPRscan
```

### 示例2：预测 Doench2016_RuleSet2 分数（使用附加特征）

```bash
# 训练时自动使用其他数值列作为特征
python train_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --train_csv ./data/train.csv \
  --dev_csv ./data/dev.csv \
  --test_csv ./data/test.csv \
  --target_col Doench2016_RuleSet2 \
  --freeze_backbone \  # 小数据集建议冻结骨干
  --lr 1e-4
```

### 示例3：批量评估多个模型

```bash
#!/bin/bash
# evaluate_all.sh
MODEL_NAMES=("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

TARGETS=("CRISPRscan" "Doench2016_RuleSet2" "E-CRISP")

for MODEL in "${MODEL_NAMES[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    echo "Evaluating $MODEL on $TARGET..."
    python evaluate_nt_regression.py \
      --model_name "$MODEL" \
      --test_csv ./data/test.csv \
      --ckpt_path "./checkpoints/${MODEL##*/}_${TARGET}.pth" \
      --target_col "$TARGET" \
      --output_dir "./results/${MODEL##*/}_${TARGET}"
  done
done
```

## 进阶配置

### 自定义模型配置

```python
# 在代码中修改模型架构
class CustomRegressionModel(nn.Module):
    def __init__(self, model_name, num_numerical_features=0, dropout=0.1):
        super().__init__()
        # 自定义回归头
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),  # 增加隐藏层维度
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
```

### 自定义绘图风格

```python
def set_custom_style():
    """自定义绘图样式"""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减小批量大小
   --batch_size 8
   
   # 减小序列长度
   --max_length 50
   
   # 使用混合精度训练
   # 在代码中添加 torch.cuda.amp.autocast()
   ```

2. **目标列不存在**
   ```bash
   # 检查 CSV 文件列名
   head -n 1 data/train.csv
   
   # 确保 --target_col 参数正确
   --target_col CRISPRscan  # 不是 CRISPRScan 或 CRISPR_SCAN
   ```

3. **模型加载失败**
   ```bash
   # 确保模型名称正确
   --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species
   
   # 使用本地模型
   --model_name "./local_models/nucleotide-transformer"
   ```

4. **绘图时警告**
   ```bash
   # 安装完整依赖
   pip install seaborn==0.12.2 matplotlib==3.7.1
   
   # 更新到最新版本
   pip install --upgrade matplotlib seaborn
   ```

### 性能优化建议

- **大数据集**：使用全量微调，增大批量大小
- **小数据集**：冻结骨干网络，使用数据增强
- **长序列**：适当增大 `--max_length`，但注意内存使用
- **多特征**：确保特征与目标列相关性高

## 引用

如使用本框架，请引用：

```bibtex
@software{nt_regression_framework,
  title = {Nucleotide Transformer Regression Framework for CRISPR sgRNA Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/nucleotide-transformer-regression},
  note = {A comprehensive framework for fine-tuning nucleotide transformers for regression tasks}
}

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

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 支持

如有问题，请：
1. 查看 [Issues](https://github.com/yourusername/nucleotide-transformer-regression/issues) 页面
2. 提交新的 Issue
3. 或联系：your.email@example.com

---

**科学、严谨、可重复** - 为生物信息学研究提供专业工具
