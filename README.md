# Nucleotide Transformer Regression Model Fine-tuning and Evaluation Framework

A fine-tuning and evaluation framework for regression tasks based on the Nucleotide Transformer pre-trained model, used for predicting CRISPR sgRNA activity and other biological sequence-related numerical values. This framework supports a complete training pipeline and professional scientific plotting output.

## Features

### 🧬 Model Support
- Supports various Nucleotide Transformer variants (NT, ESM, DNABERT, etc.)
- Flexible model architecture, expandable with additional features
- Supports freezing pre-trained backbone network

### 📊 Data Processing
- Automatic detection of sequence column and target column
- Intelligent handling of numerical features
- Automatic handling of missing values and outliers
- Supports multiple input formats

### 🔄 Training Pipeline
- Complete train-validation-test pipeline
- Learning rate scheduler support
- Early stopping mechanism
- Automatic saving of best model checkpoints

### 📈 Evaluation and Visualization
- Comprehensive regression evaluation metrics
- Professional scientific plotting (PDF format)
- Multiple visualization analyses
- Result reproducibility guarantee

## Project Structure

```
nucleotide-transformer-regression/
├── train_nt_regression.py      # Main training script
├── evaluate_nt_regression.py   # Evaluation and visualization script
├── requirements.txt            # Dependency list
├── README.md                   # This document
├── checkpoints/               # Model save directory
├── sci_plots_pdf/             # Visualization output directory
└── data/                      # Data directory (example)
    ├── train.csv
    ├── dev.csv
    └── test.csv
```

## Quick Start

### Install Dependencies

```bash
# Install basic dependencies
pip install torch transformers pandas numpy scipy scikit-learn tqdm matplotlib seaborn
```

Or use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

### Data Preparation

#### Input File Format

The model requires three CSV files: **training set, validation set, test set**. CSV files should contain the following columns:

**Required columns:**
- `sequence` or `seq`: DNA/RNA sequence string (e.g., "ATCGATCGAT")
- **Target column**: Contains the numerical labels to predict (e.g., "CRISPRscan", "Doench2016_RuleSet2", etc.)

**Optional columns:**
- Any numerical columns users can compute based on data characteristics, results will automatically be used as auxiliary features

#### Example CSV Format

Using your provided `test.csv` as an example:
| sequence | Features... |
|----------|------------|
| AGTTGGTGATTATCTGTAGG | 6 |
| GAGCATGTGTGCTACGTGCA | 7 |
| GTTGAACTTGGAGCAATGAT | 0 |

In this example:
- `sequence`: Sequence column (required)
- `CRISPRscan`: Target column (the value you want to predict)
- Other numerical columns (`EPI`, `Doench2016_RuleSet2`, `E-CRISP`, `DeepCRISPR_Approx`, `CRISPOR_Specificity`): Will be used as auxiliary features

### Train Model

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

### Evaluate and Visualize

```bash
python evaluate_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --test_csv ./data/test.csv \
  --ckpt_path ./checkpoints/best_model.pth \
  --target_col CRISPRscan \
  --output_dir ./sci_plots_pdf
```

## Detailed Parameter Description

### Training Script Parameters (`train_nt_regression.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_name` | str | **Required** | HuggingFace model name or local path |
| `--train_csv` | str | **Required** | Training set CSV file path |
| `--dev_csv` | str | **Required** | Validation set CSV file path |
| `--test_csv` | str | **Required** | Test set CSV file path |
| `--target_col` | str | "CRISPRscan" | Target column name |
| `--batch_size` | int | 16 | Training batch size |
| `--epochs` | int | 10 | Training epochs |
| `--lr` | float | 5e-5 | Learning rate |
| `--max_length` | int | 100 | Maximum sequence length |
| `--ckpt_dir` | str | "checkpoints" | Model save directory |
| `--freeze_backbone` | flag | False | Freeze pre-trained model parameters |

### Evaluation Script Parameters (`evaluate_nt_regression.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_name` | str | **Required** | HuggingFace model name or local path |
| `--test_csv` | str | **Required** | Test set CSV file path |
| `--ckpt_path` | str | **Required** | Model weight file path |
| `--target_col` | str | "CRISPRscan" | Target column name |
| `--max_length` | int | 100 | Maximum sequence length |
| `--output_dir` | str | "sci_plots_pdf" | Output directory |

## Output File Description

### Training Process Output

#### 1. Terminal Output
```
Epoch 5/10
  [Train] Loss: 0.0321
  [Val]   MSE: 0.0356 | MAE: 0.1521 | R2: 0.8523
          Pearson: 0.9234 | Spearman: 0.9125
  >>> New best model saved (Pearson: 0.9234)
```

#### 2. Model Checkpoints
```
checkpoints/
└── best_model.pth    # PyTorch model weight file (best model)
```

### Evaluation Process Output

#### 1. Prediction Results File
```
sci_plots_pdf/
└── prediction_results.csv    # Detailed prediction results
```

**prediction_results.csv example:**
| sequence | true_value | predicted_value |
|----------|------------|-----------------|
| AGTTGGTGATTATCTGTAGG | 0.83 | 0.812 |
| GAGCATGTGTGCTACGTGCA | 1.00 | 0.956 |
| GTTGAACTTGGAGCAATGAT | 0.35 | 0.324 |

#### 2. Scientific Visualization Charts (PDF Format)

| Filename | Chart Type | Description |
|----------|------------|-------------|
| **Fig1_DensityScatter.pdf** | Density Scatter Plot | Scatter plot of predicted vs. true values, including regression line and main metrics |
| **Fig2_Residuals.pdf** | Residual Plot | Residual analysis to check model bias |
| **Fig3_Distribution.pdf** | Distribution Comparison Plot | Kernel density estimation of predicted vs. true value distributions |
| **Fig4_QuartileBoxplot.pdf** | Quartile Boxplot | Prediction performance grouped by true value quartiles |
| **Fig5_MetricsBar.pdf** | Metrics Bar Chart | Bar chart display of main evaluation metrics |

## Evaluation Metrics Explanation

| Metric | Range | Interpretation | Applicable Scenarios |
|--------|-------|----------------|----------------------|
| **MSE/RMSE** | [0, +∞) | Mean Squared Error/Root Mean Squared Error, penalizes large errors | High numerical precision requirements |
| **MAE** | [0, +∞) | Mean Absolute Error, intuitive error magnitude | High robustness requirements |
| **R²** | (-∞, 1] | Coefficient of determination, model explanatory power | Model goodness of fit |
| **Pearson R** | [-1, 1] | Linear correlation coefficient | Linear trend prediction |
| **Spearman R** | [-1, 1] | Rank correlation coefficient | Ranking/ordering prediction |

## Usage Examples

### Example 1: Predicting CRISPRscan Scores

```bash
# 1. Train model
python train_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --train_csv ./data/train.csv \
  --dev_csv ./data/dev.csv \
  --test_csv ./data/test.csv \
  --target_col CRISPRscan \
  --epochs 15 \
  --batch_size 32

# 2. Evaluate model
python evaluate_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --test_csv ./data/test.csv \
  --ckpt_path ./checkpoints/best_model.pth \
  --target_col CRISPRscan \
  --output_dir ./results_CRISPRscan
```

### Example 2: Predicting Doench2016_RuleSet2 Scores (Using Additional Features)

```bash
# Automatically use other numerical columns as features during training
python train_nt_regression.py \
  --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --train_csv ./data/train.csv \
  --dev_csv ./data/dev.csv \
  --test_csv ./data/test.csv \
  --target_col Doench2016_RuleSet2 \
  --freeze_backbone \  # Recommend freezing backbone for small datasets
  --lr 1e-4
```

### Example 3: Batch Evaluate Multiple Models

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

## Advanced Configuration

### Custom Model Configuration

```python
# Modify model architecture in code
class CustomRegressionModel(nn.Module):
    def __init__(self, model_name, num_numerical_features=0, dropout=0.1):
        super().__init__()
        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 512),  # Increase hidden layer dimension
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
```

### Custom Plotting Style

```python
def set_custom_style():
    """Custom plotting style"""
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

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 8
   
   # Reduce sequence length
   --max_length 50
   
   # Use mixed precision training
   # Add torch.cuda.amp.autocast() in code
   ```

2. **Target Column Doesn't Exist**
   ```bash
   # Check CSV file column names
   head -n 1 data/train.csv
   
   # Ensure --target_col parameter is correct
   --target_col CRISPRscan  # Not CRISPRScan or CRISPR_SCAN
   ```

3. **Model Loading Failed**
   ```bash
   # Ensure model name is correct
   --model_name InstaDeepAI/nucleotide-transformer-2.5b-multi-species
   
   # Use local model
   --model_name "./local_models/nucleotide-transformer"
   ```

4. **Warnings During Plotting**
   ```bash
   # Install complete dependencies
   pip install seaborn==0.12.2 matplotlib==3.7.1
   
   # Update to latest versions
   pip install --upgrade matplotlib seaborn
   ```

### Performance Optimization Suggestions

- **Large datasets**: Use full fine-tuning, increase batch size
- **Small datasets**: Freeze backbone network, use data augmentation
- **Long sequences**: Appropriately increase `--max_length`, but be mindful of memory usage
- **Multiple features**: Ensure features have high correlation with target column

## Citation

If using this framework, please cite:

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

## License

This project uses the MIT License. See the [LICENSE](LICENSE) file for details.

## Contribution Guidelines

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

If you encounter issues, please:
1. Check the [Issues](https://github.com/yourusername/nucleotide-transformer-regression/issues) page
2. Submit a new Issue
3. Or contact: your.email@example.com

---

**Scientific, Rigorous, Reproducible** - Providing professional tools for bioinformatics research
