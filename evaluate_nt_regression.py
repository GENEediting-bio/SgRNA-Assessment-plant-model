import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ================= 绘图风格设置 (SCI 标准 PDF) =================
def set_sci_style():
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

# ================= 模型定义 =================
class NTRegressionModel(nn.Module):
    def __init__(self, model_name, num_numerical_features=0, dropout=0.1):
        super(NTRegressionModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        if hasattr(self.backbone.config, "hidden_size"):
            self.hidden_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "d_model"):
            self.hidden_dim = self.backbone.config.d_model
        else:
            dummy_input = torch.zeros((1, 10), dtype=torch.long)
            with torch.no_grad():
                out = self.backbone(dummy_input)
                self.hidden_dim = out[0].shape[-1]
        
        self.num_numerical_features = num_numerical_features
        if num_numerical_features > 0:
            self.numerical_proj = nn.Sequential(
                nn.Linear(num_numerical_features, 64),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            combined_dim = self.hidden_dim + 64
        else:
            combined_dim = self.hidden_dim

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, numerical_features=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]
            
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        seq_repr = sum_embeddings / sum_mask

        if self.num_numerical_features > 0 and numerical_features is not None:
            num_repr = self.numerical_proj(numerical_features)
            final_repr = torch.cat((seq_repr, num_repr), dim=1)
        else:
            final_repr = seq_repr

        score = self.regressor(final_repr)
        return score.squeeze(-1)

# ================= 数据集类 =================
class CRISPRDataset(Dataset):
    def __init__(self, sequences, features, labels, tokenizer, max_length=100):
        self.sequences = sequences
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        label = float(self.labels[idx]) if self.labels is not None else 0.0
        
        encoding = self.tokenizer(
            seq, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }
        if self.features is not None and len(self.features) > 0:
            item["numerical_features"] = torch.tensor(self.features[idx], dtype=torch.float)
        return item

# ================= 数据读取 =================
def read_data(file_path, target_col):
    df = pd.read_csv(file_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    
    if 'sequence' in df.columns: seq_col = 'sequence'
    elif 'seq' in df.columns: seq_col = 'seq'
    else: seq_col = df.columns[0]

    df = df.dropna(subset=[seq_col, target_col])
    
    sequences = df[seq_col].tolist()
    labels = df[target_col].astype(float).tolist()
    
    exclude = [seq_col, target_col]
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude]
    
    if len(feature_cols) > 0:
        features = df[feature_cols].fillna(0).values.tolist()
    else:
        features = []
        
    return sequences, features, labels, len(feature_cols)

# ================= 绘图函数 (已修复 edgecolor 问题) =================

def plot_density_scatter(y_true, y_pred, metrics, output_path, target_name):
    plt.figure(figsize=(7, 6))
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = np.array(y_true)[idx], np.array(y_pred)[idx], z[idx]
    except:
        x, y = y_true, y_pred
        z = 'blue'

    # === 修复点在这里: edgecolor='none' ===
    plt.scatter(x, y, c=z, s=30, cmap='Spectral_r', alpha=0.8, edgecolor='none')
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.6, label='Ideal Fit')
    
    poly = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(poly)
    plt.plot(y_true, p(y_true), "r-", alpha=0.5, lw=2, label='Regression Line')

    textstr = '\n'.join((
        r'$\mathbf{R^2 = %.3f}$' % (metrics["r2"], ),
        r'$Pearson\ R = %.3f$' % (metrics["pearson"], ),
        r'$Spearman\ R = %.3f$' % (metrics["spearman"], ),
        r'$RMSE = %.3f$' % (np.sqrt(metrics["mse"]), ),
        r'$MAE = %.3f$' % (metrics["mae"], )
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.xlabel(f'Experimental {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title('Predicted vs Experimental', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 密度散点图已保存: {output_path}")

def plot_residuals(y_true, y_pred, output_path, target_name):
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=residuals, alpha=0.7, color='#3b5b92', edgecolor='w', s=50)
    plt.axhline(0, color='#d62728', linestyle='--', lw=2)
    plt.xlabel(f'Experimental {target_name}')
    plt.ylabel('Residuals (Exp - Pred)')
    plt.title('Residual Analysis', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 残差图已保存: {output_path}")

def plot_distribution_comparison(y_true, y_pred, output_path, target_name):
    plt.figure(figsize=(7, 5))
    sns.kdeplot(y_true, fill=True, color="#1f77b4", label='Experimental', alpha=0.3, linewidth=2)
    sns.kdeplot(y_pred, fill=True, color="#ff7f0e", label='Predicted', alpha=0.3, linewidth=2)
    plt.xlabel(f'{target_name} Score')
    plt.ylabel('Density')
    plt.title('Distribution Comparison', fontweight='bold', pad=15)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 分布对比图已保存: {output_path}")

def plot_quartile_boxplot(y_true, y_pred, output_path, target_name):
    df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    try:
        df['Group'] = pd.qcut(df['True'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    except ValueError:
        df['Group'] = pd.cut(df['True'], bins=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])

    plt.figure(figsize=(7, 6))
    sns.boxplot(x='Group', y='Predicted', data=df, palette="viridis", width=0.6, linewidth=1.5)
    plt.xlabel('Experimental Quartiles')
    plt.ylabel(f'Predicted {target_name}')
    plt.title('Performance by Quartiles', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 箱线图已保存: {output_path}")

def plot_metrics_bar(metrics, output_path):
    names = ['R²', 'Pearson R', 'Spearman R']
    values = [metrics['r2'], metrics['pearson'], metrics['spearman']]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    plt.figure(figsize=(6, 5))
    bars = plt.bar(names, values, color=colors, alpha=0.8, width=0.6, edgecolor='black', linewidth=1)
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.title('Model Performance Metrics', fontweight='bold', pad=15)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 指标柱状图已保存: {output_path}")

# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="CRISPRscan")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="sci_plots_pdf")
    args = parser.parse_args()

    set_sci_style() 
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(">>> 正在加载数据...")
    seqs, feats, labels, num_feats = read_data(args.test_csv, args.target_col)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    print(">>> 正在初始化模型架构 (CPU)...")
    model = NTRegressionModel(args.model_name, num_numerical_features=num_feats)
    
    print(f">>> 正在加载权重: {args.ckpt_path} (到 CPU)...")
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    del state_dict
    
    print(">>> 正在将完整模型移动到 GPU...")
    model.to(device)
    model.eval()
    
    torch.cuda.empty_cache()

    ds = CRISPRDataset(seqs, feats, labels, tokenizer, args.max_length)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print(">>> 正在进行推理...")
    with torch.no_grad():
        for batch in tqdm(dl):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            num_f = batch.get("numerical_features", None)
            if num_f is not None: num_f = num_f.to(device)
            
            preds = model(input_ids, mask, num_f)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    metrics = {
        "mse": mean_squared_error(all_labels, all_preds),
        "mae": mean_absolute_error(all_labels, all_preds),
        "r2": r2_score(all_labels, all_preds),
        "pearson": pearsonr(all_labels, all_preds)[0],
        "spearman": spearmanr(all_labels, all_preds)[0]
    }
    
    res_df = pd.DataFrame({
        "sequence": seqs, 
        "true_value": all_labels, 
        "predicted_value": all_preds
    })
    res_df.to_csv(os.path.join(args.output_dir, "prediction_results.csv"), index=False)

    print("\n>>> 正在生成 PDF 图表...")
    
    plot_density_scatter(
        all_labels, all_preds, metrics, 
        os.path.join(args.output_dir, "Fig1_DensityScatter.pdf"), 
        args.target_col
    )
    
    plot_residuals(
        all_labels, all_preds,
        os.path.join(args.output_dir, "Fig2_Residuals.pdf"),
        args.target_col
    )
    
    plot_distribution_comparison(
        all_labels, all_preds,
        os.path.join(args.output_dir, "Fig3_Distribution.pdf"),
        args.target_col
    )
    
    plot_quartile_boxplot(
        all_labels, all_preds,
        os.path.join(args.output_dir, "Fig4_QuartileBoxplot.pdf"),
        args.target_col
    )
    
    plot_metrics_bar(
        metrics,
        os.path.join(args.output_dir, "Fig5_MetricsBar.pdf")
    )

    print(f"\n✅ 所有 PDF 结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
