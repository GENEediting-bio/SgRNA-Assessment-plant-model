import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ================= 自定义回归模型 =================
class NTRegressionModel(nn.Module):
    def __init__(self, model_name, num_numerical_features=0, dropout=0.1):
        super(NTRegressionModel, self).__init__()
        
        # 加载预训练的 Nucleotide Transformer Backbone
        print(f"正在加载预训练模型: {model_name} ...")
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # 获取隐藏层维度
        if hasattr(self.backbone.config, "hidden_size"):
            self.hidden_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "d_model"): # 某些 ESM/NT 变体
            self.hidden_dim = self.backbone.config.d_model
        else:
            # 尝试通过一次前向传播推断维度
            print("无法从配置获取 hidden_size，尝试自动推断...")
            dummy_input = torch.zeros((1, 10), dtype=torch.long)
            with torch.no_grad():
                out = self.backbone(dummy_input)
                # out[0] is usually last_hidden_state
                self.hidden_dim = out[0].shape[-1]
        
        print(f"Backbone 隐藏层维度: {self.hidden_dim}")
        
        # 数值特征处理层 (如果有额外特征)
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

        # 回归头 (Regression Head)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # 回归任务：输出维度为 1
        )

    def forward(self, input_ids, attention_mask, numerical_features=None):
        # 1. 获取序列特征
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取 last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]
            
        # 池化策略：取 Mean Pooling 作为序列的整体表示
        # (Batch, Seq_Len, Hidden) -> (Batch, Hidden)
        # 忽略 padding 部分
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        seq_repr = sum_embeddings / sum_mask

        # 2. 融合数值特征 (如果有)
        if self.num_numerical_features > 0 and numerical_features is not None:
            num_repr = self.numerical_proj(numerical_features)
            final_repr = torch.cat((seq_repr, num_repr), dim=1)
        else:
            final_repr = seq_repr

        # 3. 预测分数
        score = self.regressor(final_repr)
        return score.squeeze(-1) # (Batch, 1) -> (Batch,)

# ================= 数据集处理 =================
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
        label = float(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }
        
        if self.features is not None and len(self.features) > 0:
            item["numerical_features"] = torch.tensor(self.features[idx], dtype=torch.float)
            
        return item

# ================= 工具函数：读取数据 =================
def read_data(file_path, target_col):
    """
    读取 CSV，自动识别 sequence 列，指定 target 列，其余数值列作为特征
    """
    print(f"读取文件: {file_path}")
    try:
        # 自动推断分隔符
        df = pd.read_csv(file_path, sep=None, engine='python')
    except Exception as e:
        raise ValueError(f"读取CSV失败: {e}")

    # 清洗列名空格
    df.columns = df.columns.str.strip()
    
    # 确定序列列
    if 'sequence' in df.columns:
        seq_col = 'sequence'
    elif 'seq' in df.columns:
        seq_col = 'seq'
    else:
        # 如果找不到，假设第一列是序列
        seq_col = df.columns[0]
        print(f"警告: 未找到 'sequence' 列，默认使用第一列 '{seq_col}' 作为序列。")

    # 检查目标列
    if target_col not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"错误: 在文件中未找到目标列 '{target_col}'。\n可用列: {available}")

    # 数据清洗：删除目标列或序列列为空的行
    initial_len = len(df)
    df = df.dropna(subset=[seq_col, target_col])
    if len(df) < initial_len:
        print(f"  - 已删除 {initial_len - len(df)} 行包含空值的记录")

    # 提取数据
    sequences = df[seq_col].tolist()
    labels = df[target_col].astype(float).tolist()
    
    # 提取额外特征 (除了序列和目标列之外的所有数值列)
    exclude = [seq_col, target_col]
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude]
    
    if len(feature_cols) > 0:
        print(f"  - 检测到辅助数值特征 ({len(feature_cols)}列): {feature_cols}")
        # 填充特征列的空值为 0
        features = df[feature_cols].fillna(0).values.tolist()
    else:
        print("  - 未检测到辅助数值特征，仅使用序列。")
        features = []

    return sequences, features, labels, len(feature_cols)

# ================= 训练与评估流程 =================
def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        numerical_features = None
        if "numerical_features" in batch:
            numerical_features = batch["numerical_features"].to(device)
            
        optimizer.zero_grad()
        
        # Forward
        preds = model(input_ids, attention_mask, numerical_features)
        
        # Loss (MSE)
        loss = loss_fn(preds, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix({"mse_loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            numerical_features = None
            if "numerical_features" in batch:
                numerical_features = batch["numerical_features"].to(device)
                
            preds = model(input_ids, attention_mask, numerical_features)
            loss = loss_fn(preds, labels)
            
            total_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    # 计算多种指标
    if len(all_preds) > 1:
        # 1. 相关性指标 (关注趋势)
        # 处理常数预测导致的相关系数计算警告
        try:
            pearson_corr, _ = pearsonr(all_labels, all_preds)
        except:
            pearson_corr = 0.0
            
        try:
            spearman_corr, _ = spearmanr(all_labels, all_preds)
        except:
            spearman_corr = 0.0
        
        # 2. 误差指标 (关注准确度)
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        
        # 3. 拟合优度 (关注解释力)
        r2 = r2_score(all_labels, all_preds)
    else:
        pearson_corr, spearman_corr, mse, mae, r2 = 0, 0, 0, 0, 0
        
    # 返回所有指标
    return avg_loss, mse, mae, r2, pearson_corr, spearman_corr

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser(description="Nucleotide Transformer Regression Finetuning")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace 模型名称或路径")
    parser.add_argument("--train_csv", type=str, required=True, help="训练集路径")
    parser.add_argument("--dev_csv", type=str, required=True, help="验证集路径")
    parser.add_argument("--test_csv", type=str, required=True, help="测试集路径")
    parser.add_argument("--target_col", type=str, default="CRISPRscan", help="要预测的目标列名 (e.g. CRISPRscan, Doench2016)")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--freeze_backbone", action="store_true", help="是否冻结预训练模型参数，只训练回归头")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 读取数据
    print(f"正在加载数据，目标列: {args.target_col} ...")
    train_seqs, train_feats, train_labels, num_feats = read_data(args.train_csv, args.target_col)
    val_seqs, val_feats, val_labels, _ = read_data(args.dev_csv, args.target_col)
    test_seqs, test_feats, test_labels, _ = read_data(args.test_csv, args.target_col)
    
    print(f"训练集大小: {len(train_seqs)}")
    if num_feats > 0:
        print(f"注意: 模型将使用 {num_feats} 个辅助特征进行预测。")
    
    # 2. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 3. 创建 Dataset 和 DataLoader
    train_ds = CRISPRDataset(train_seqs, train_feats, train_labels, tokenizer, args.max_length)
    val_ds = CRISPRDataset(val_seqs, val_feats, val_labels, tokenizer, args.max_length)
    test_ds = CRISPRDataset(test_seqs, test_feats, test_labels, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # 4. 初始化模型
    model = NTRegressionModel(args.model_name, num_numerical_features=num_feats).to(device)
    
    if args.freeze_backbone:
        print("提示: 已启用 --freeze_backbone，将冻结 Transformer 参数，只训练回归头。")
        for param in model.backbone.parameters():
            param.requires_grad = False
            
    # 5. 优化器与损失函数
    # 如果冻结了backbone，建议稍微调大一点学习率给 regression head，或者保持 1e-4
    # 如果全量微调，建议 1e-5 到 5e-5
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    best_val_score = -1.0 # 使用 Pearson R 作为最佳模型标准
    
    # 创建保存目录
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # 6. 训练循环
    print("开始训练...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, None, device, loss_fn)
        
        # 验证
        val_loss, val_mse, val_mae, val_r2, val_pearson, val_spearman = evaluate(model, val_loader, device, loss_fn)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  [Train] Loss: {train_loss:.4f}")
        print(f"  [Val]   MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | R2: {val_r2:.4f}")
        print(f"          Pearson: {val_pearson:.4f} | Spearman: {val_spearman:.4f}")
        
        # 这里的标准选用了 Pearson 相关系数 (Correlation)，因为 sgRNA 任务通常看重排序能力
        # 如果你更看重数值精准度，可以改成 if val_r2 > best_val_score
        current_score = val_pearson
        
        if current_score > best_val_score:
            best_val_score = current_score
            save_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> 新的最佳模型已保存 (Pearson: {best_val_score:.4f})")
            
    # 7. 最终测试
    print("\n" + "="*40)
    print("训练结束，正在加载最佳模型进行最终测试...")
    best_model_path = os.path.join(args.ckpt_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("未找到保存的模型，使用当前模型进行测试。")
        
    test_loss, test_mse, test_mae, test_r2, test_pearson, test_spearman = evaluate(model, test_loader, device, loss_fn)
    
    print(f"最终测试结果 (目标列: {args.target_col}):")
    print(f"  MSE Loss:    {test_mse:.4f}  (越低越好)")
    print(f"  MAE Loss:    {test_mae:.4f}  (越低越好)")
    print(f"  R2 Score:    {test_r2:.4f}   (越接近1越好)")
    print("-" * 30)
    print(f"  Pearson R:   {test_pearson:.4f} (线性相关性)")
    print(f"  Spearman R:  {test_spearman:.4f} (排名相关性)")
    print("="*40)

if __name__ == "__main__":
    main()
