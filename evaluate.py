import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import Config
from models.dit_classifier import DiTClassifier

def evaluate():
    print("========== Starting Evaluation ==========")
    device = Config.DEVICE
    
    # 1. Load test features and optimal mask
    test_feat_path = os.path.join(Config.CACHE_DIR, "features_test.npy")
    test_lbl_path = os.path.join(Config.CACHE_DIR, "labels_test.npy")
    mask_path = os.path.join(Config.CACHE_DIR, "optimal_mask.npy")
    
    if not os.path.exists(test_feat_path) or not os.path.exists(mask_path):
        print("Test features or optimal mask not found. Run train.py first.")
        return
        
    features_test = np.load(test_feat_path)
    labels_test = np.load(test_lbl_path)
    optimal_mask = np.load(mask_path)
    
    print(f"Loaded {features_test.shape[0]} test samples.")
    
    # Apply mask
    masked_test = features_test * optimal_mask
    
    # 2. Setup DataLoader
    test_dataset = TensorDataset(torch.tensor(masked_test, dtype=torch.float32), 
                                 torch.tensor(labels_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=Config.FINAL_BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_dit_model.pth")
    if not os.path.exists(model_path):
        print("Best model checkpoint not found. Run train.py first.")
        return
        
    model = DiTClassifier(
        feature_dim=Config.FEATURE_DIM,
        hidden_dim=Config.FINAL_DIT_HIDDEN_DIM,
        depth=Config.FINAL_DIT_DEPTH,
        num_heads=Config.FINAL_DIT_HEADS,
        mlp_ratio=Config.FINAL_DIT_MLP_RATIO
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Predict
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy() # probability of positive class
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(y_batch.numpy())
            
    # 5. Calculate Metrics
    acc = accuracy_score(all_targets, all_preds) * 100 # percentage
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    from sklearn.metrics import matthews_corrcoef
    import scipy.stats as st
    
    print("\n========== Test Set Metrics ==========")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # Specificity
    TN, FP, FN, TP = confusion_matrix(all_targets, all_preds).ravel()
    specificity = TN / (TN + FP)
    print(f"Specificity: {specificity:.4f}")

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(all_targets, all_preds)
    print(f"MCC: {mcc:.4f}")

    # 95% Confidence Interval for Accuracy
    n = len(all_targets)
    acc_raw = accuracy_score(all_targets, all_preds)
    ci_low, ci_high = st.binom.interval(0.95, n, acc_raw)
    print(f"95% CI: ({ci_low/n:.4f}, {ci_high/n:.4f})")
    
    # Save to results.txt
    results_path = os.path.join(Config.LOG_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy:  {acc:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"95% CI: ({ci_low/n:.4f}, {ci_high/n:.4f})\n")
    
    # 6. Comparison Table
    print("\n========== Comparison with Previous Works ==========")
    data = [
        ["AlexNet", "89.90%"],
        ["CNN Global Attention", "96.50%"],
        ["Multi-CNN+CCA", "99.10%"],
        ["EfficientNet+Transformer", "99.87%"],
        ["Hybrid BOA-WOA + DiT (Ours)", f"{acc:.2f}%"]
    ]
    df = pd.DataFrame(data, columns=["Model", "Accuracy"])
    print(df.to_string(index=False))
    
    # 7. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Adenocarcinoma'], 
                yticklabels=['Benign', 'Adenocarcinoma'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(Config.LOG_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    print(f"\nSaved confusion matrix to {cm_path}")
    print("==========================================")

if __name__ == "__main__":
    evaluate()
