import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns

# --------------------- Configuration ---------------------
FEATURE_FOLDER = r"c:\Users\KIIT0001\Desktop\IIT BBSR\trident\features_uniformer_mix - other"
INPUT_DIM = 512
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# --------------------- Dataset ---------------------
class FeatureDataset(Dataset):
    def __init__(self, root_folder):
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if not os.path.isdir(root_folder):
            print(f"Error: Root folder not found at {root_folder}")
            return

        for class_name in sorted(os.listdir(root_folder)):
            class_folder = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_folder):
                continue

            if class_name not in self.class_to_idx:
                idx = len(self.class_to_idx)
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name

            label_idx = self.class_to_idx[class_name]

            for file in os.listdir(class_folder):
                if file.endswith((".pt", ".npy")):
                    self.samples.append((os.path.join(class_folder, file), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        if file_path.endswith(".pt"):
            x = torch.load(file_path)
        else:
            x = torch.tensor(np.load(file_path), dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x, label

# --------------------- Padding Function ---------------------
def collate_with_padding(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.LongTensor(labels)
    return features_padded, labels

# --------------------- Model ---------------------
class VariableLengthClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        projected_x = torch.relu(self.feature_proj(x))
        _, hn = self.gru(projected_x)
        sequence_embedding = hn.squeeze(0)
        return self.fc(sequence_embedding)

# --------------------- Evaluation ---------------------
def get_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for x_padded, y in dataloader:
            x_padded, y = x_padded.to(device), y.to(device)
            outputs = model(x_padded)
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(y.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

            total_samples += y.size(0)
            correct_predictions += (predicted == y).sum().item()

    accuracy = (correct_predictions / total_samples) * 100
    return np.array(all_labels), np.array(all_scores), accuracy

def plot_roc_curve(y_true, y_score, num_classes, class_names, title):
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

# --------------------- Main ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = FeatureDataset(FEATURE_FOLDER)

if len(dataset) == 0:
    raise ValueError(f"No data found in {FEATURE_FOLDER}")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_padding)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_with_padding)

num_classes = len(dataset.class_to_idx)
model = VariableLengthClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"✅ Found {len(dataset)} samples: {len(train_dataset)} train, {len(test_dataset)} test")

# --------------------- Training ---------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_padded, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x_padded, y = x_padded.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x_padded)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if len(train_loader) > 0:
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

print("✅ Training Completed!")

# --------------------- Evaluation ---------------------
def evaluate_model(loader, dataset_name="Test"):
    labels, scores, accuracy = get_predictions(model, loader, device)
    preds = np.argmax(scores, axis=1)

    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    print(f"\n--- {dataset_name} Metrics ---")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    plot_confusion_matrix(labels, preds, dataset.idx_to_class, f"{dataset_name} Confusion Matrix")
    plot_roc_curve(labels, scores, num_classes, dataset.idx_to_class, f"ROC Curve - {dataset_name} Set")

# Evaluate on training set
evaluate_model(train_loader, "Train")

# Evaluate on test set
evaluate_model(test_loader, "Test")
