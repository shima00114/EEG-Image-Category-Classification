import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. 脳波（EEG）画像カテゴリ分類モデル
# ==========================================
class EEGNetClassifier(nn.Module):
    def __init__(self, num_channels=64, num_time_steps=250, num_classes=50, dropout_rate=0.4):
        super(EEGNetClassifier, self).__init__()
        
        # 時系列畳み込み (カーネルサイズ: 5)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # 空間畳み込み (電極・チャンネル方向)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )
        
        # 特徴抽出層 (深層畳み込み)
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )
        
        # 分類ヘッド
        out_time_steps = num_time_steps // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * out_time_steps, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 入力形状: (Batch, Channels, Time_Steps) -> (Batch, 1, Channels, Time_Steps)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. 学習ルーチン (train.py)
# ==========================================
def train_model(
    batch_size=256,
    epochs=50,
    learning_rate=1e-3,
    label_smoothing=0.1,
    num_channels=64,
    num_time_steps=250,
    num_classes=50,
    save_dir="checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # サンプルデータの生成 (前処理済みデータセット読み込みに置換可能)
    # 入力形状: (サンプル数, チャンネル数, タイムステップ数)
    dummy_x = torch.randn(1000, num_channels, num_time_steps)
    dummy_y = torch.randint(0, num_classes, (1000,))
    
    dataset = TensorDataset(dummy_x, dummy_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデルとハイパーパラメータ設定
    model = EEGNetClassifier(
        num_channels=num_channels,
        num_time_steps=num_time_steps,
        num_classes=num_classes,
        dropout_rate=0.4
    ).to(device)

    # Loss (Label Smoothing) & Optimizer (AdamW) & Scheduler (CosineAnnealingLR)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    print("学習を開始します...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        # チェックポイント保存 (最良モデル)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, checkpoint_path)
            print(f"  -> チェックポイントを更新・保存しました: {checkpoint_path}")

    print("学習処理が正常に終了しました。")

if __name__ == "__main__":
    train_model()
