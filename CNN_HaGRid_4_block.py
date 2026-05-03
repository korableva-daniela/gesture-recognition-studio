# train_4blocks_optimized.py
"""
CNN с 4 сверточными блоками для распознавания жестов рук
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ для скорости
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import time
from datetime import datetime
import gc
import warnings

warnings.filterwarnings('ignore')

# ==================== НАСТРОЙКИ (ОПТИМИЗИРОВАННЫЕ) ====================
DATA_DIR = 'hagrid_uint8'
BATCH_SIZE = 32  # Уменьшил с 64 до 32 для скорости
NUM_EPOCHS = 15  # Уменьшил с 30 до 15
LEARNING_RATE = 0.001
NUM_WORKERS = 2  # Увеличил с 0 до 2
GRADIENT_CLIP = 1.0
PATIENCE = 5  # Уменьшил patience для ранней остановки

# ==================== ПРОВЕРКА GPU ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print(" CNN С 4 СВЕРТОЧНЫМИ БЛОКАМИ (ОПТИМИЗИРОВАННАЯ)")
print("=" * 80)
print(f"\nPyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
print(f"Using device: {device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Num workers: {NUM_WORKERS}")


# ==================== ДАТАСЕТ (ОПТИМИЗИРОВАННЫЙ) ====================
class FastMemoryDataset(Dataset):
    """Оптимизированный датасет с кэшированием"""

    def __init__(self, data_dir, subset='train'):
        self.data_dir = data_dir
        self.subset = subset

        # Загружаем метаданные
        metadata_path = os.path.join(data_dir, 'metadata.npy')
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.num_classes = metadata['num_classes']
        self.gesture_map = metadata['gesture_map']

        # Находим все файлы
        self.X_files = sorted([f for f in os.listdir(data_dir)
                               if f.startswith(f'X_{subset}') and f.endswith('.npy')])
        self.y_files = sorted([f for f in os.listdir(data_dir)
                               if f.startswith(f'y_{subset}') and f.endswith('.npy')])

        # Создаем индекс
        self.indices = []
        for file_idx, X_file in enumerate(self.X_files):
            X_path = os.path.join(data_dir, X_file)
            X_data = np.load(X_path, mmap_mode='r')
            size = X_data.shape[0]
            for i in range(size):
                self.indices.append((file_idx, i))

        self.total_size = len(self.indices)
        print(f"  {subset.upper()}: {self.total_size:,} изображений")

        # Кэш для файлов
        self.cache = {}
        self.cache_size = 3

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        file_idx, local_idx = self.indices[idx]

        if file_idx not in self.cache:
            if len(self.cache) >= self.cache_size:
                oldest = next(iter(self.cache.keys()))
                del self.cache[oldest]
                gc.collect()

            X_path = os.path.join(self.data_dir, self.X_files[file_idx])
            y_path = os.path.join(self.data_dir, self.y_files[file_idx])

            self.cache[file_idx] = (
                np.load(X_path, mmap_mode='r'),
                np.load(y_path, mmap_mode='r')
            )

        X_data, y_data = self.cache[file_idx]
        img = X_data[local_idx].copy()
        label = y_data[local_idx].copy()

        # Преобразование в тензор
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return img, label


# ==================== 4-БЛОЧНАЯ АРХИТЕКТУРА ====================
class FourBlockGestureCNN(nn.Module):
    """CNN с 4 сверточными блоками"""

    def __init__(self, num_classes=18, input_channels=3):
        super(FourBlockGestureCNN, self).__init__()

        # Блок 1: 128x128 → 64x64
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Блок 2: 64x64 → 32x32
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Блок 3: 32x32 → 16x16
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Блок 4: 16x16 → 8x8 → Global Average Pooling
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.25)
        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss / len(loader):.3f}',
            'Acc': f'{100. * correct / total:.1f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', ncols=100, leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss / len(loader):.3f}',
                'Acc': f'{100. * correct / total:.1f}%'
            })

    return running_loss / len(loader), 100. * correct / total


def plot_training_history(history, test_acc, best_val_acc, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[0].set_title('Точность (Accuracy)')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_loss'], label='Train', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Потери (Loss)')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'4-блочная CNN | Тест: {test_acc:.2f}% | Val: {best_val_acc:.2f}%')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ==================== MAIN ====================
def main():
    print("\n" + "=" * 80)
    print(" ЗАГРУЗКА ДАННЫХ")
    print("=" * 80)

    train_dataset = FastMemoryDataset(DATA_DIR, 'train')
    val_dataset = FastMemoryDataset(DATA_DIR, 'val')
    test_dataset = FastMemoryDataset(DATA_DIR, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    num_classes = train_dataset.num_classes
    print(f"\nКлассов: {num_classes}")
    print(f"Train: {len(train_dataset):,} изображений")
    print(f"Val:   {len(val_dataset):,} изображений")
    print(f"Train batches: {len(train_loader):,}")

    print("\n" + "=" * 80)
    print(" СОЗДАНИЕ МОДЕЛИ")
    print("=" * 80)

    model = FourBlockGestureCNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("\n" + "=" * 80)
    print(" НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 80)

    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'=' * 50}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
            }, f'models/best_model_4blocks_{timestamp}.pth')

            torch.save(model, f'models/best_model_4blocks_full_{timestamp}.pth')
            print(f"  ✓ Сохранена (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n Early stopping на эпохе {epoch + 1}")
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\nОбучение завершено за {total_time / 60:.1f} минут")

    # Тестирование
    print("\n" + "=" * 80)
    print(" ОЦЕНКА НА ТЕСТЕ")
    print("=" * 80)

    model.load_state_dict(torch.load(f'models/best_model_4blocks_{timestamp}.pth')['model_state_dict'])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataset, desc='Testing'):
            inputs = inputs.unsqueeze(0).to(device)
            labels = torch.tensor([labels]).to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    plot_training_history(history, test_acc, best_val_acc,
                          f'logs/training_4blocks_{timestamp}.png')

    print("\n" + "=" * 80)
    print(" ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    main()