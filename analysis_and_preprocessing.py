import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict
import gc
import psutil
import time
from datetime import datetime
from tqdm import tqdm  # Добавляем tqdm для прогресс-бара


# Для отслеживания памяти
def get_memory_usage():
    """Возвращает использование памяти в ГБ"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024


def format_time(seconds):
    """Форматирует время в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.1f} сек"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} мин"
    else:
        return f"{seconds / 3600:.1f} час"


print(f"Память до начала: {get_memory_usage():.2f} ГБ")


def analyze_hagrid_structure(base_path='data/hagrid-classification-512p'):
    """Анализирует структуру датасета Hagrid"""

    print("=" * 70)
    print(" АНАЛИЗ СТРУКТУРЫ ДАТАСЕТА HAGRID")
    print("=" * 70)

    if not os.path.exists(base_path):
        print(f"Путь не существует: {base_path}")
        return None, None

    gesture_folders = [d for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d))]
    gesture_folders.sort()

    print(f"\n НАЙДЕНО ЖЕСТОВ: {len(gesture_folders)}")

    gesture_stats = {}
    total_images = 0

    for gesture in gesture_folders:
        gesture_path = os.path.join(base_path, gesture)
        image_files = [f for f in os.listdir(gesture_path)
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        count = len(image_files)
        gesture_stats[gesture] = count
        total_images += count

        print(f"  {gesture}: {count} изображений")

    print(f"\n ИТОГОВАЯ СТАТИСТИКА:")
    print(f"  Всего жестов: {len(gesture_folders)}")
    print(f"  Всего изображений: {total_images:,}")

    return gesture_folders, gesture_stats


def create_dataset_splits(base_path='data/hagrid-classification-512p',
                          test_size=0.2,
                          val_size=0.1,
                          seed=42):
    print("\n" + "=" * 70)
    print(" СОЗДАНИЕ РАЗДЕЛЕНИЯ ДАТАСЕТА")
    print("=" * 70)

    # Получаем список жестов
    gesture_folders = [d for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d))]
    gesture_folders.sort()

    # Создаем маппинг
    gesture_map = {gesture: idx for idx, gesture in enumerate(gesture_folders)}
    gesture_map_inv = {idx: gesture for gesture, idx in gesture_map.items()}

    print(f"\n МАППИНГ ЖЕСТОВ НА МЕТКИ:")
    for gesture, label in list(gesture_map.items())[:5]:
        print(f"  {gesture} → класс {label}")
    print(f"  ... и еще {len(gesture_map) - 5} жестов")

    # Собираем все пути к изображениям
    print(f"\n Сбор путей к изображениям...")
    start_time = time.time()

    all_image_paths = []
    all_labels = []
    class_counts = {}

    for gesture in gesture_folders:
        gesture_path = os.path.join(base_path, gesture)
        label = gesture_map[gesture]

        image_files = [f for f in os.listdir(gesture_path)
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        class_counts[gesture] = len(image_files)

        for img_file in image_files:
            all_image_paths.append(os.path.join(gesture_path, img_file))
            all_labels.append(label)

        print(f"  {gesture}: {len(image_files)} изображений")

    total_images = len(all_image_paths)
    elapsed = time.time() - start_time
    print(f"\n Всего изображений: {total_images:,}")
    print(f"Время сбора путей: {format_time(elapsed)}")
    print(f"Память после сбора путей: {get_memory_usage():.2f} ГБ")

    # Разделение на train/val/test
    print(f"\n Разделение на train/val/test...")
    start_time = time.time()

    # Сначала выделяем тест
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_image_paths, all_labels,
        test_size=test_size,
        random_state=seed,
        stratify=all_labels
    )

    # Очищаем исходные списки для экономии памяти
    del all_image_paths, all_labels
    gc.collect()

    # Потом из оставшегося выделяем валидацию
    val_ratio = val_size / (1 - test_size)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val_labels
    )

    # Очищаем промежуточные данные
    del train_val_paths, train_val_labels
    gc.collect()

    elapsed = time.time() - start_time
    print(f"\n РАЗМЕРЫ ВЫБОРОК:")
    print(f"  Train: {len(train_paths):,} изображений")
    print(f"  Val:   {len(val_paths):,} изображений")
    print(f"  Test:  {len(test_paths):,} изображений")
    print(f"Время разделения: {format_time(elapsed)}")
    print(f"Память после разделения: {get_memory_usage():.2f} ГБ")

    return {
        'train_paths': train_paths,
        'val_paths': val_paths,
        'test_paths': test_paths,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'gesture_map': gesture_map,
        'gesture_map_inv': gesture_map_inv,
        'class_counts': class_counts
    }


def process_and_save_in_batches(split_data,
                                img_size=(128, 128),
                                output_dir='hagrid_uint8',  # Изменил название папки
                                batch_size=1000):
    """
    Обрабатывает и сохраняет датасет пакетами в формате uint8
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" ПАКЕТНАЯ ОБРАБОТКА ИЗОБРАЖЕНИЙ (UINT8)")
    print("=" * 70)
    print(f"Формат сохранения: uint8 (0-255) - экономия памяти в 4 раза")
    print(f"Размер изображений: {img_size}")

    # Функция для обработки одного пакета
    def process_batch(paths, labels, batch_num, total_batches, subset_name):
        X_batch = []
        y_batch = []

        for i, (path, label) in enumerate(zip(paths, labels)):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # СОХРАНЯЕМ В UINT8 (0-255) ВМЕСТО FLOAT32
                img = img.astype(np.uint8)  # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
                X_batch.append(img)
                y_batch.append(label)

        # Сохраняем пакет
        X_batch = np.array(X_batch, dtype=np.uint8)  # <--- ЯВНО УКАЗЫВАЕМ dtype
        y_batch = np.array(y_batch)

        batch_file = os.path.join(output_dir, f'X_{subset_name}_batch_{batch_num:04d}.npy')
        label_file = os.path.join(output_dir, f'y_{subset_name}_batch_{batch_num:04d}.npy')

        np.save(batch_file, X_batch)
        np.save(label_file, y_batch)

        return len(X_batch)

    # Обрабатываем train
    print(f"\n Обработка TRAIN ({len(split_data['train_paths']):,} изображений)...")
    train_batches = len(split_data['train_paths']) // batch_size + 1
    train_total = 0

    # Создаем прогресс-бар для train
    with tqdm(total=len(split_data['train_paths']), desc="Train", unit="img", ncols=100) as pbar:
        for i in range(0, len(split_data['train_paths']), batch_size):
            batch_start_time = time.time()

            batch_paths = split_data['train_paths'][i:i + batch_size]
            batch_labels = split_data['train_labels'][i:i + batch_size]
            batch_num = i // batch_size + 1

            processed = process_batch(batch_paths, batch_labels, batch_num, train_batches, 'train')
            train_total += processed

            # Обновляем прогресс-бар
            pbar.update(len(batch_paths))

            # Показываем дополнительную информацию
            if (batch_num) % 10 == 0 or batch_num == train_batches:
                pbar.set_postfix({
                    'Батч': f'{batch_num}/{train_batches}',
                    'Скорость': f'{len(batch_paths) / (time.time() - batch_start_time):.0f} img/s'
                })

            # Очищаем память после каждого пакета
            gc.collect()

    # Обрабатываем val
    print(f"\n Обработка VAL ({len(split_data['val_paths']):,} изображений)...")
    val_batches = len(split_data['val_paths']) // batch_size + 1

    with tqdm(total=len(split_data['val_paths']), desc="Val", unit="img", ncols=100) as pbar:
        for i in range(0, len(split_data['val_paths']), batch_size):
            batch_paths = split_data['val_paths'][i:i + batch_size]
            batch_labels = split_data['val_labels'][i:i + batch_size]
            batch_num = i // batch_size + 1

            process_batch(batch_paths, batch_labels, batch_num, val_batches, 'val')
            pbar.update(len(batch_paths))
            gc.collect()

    # Обрабатываем test
    print(f"\n Обработка TEST ({len(split_data['test_paths']):,} изображений)...")
    test_batches = len(split_data['test_paths']) // batch_size + 1

    with tqdm(total=len(split_data['test_paths']), desc="Test", unit="img", ncols=100) as pbar:
        for i in range(0, len(split_data['test_paths']), batch_size):
            batch_paths = split_data['test_paths'][i:i + batch_size]
            batch_labels = split_data['test_labels'][i:i + batch_size]
            batch_num = i // batch_size + 1

            process_batch(batch_paths, batch_labels, batch_num, test_batches, 'test')
            pbar.update(len(batch_paths))
            gc.collect()

    # Сохраняем метаданные
    metadata = {
        'img_size': img_size,
        'num_classes': len(split_data['gesture_map']),
        'train_size': len(split_data['train_paths']),
        'val_size': len(split_data['val_paths']),
        'test_size': len(split_data['test_paths']),
        'train_batches': train_batches,
        'val_batches': val_batches,
        'test_batches': test_batches,
        'batch_size': batch_size,
        'data_format': 'uint8',  # Добавляем информацию о формате
        'gesture_map': split_data['gesture_map'],
        'class_counts': split_data['class_counts']
    }

    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)

    # Сохраняем маппинг жестов в текстовом формате
    with open(os.path.join(output_dir, 'gesture_mapping.txt'), 'w') as f:
        for gesture, label in split_data['gesture_map'].items():
            f.write(f"{label}: {gesture}\n")

    # Подсчитываем экономию места
    train_size_gb = (metadata['train_size'] * 128 * 128 * 3 * 1) / 1024 ** 3  # uint8 = 1 байт
    train_size_float_gb = (metadata['train_size'] * 128 * 128 * 3 * 4) / 1024 ** 3  # float32 = 4 байта

    print(f"\n ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"  Всего сохранено пакетов: {train_batches + val_batches + test_batches}")
    print(f"  Формат: uint8 (экономия места в 4 раза по сравнению с float32)")
    print(f"  Train размер: {train_size_gb:.1f} GB (было бы {train_size_float_gb:.1f} GB в float32)")
    print(f"  Память после обработки: {get_memory_usage():.2f} ГБ")

    return metadata


def create_data_generator(output_dir='hagrid_uint8',  # Изменил название папки
                          subset='train',
                          batch_size=32,
                          shuffle=True):
    """
    Создает генератор для загрузки данных пакетами во время обучения
    """

    metadata = np.load(os.path.join(output_dir, 'metadata.npy'), allow_pickle=True).item()

    # Находим все файлы для данного subset
    X_files = sorted([f for f in os.listdir(output_dir)
                      if f.startswith(f'X_{subset}') and f.endswith('.npy')])
    y_files = sorted([f for f in os.listdir(output_dir)
                      if f.startswith(f'y_{subset}') and f.endswith('.npy')])

    def generator():
        while True:
            # Создаем порядок пакетов
            batch_indices = list(range(len(X_files)))
            if shuffle:
                np.random.shuffle(batch_indices)

            for idx in batch_indices:
                # Загружаем пакет (uint8)
                X_batch = np.load(os.path.join(output_dir, X_files[idx]))
                y_batch = np.load(os.path.join(output_dir, y_files[idx]))

                # Нормализуем на лету (конвертируем в float32 и делим на 255)
                X_batch = X_batch.astype(np.float32) / 255.0

                # Разбиваем на мини-пакеты для обучения
                for i in range(0, len(X_batch), batch_size):
                    end_idx = min(i + batch_size, len(X_batch))
                    yield X_batch[i:end_idx], y_batch[i:end_idx]

    steps_per_epoch = metadata[f'{subset}_size'] // batch_size

    return generator(), steps_per_epoch


def visualize_samples_from_batch(output_dir='hagrid_uint8', num_samples=15):
    """Визуализирует примеры из первого пакета"""

    # Загружаем метаданные
    metadata = np.load(os.path.join(output_dir, 'metadata.npy'), allow_pickle=True).item()
    gesture_map_inv = metadata['gesture_map']

    # Загружаем первый пакет train (uint8)
    X_batch = np.load(os.path.join(output_dir, 'X_train_batch_0001.npy'))
    y_batch = np.load(os.path.join(output_dir, 'y_train_batch_0001.npy'))

    # Создаем обратный маппинг (label -> gesture)
    inv_map = {v: k for k, v in gesture_map_inv.items()}

    # Показываем примеры (данные уже в uint8, не нужно нормализовать)
    n_samples = min(num_samples, len(X_batch))
    fig, axes = plt.subplots(3, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(X_batch[i])  # uint8 работает напрямую с imshow
        gesture_name = inv_map[y_batch[i]]
        axes[i].set_title(f'{gesture_name}', fontsize=14)
        axes[i].axis('off')

    plt.suptitle('Примеры изображений из датасета Hagrid (uint8)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n Распределение классов в первом пакете:")
    unique, counts = np.unique(y_batch, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Класс {label} ({inv_map[label]}): {count}")
    print(f"\n Формат данных: uint8 (значения 0-255)")


if __name__ == "__main__":
    dataset_path = 'data/hagrid-classification-512p'

    print("=" * 70)
    print(" ПАКЕТНАЯ ОБРАБОТКА ДАТАСЕТА HAGRID (UINT8)")
    print("=" * 70)
    print(f"Начальная память: {get_memory_usage():.2f} ГБ")

    # 1. Анализируем структуру
    gesture_folders, gesture_stats = analyze_hagrid_structure(dataset_path)

    if not gesture_folders:
        print(" Не удалось проанализировать датасет")
        exit(1)

    # 2. Создаем только разделение (без загрузки изображений)
    split_data = create_dataset_splits(
        base_path=dataset_path,
        test_size=0.2,
        val_size=0.1,
        seed=42
    )

    # 3. Обрабатываем и сохраняем пакетами в uint8
    metadata = process_and_save_in_batches(
        split_data=split_data,
        img_size=(128, 128),
        output_dir='hagrid_uint8',  # Новая папка для uint8
        batch_size=1000
    )

    # 4. Визуализируем примеры из первого пакета
    visualize_samples_from_batch('hagrid_uint8')

    print("\n" + "=" * 70)
    print(" ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"\n Теперь у вас есть:")
    print(f"  - Старые данные: hagrid_processed/ (float32, большой размер)")
    print(f"  - Новые данные:  hagrid_uint8/   (uint8, в 4 раза меньше)")
    print(f"\n Для обучения используйте папку: hagrid_uint8")