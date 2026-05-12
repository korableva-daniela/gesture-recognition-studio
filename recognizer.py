"""
Модуль распознавания жестов
Классы: SimpleCNN, GestureRecognizer
"""

import torch
import torch.nn as nn
import os
from typing import Optional, Tuple, List

# Список 18 классов жестов
GESTURE_CLASSES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]


class SimpleCNN(nn.Module):
    """Простая CNN для распознавания жестов (3 свёрточных слоя)"""

    def __init__(self, num_classes: int = 18):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Первый свёрточный слой
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Второй свёрточный слой
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Третий свёрточный слой
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class GestureRecognizer:
    """Распознавание жестов с использованием свёрточной нейронной сети"""

    def __init__(self, model_path: str = "models/gesture_model.pth", confidence_threshold: float = 50.0):
        """
        Инициализация распознавателя жестов

        Args:
            model_path: путь к файлу обученной модели
            confidence_threshold: порог уверенности (0-100%)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.frame_buffer: List[Tuple[Optional[str], float]] = []
        self.model = None
        self.gesture_classes = GESTURE_CLASSES
        self.load_model()

    def load_model(self) -> bool:
        """Загрузка обученной модели CNN из файла"""
        try:
            self.model = SimpleCNN(num_classes=len(self.gesture_classes))

            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                print(f"Модель загружена из {self.model_path}")
            else:
                print(f"Файл модели {self.model_path} не найден. Используется неподготовленная модель.")

            self.model.eval()
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def predict(self, frame_tensor: torch.Tensor) -> Tuple[Optional[str], float]:
        """Распознавание жеста на одном кадре"""
        if frame_tensor is None or self.model is None:
            return None, 0.0

        with torch.no_grad():
            outputs = self.model(frame_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        confidence_percent = confidence.item() * 100
        gesture_type = self.gesture_classes[predicted_idx.item()]

        return gesture_type, confidence_percent

    def filter_predictions(self, gesture: Optional[str], confidence: float) -> Tuple[Optional[str], float, bool]:
        """Фильтрация ложных срабатываний"""
        if confidence < self.confidence_threshold:
            return None, 0.0, False

        if gesture is None:
            return None, 0.0, False

        return gesture, confidence, True

    def smooth_predictions(self, gesture: str, confidence: float) -> Tuple[Optional[str], float]:
        """Сглаживание временной последовательности"""
        self.frame_buffer.append((gesture, confidence))

        if len(self.frame_buffer) > 5:
            self.frame_buffer.pop(0)

        if not self.frame_buffer:
            return gesture, confidence

        gestures = [g for g, _ in self.frame_buffer if g is not None]
        if not gestures:
            return None, 0.0

        most_common = max(set(gestures), key=gestures.count)
        avg_confidence = sum(c for g, c in self.frame_buffer if g == most_common) / gestures.count(most_common)

        return most_common, avg_confidence