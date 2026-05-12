"""
Модуль работы с видеокамерой
Класс: Camera
"""

import cv2
import torch
import numpy as np
from typing import Optional, Tuple


class Camera:
    """Управление видеокамерой и захват кадров"""

    def __init__(self, camera_index: int = 0, fps: int = 30, resolution: Tuple[int, int] = (1280, 720)):
        """
        Инициализация параметров камеры

        Args:
            camera_index: индекс камеры (по умолчанию 0)
            fps: частота кадров
            resolution: разрешение кадра (ширина, высота)
        """
        self.camera_index = camera_index
        self.fps = fps
        self.resolution = resolution
        self.is_active = False
        self.cap = None

    def init_camera(self) -> bool:
        """Инициализация и настройка камеры"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Ошибка: Камера {self.camera_index} не доступна")
                return False

            # Установка параметров камеры
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            self.is_active = True
            print(f"Камера инициализирована: {self.resolution[0]}x{self.resolution[1]}, {self.fps} fps")
            return True
        except Exception as e:
            print(f"Ошибка инициализации камеры: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Захват очередного кадра"""
        if not self.is_active or self.cap is None:
            print("Камера не активна")
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            return None

        return frame

    def preprocess_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Предобработка кадра для нейронной сети"""
        if frame is None:
            return None

        # Изменение размера до 128x128
        resized = cv2.resize(frame, (128, 128))

        # Преобразование цветового пространства BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Нормализация значений пикселей [0, 1]
        normalized = rgb / 255.0

        # Преобразование в тензор PyTorch
        tensor = torch.from_numpy(normalized).float().permute(2, 0, 1).unsqueeze(0)

        return tensor

    def release_camera(self):
        """Освобождение ресурсов камеры"""
        if self.cap is not None:
            self.cap.release()
        self.is_active = False
        print("Ресурсы камеры освобождены")