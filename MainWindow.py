"""
Модуль графического интерфейса
Класс: MainWindow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from camera import Camera
from recognizer import GestureRecognizer
from logger import Logger


class MainWindow:
    """Главное окно приложения"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Recognition Studio")
        self.root.geometry("800x600")

        # Инициализация компонентов
        self.camera = Camera()
        self.recognizer = GestureRecognizer()
        self.logger = Logger()

        self.is_running = False
        self.current_performer = tk.StringVar(value="Исполнитель")

        self._setup_ui()

    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Меню
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Экспорт журнала", command=self.open_log_viewer)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_menu)

        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Отображение жеста
        display_frame = ttk.LabelFrame(main_frame, text="Распознанный жест", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.gesture_label = ttk.Label(display_frame, text="---", font=("Arial", 48))
        self.gesture_label.pack(pady=20)

        # Индикатор уверенности
        confidence_frame = ttk.Frame(display_frame)
        confidence_frame.pack(fill=tk.X, pady=10)

        ttk.Label(confidence_frame, text="Уверенность:").pack(side=tk.LEFT, padx=5)
        self.confidence_bar = ttk.Progressbar(confidence_frame, length=300, mode='determinate')
        self.confidence_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.confidence_label = ttk.Label(confidence_frame, text="0%")
        self.confidence_label.pack(side=tk.LEFT, padx=5)

        # Предупреждения
        self.warning_label = ttk.Label(display_frame, text="", foreground="red", font=("Arial", 12))
        self.warning_label.pack(pady=10)

        # Управление
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Исполнитель:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.current_performer, width=30).grid(row=0, column=1, padx=5, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.start_button = ttk.Button(button_frame, text="Старт", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Стоп", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Экспорт журнала", command=self.open_log_viewer).pack(side=tk.LEFT, padx=5)

    def update_display(self, gesture: str, confidence: float):
        """Обновление отображения"""
        if gesture:
            self.gesture_label.config(text=gesture.upper())
        else:
            self.gesture_label.config(text="???")

        self.confidence_bar['value'] = confidence
        self.confidence_label.config(text=f"{confidence:.1f}%")

        if confidence >= 85:
            self.gesture_label.config(foreground="green")
        elif confidence >= 50:
            self.gesture_label.config(foreground="orange")
        else:
            self.gesture_label.config(foreground="red")

    def show_warning(self, message: str):
        """Отображение предупреждения"""
        self.warning_label.config(text=message)
        self.root.after(3000, lambda: self.warning_label.config(text=""))

    def show_menu(self):
        """Меню 'О программе'"""
        messagebox.showinfo("О программе",
                            "Gesture Recognition Studio\nВерсия 1.0\n\n"
                            "Распознавание 18 типов статических жестов\n"
                            "с использованием свёрточной нейронной сети")

    def open_log_viewer(self):
        """Просмотр журнала"""
        result = self.logger.export_log()
        messagebox.showinfo("Экспорт", result)

    def start_recognition(self):
        """Запуск распознавания"""
        if not self.camera.init_camera():
            self.show_warning("Не удалось инициализировать камеру")
            return

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self._recognition_loop()

    def stop_recognition(self):
        """Остановка распознавания"""
        self.is_running = False
        self.camera.release_camera()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def _recognition_loop(self):
        """Основной цикл"""
        if not self.is_running:
            return

        frame = self.camera.capture_frame()
        if frame is None:
            self.root.after(33, self._recognition_loop)
            return

        tensor = self.camera.preprocess_frame(frame)
        if tensor is None:
            self.root.after(33, self._recognition_loop)
            return

        gesture, confidence = self.recognizer.predict(tensor)
        gesture, confidence, is_valid = self.recognizer.filter_predictions(gesture, confidence)

        if is_valid:
            gesture, confidence = self.recognizer.smooth_predictions(gesture, confidence)
            self.update_display(gesture, confidence)
            self.logger.log_event(
                performer_name=self.current_performer.get(),
                gesture_type=gesture,
                confidence=confidence,
                is_abnormal=False
            )
        else:
            self.update_display(None, confidence)
            self.show_warning("Жест не распознан, повторите")
            self.logger.log_event(
                performer_name=self.current_performer.get(),
                gesture_type="unknown",
                confidence=confidence,
                is_abnormal=True,
                abnormal_reason="low_confidence"
            )

        self.root.after(33, self._recognition_loop)

    def run(self):
        """Запуск приложения"""
        self.root.mainloop()