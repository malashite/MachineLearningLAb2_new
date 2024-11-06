from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QSpinBox, QDoubleSpinBox, QLabel,
                            QProgressBar, QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal
from data_loader import CustomDataLoader
from drawing_canvas import DrawingCanvas
from training_visualizer import TrainingVisualizer
from neural_network import NeuralNetwork
import numpy as np

# Класс для выполнения обучения в отдельном потоке
class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    
    def __init__(self, neural_network, data_loader, epochs, learning_rate):
        super().__init__()
        self.neural_network = neural_network
        self.data_loader = data_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def run(self):
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_prec': [],
            'val_prec': [],
            'train_rec': [],
            'val_rec': []
        }
        
        for epoch in range(self.epochs):
            # Обучение на одной эпохе
            train_loss, train_acc, train_prec, train_rec = self.neural_network.train_epoch(
                self.data_loader.X_train,
                self.data_loader.y_train,
                self.learning_rate
            )
            
            # Валидация
            val_loss, val_acc, val_prec, val_rec = self.neural_network.validate(
                self.data_loader.X_val,
                self.data_loader.y_val
            )
            
            # Сохраняем метрики
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_prec'].append(train_prec)
            history['val_prec'].append(val_prec)
            history['train_rec'].append(train_rec)
            history['val_rec'].append(val_rec)
            
            # Обновляем прогресс
            progress = int((epoch + 1) / self.epochs * 100)
            self.progress.emit(progress)
            
            # Выводим текущие метрики
            print(f"Эпоха {epoch+1}/{self.epochs}:")
            print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_prec: {train_prec:.4f}, train_rec: {train_rec:.4f}")
            print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_prec: {val_prec:.4f}, val_rec: {val_rec:.4f}")
        
        self.finished.emit(history)

class MainWindow(QMainWindow):
    def __init__(self, neural_network, data_loader):
        QMainWindow.__init__(self)
        self.neural_network = neural_network
        self.data_loader = data_loader
        self.last_point = None
        self.setup_ui()
        
    def setup_ui(self):
        # Создаем главный виджет с горизонтальным layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # Левая панель с настройками
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Группа параметров обучения
        params_group = QGroupBox("Параметры обучения")
        params_layout = QVBoxLayout()
        
        # Количество эпох
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Количество эпох:"))
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(50)
        epochs_layout.addWidget(self.epochs_spinbox)
        params_layout.addLayout(epochs_layout)
        
        # Скорость обучения
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Скорость обучения:"))
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 1.0)
        self.lr_spinbox.setValue(0.01)
        self.lr_spinbox.setSingleStep(0.001)
        self.lr_spinbox.setDecimals(4)
        lr_layout.addWidget(self.lr_spinbox)
        params_layout.addLayout(lr_layout)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Кнопка обучения
        self.train_button = QPushButton("Начать обучение")
        self.train_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_button)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # Группа управления весами
        weights_group = QGroupBox("Управление весами")
        weights_layout = QVBoxLayout()
        
        self.save_weights_button = QPushButton("Сохранить веса")
        self.save_weights_button.clicked.connect(self.save_weights)
        weights_layout.addWidget(self.save_weights_button)
        
        self.load_weights_button = QPushButton("Загрузить веса")
        self.load_weights_button.clicked.connect(self.load_weights)
        weights_layout.addWidget(self.load_weights_button)
        
        weights_group.setLayout(weights_layout)
        left_layout.addWidget(weights_group)
        
        # Группа рисования
        draw_group = QGroupBox("Рисование")
        draw_layout = QVBoxLayout()
        
        # Используем DrawingCanvas вместо QLabel
        self.canvas = DrawingCanvas(size=28)
        draw_layout.addWidget(self.canvas)
        
        # Кнопки управления рисованием
        buttons_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.canvas.clear)  # Используем метод из DrawingCanvas
        buttons_layout.addWidget(self.clear_button)
        
        # Изначально кнопка распознавания неактивна
        self.recognize_button = QPushButton("Распознать")
        self.recognize_button.setEnabled(False)
        self.recognize_button.clicked.connect(self.recognize_digit)
        buttons_layout.addWidget(self.recognize_button)
        
        draw_layout.addLayout(buttons_layout)
        draw_group.setLayout(draw_layout)
        left_layout.addWidget(draw_group)
        
        # Добавляем растягивающийся пробел в конце
        left_layout.addStretch()
        
        # Правая панель с графиками
        self.visualizer = TrainingVisualizer()
        
        # Добавляем панели в главный layout
        main_layout.addWidget(left_panel, stretch=1)  # 1 часть ширины
        main_layout.addWidget(self.visualizer, stretch=4)  # 4 части ширины
        
        # Устанавливаем размер окна
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle('Обучение нейронной сети')
        
    def start_training(self):
        """Начало процесса обучения"""
        # Сбрасываем веса сети перед новым обучением
        self.neural_network.reset()
        
        # Получаем параметры
        epochs = self.epochs_spinbox.value()
        learning_rate = self.lr_spinbox.value()
        
        print("\nНачинаем обучение:")
        print(f"Количество эпох: {epochs}")
        print(f"Скорость обучения: {learning_rate}")
        
        # Сбрасываем графики перед новым обучением
        self.visualizer.clear_plots()
        
        # Отключаем кнопку на время обучения
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Создаем и запускаем поток обучения
        print("Запускаем поток обучения...\n")
        self.training_thread = TrainingThread(
            self.neural_network,
            self.data_loader,
            epochs,
            learning_rate
        )
        
        # Подключаем сигналы
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        
        # Запускаем обучение
        self.training_thread.start()

    def update_progress(self, value):
        """Обновление прогресс-бара"""
        self.progress_bar.setValue(value)

    def training_finished(self, history):
        """Обработка завершения обучения"""
        # Включаем кнопку обратно
        self.train_button.setEnabled(True)
        self.recognize_button.setEnabled(True)  # Активируем кнопку после обучения
        
        # Обновляем графики
        self.visualizer.update_plots(history)
        
        # Выводим сообщение о завершении
        print("\nОбучение завершено!")

    def recognize_digit(self):
        """Распознавание нарисованной цифры"""
        if not hasattr(self, 'neural_network'):
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите веса сети")
            return
        
        if not hasattr(self.neural_network, 'is_trained') or not self.neural_network.is_trained:
            QMessageBox.warning(self, "Предупреждение", "Сеть не обучена. Сначала обучите сеть или загрузите веса.")
            return
        
        # Получаем данные изображения из canvas
        image_data = self.canvas.get_image_data()
        
        # Преобразуем в одномерный массив
        pixels = image_data.flatten()
        
        # Получаем вероятности для всех классов
        probabilities = self.neural_network.predict_proba(pixels)
        
        # Находим класс с максимальной вероятностью
        predicted_class = np.argmax(probabilities) + 1  # +1 так как классы от 1 до 10
        
        # Формируем сообщение с вероятностями
        message = f"Распознанная цифра: {predicted_class}\n\nВероятности:\n"
        for i, prob in enumerate(probabilities[0]):
            digit = i + 1  # +1 так как классы от 1 до 10
            message += f"Цифра {digit}: {prob*100:.2f}%\n"
        
        # Показываем результат
        QMessageBox.information(self, "Результат", message)

    def save_weights(self):
        """Обработчик сохранения весов"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить веса",
            "",
            "Weight Files (*.weights);;All Files (*)"
        )
        if filepath:
            if not filepath.endswith('.weights'):
                filepath += '.weights'
            try:
                self.neural_network.save_weights(filepath)
                QMessageBox.information(self, "Успех", "Веса успешно сохранены")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении весов: {str(e)}")

    def load_weights(self):
        """Обработчик загрузки весов"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить веса",
            "",
            "Weight Files (*.weights);;All Files (*)"
        )
        if filepath:
            try:
                new_network = NeuralNetwork.load_weights(filepath)
                self.neural_network = new_network
                self.recognize_button.setEnabled(True)  # Активируем кнопку после загрузки весов
                QMessageBox.information(self, "Успех", "Веса успешно загружены")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке весов: {str(e)}")