from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class TrainingVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Создаем график с большим размером и отступами
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Создаем подграфики с отступами
        self.ax1 = self.figure.add_subplot(221)
        self.ax2 = self.figure.add_subplot(222)
        self.ax3 = self.figure.add_subplot(223)
        self.ax4 = self.figure.add_subplot(224)

        # Начальная настройка графиков
        self._setup_plots()
        
    def _setup_plots(self):
        """Начальная настройка внешнего вида графиков"""
        # Настройка первого графика
        self.ax1.set_title('Функция Loss', fontsize=12)
        self.ax1.set_xlabel('Эпоха', fontsize=10)
        self.ax1.set_ylabel('Loss', fontsize=10)
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Настройка второго графика
        self.ax2.set_title('Accuracy', fontsize=12)
        self.ax2.set_xlabel('Эпоха', fontsize=10)
        self.ax2.set_ylabel('Accuracy', fontsize=10)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Настройка третьего графика
        self.ax3.set_title('Precision', fontsize=12)
        self.ax3.set_xlabel('Эпоха', fontsize=10)
        self.ax3.set_ylabel('Precision', fontsize=10)
        self.ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Настройка четвертого графика
        self.ax4.set_title('Recall', fontsize=12)
        self.ax4.set_xlabel('Эпоха', fontsize=10)
        self.ax4.set_ylabel('Recall', fontsize=10)
        self.ax4.grid(True, linestyle='--', alpha=0.7)
        
        # Установка отступов между графиками
        self.figure.tight_layout(pad=3.0)
        
    def update_plots(self, history):
        # Очистка графиков
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        # Восстановление настроек
        self._setup_plots()
        
        # График функции loss
        self.ax1.plot(history['train_loss'], label='Обучение', linewidth=2)
        self.ax1.plot(history['val_loss'], label='Валидация', linewidth=2)
        self.ax1.legend(fontsize=10)
        
        # График accuracy
        self.ax2.plot(history['train_acc'], label='Обучение', linewidth=2)
        self.ax2.plot(history['val_acc'], label='Валидация', linewidth=2)
        self.ax2.legend(fontsize=10)
        
        # График precision
        self.ax3.plot(history['train_prec'], label='Обучение', linewidth=2)
        self.ax3.plot(history['val_prec'], label='Валидация', linewidth=2)
        self.ax3.legend(fontsize=10)

        # График recall
        self.ax4.plot(history['train_rec'], label='Обучение', linewidth=2)
        self.ax4.plot(history['val_rec'], label='Валидация', linewidth=2)
        self.ax4.legend(fontsize=10)
        
        # Обновление графика
        self.canvas.draw() 

    def clear_plots(self):
        """Очистка графиков"""
        # Очищаем данные
        self.train_loss_data = []
        self.val_loss_data = []
        self.train_acc_data = []
        self.val_acc_data = []
        self.train_prec_data = []
        self.val_prec_data = []
        self.train_rec_data = []
        self.val_rec_data = []
        
        # Очищаем графики
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            
            # Восстанавливаем настройки графиков
            ax.grid(True)
            ax.set_xlabel('Эпоха')
        
        # Устанавливаем заголовки
        self.ax1.set_title('Функция Loss')
        self.ax1.set_ylabel('Потери')
        
        self.ax2.set_title('Accuracy')
        self.ax2.set_ylabel('Точность')

        self.ax3.set_title('Precision')
        self.ax3.set_ylabel('Точность')
        
        self.ax4.set_title('Recall')
        self.ax4.set_ylabel('Полнота')
        
        # Обновляем холст
        self.canvas.draw()