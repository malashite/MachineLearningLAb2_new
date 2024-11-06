from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QImage
import numpy as np

class DrawingCanvas(QWidget):
    def __init__(self, size=28):  # размер можно изменить в соответствии с вашими данными
        super().__init__()
        self.size = size
        self.image = QImage(size, size, QImage.Format_Grayscale8)
        self.image.fill(Qt.white)
        self.last_point = None
        self.setFixedSize(280, 280)  # размер виджета оставляем большим для удобства
        
    def paintEvent(self, event):
        painter = QPainter(self)
        # Масштабируем изображение до размера виджета
        painter.drawImage(self.rect(), self.image)
        
    def mousePressEvent(self, event):
        self.last_point = self.scale_point(event.pos())
        
    def mouseMoveEvent(self, event):
        current_point = self.scale_point(event.pos())
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter.drawLine(self.last_point, current_point)
        self.last_point = current_point
        self.update()
        
    def scale_point(self, pos):
        # Масштабируем координаты из размера виджета в размер изображения
        return QPoint(int(pos.x() * self.size / self.width()),
                     int(pos.y() * self.size / self.height()))
        
    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        
    def get_image_data(self):
        # Преобразуем изображение в массив numpy
        ptr = self.image.bits()
        ptr.setsize(self.image.byteCount())
        arr = np.array(ptr).reshape(self.size, self.size)
        # Нормализуем и преобразуем в бинарный формат (0/1)
        return np.where((255 - arr) > 127, 1, 0)