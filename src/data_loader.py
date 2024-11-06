import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class CustomDataLoader:
    def __init__(self, csv_path='annotations.csv'):
        print("Загрузка данных из", csv_path)
        
        # Загружаем данные из CSV
        self.data = pd.read_csv(csv_path)
        
        # Получаем уникальные классы
        self.classes = sorted(self.data['class'].unique())
        self.num_classes = len(self.classes)
        print(f"Найдено {self.num_classes} классов: {self.classes}")
        
        # Преобразуем данные в нужный формат
        X = []
        y = []
        
        for _, row in self.data.iterrows():
            # Преобразуем строку пикселей в массив, используя запятую как разделитель
            pixels = np.array([int(p) for p in row['pixels'].split(',')])
            X.append(pixels)
            y.append(row['class'])
            
        X = np.array(X, dtype='float32')
        
        # Преобразуем метки в one-hot encoding
        y = np.array(y)
        y_onehot = self.preprocess_labels(y)
        
        # Разделяем данные на обучающую и валидационную выборки,
        # сохраняя пропорции классов
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y_onehot, 
            test_size=0.2,
            stratify=y,  # сохраняем пропорции классов
            random_state=42
        )
        
        print(f"Размер обучающей выборки: {self.X_train.shape[0]} изображений")
        print(f"Размер валидационной выборки: {self.X_val.shape[0]} изображений")
        print(f"Размерность входных данных: {self.X_train.shape[1]} пикселей")
        print("Данные загружены успешно!")
    
    def get_class_name(self, class_index):
        """Возвращает название класса по его индексу"""
        return self.classes[class_index]
    
    def preprocess_labels(self, labels):
        """Преобразование меток в one-hot encoding"""
        # Преобразуем метки в индексы от 0 до 9
        labels = labels - 1  # вычитаем 1, так как метки в датасете от 1 до 10
        
        # Создаем one-hot encoding
        n_classes = len(np.unique(labels))
        one_hot = np.zeros((len(labels), n_classes))
        for i, label in enumerate(labels):
            one_hot[i, label] = 1
        
        return one_hot