import numpy as np
import json
import os

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        print(f"Инициализация сети: вход={input_size}, скрытые слои={hidden_layers}, выход={output_size}")
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.is_trained = False  # Флаг обученности
        
        # Увеличим размер скрытых слоев
        layer_sizes = [input_size] + [256, 128, 64] + [output_size]
        
        # Инициализация весов и смещений
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Инициализация He с небольшим шумом
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            weight += np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            bias = np.zeros(layer_sizes[i+1])
            
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        self.activations = [X]
        current_activation = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            
            # Проверка на численную стабильность
            if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                print(f"Предупреждение: обнаружены NaN или Inf значения в слое {i}")
            
            if i == len(self.weights) - 1:
                current_activation = self.softmax(z)
            else:
                current_activation = self.relu(z)
            
            self.activations.append(current_activation)
        
        return current_activation

    def backward(self, X, y, output, learning_rate):
        batch_size = X.shape[0]
        
        # y уже в формате one-hot encoding, не нужно преобразовывать
        delta = output - y
        
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            weight_grad = np.dot(self.activations[i].T, delta)
            bias_grad = np.sum(delta, axis=0, keepdims=True)
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= (self.activations[i] > 0)
        
        # Обновление весов и смещений
        for j in range(len(self.weights)):
            self.weights[j] -= learning_rate * weight_gradients[j]
            self.biases[j] -= learning_rate * bias_gradients[j]
        
        return weight_gradients, bias_gradients

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32, learning_rate=0.01, progress_callback=None):
        # Проверка размерностей данных
        print(f"Размерности данных:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"")
        
        history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'train_prec': [], 'val_prec': [],
        'train_rec': [], 'val_rec': []
        }
        n_samples = X_train.shape[0]
        
        print(f"\nПараметры: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}\n")
        
        for epoch in range(epochs):
            print(f"Эпоха {epoch + 1}/{epochs}")
            
            # Перемешиваем данные
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Обучение по мини-батчам
            for i in range(0, n_samples, batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                # Прямое распространение
                output = self.forward(batch_X)
                
                # Обратное распространение
                weight_grads, bias_grads = self.backward(batch_X, batch_y, output, learning_rate)
                
                # Обновление весов и смещений
                for j in range(len(self.weights)):
                    self.weights[j] -= learning_rate * weight_grads[j]
                    self.biases[j] -= learning_rate * bias_grads[j]
            
            # Вычисление ошибки и точности
            train_loss, train_acc, train_prec, train_rec = self.evaluate(X_train, y_train)
            val_loss, val_acc, val_prec, val_rec = self.evaluate(X_val, y_val)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_prec'].append(train_prec)
            history['val_prec'].append(val_prec)
            history['train_rec'].append(train_rec)
            history['val_rec'].append(val_rec)
            print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_prec: {train_prec:.4f}, train_rec: {train_rec:.4f}")
            print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_prec: {val_prec:.4f}, val_rec: {val_rec:.4f}\n")
            
            if progress_callback:
                progress_callback(int((epoch + 1) / epochs * 100))
        
        return history

    def evaluate(self, X, y):
        predictions = self.forward(X)
        
        # Вычисление функции потерь (cross-entropy)
        loss = -np.mean(np.sum(y * np.log(predictions + 1e-10), axis=1))
        
        # Вычисление точности
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)

        # Вычисление precision
        precision = self.calculate_precision(predictions, y)

        # Вычисление recall
        recall = self.calculate_recall(predictions, y)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

    def relu(self, x):
        """ReLU активация"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Производная ReLU"""
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        # Проверяем размерность входного массива
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def save_weights(self, filepath):
        """Сохранение весов и параметров сети в файл"""
        # Создаем словарь с параметрами сети и весами
        network_data = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size
            },
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'is_trained': self.is_trained  # Сохраняем флаг обученности
        }
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Сохраняем в файл
        with open(filepath, 'w') as f:
            json.dump(network_data, f)
        
        print(f"Веса сохранены в файл: {filepath}")
    
    @classmethod
    def load_weights(cls, filepath):
        """Загрузка весов из файла и создание новой сети"""
        with open(filepath, 'r') as f:
            network_data = json.load(f)
        
        # Создаем новую сеть с той же архитектурой
        network = cls(
            network_data['architecture']['input_size'],
            network_data['architecture']['hidden_layers'],
            network_data['architecture']['output_size']
        )
        
        # Загружаем веса и смещения
        network.weights = [np.array(w) for w in network_data['weights']]
        network.biases = [np.array(b) for b in network_data['biases']]
        network.is_trained = network_data.get('is_trained', True)  # Загружаем флаг обученности
        
        print(f"Веса загружены из файла: {filepath}")
        return network

    def predict(self, X):
        """Предсказание класса"""
        # Нормализация входных данных
        X = X.astype('float32') / 255.0
        
        # Применяем ту же предобработку, что и при обучении
        mean = np.mean(X)
        std = np.std(X) + 1e-8
        X = (X - mean) / std
        
        # Прямое распространение
        current_input = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self.relu(z)
        
        # Последний слой с softmax
        z_last = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        probabilities = self.softmax(z_last)
        
        # Возвращаем класс (добавляем 1, так как классы начинаются с 1)
        return np.argmax(probabilities) + 1

    def predict_proba(self, X):
        """Получение вероятностей для всех классов"""
        # Нормализация входных данных
        X = X.astype('float32') / 255.0
        
        # Применяем ту же предобработку, что и при обучении
        mean = np.mean(X)
        std = np.std(X) + 1e-8
        X = (X - mean) / std
        
        # Прямое распространение
        current_input = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self.relu(z)
        
        # Последний слой с softmax
        z_last = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        return self.softmax(z_last)

    def train_epoch(self, X, y, learning_rate):
        """Обучение на одной эпохе"""
        batch_size = len(X)
        
        # Прямое распространение
        activations = [X]
        z_values = []
        
        # Проходим через скрытые слои с ReLU
        current_input = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            current_input = self.relu(z)
            activations.append(current_input)
        
        # Выходной слой с softmax
        z_last = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_last)
        output = self.softmax(z_last)
        activations.append(output)
        
        # Обратное распространение
        deltas = [output - y]  # Для softmax + cross-entropy
        
        # Для скрытых слоев используем производную ReLU
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.relu_derivative(z_values[i-1])
            deltas.insert(0, delta)
        
        # Обновляем веса и смещения
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i]) / batch_size
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0) / batch_size
        
        # Вычисляем метрики
        loss = self.calculate_loss(output, y)
        accuracy = self.calculate_accuracy(output, y)
        precision = self.calculate_precision(output, y)
        recall = self.calculate_recall(output, y)
        
        self.is_trained = True
        return loss, accuracy, precision, recall

    def validate(self, X, y):
        """Валидация на тестовых данных"""
        # Прямое распространение
        output = self.forward(X)
        
        # Вычисляем метрики
        loss = self.calculate_loss(output, y)
        accuracy = self.calculate_accuracy(output, y)
        precision = self.calculate_precision(output, y)
        recall = self.calculate_recall(output, y)
        
        return loss, accuracy, precision, recall

    def calculate_loss(self, output, y):
        """Категориальная кросс-энтропия"""
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y * np.log(output), axis=1))

    def calculate_accuracy(self, output, y):
        """Вычисление точности"""
        predicted = np.argmax(output, axis=1)
        true = np.argmax(y, axis=1)
        return np.mean(predicted == true)
    
    def calculate_precision(self, output, y):
        """
        Вычисление precision как TP/(TP+TN)
        TP - количество истинно положительных прогнозов
        TN - количество истинно отрицательных прогнозов
        """
        predicted = np.argmax(output, axis=1)
        true = np.argmax(y, axis=1)
        
        # Маска для правильных предсказаний
        correct_predictions = (predicted == true)
        
        # Правильные положительные прогнозы (TP)
        tp = np.sum(correct_predictions)
        
        # Правильные отрицательные прогнозы (TN)
        # Когда предсказание и истинное значение оба отрицательные
        tn = len(predicted) - tp
        
        # Precision = TP / (TP + TN)
        precision = tp / (tp + tn + 1e-10)
        
        return precision
    
    def calculate_recall(self, output, y):
        """
        Вычисление recall как TP/(TP+FN)
        TP - истинно положительные предсказания
        FN - ложноотрицательные предсказания (случаи, которые модель пропустила)
        """
        predicted = np.argmax(output, axis=1)
        true = np.argmax(y, axis=1)
        
        # Истинно положительные (TP) - правильные предсказания
        tp = np.sum(predicted == true)
        
        # Ложноотрицательные (FN) - случаи, когда истинное значение положительное,
        # но модель предсказала отрицательное
        total_positives = np.sum(true)  # Общее количество положительных случаев
        fn = total_positives - tp
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-10)
        
        return recall
    

    def sigmoid(self, x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip для численной стабильности
