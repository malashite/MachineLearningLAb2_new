import sys
from PyQt5.QtWidgets import QApplication
from data_loader import CustomDataLoader
from neural_network import NeuralNetwork
from main_window import MainWindow

def main():
    # Создаем приложение
    app = QApplication(sys.argv)
    
    # Загружаем данные
    data_loader = CustomDataLoader('annotations.csv')
    
    # Создаем нейронную сеть
    network = NeuralNetwork(
        input_size=data_loader.X_train.shape[1],
        hidden_layers=[512, 256, 128],
        output_size=data_loader.num_classes
    )
    
    # Создаем и показываем главное окно
    window = MainWindow(network, data_loader)
    window.show()
    
    # Запускаем приложение
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 