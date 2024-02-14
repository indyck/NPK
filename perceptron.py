import numpy as np

EPOCHS = 1000

# Функция активации - пороговая функция
def activation_function(x: any) -> int:
    return 0 if x < 0 else 1

# Реализация перцептрона
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def predict(self, input: any) -> int:
        weighted_sum = np.dot(self.weights, input) + self.bias
        return activation_function(weighted_sum)

# Задаем обучающие данные для логических операций AND, OR и эквивалентность
training_data = {
    'AND': [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
    'OR': [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
}

# Обучение и тестирование перцептрона
for operation, data in training_data.items():
    perceptron = Perceptron(input_size=2)
    print(f"Операция {operation}:")
    
    # Обучение перцептрона
    for _ in range(EPOCHS):
        for input_data, output in data:
            input_data_np = np.array(input_data)
            prediction = perceptron.predict(input_data_np)
            error = output - prediction
            perceptron.weights += error * input_data_np
            perceptron.bias += error

    # Тестирование перцептрона
    for input_data, output in data:
        input_data_np = np.array(input_data)
        prediction = perceptron.predict(input_data_np)
        print(f"Входные данные: {input_data}, Предсказание: {prediction}")
