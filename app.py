from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Загружаем модель ОДИН РАЗ при старте сервера, а не при каждом запросе!
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Написать метод get к основной странице
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to our server!'}), 200

# Создаем эндпоинт (маршрут) для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Получаем JSON из запроса (помните слайд про request?)
    data = request.get_json()

    # todo - Написать простую валидацию, не забыть код ошибки

    # 2. Достаем фичи (ожидаем, что клиент пришлет ключи: age, income, history)
    try:
        age = data['age']
        income = data['income']
        history = data['history']
    except KeyError:
        # todo - Написать ошибку неверного ключа
        print("00000psie")

    # 3. Преобразуем в 2D-массив NumPy для sklearn (1 строка, 3 столбца)
    features = np.array([[age, income, history]])

    # 4. Делаем предсказание (predict_proba вернет вероятность класса)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # вероятность класса 1

    # 5. Формируем ответ клиенту в виде словаря
    # todo - Написать словарь с prediction и probability
    result = {}

    # Возвращаем JSON и статус 200 (OK)
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
