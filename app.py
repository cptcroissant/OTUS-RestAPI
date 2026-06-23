from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask('app')

# Загружаем модель ОДИН РАЗ при старте сервера
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Главная страница с формой
@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')


# Эндпоинт для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем данные из формы
        age = int(request.form.get('age'))
        income = int(request.form.get('income'))
        history = int(request.form.get('history'))

        # Преобразуем в 2D-массив NumPy
        features = np.array([[age, income, history]])

        # Делаем предсказание
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Возвращаем HTML с результатом
        return render_template(
            'result.html',
            approved=bool(prediction),
            probability=round(float(probability), 2)
        )


    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid data format', 'details': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Server error', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
