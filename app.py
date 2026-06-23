from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Загружаем модель ОДИН РАЗ при старте сервера
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Определяем тип входящих данных
        if request.is_json:
            # JSON запрос (API)
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            try:
                age = data['age']
                income = data['income']
                history = data['history']
            except KeyError:
                return jsonify({'error': 'Missing required fields'}), 400
        else:
            # Form data запрос (HTML форма)
            age = int(request.form.get('age'))
            income = int(request.form.get('income'))
            history = int(request.form.get('history'))

        # Преобразуем в 2D-массив NumPy
        features = np.array([[age, income, history]])

        # Делаем предсказание
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Возвращаем ответ в зависимости от типа запроса
        if request.is_json:
            # JSON ответ для API
            return jsonify({
                'approved': bool(prediction),
                'probability': round(float(probability), 2)
            }), 200
        else:
            # HTML ответ для веб-интерфейса
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