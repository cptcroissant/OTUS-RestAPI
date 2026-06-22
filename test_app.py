import requests

# Адрес нашего микросервиса
url = 'http://127.0.0.1:5000/predict'

# Данные нового клиента (JSON)
client_data = {
    "age": 10,
    "income": 10000,
    "history": 0
}

# Отправляем POST запрос
print("Отправляем запрос на сервер...")
response = requests.post(url, json=client_data)

# Печатаем ответ сервера
print(f"Статус код: {response.status_code}")
print(f"Ответ модели: {response.json()}")
