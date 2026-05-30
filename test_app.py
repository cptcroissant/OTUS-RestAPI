import requests

# Адрес нашего микросервиса
url = 'http://127.0.0.1:5000/predict'

# Данные нового клиента (JSON)
client_data = {
    "age": 30,
    "income": 120000,
    "history": 1
}

# Отправляем POST запрос
print("Отправляем запрос на сервер...")
response = requests.post(url, json=client_data)

# Печатаем ответ сервера
print(f"Статус код: {response.status_code}")
print(f"Ответ модели: {response.json()}")
