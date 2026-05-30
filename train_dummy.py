import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Фичи: [возраст, зарплата, есть_ли_кредитная_история(1/0)]
X = np.array([
    [25, 50000, 1],
    [40, 150000, 1],
    [22, 20000, 0],
    [55, 80000, 0],
    [23, 30000, 1],
    [40, 75000, 0],
    [26, 120000, 1],
    [30, 60000, 0]
])
# Таргет: 1 - выдать кредит, 0 - отказать
y = np.array([1, 1, 0, 0, 0, 1, 1, 0])

# Обучаем простейшую модель
model = LogisticRegression()
model.fit(X, y)

# Сохраняем модель (имитируем артефакт из MLflow/папки)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель успешно обучена и сохранена в model.pkl!")