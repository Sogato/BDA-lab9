import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_and_prepare_data(filepath):
    """Загрузка и подготовка данных."""
    # Загрузка данных
    sales_dist = pd.read_csv(filepath)

    # Переименование столбцов
    sales_dist = sales_dist.rename(columns={'annual net sales': 'sales', 'number of stores in district': 'stores'})
    print("Первые пять записей после переименования столбцов:")
    print(sales_dist.head())

    # Удаление столбца 'district'
    sales = sales_dist.drop('district', axis=1)
    print("Первые пять записей после удаления столбца 'district':")
    print(sales.head())

    return sales


def fit_polynomial_regression(X, y, order=1):
    """Выполнение полиномиальной регрессии и вывод результатов."""
    # Вычисление коэффициентов полиномиальной регрессии
    p = np.poly1d(np.polyfit(X, y, order))

    print(
        "\nМассив p(x) хранит вычисленные значения y от модели полиномиальной регрессии для каждого значения x:\n\n{}.".format(
            p(X)))
    print("\nВектор коэффициентов p описывает эту регрессионную модель: {}.".format(p))
    print("\nКоэффициент нулевого порядка (свободный член или b) хранится в p[0]: {}.".format(p[0]))
    print("Коэффициент первого порядка (угловой коэффициент или m) хранится в p[1]: {}.".format(p[1]))

    # Расчёт метрик
    r2 = r2_score(y, p(X))
    print("\nКоэффициент детерминации (R^2):", r2)

    mse = mean_squared_error(y, p(X))
    print("Среднеквадратичная ошибка (MSE):", mse)

    mae = mean_absolute_error(y, p(X))
    print("Средняя абсолютная ошибка (MAE):", mae)


# Основной блок кода
if __name__ == "__main__":
    filepath = 'stores-dist.txt'
    sales = load_and_prepare_data(filepath)

    # Зависимая переменная для оси Y и независимая переменная для оси X
    y = sales.sales
    X = sales.stores

    # Выполнение регрессионного анализа
    fit_polynomial_regression(X, y)
