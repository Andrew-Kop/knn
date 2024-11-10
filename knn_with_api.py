import pandas as pd
import tkinter as tk
import numpy as np
import joblib
import os
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import openpyxl
from openpyxl.styles import PatternFill
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

scaler = StandardScaler()

def preprocessing_of_data(file_path):
    data = pd.read_excel(file_path)
    data = data.drop_duplicates()
    base_columns = [
    'Город', 'Адрес', 'Широта', 'Долгота', 'Функциональная зона', 
    'Общая площадь,\nкв.м', 'Год постройки', 'Материал стен', 'Класс', 
    'Тип здания', 'Этаж', 'Этажность', 'Состояние ремонта', 
    'Высота потолков, м', 'Планировка', 'Отдельный вход', 'Доступ', 
    'Охрана', 'Парковка', 'Красная линия']
    extra_columns = ['Цена предложения,\n руб.', 'Удельная цена, руб./кв.м']

    # Проверяем, присутствуют ли дополнительные столбцы в данных
    columns_to_check = base_columns + [col for col in extra_columns if col in data.columns]

    data = data.drop_duplicates(subset=columns_to_check, keep='first')

    columns_to_process = 'Цена предложения,\n руб.', 'Удельная цена, руб./кв.м'
    for column in columns_to_process:
        if column in data.columns:
            # Преобразуем в числовые данные
            data[column] = pd.to_numeric(data[column], errors='coerce')

    #преобразуем в булевый формат
    data['Охрана'] = data['Охрана'].replace({'есть': True, 'ЛОЖЬ': False})
    data['Охрана'] = pd.to_numeric(data['Охрана'], errors='coerce')
    data['Охрана'] = data['Охрана'].fillna(False) #заполняем пустые значения ложью
    data['Охрана'] = data['Охрана'].astype(bool)

    data['Отдельный вход'] = data['Отдельный вход'].replace({'Есть': True, 'нет': False})
    data['Отдельный вход'] = pd.to_numeric(data['Отдельный вход'], errors='coerce')
    data['Отдельный вход'] = data['Отдельный вход'].fillna(False)
    data['Отдельный вход'] = data['Отдельный вход'].astype(bool)

    data['Парковка'] = data['Парковка'].replace({'Есть': True, 'Организованная наземная открытая': True, 
                                                 'Стихийная наземная': True, 'Организованная подземная': True,
                                                 'Для грузового транспорта': True, 'Организованная наземная крытая': True })
    data['Парковка'] = pd.to_numeric(data['Парковка'], errors='coerce')
    data['Парковка'] = data['Парковка'].fillna(False)
    data['Парковка'] = data['Парковка'].astype(bool)

    data['Состояние ремонта'] = data['Состояние ремонта'].replace({'Типовой ремонт': True, 'Комфортный ремонт': True, 'Требуется косметический ремонт': False})
    data['Состояние ремонта'] = pd.to_numeric(data['Состояние ремонта'], errors='coerce')
    data['Состояние ремонта'] = data['Состояние ремонта'].fillna(False)
    data['Состояние ремонта'] = data['Состояние ремонта'].astype(bool)

    key_columns = ['Класс', 'Охрана', 'Парковка', 'Состояние ремонта', 'Отдельный вход', 'Широта', 'Долгота', 'Общая площадь,\nкв.м']
    data = data.dropna(subset=key_columns)
    
    data['Широта'] = pd.to_numeric(data['Широта'], errors='coerce')
    data['Долгота'] = pd.to_numeric(data['Долгота'], errors='coerce')
    data['Общая площадь,\nкв.м'] = pd.to_numeric(data['Общая площадь,\nкв.м'], errors='coerce')\
    
    #Преобразует для класса в числовые данные
    label_encoder = LabelEncoder()
    data['Класс'] = label_encoder.fit_transform(data['Класс'])
    return data
    # Создание DataFrame и сохранение в Excel
    #df = pd.DataFrame(data)
    #data_show=df.to_excel('data.xlsx', index=False)
    #os.startfile('data.xlsx')

def find_best_k(X_train, X_test, Y_train, Y_test):
    best_k, best_error = 0, float('inf')
    for k in range(2, 52):
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, Y_train)
        Y_pred = np.exp(knn.predict(X_test))
        error = np.median(np.abs(np.exp(Y_test) - Y_pred) / np.exp(Y_test) * 100)
        if error < best_error:
            best_k, best_error = k, error
    return best_k, best_error

def open_file_dialog(scaler):
    file_path = filedialog.askopenfilename(
        title="Выберите файл", 
        filetypes=(("Excel файлы", "*.xlsx *.xls"), ("Все файлы", "*.*"))
    )

    if file_path:
        file_path_converted = file_path.replace('/', '\\')  # Изменяем вид пути
        original_data = preprocessing_of_data('C:\\Users\\andrew\\Downloads\\office_nn.xlsx')
        data = preprocessing_of_data(file_path_converted)  # Используем выбранный файл
         # Проверяем наличие данных для расчета
        if not data['Удельная цена, руб./кв.м'].isnull().all() and (data['Удельная цена, руб./кв.м'] != 0).any():
            result_table, predicted_values, actual_values = knn_predict(data, original_data, scaler)  # Вызываем предсказание
            
            # Выводим ошибку
            error_message = error_output(predicted_values, actual_values)
            text_output.delete(1.0, tk.END)  # Очищаем текстовое поле
            text_output.insert(tk.END, error_message)  # Вставляем текст ошибки
            
            # Строим график
            plot_feature_importance(data, scaler, frame)
        else: 
            text_output.delete(1.0, tk.END)  # Очищаем текстовое поле
            text_output.insert(tk.END, "Нет данных для вывода значения ошибки и построения графика приоритетов.")  # Сообщение об отсутствии данных

# Функция для изучения KNN
def knn_study(data):
    # Определяем признаки и целевую переменную
    X_target = data[['Класс', 'Охрана', 'Парковка', 'Состояние ремонта', 'Отдельный вход', 'Широта', 'Долгота', 'Общая площадь,\nкв.м']]
    Y_target = np.log(data['Удельная цена, руб./кв.м'])  # Используем одну квадратную скобку для выбора Series
    # Масштабируем данные
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_target)
    # Разделяем данные на обучающую и тестовую выборки
    X_learn, X_test, Y_learn, Y_test = train_test_split(X_scaled, Y_target, test_size=0.2, random_state=12)
    # Находим лучший K и ошибку
    k, error = find_best_k(X_learn, X_test, Y_learn, Y_test)  
    # Обучаем KNN
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')
    knn.fit(X_learn, Y_learn)
    print(f'best_k = {k}, best_error = {error:.4f}%')
    # Сохраняем модель
    joblib.dump(knn, 'knn_model.joblib')
    return scaler 

data_study = preprocessing_of_data('C:\\Users\\andrew\\Downloads\\office_nn.xlsx')
scaler = knn_study(data_study)

# Функция для предсказания
def knn_predict(data, original_data, scaler):
    # Загружаем модель
    knn = joblib.load('knn_model.joblib')
    
    # Определяем признаки
    X_target = data[['Класс', 'Охрана', 'Парковка', 'Состояние ремонта', 'Отдельный вход', 'Широта', 'Долгота', 'Общая площадь,\nкв.м']]
    # Масштабируем данные (используем transform)
    X_scaled = scaler.transform(X_target)
    # Получаем предсказания
    predicted_value = knn.predict(X_scaled)
    predicted_value = np.exp(predicted_value)  # Возвращаем обратно в исходные единицы

    if 'Удельная цена, руб./кв.м' in data.columns and not data['Удельная цена, руб./кв.м'].isnull().all():
        actual_prices = data['Удельная цена, руб./кв.м'].values
        # Вычисляем ошибку
        error = np.median(np.abs(actual_prices - predicted_value) / actual_prices * 100)  
        print(f'Ошибка предсказания: {error:.4f}%')

    # Получение массива расстояний и индексов для ближайших соседей
    distances, indices = knn.kneighbors(X_scaled)  # Используем X_scaled
    # Создание DataFrame для предсказанных значений
    predictions_df = pd.DataFrame(predicted_value, columns=['Предсказанная цена, руб.'])
    # Создание нового DataFrame для результатов
    new_rows = []
    
    k = knn.n_neighbors
    # Добавляем индексы объектов для сопоставления с оригинальными данными
    for i in range(len(data)):
        row = data.iloc[i].copy()
        row['Предсказанная цена, руб.'] = predictions_df.iloc[i, 0]
        new_rows.append(row)

        # Проверка на наличие точного совпадения
        exact_match = original_data[
            (original_data[['Класс', 'Охрана', 'Парковка', 'Состояние ремонта', 'Отдельный вход', 'Широта', 'Долгота', 'Общая площадь,\nкв.м','Цена предложения,\n руб.', 'Удельная цена, руб./кв.м']] 
            == row[['Класс', 'Охрана', 'Парковка', 'Состояние ремонта', 'Отдельный вход', 'Широта', 'Долгота', 'Общая площадь,\nкв.м', 'Цена предложения,\n руб.', 'Удельная цена, руб./кв.м']].values).all(axis=1)
        ]

        if not exact_match.empty:
            # Добавляем только первое точное совпадение
            match_row_copy = exact_match.iloc[0].copy()
            match_row_copy['Предсказанная цена, руб.'] = None
            new_rows.append(match_row_copy)
            
            # Добавляем k-1 ближайших соседей
            indices_neighbors = indices[i][1:k] 
        else:
            # Если точного совпадения нет, добавляем k ближайших соседей
            indices_neighbors = indices[i][:k]

        for neighbor_index in indices_neighbors:
            neighbor_row = original_data.iloc[neighbor_index].copy()
            neighbor_row['Предсказанная цена, руб.'] = None
            new_rows.append(neighbor_row)
                        
    # Создаем DataFrame из собранных строк
    result_df = pd.DataFrame(new_rows)
    result_df.reset_index(drop=True, inplace=True)

    # Сохранение результата в Excel файл
    result_df.to_excel('data.xlsx', index=False, engine='openpyxl')

    # Открытие файла Excel для редактирования и выделения предсказанных строк
    wb = openpyxl.load_workbook('data.xlsx')
    ws = wb.active
    yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
    
    # Итерация по строкам DataFrame для выделения
    for i in range(len(result_df)):
        # Проверяем, что в строке есть предсказанная цена (и она не NaN)
        if pd.notna(result_df.iloc[i]['Предсказанная цена, руб.']):
            # Проходим по всем ячейкам в строке i + 2
            for col in range(1, len(result_df.columns) + 1):  # Применяем заливку ко всем столбцам этой строки
                ws.cell(row=i + 2, column=col).fill = yellow_fill  # +2 для пропуска заголовков

    # Сохранение изменений в файле
    wb.save('data.xlsx')
    wb.close()
    os.startfile('data.xlsx')

    # Возврат результата
    if 'Удельная цена, руб./кв.м' in data.columns and not data['Удельная цена, руб./кв.м'].isnull().all():
        actual_prices = data['Удельная цена, руб./кв.м'].values
        return result_df, predicted_value, actual_prices
    else:
        return result_df

def plot_feature_importance(data, scaler, frame): 
    # Определяем признаки
    X_target = data[['Класс', 'Охрана', 'Парковка', 'Состояние ремонта', 'Отдельный вход', 'Широта', 'Долгота', 'Общая площадь,\nкв.м']]
    
    # Масштабируем данные (используем transform)
    X_scaled = scaler.transform(X_target)

    y = data['Цена предложения,\n руб.']
    model = joblib.load('knn_model.joblib')

    # Оценка важности признаков с помощью Permutation Importance
    results = permutation_importance(model, X_scaled, y, n_repeats=30, random_state=42)
    
    # Получение важности признаков
    feature_importance = results.importances_mean
    feature_names = X_target.columns.tolist()

    # Очистка предыдущего графика
    plt.clf() 
    # Создание графика
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel("Важность признаков")
    plt.title("Важность признаков для модели KNN")
    
    # Отображение графика в Tkinter
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)  # Используем текущую фигуру
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=3, column=0, padx=10, pady=10)  # Позиционирование графика в фрейме
    canvas.draw()  # Отображаем график

def error_output(predicted_values, actual_prices):
     error = np.median(np.abs(actual_prices - predicted_values) / actual_prices * 100)  
     error_message = f'Ошибка предсказания: {error:.4f}%'
     return error_message

# Создаём новый объект Tk
root = tk.Tk()
root.title("Knn Нижний Новгород")

# Настраиваем позицию на экране и размеры окна
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_widht = 800  # Увеличен размер окна для графика
window_height = 600
x_centre = (screen_width//2) - (window_widht//2)
y_centre = (screen_height//2) - (window_height//2)
root.geometry(f"{window_widht}x{window_height}+{x_centre}+{y_centre}")

# Создаём новый frame
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Текстовая метка и кнопка выбора файла
label1 = tk.Label(frame, text="Это программа оценивает стоймость офисов. Выберите файл с офисами, которые нужно оценить", wraplength=300)
button1 = tk.Button(frame, text="Выберите файл excel", command=lambda: open_file_dialog(scaler))
label1.grid(row=0, column=0, padx=10)
button1.grid(row=1, column=0, padx=10, pady=15)

# Поле для вывода текста
text_output = tk.Text(frame, height=10, width=50)
text_output.grid(row=2, column=0, padx=10, pady=10)


root.mainloop()
