# AgroAI · Crop Yield Predictor 

**Интерактивное веб-приложение для прогнозирования урожайности сельскохозяйственных культур с использованием RandomForestRegressor и визуализацией в Plotly.**

[![Streamlit App Status](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Описание проекта

**AgroAI** — это ML-приложение для предсказания урожайности культур (пшеница, рис, просо) в штате Пенджаб (Индия) на основе исторических данных. Модель анализирует ключевые параметры: год, тип культуры, осадки, тип почвы и площадь орошения.

**Основная задача**: Предоставить фермерам и агрономам точные прогнозы урожайности с доверительными интервалами и визуальной аналитикой.

### Главная страница
![Главная страница](screenshots/main_dashboard.png)
*Форма ввода параметров слева, результаты предсказания справа,под ними виузаулизации*

### Результаты предсказания
Predicted yield · 2025: 4,256 kg/ha
vs. Wheat average (4,123 kg/ha): +133 kg/ha
Model confidence: 87.4%
Range: 4,189 – 4,323 kg/ha (±37 kg/ha)


### Визуализации (8 графиков):
1. **Сравнение урожайности по культурам** (гистограмма с доверительными интервалами)
2. **Распределение предсказаний** (гистограмма 200 деревьев RandomForest)
3. **Историческая динамика** выбранной культуры + прогноз
4. **Осадки vs Урожайность** (scatter plot по культурам)
5. **Важность признаков** (горизонтальная столбчатая диаграмма)
6. **Доверительный gauge** (индикатор с цветовой градацией)
7. **Тепловая карта** (урожайность × год × уровень осадков)
8. **Орошение vs Урожай** + Box plot распределения

## Технологический стек

| Категория      | Технологии                                    |
|----------------|-----------------------------------------------|
| **ML**         | scikit-learn (RandomForestRegressor, n_estimators=200) |
| **Веб**        | Streamlit, Plotly Graph Objects               |
| **Обработка**  | pandas, numpy, sklearn.preprocessing          |
| **Кэширование**| `@st.cache_resource`                          |
| **Стилизация** | Custom CSS + Google Fonts (Playfair Display, DM Sans, DM Mono) |

### Входные параметры
Year (2024-2030)
State (Punjab)
Crop_Type (Wheat, Rice, Bajra)
Rainfall (mm/yr, 0-1000)
Soil_Type (Loamy, alluvial)
Irrigation_Area (ha, 0-4000)


### Исторические данные
- **55 записей** по Пенджабу (2000-2021)
- **Целевая переменная**: Crop_Yield (kg/ha)
- **Предобработка**: LabelEncoder, StandardScaler

### Метрики модели
✓ Доверительный интервал (±1σ по деревьям)
✓ Feature importance ranking
✓ Сравнение с историческим средним
✓ Confidence score


## Быстрый старт

# Клонировать репозиторий
git clone https://github.com/leramzor/AgroAI-ADA-2403M-Aruzhan-Zhanerke-Valeriya-Karina-AlisherSh.git
cd AgroAI-Crop-Yield-Predictor

# Установить зависимости
pip install -r requirements.txt

# Запустить приложение
streamlit run app.py

# Авторы
ADA-2403M | Astana IT University
Бисимбаева Аружан
Дуйсен Жанерке
Казагашева Валерия
Жумагулова Карина
Шаймуран Алишер
Проект для: НИЦ AgroTech 05.04.2026
