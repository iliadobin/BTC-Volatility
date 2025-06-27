# Документация проекта HAR-RV для прогнозирования волатильности биткойна

## Основные функции

### `parse_date(date_int)`
Преобразует целочисленное представление даты (формат Finam) в строку формата `YYYY-MM-DD`.

**Параметры:**
- `date_int` (int): Дата в формате Finam (YYMMDD)

**Возвращает:**
- `str`: Дата в формате `YYYY-MM-DD`

---

### `parse_time(time_int)`
Преобразует целочисленное представление времени в объект `time`.

**Параметры:**
- `time_int` (int): Время в формате HHMMSS

**Возвращает:**
- `datetime.time`: Объект времени

---

### `process_finam_data(df)`
Обрабатывает сырые данные из Finam, преобразуя даты и время.

**Параметры:**
- `df` (pd.DataFrame): Исходный DataFrame с колонками `<DATE>` и `<TIME>`

**Возвращает:**
- `pd.DataFrame`: Обработанный DataFrame с корректными типами дат и времени

---

### `compute_rv(data)`
Вычисляет реализованную волатильность (RV) и логарифмические доходности из минутных данных.

**Параметры:**
- `data` (pd.DataFrame): Минутные данные цен с колонками `<DATE>`, `<TIME>`, `<CLOSE>`

**Возвращает:**
- `pd.DataFrame`: DataFrame с колонками:
  - `date`: Дата
  - `rv_ne_shk`: Нескорректированная RV
  - `rr_ne_shk`: Логарифмическая доходность
  - `rv_shk`: Скорректированная RV
  - `rr_shk`: Скорректированная доходность
  - и другие технические колонки

---

### `process_har_rv_df(raw_rv)`
Подготавливает DataFrame для HAR-RV модели.

**Параметры:**
- `raw_rv` (pd.DataFrame): DataFrame с реализованной волатильностью

**Возвращает:**
- `pd.DataFrame`: DataFrame с колонками:
  - `rv_day`: RV за день
  - `rv_day_lag1`: RV за предыдущий день
  - `rv_week`: Средняя RV за неделю
  - `rv_month`: Средняя RV за месяц

---

### `rolling_har_regression(df, window_size=300, ...)`
Выполняет скользящую регрессию HAR-RV.

**Основные параметры:**
- `df`: Входной DataFrame
- `window_size`: Размер окна для обучения (по умолчанию 300)
- `target_col`: Целевая переменная (по умолчанию 'rv_day')
- `base_predictors`: Базовые предикторы
- `add_predictors`: Дополнительные предикторы
- `log_target`: Логарифмировать целевую переменную
- `log_predictors`: Логарифмировать предикторы

**Возвращает:**
- `tuple`: (df_with_forecasts, metrics_dict)
  - `df_with_forecasts`: DataFrame с прогнозами
  - `metrics_dict`: Словарь с метриками качества

---

### `evaluate_all_har_models(df, additional_features=[], window_size=300)`
Оценивает различные варианты HAR-RV моделей.

**Параметры:**
- `df`: Входной DataFrame
- `additional_features`: Список дополнительных признаков
- `window_size`: Размер окна

**Возвращает:**
- `tuple`: (results_df, forecasts_dict)
  - `results_df`: DataFrame с результатами всех моделей
  - `forecasts_dict`: Словарь с прогнозами всех моделей
