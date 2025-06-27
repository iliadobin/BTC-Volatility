import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from copy import deepcopy

from datetime import time

import math
import pandas as pd
import numpy as np
import datetime
from datetime import time
from datetime import timedelta


def parse_date(date_int):
    date_str = str(date_int)
    day = int(date_str[4:6])
    month = int(date_str[2:4])
    year = 2000 + int(date_str[:2])
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_time(time_int):
    time_str = f"{time_int:06d}"
    hour = int(time_str[:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    return time(hour, minute, second)


def process_finam_data(df):
    df = df.copy()
    df['<DATE>'] = pd.to_datetime(df['<DATE>'].apply(parse_date))
    df['<TIME>'] = df['<TIME>'].apply(parse_time)
    return df


def compute_rv(data):
    data = process_finam_data(data)
    
    rv_tmp=0
    rr_itog=[]
    count=0
    rv_itog=[]
    count_itog=[]
    date=[]
    r_date=[]

    t_time=[]

    rv_itog_ne_shk=[]

    r_kon_date=[]
    r_nach_date=[]
    time_nachal=[]
    time_kon=[]

    datetimeFormat = '%H:%M:%S'    
    time1 = '00:00:00'
    time2 = '23:55:00'  
    timedelta = datetime.datetime.strptime(time1, datetimeFormat) - datetime.datetime.strptime(time2,datetimeFormat)

    time_nach=[]
    time_nach.append(0)


    for i in range(0,len(data)-1):
        if str(data['<DATE>'][i+1])==str(data['<DATE>'][i]):  
            z=0
        else:
            time_nach.append(i+1)



    for i in range(0,len(time_nach)-1):
        rv_tmp=0
        count=0

        for j in range(time_nach[i], time_nach[i+1]-1):
            rv_tmp=rv_tmp+(math.log(float(data['<CLOSE>'][j+1]))-math.log(float(data['<CLOSE>'][j])))**2
            count=count+1            
            if str(data['<TIME>'][j+2])==str(time1):
                    j=time_nach[i+1]-1
                    rv_tmp=rv_tmp+(math.log(float(data['<CLOSE>'][j+1]))-math.log(float(data['<CLOSE>'][j])))**2
                    count=count+1

 
        if str(data['<TIME>'][time_nach[i+1]])==str(time1) and str(data['<TIME>'][time_nach[i]])==str(time1):
            rr_tmp=(math.log(float(data['<CLOSE>'][time_nach[i+1]]))-math.log(float(data['<CLOSE>'][time_nach[i]+1])))
        elif str(data['<TIME>'][time_nach[i+1]])!=str(time1) and str(data['<TIME>'][time_nach[i]])==str(time1):
            rr_tmp=(math.log(float(data['<CLOSE>'][time_nach[i+1]-1]))-math.log(float(data['<CLOSE>'][time_nach[i]+1])))
        elif str(data['<TIME>'][time_nach[i+1]])!=str(time1) and str(data['<TIME>'][time_nach[i]])!=str(time1):
            rr_tmp=(math.log(float(data['<CLOSE>'][time_nach[i+1]-1]))-math.log(float(data['<CLOSE>'][time_nach[i]])))
        elif str(data['<TIME>'][time_nach[i+1]])==str(time1) and str(data['<TIME>'][time_nach[i]])!=str(time1):
            rr_tmp=(math.log(float(data['<CLOSE>'][time_nach[i+1]]))-math.log(float(data['<CLOSE>'][time_nach[i]])))
        else:
            zz=0
        
        rv_itog.append(rv_tmp)
        rr_itog.append(rr_tmp)
        date.append(data['<DATE>'][time_nach[i]])
        count_itog.append(count)

        razn=data['<DATE>'][time_nach[i+1]]-data['<DATE>'][time_nach[i]]
        r_date.append(razn.days-1)

        r2=datetime.datetime.strptime(str(data['<TIME>'][time_nach[i]]), datetimeFormat) - datetime.datetime.strptime(time1,datetimeFormat)
        r_nach_date.append((r2.seconds)/(60*5))
        

        if str(data['<TIME>'][time_nach[i+1]])==str(time1):
            r1=datetime.datetime.strptime(time1, datetimeFormat) - datetime.datetime.strptime(str(data['<TIME>'][time_nach[i+1]]),datetimeFormat)
            r_kon_date.append((r1.seconds)/(60*5))
        else:
            r1=datetime.datetime.strptime(time1, datetimeFormat) - datetime.datetime.strptime(str(data['<TIME>'][time_nach[i+1]-1]),datetimeFormat)
            r_kon_date.append((r1.seconds)/(60*5))


        if str(data['<TIME>'][time_nach[i+1]])==str(time1):
            time_nachal.append(time((data['<TIME>'][time_nach[i]]).hour,(data['<TIME>'][time_nach[i]]).minute,(data['<TIME>'][time_nach[i]]).second))
            time_kon.append(time((data['<TIME>'][time_nach[i+1]]).hour,(data['<TIME>'][time_nach[i+1]]).minute,(data['<TIME>'][time_nach[i+1]]).second))
        else:
            time_nachal.append(time((data['<TIME>'][time_nach[i]]).hour,(data['<TIME>'][time_nach[i]]).minute,(data['<TIME>'][time_nach[i]]).second))
            time_kon.append(time((data['<TIME>'][time_nach[i+1]-1]).hour,(data['<TIME>'][time_nach[i+1]-1]).minute,(data['<TIME>'][time_nach[i+1]-1]).second))

    
    rv_tmp=0
    count=0
    for i in range(time_nach[len(time_nach)-1],len(data)-1):
            rv_tmp=rv_tmp+(math.log(float(data['<CLOSE>'][i+1]))-math.log(float(data['<CLOSE>'][i])))**2
            count=count+1

    rv_itog.append(rv_tmp)
    
    if str(data['<TIME>'][time_nach[len(time_nach)-1]])==str(time1):
        rr_tmp=(math.log(float(data['<CLOSE>'][len(data)-1]))-math.log(float(data['<CLOSE>'][time_nach[len(time_nach)-1]+1])))
    elif str(data['<TIME>'][time_nach[len(time_nach)-1]])!=str(time1):
        rr_tmp=(math.log(float(data['<CLOSE>'][len(data)-1]))-math.log(float(data['<CLOSE>'][time_nach[len(time_nach)-1]])))
    else:
        zz=0
    
    rr_itog.append(rr_tmp)
    date.append(data['<DATE>'][time_nach[-1]])
    count_itog.append(count)

    razn=data['<DATE>'][time_nach[-1]]-data['<DATE>'][time_nach[-2]]
    r_date.append(razn.days-1)

    r2=datetime.datetime.strptime(str(data['<TIME>'][time_nach[-1]]), datetimeFormat) - datetime.datetime.strptime(time1,datetimeFormat)
    r_nach_date.append((r2.seconds)/(60*5))

    r1=datetime.datetime.strptime(time2, datetimeFormat) - datetime.datetime.strptime(str(data['<TIME>'][len(data)-1]),datetimeFormat)
    r_kon_date.append((r1.seconds)/(60*5))

    time_nachal.append(time((data['<TIME>'][time_nach[-1]]).hour,(data['<TIME>'][time_nach[-1]]).minute,(data['<TIME>'][time_nach[-1]]).second))
    time_kon.append(time((data['<TIME>'][len(data)-1]).hour,(data['<TIME>'][len(data)-1]).minute,(data['<TIME>'][len(data)-1]).second))


    na_vivod_rv=pd.DataFrame({'date': date,
                                'rv_ne_shk': rv_itog,
                                'rr_ne_shk': rr_itog,
                                'razn': r_date,
                                'count_itog':count_itog,
                                'r_nach_date': r_nach_date,
                                'r_kon_date': r_kon_date,
                                'time_nach': time_nachal,
                                'time_kon': time_kon
                                     })
    na_vivod_rv['rv_shk']=288*na_vivod_rv.rv_ne_shk/(288-na_vivod_rv.r_nach_date-na_vivod_rv.r_kon_date)
    na_vivod_rv['rr_shk']=288*na_vivod_rv.rr_ne_shk/(288-na_vivod_rv.r_nach_date-na_vivod_rv.r_kon_date)

    na_vivod_rv['shk']=(288-na_vivod_rv.r_nach_date-na_vivod_rv.r_kon_date)

    return na_vivod_rv


def process_har_rv_df(raw_rv):
    har_rv_df = raw_rv[['date','rv_ne_shk']]

    har_rv_df = har_rv_df.rename(columns={'rv_ne_shk' : 'rv_day'})
    har_rv_df['rv_day_lag1'] = har_rv_df['rv_day'].shift(1)
    har_rv_df['rv_week'] = har_rv_df['rv_day'].shift(1).rolling(window=7).mean()
    har_rv_df['rv_month'] = har_rv_df['rv_day'].shift(1).rolling(window=30).mean()
    har_rv_df = har_rv_df.dropna()
    har_rv_df = har_rv_df.reset_index(drop=True)
    return har_rv_df


def rolling_har_regression(df, window_size=300, 
                          target_col='rv_day',
                          base_predictors=['rv_day_lag1', 'rv_week', 'rv_month'],
                          add_predictors=None,
                          pred_col_name='tomorrow_pred',
                          models_col_name='models',
                          log_target=False,
                          log_predictors=False,
                          add_constant=True):
    forecasts = []
    models = []
    
    train_mse_list = []
    train_mae_list = []
    train_mape_list = []
    
    predictors = base_predictors.copy()
    if add_predictors is not None:
        predictors.extend(add_predictors)
    
    temp_df = df.copy()
    
    if log_target:
        temp_df[target_col] = np.log(temp_df[target_col])
        
    if log_predictors:
        for predictor in predictors:
            if predictor in temp_df.columns:
                temp_df[predictor] = np.log(temp_df[predictor] + 1e-10)
    
    for t in range(window_size, len(temp_df)):
        train_data = temp_df.iloc[t - window_size:t]
        X_train = train_data[predictors]
        y_train = train_data[target_col]
        
        if add_constant:
            X_train = sm.add_constant(X_train)
        
        model = sm.OLS(y_train, X_train).fit()
        
        # Расчет метрик на тренировочной выборке
        y_train_pred = model.predict(X_train)
                
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-10))) * 100
        
        train_mse_list.append(train_mse)
        train_mae_list.append(train_mae)
        train_mape_list.append(train_mape)
        
        # Прогнозирование на тестовой выборке
        X_test = temp_df[predictors].iloc[t:t+1]
        if add_constant:
            X_test = sm.add_constant(X_test, has_constant='add')
        
        forecast = model.predict(X_test)
        
        if log_target:
            forecast = np.exp(forecast)
            
        forecasts.append(forecast.iloc[0])
        models.append(model)
    
    df[pred_col_name] = np.nan
    df[models_col_name] = np.nan
    
    df.loc[window_size:, pred_col_name] = forecasts
    df.loc[window_size:, models_col_name] = models
    
    # Расчет метрик на тестовой выборке
    actuals = df[target_col][window_size:len(df)-1]
    predictions = df[pred_col_name][window_size:len(df)-1]
    
    valid_mask = (~actuals.isna()) & (~predictions.isna())
    actuals = actuals[valid_mask]
    predictions = predictions[valid_mask]
    
    train_metrics = {
        'Train_MSE': np.mean(train_mse_list),
        'Train_MAE': np.mean(train_mae_list),
        'Train_MAPE': np.mean(train_mape_list)
    }
    
    test_metrics = {
        'Test_RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
        'Test_MAE': mean_absolute_error(actuals, predictions),
        'Test_MAPE': np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
    }
    
    metrics = {**train_metrics, **test_metrics}
    
    return df, metrics

def evaluate_all_har_models(df, additional_features=[], window_size=300):
    """
    Оценивает HAR-RV модели с каждым признаком отдельно
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм со всеми необходимыми признаками для моделей: (rv_day, rv_day_lag1, rv_week, rv_month) + кастомные
    additional_features : list
        Список дополнительных признаков для тестирования (каждый будет добавлен отдельно)
    window_size : int
        Размер скользящего окна
    
    Возвращает:
    -----------
    results_df : pd.DataFrame
        Таблица с результатами всех моделей
    forecasts_dict : dict
        Словарь с прогнозами всех моделей
    """
    
    base_features = ['rv_day_lag1', 'rv_week', 'rv_month']
    
    # Создаем список всех моделей для тестирования:
    # 1. Базовая HAR-RV
    # 2. HAR-RV + каждый дополнительный признак по отдельности
    models_to_test = [{'name': 'HAR-RV', 'features': base_features}]
    
    for feature in additional_features:
        models_to_test.append({
            'name': f'HAR-RV + {feature}',
            'features': base_features + [feature]
        })
    
    log_options = [
        {'name_suffix': '', 'log_target': False, 'log_predictors': False},
        {'name_suffix': ' (log RV)', 'log_target': True, 'log_predictors': False},
        {'name_suffix': ' (log predictors)', 'log_target': False, 'log_predictors': True},
        {'name_suffix': ' (log all)', 'log_target': True, 'log_predictors': True}
    ]
    
    results = []
    forecasts_dict = {}
    
    for model in models_to_test:
        for log_opt in log_options:
            temp_df = deepcopy(df)
            
            model_name = model['name'] + log_opt['name_suffix']
            
            # Формируем математическое представление модели
            if log_opt['log_target']:
                formula = r"$\log{RV_{t+1}}$ = "
            else:
                formula = r"$RV_{t+1}$ = "
            
            formula += r"$\beta_0$ + $\beta_d$"
            if log_opt['log_predictors']:
                formula += r"$\log{RV_t^d}$ + $\beta_w$" + r"$\log{RV_t^w}$ + $\beta_m$" + r"$\log{RV_t^m}$"
            else:
                formula += r"$RV_t^d$ + $\beta_w$" + r"$RV_t^w$ + $\beta_m$" + r"$RV_t^m$"
            
            for feat in model['features']:
                if feat not in base_features:
                    formula += f" + $\\beta_{{{feat}}}$"
                    if log_opt['log_predictors']:
                        formula += f"$\\log{{{feat}}}$"
                    else:
                        formula += f"${feat}$"
            
            # Запускаем регрессию
            model_df, metrics = rolling_har_regression(
                temp_df,
                window_size=window_size,
                add_predictors=[f for f in model['features'] if f not in base_features],
                log_target=log_opt['log_target'],
                log_predictors=log_opt['log_predictors']
            )
            
            results.append({
                'model_name': model_name,
                'formula': formula,
                'features': ', '.join(model['features']),
                'log_target': log_opt['log_target'],
                'log_predictors': log_opt['log_predictors'],
                **metrics
            })
            
            forecasts_dict[model_name] = {
                'date': model_df['date'].iloc[window_size:],
                'actual': model_df['rv_day'].iloc[window_size:],
                'predicted': model_df['tomorrow_pred'].iloc[window_size:],
                'model': model_name,
                'formula': formula
            }
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values('Test_RMSE')
    
    return results_df, forecasts_dict

