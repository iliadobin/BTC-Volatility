import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def compare_model_families(results_df, feature_name, xlim=True):
    """
    Сравнивает семейство default HAR-RV моделей с семейством моделей, содержащих указанный признак
    с фиксированным порядком моделей на всех графиках
    
    Параметры:
    ----------
    results_df : pd.DataFrame
        DataFrame с результатами всех моделей
    feature_name : str
        Название признака для сравнения (например 'NormVolume')
    """
    
    metrics = ['Test_RMSE', 'Test_MAE', 'Test_MAPE']
    metric_names = ['RMSE', 'MAE', 'MAPE (%)']
    
    feature_models = results_df[results_df['features'].str.contains(feature_name)]
    
    model_order = [
        f"HAR-RV + {feature_name}",
        f"HAR-RV + {feature_name} (log RV)",
        f"HAR-RV + {feature_name} (log predictors)",
        f"HAR-RV + {feature_name} (log all)"
    ]
    
    existing_models = [m for m in model_order if m in feature_models['model_name'].values]
    
    default_models = results_df[results_df['features'] == 'rv_day_lag1, rv_week, rv_month']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Сравнение семейств моделей: HAR-RV vs HAR-RV + {feature_name}\n'
                f'Фиксированный порядок моделей на всех графиках', 
                y=1.05, fontsize=14)
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        if xlim:
            ax.set_xlim(left=-5, right=5)
        
        best_default = default_models.loc[default_models[metric].idxmin()]
        best_default_value = best_default[metric]
        
        differences = []
        colors = []
        
        for model_name in existing_models:
            model_row = feature_models[feature_models['model_name'] == model_name].iloc[0]
            diff_pct = 100 * (model_row[metric] - best_default_value) / best_default_value
            differences.append(diff_pct)
            colors.append('green' if diff_pct < 0 else 'red')
        
        y_pos = np.arange(len(existing_models))
        bars = ax.barh(y_pos, differences, color=colors, alpha=0.7)
        
        for bar in bars:
            width = bar.get_width()
            x_pos = min(differences)-2
            ax.text(x_pos, 
                    bar.get_y() + bar.get_height()/2., 
                    f'{width:.1f}%',
                    ha='left',
                    va='center',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3))
        
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.replace(f'HAR-RV + {feature_name}', '').strip() for m in existing_models])
        ax.set_xlabel(f'Разница в {metric_name} относительно лучшей HAR-RV модели (%)')
        ax.set_title(f'{metric_name} сравнение\n(лучшая HAR-RV: {best_default_value:.4f})')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Лучшие default HAR-RV модели для каждой метрики:")
    for metric, metric_name in zip(metrics, metric_names):
        best_default = default_models.loc[default_models[metric].idxmin()]
        print(f"\n{metric_name}:")
        print(f"  Название: {best_default['model_name']}")
        print(f"  Формула: {best_default['formula']}")
        print(f"  Значение: {best_default[metric]:.4f}")

def plot_volatility_comparison(forecasts_dict, model1_name, model2_name):
    """
    Сравнивает прогнозы двух моделей на одном графике с использованием Plotly
    
    Параметры:
    ----------
    forecasts_dict : dict
        Словарь с прогнозами всех моделей
    model1_name : str
        Название первой модели для сравнения
    model2_name : str
        Название второй модели для сравнения
    """
    
    dates = forecasts_dict[model1_name]['date']
    actual = forecasts_dict[model1_name]['actual']
    pred1 = forecasts_dict[model1_name]['predicted']
    pred2 = forecasts_dict[model2_name]['predicted']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        name='Actual RV',
        line=dict(color='green', width=2),
        opacity=0.8
    ))
    
    # прогноз первой модели
    fig.add_trace(go.Scatter(
        x=dates,
        y=pred1,
        name=f'{model1_name}',
        line=dict(color='blue', width=1.5),
        opacity=0.7
    ))
    
    # прогноз второй модели
    fig.add_trace(go.Scatter(
        x=dates,
        y=pred2,
        name=f'{model2_name}',
        line=dict(color='red', width=1.5),
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f'Сравнение моделей: {model1_name} vs {model2_name}',
        xaxis_title='Дата',
        yaxis_title='Реализованная волатильность',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        template="plotly_white",
        height=600,
        margin=dict(l=50, r=50, b=50, t=80)
    )
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    fig.show()
    