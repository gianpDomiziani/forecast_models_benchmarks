import greykite
from prophet import Prophet
import prophet.diagnostics
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
from time import time
from visualization import total_df, states


TIME_HORIZON = 28
END_TRAIN = 1913


def fbprophet_train(train_df, time_horizon, end_train, path=None):
    train_df = train_df.rename(columns={'date': 'ds', 'sales': 'y'})
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    mapes = []  # Store the MAPEs for each params here
    fit_times = []
    cv_times = []

    for params in tqdm(all_params):
        print(params)

        start_time = time()
        estimator = Prophet(**params, daily_seasonality=True).fit(train_df)
        end_time = time()
        fit_time = end_time - start_time
        fit_times.append(fit_time)

        start_time = time()
        df_cv = prophet.diagnostics.cross_validation(estimator,
                                                     initial=f'{end_train-(time_horizon+1)} days',
                                                     period=f'{time_horizon} days',
                                                     horizon=f'{time_horizon} days')
        end_time = time()
        cv_time = end_time - start_time
        cv_times.append(cv_time)
        # print(df_cv)

        df_p = prophet.diagnostics.performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])
        mapes.append(df_p['mape'].values[0])

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results['mape'] = mapes
    tuning_results['fit_time'] = fit_times
    tuning_results['cv_time'] = cv_times
    print(tuning_results)

    if path is not None:
        tuning_results.to_csv(path)


def fbprophet_predict(train_df, time_horizon, results_csv, path=None):
    tuning_results = pd.read_csv(results_csv)
    best_params = tuning_results[tuning_results.mape == tuning_results.mape.min()]
    best_params = best_params[best_params.fit_time == best_params.fit_time.min()]
    best_params = best_params[['changepoint_prior_scale', 'seasonality_prior_scale', 'holidays_prior_scale', 'seasonality_mode']]
    best_params = best_params.to_dict('records')[0]

    train_df = train_df.rename(columns={'date': 'ds', 'sales': 'y'})
    train_df['ds'] = pd.to_datetime(train_df['ds'])

    start_time = time()
    estimator = Prophet(**best_params, daily_seasonality=True).fit(train_df)
    end_time = time()
    fit_time = end_time - start_time
    print(fit_time)

    future = estimator.make_future_dataframe(periods=time_horizon)
    forecast = estimator.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    estimator.plot(forecast)

    if path is not None:
        forecast = forecast[['ds', 'yhat']]
        forecast.to_csv(path)


fbprophet_train(total_df, TIME_HORIZON, END_TRAIN, path='tuning_results\\total_level_dailyseasonailty_results.csv')

fbprophet_predict(total_df, TIME_HORIZON,
                  results_csv='tuning_results\\total_level_dailyseasonailty_results.csv',
                  path='predictions\\total_level_dailyseasonailty_predictions.csv')

# for state in states:
#     print(state)
#     state_df = states.get(state)[['date', 'sales']]
#     fbprophet_train(state_df, TIME_HORIZON, END_TRAIN, path=f'tuning_results\\{state}_results.csv')



