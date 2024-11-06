import pandas as pd
from pmdarima import auto_arima
import numpy as np
from datetime import datetime, timedelta
import calendar
from tqdm.auto import tqdm
import sys

def preprocess_chunk(chunk):
    # 'Date' 컬럼을 MESURE_YEAR, MESURE_MT, MESURE_DTTM의 조합으로 생성합니다.
    chunk['Date'] = pd.to_datetime(chunk['MESURE_YEAR'].astype(str) + '-' +
                                   chunk['MESURE_MT'].astype(str).str.zfill(2) + '-' +
                                   chunk['MESURE_DTTM'].astype(str).str.zfill(2),
                                   format='%Y-%m-%d').dt.strftime('%Y%m%d')
    # HLDY_AT 컬럼을 정수형으로 변환: 'Y'는 1, 'N'은 0으로 처리합니다.
    chunk['HLDY_AT'] = chunk['HLDY_AT'].map({'Y': 1, 'N': 0})
    return chunk

def load_and_preprocess_data(filepath, chunksize=100000):
    tqdm.pandas(desc="Loading & Preprocessing Data")
    # HLDY_AT 컬럼 포함하여 데이터 로드
    iterator = pd.read_csv(filepath, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7], names=['CNSMR_NO', 'LEGALDONG_CD', 'MESURE_YEAR', 'MESURE_MT', 'MESURE_DTTM', 'Hour', 'HLDY_AT', 'SGPowerUsage'], chunksize=chunksize)
    df = pd.concat([preprocess_chunk(chunk) for chunk in tqdm(iterator)], ignore_index=True)
    return df

def auto_select_order_and_predict(df, n_periods):
    results = []
    for (leg_cd, cns_no), group in tqdm(df.groupby(['LEGALDONG_CD', 'CNSMR_NO']), desc="Model Fitting & Prediction"):
        series = group['SGPowerUsage'].values
        # HLDY_AT 컬럼을 외부 설명 변수로 사용
        exogenous = group['HLDY_AT'].values.reshape(-1, 1)
        auto_model = auto_arima(series, seasonal=True, m=12, exogenous=exogenous, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        forecast = auto_model.predict(n_periods=n_periods, exogenous=np.tile(exogenous[-1], (n_periods, 1)))
        next_month_date = pd.date_range(start=group['Date'].max(), periods=n_periods + 1, freq='h')[1:]
        for i, date in enumerate(next_month_date):
            results.append([leg_cd, cns_no, date.strftime('%Y%m%d'), date.strftime('%H'), forecast[i]])
    return results

def save_forecast(results, output_file):
    forecast_df = pd.DataFrame(results, columns=['LEGALDONG_CD', 'CNSMR_NO', 'Date', 'Hour', 'SGPowerUsage'])
    forecast_df.to_csv(output_file, index=False, header=False)

def main(input_csv_file, output_csv_file):
    df = load_and_preprocess_data(input_csv_file)
    last_date_str = df['Date'].max()
    last_date = datetime.strptime(last_date_str, '%Y%m%d')
    next_month = last_date + pd.DateOffset(months=1)
    start_of_next_month = datetime(next_month.year, next_month.month, 1)
    _, last_day_of_next_month = calendar.monthrange(start_of_next_month.year, start_of_next_month.month)
    n_periods = last_day_of_next_month * 24  # 24 hours for each day of the month

    results = auto_select_order_and_predict(df, n_periods)
    save_forecast(results, output_csv_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_file> <output_csv_file>")
        sys.exit(1)

    input_csv_file = sys.argv[1]
    output_csv_file = sys.argv[2]
    main(input_csv_file, output_csv_file)
