import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import sys
from datetime import datetime, timedelta
import calendar

# 데이터 전처리 함수
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['SGPowerUsage_scaled'] = scaler.fit_transform(df[['SGPowerUsage']])
    return df, scaler

# 시퀀스 생성 함수
def create_sequences(df, n_steps):
    X, y = [], []
    for i in range(n_steps, len(df)):
        X.append(df['SGPowerUsage_scaled'].iloc[i-n_steps:i].values)
        y.append(df['SGPowerUsage_scaled'].iloc[i])
    return np.array(X), np.array(y)

# LSTM 모델 생성 함수
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 각 그룹별로 다음달 예측 수행 함수
def predict_for_each_group(group, n_steps, model, scaler):
    predictions = []
    last_date_str = group['Date'].iloc[-1]
    last_date = datetime.strptime(last_date_str, '%Y%m%d')
    start_next_month = (last_date.replace(day=28) + timedelta(days=4)).replace(day=1)
    end_next_month = (start_next_month + timedelta(days=31)).replace(day=1) - timedelta(days=1)
    total_hours_next_month = pd.date_range(start=start_next_month, end=end_next_month, freq='H')

    # 시퀀스 데이터 준비
    X_new = group['SGPowerUsage_scaled'].values[-n_steps:].reshape((1, n_steps, 1))
    
    for date in total_hours_next_month:
        predicted_scaled = model.predict(X_new)
        predicted = scaler.inverse_transform(predicted_scaled).flatten()[0]

        predictions.append((date.strftime('%Y%m%d'), date.strftime('%H'), predicted))

        # 다음 시퀀스 데이터 준비
        new_row = [predicted_scaled.flatten()[-1]]
        X_new = np.append(X_new.flatten()[1:], new_row).reshape((1, n_steps, 1))

    return predictions

# 메인 함수
def main(input_csv_file, output_csv_file, n_steps=24):
    df = pd.read_csv(input_csv_file, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                     names=['CNSMR_NO', 'LEGALDONG_CD', 'MESURE_YEAR', 'MESURE_MT', 'MESURE_DTTM', 'Hour', 'HLDY_AT', 'SGPowerUsage'])
    df['Date'] = pd.to_datetime(df[['MESURE_YEAR', 'MESURE_MT', 'MESURE_DTTM']].astype(str).agg('-'.join, axis=1)).dt.strftime('%Y%m%d')
    df, scaler = preprocess_data(df)

    grouped = df.groupby(['LEGALDONG_CD', 'CNSMR_NO'])
    results = []

    for (legal_cd, cns_no), group in tqdm(grouped, desc="Processing Groups"):
        group_sorted = group.sort_values('Date').reset_index(drop=True)
        if len(group_sorted) >= n_steps:
            X, y = create_sequences(group_sorted, n_steps)
            model = build_lstm_model((n_steps, 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

            predictions = predict_for_each_group(group_sorted, n_steps, model, scaler)
            for pred in predictions:
                results.append([legal_cd, cns_no] + list(pred))

    results_df = pd.DataFrame(results, columns=['LEGALDONG_CD', 'CNSMR_NO', 'Date', 'Hour', 'SGPowerUsage'])
    results_df.to_csv(output_csv_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_file> <output_csv_file>")
        sys.exit(1)

    input_csv_file = sys.argv[1]
    output_csv_file = sys.argv[2]
    main(input_csv_file, output_csv_file)