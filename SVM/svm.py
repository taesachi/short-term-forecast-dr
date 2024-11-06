import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import calendar
import sys
from datetime import datetime, timedelta

def preprocess_data(filepath):
    df = pd.read_csv(filepath, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                     names=['CNSMR_NO', 'LEGALDONG_CD', 'MESURE_YEAR', 'MESURE_MT', 'MESURE_DTTM', 'Hour', 'HLDY_AT', 'SGPowerUsage'])
    df['Date'] = pd.to_datetime(df[['MESURE_YEAR', 'MESURE_MT', 'MESURE_DTTM']].astype(str).agg('-'.join, axis=1)).dt.strftime('%Y%m%d')
    df['Hour'] = df['Hour'].astype(int)  # Ensure 'Hour' is an integer for modeling
    # We skip the HLDY_AT column as per instructions
    return df

def generate_future_dates(last_date_str, n_days):
    last_date = datetime.strptime(last_date_str, '%Y%m%d')
    start_of_next_month = datetime(last_date.year + (last_date.month // 12), ((last_date.month % 12) + 1), 1)
    end_date = start_of_next_month + timedelta(days=n_days)
    future_dates = pd.date_range(start=start_of_next_month, end=end_date, freq='H')[:-1]
    return future_dates

def prepare_future_dataset(df, future_dates):
    future_data_list = []
    for date in future_dates:
        df_temp = df[['CNSMR_NO', 'LEGALDONG_CD']].drop_duplicates().copy()
        df_temp['Date'] = date.strftime('%Y%m%d')
        df_temp['Hour'] = date.hour
        future_data_list.append(df_temp)
    future_df = pd.concat(future_data_list, ignore_index=True)
    return future_df

def train_and_predict(df):
    predictions_list = []
    grouped = df.groupby(['CNSMR_NO', 'LEGALDONG_CD'])
    for name, group in grouped:
        # 특성과 타겟 분리
        features = group[['Hour']]
        target = group['SGPowerUsage']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
        
        # 모델 훈련
        model = SVR(C=1.0, gamma='scale', kernel='rbf')
        model.fit(X_train, y_train)
        
        # 미래 데이터에 대한 예측을 위한 데이터셋 준비 및 예측
        condition = (df['CNSMR_NO'] == name[0]) & (df['LEGALDONG_CD'] == name[1])
        future_dates = generate_future_dates(df['Date'].max(), 30)
        future_df = prepare_future_dataset(df.loc[condition], future_dates)
        
        future_features = future_df[['Hour']]
        future_features_scaled = scaler.transform(future_features)
        predictions = model.predict(future_features_scaled)
        future_df['SGPowerUsage'] = predictions
        
        predictions_list.append(future_df)
    predictions_df = pd.concat(predictions_list, ignore_index=True)
    return predictions_df


def main(input_csv_file, output_csv_file):
    df = preprocess_data(input_csv_file)
    predicted_df = train_and_predict(df)
    predicted_df.to_csv(output_csv_file, index=False, columns=['LEGALDONG_CD', 'CNSMR_NO', 'Date', 'Hour', 'SGPowerUsage'])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_file> <output_csv_file>")
        sys.exit(1)
    
    input_csv_file = sys.argv[1]
    output_csv_file = sys.argv[2]
    main(input_csv_file, output_csv_file)
