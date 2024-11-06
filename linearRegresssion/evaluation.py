import pandas as pd
import numpy as np
import sys

def calculate_error_metrics_per_consumer(predicted_csv_path, actual_csv_path):
    # 헤더가 없는 CSV 파일 로드, 필요한 열만 선택
    predicted_data = pd.read_csv(predicted_csv_path, header=None, usecols=[0, 1, 2, 3, 4])
    actual_data = pd.read_csv(actual_csv_path, header=None, usecols=[0, 1, 2, 3, 4])
    
    # 열 이름 지정
    columns = ['LEGALDONG_CD', 'CNSMR_NO', 'Date', 'Hour', 'SGPowerUsage']
    predicted_data.columns = columns
    actual_data.columns = columns
    
    # 시간대별로 예측 데이터와 실제 데이터 병합
    merged_data = pd.merge(predicted_data, actual_data, on=['CNSMR_NO', 'Date', 'Hour'], suffixes=('_pred', '_actual'))
    
    # CNSMR_NO 별로 그룹화하여 각 오차 메트릭 계산
    error_metrics = {}
    for consumer_no, group in merged_data.groupby('CNSMR_NO'):
        # SGPowerUsage 값의 차이 계산
        group['error'] = group['SGPowerUsage_pred'] - group['SGPowerUsage_actual']
        
        # MSE 계산
        mse = np.mean(np.square(group['error']))
        
        # MAE 계산
        mae = np.mean(np.abs(group['error']))
        
        # RMSE 계산
        rmse = np.sqrt(mse)
        
        # 결과 저장
        error_metrics[consumer_no] = (mse, mae, rmse)
        
    return error_metrics

if __name__ == "__main__":
    predicted_csv_path = sys.argv[1]
    actual_csv_path = sys.argv[2]
    
    error_metrics = calculate_error_metrics_per_consumer(predicted_csv_path, actual_csv_path)
    for consumer_no, metrics in error_metrics.items():
        mse, mae, rmse = metrics
        print(f"CNSMR_NO {consumer_no}: MSE: {mse}, MAE: {mae}, RMSE: {rmse}")