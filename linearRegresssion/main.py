import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import numpy as np
import sys
from tqdm import tqdm

def simple_linear_regression(x, y):
    slope, intercept, _, _, _ = linregress(x, y)
    return slope, intercept

def trend_line(x, slope, intercept):
    return slope * x + intercept

def calculate_x_values(group):
    # 동일한 Hour값을 가진 그룹별로 처리
    grouped = group.groupby(['Date', 'Hour'])
    x_values = np.zeros(len(group))
    for _, subgroup in grouped:
        count = len(subgroup)
        for i, (_, row) in enumerate(subgroup.iterrows()):
            idx = group.index.get_loc(row.name)
            #x_values[idx] = int(row['Date'].strftime('%Y%m%d')) + (i * (1.0 / count))
            x_values[idx] = int(row['Date']) + (i * (1.0 / count))
    return x_values

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, header=None, usecols=[0, 1, 2, 3, 4], names=['LEGALDONG_CD', 'CNSMR_NO', 'Date', 'Hour', 'SGPowerUsage'])

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    # Exclude specific dates
    excluded_dates = ['1130', '1231']
    df['MonthDay'] = df['Date'].dt.strftime('%m%d')
    df = df[~df['MonthDay'].isin(excluded_dates)]
    
    # Drop the 'MonthDay' column as it's no longer needed
    df.drop('MonthDay', axis=1, inplace=True)
    
    df['Date'] = df['Date'].dt.strftime('%Y%m%d')

    df.sort_values(by=['CNSMR_NO', 'Date', 'Hour'], inplace=True)
    # Exclude data for November 30th and December 31st
    #df = df[~df['Date'].isin(['20211130', '20211231'])]

    return df

def plot_and_save(df, output_dir, filename_without_extension):
    slope_dict = {}  # CNSMR_NO별 기울기 저장을 위한 딕셔너리
    for (cns_no, leg_cd), group in tqdm(df.groupby(['CNSMR_NO', 'LEGALDONG_CD']), desc='Overall Progress'):
        # X 축 값 계산
        x = calculate_x_values(group)
        y = group['SGPowerUsage'].values
        slope, intercept = simple_linear_regression(x, y)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='SG Power Usage')
        plt.plot(x, trend_line(x, slope, intercept), color='red', label='Trend Line')
        plt.xlabel('Time')
        plt.ylabel('SG Power Usage')
        plt.title(f'SG Power Usage Trend for {cns_no}')
        plt.legend()

        leg_output_dir = os.path.join(output_dir, f'{leg_cd}')
        os.makedirs(leg_output_dir, exist_ok=True)
        output_filepath = os.path.join(leg_output_dir, f'{cns_no}_{filename_without_extension}.png')

        plt.savefig(output_filepath)
        plt.close()
        
        group.to_csv(os.path.join(leg_output_dir, f'{cns_no}_{filename_without_extension}.csv'), index=False, header=False)

        slope_dict[cns_no] = [slope, intercept]  # 기울기 저장
    
    # CNSMR_NO별 기울기 출력
    for cns_no, formula in slope_dict.items():
        print(f"The slope (trend) for CNSMR_NO {cns_no} is y={formula[0]}x + {formula[1]}, indicating {'an increase' if slope > 0 else 'a decrease' if slope < 0 else 'no change'} in SG Power Usage over time.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    output_dir = 'output'
    filename_without_extension = os.path.splitext(os.path.basename(csv_file_path))[0]
    #output_dir = filename_without_extension
    
    df = load_and_preprocess_data(csv_file_path)
    plot_and_save(df, output_dir, filename_without_extension)

if __name__ == "__main__":
    main()
