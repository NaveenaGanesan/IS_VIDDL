import pandas as pd
import os

directory_path = '/Users/naveenaganesan/Projects/IS/IS_VIDDL/Data/'

# Function to rename columns based on file name
def rename_columns(df, prefix):
    # Rename all columns except 'TimeSeries ID'
    df.rename(columns={col: f"{prefix}" for col in df.columns if col != 'TimeSeries ID'}, inplace=True)
    return df

def clean_and_combine_data():
    data_frames = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            data_frame = pd.read_csv(file_path)
            if 'TimeSeries ID' in data_frame.columns:
                data_frame = data_frame[~data_frame['TimeSeries ID'].isin(['project_id', 'location'])]
            prefix = os.path.splitext(filename)[0]
            print(f"Prefix: {prefix}")
            data_frame = rename_columns(data_frame, prefix)
            data_frames.append(data_frame)

        combined_data = data_frames[0]
        for df in data_frames[1:]:
            combined_data = pd.merge(combined_data, df, on='TimeSeries ID', how='outer')

        output_path = os.path.join(directory_path, 'combined_data.csv')
        combined_data.to_csv(output_path, index=False)

    print(f'Combined data saved to {output_path}')

if __name__ == "__main__":
    clean_and_combine_data()


