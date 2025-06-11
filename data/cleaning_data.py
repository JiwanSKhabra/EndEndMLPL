import pandas as pd
import numpy as np

csv_file_path = 'AB_NYC_2019.csv'

df = pd.read_csv(csv_file_path)
print(df.head())