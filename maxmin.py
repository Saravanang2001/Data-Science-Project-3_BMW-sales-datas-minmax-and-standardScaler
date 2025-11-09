import matplotlib.pyplot as pt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as ps
bmw = ps.read_csv("j:/Data Science Project/minmax and standard scaler/BMW_Datas.csv")
print(bmw)
print("Min Max Scaler")
numeric_col = bmw.select_dtypes(include=['int64', 'float64']).columns
minmax_scaler = MinMaxScaler()
bmw_normalized = ps.DataFrame(minmax_scaler.fit_transform(bmw[numeric_col]), columns=numeric_col)
print(bmw_normalized.head())
print("Standard Scaler")
standard_scaler = StandardScaler()
bmw_standardized = ps.DataFrame(standard_scaler.fit_transform(bmw[numeric_col]), columns=numeric_col)
print(bmw_standardized.head())
pt.figure(figsize=(12, 6))
pt.hist(bmw['Sales_Volume'], bins=5)
pt.title("Distribution of Sales of BMW")
pt.xlabel("Sales Volume")
pt.ylabel("Frequency")
pt.show()
