import pandas as pd

# Загрузка данных из Parquet
df = pd.read_parquet('results/results.parquet')

# Сохранение в CSV
df.to_csv('results/results.csv', index=False)

print("Данные успешно")