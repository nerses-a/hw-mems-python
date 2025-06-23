import pandas as pd
from . import config

df = pd.read_csv(config.file_path_dreif, sep=' ')

df.drop([0, 1], inplace=True)
df.reset_index(drop=True, inplace=True)

if df is not None:
    #print("\nСодержимое DataFrame для анализа дрейфа гироскопа:")
    #print(df)
    #print(df.info())
    df.to_excel(r'Data\Дрейф.xlsx', index=False)
    #print("\nСодержимое успешно скопировано в 'Data\Дрейф.xlsx'")
