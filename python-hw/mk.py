import pandas as pd
import config

df = pd.read_csv(config.file_path_mk, sep=' ')

df.drop([0, 1], inplace=True)
df.reset_index(drop=True, inplace=True)

if df is not None:
    #print("\nСодержимое DataFrame для анализа масштабного коэффициента гироскопа:")
    #print(df)
    #print(df.info())
    df.to_excel(r'Data\MK.xlsx', index=False)
    #print("\nСодержимое успешно скопировано в 'Data\МК.xlsx'")
