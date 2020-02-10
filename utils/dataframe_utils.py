import pandas as pd

def show_column_options(df):
    print('Column Values:')
    cols = df.columns
    for col in cols:
        print(col,':',df[col].unique())