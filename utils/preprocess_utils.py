import pandas as pd
import scipy.stats as sp
import numpy as np

def remove_outliers(df,columns_to_analyse,threshold = 4):
    z_score = np.abs(sp.zscore(df[columns_to_analyse]))
    df_no_outlier = df.loc[(z_score < threshold).all(axis=1),:].copy()
    print(df_no_outlier.shape)
    return df_no_outlier

def encode_with_nan(df_input,categorical_columns,ordinal_encoder):
    df = df_input.copy()
    for category,col in zip(ordinal_encoder.categories_,categorical_columns):
        for index, label in enumerate(category):
            df.loc[df[col] == label,col] = index
    return df
