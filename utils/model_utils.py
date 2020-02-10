import pandas as pd

def predict(model,X,y):
    df_result = pd.DataFrame(columns = ['TrueClass','Predicted'])
    df_result.Predicted = model.predict(X.values)
    df_result.TrueClass = y.values.ravel()
    return df_result