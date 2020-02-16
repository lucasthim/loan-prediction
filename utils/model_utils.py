import time

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

def predict(model,X,y):
    df_result = pd.DataFrame(columns = ['TrueClass','Predicted'])
    df_result.Predicted = model.predict(X.values)
    df_result.TrueClass = y.values.ravel()
    return df_result


def find_best_classification_model_with_cross_validation(model,parameters,X_train,y_train,k_folds = 10,metric = 'f1'):
    start = time.time()
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = parameters,
        cv = k_folds,
        scoring = metric, 
        verbose = 1, 
        n_jobs = -1)
    grid_search.fit(X_train,y_train)

    print("--- Ellapsed time: %s seconds ---" % (time.time() - start))
    print('Best params: ',grid_search.best_params_)
    print('Best score (%s)' % metric,grid_search.best_score_)
    return grid_search.best_estimator_,grid_search.best_params_, grid_search.best_score_