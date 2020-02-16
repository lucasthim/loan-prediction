import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix,recall_score, classification_report, auc, roc_curve,roc_auc_score

plt.style.use('seaborn')

def evalute_model_performance(model,model_name,X,y,df_result):
    plot_confusion_matrix(df_result,model_name)
    acc_report = classification_report(df_result.TrueClass, df_result.Predicted,target_names =['Rejected', 'Approved'])
    print(acc_report)
    try:
        plot_ROC(model, model_name, X, y)
    except:
        print('Could not print ROC AUC curve.')


def plot_confusion_matrix(df,title,labels = ['Rejected', 'Approved'],set_type = 'Validation'):
    conf_matrix = confusion_matrix(df.TrueClass, df.Predicted)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    plt.title('{0} - Confusion matrix - {1} set'.format(title,set_type), fontsize = 20)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()
    return conf_matrix.ravel()

    

def plot_ROC(model,model_name,X_test,y_test):
    
    naive_probs = [0 for _ in range(len(y_test))]
    
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]

    naive_auc = roc_auc_score(y_test, naive_probs)
    model_auc = roc_auc_score(y_test, probs)

    print('No Skill: ROC AUC=%.3f' % (naive_auc))
    print(model_name,': ROC AUC=%.3f' % (model_auc))

    naive_fpr, naive_tpr, _ = roc_curve(y_test, naive_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    
    plt.plot(naive_fpr, naive_tpr, linestyle='--', label='Naive')
    plt.plot(model_fpr, model_tpr, marker='.', label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    