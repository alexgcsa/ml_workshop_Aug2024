from math import sqrt
from matplotlib import pyplot as plt
import plotly.express as px
from plotly.offline import iplot
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay
from tabulate import tabulate
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm

def scatter_plot(y, y_pred, label=""):
    sns.regplot(x=y, y=y_pred, fit_reg=True)
    plt.title(label)
    plt.ylabel('actual')
    plt.xlabel('prediction')
    plt.tight_layout()
    plt.show()

def regression_metrics(y_train, y_pred_train, y_test, y_pred_test):
    rmse_train = round(sqrt(mean_squared_error(y_train, y_pred_train)), 2)
    pearsons_train = round(pearsonr(y_train, y_pred_train)[0], 2)
    kendalls_train = round(kendalltau(y_train, y_pred_train)[0], 2)
    spearmans_train = round(spearmanr(y_train, y_pred_train)[0], 2)
    rmse_test = round(sqrt(mean_squared_error(y_test, y_pred_test)), 2)
    pearsons_test = round(pearsonr(y_test, y_pred_test)[0], 2)
    kendalls_test = round(kendalltau(y_test, y_pred_test)[0], 2)
    spearmans_test = round(spearmanr(y_test, y_pred_test)[0], 2)

    d = [ ["RMSE", rmse_train, rmse_test],
         ["Pearson's", pearsons_train, pearsons_test],
         ["Kendall's", kendalls_train, kendalls_test],
         ["Spearman's", spearmans_train, spearmans_test]]

    print(tabulate(d, headers=["Metric", "Training", "Test"]))
    return True

def classification_metrics(y, y_pred):
    rmse = round(sqrt(mean_squared_error(y, y_pred)), 2)
    pearsons = round(pearsonr(y, y_pred)[0], 2)
    kendalls = round(kendalltau(y, y_pred)[0], 2)
    spearmans = round(spearmanr(y, y_pred)[0], 2)
    print("RMSE:\t\t{}".format(rmse))
    print("Pearson's:\t{}".format(pearsons))
    print("Kendall's:\t{}".format(kendalls))
    print("Spearman's:\t{}".format(spearmans))
    return True

def feat_importance(model):
    labels = []
    scores = []
    try:
        for feature,score in zip(model.feature_names_in_, model.feature_importances_):
            if score != 0:
                labels.append(feature)
                scores.append(round(score,2))
    except AttributeError as e:
        for feature,score in zip(range(0, len(model.feature_importances_)), model.feature_importances_):
            if score != 0:
                labels.append(feature)
                scores.append(round(score,2))
    fig = px.bar(x=labels, y=scores, title="Feature importance")
    fig.update_layout(yaxis_title="Importance score", xaxis_title="Features")
    iplot(fig)

def gen_train_test_performances(y_train, pred_train, y_test, pred_test, classes):
    print("\033[1m" + "Performance on Training (k-fold cross validation)" + "\033[0m")
    output_dict_train = classification_report(y_train, pred_train, output_dict=True)
    output_train_df = pd.DataFrame(output_dict_train)
    output_train_df = output_train_df.drop("accuracy",axis=1)
    output_train_df = output_train_df.drop("support",axis=0)
    print(output_train_df.round(3))    
    
    print("\n\n")
    print("\033[1m" + "Performance on Test set" +  "\033[0m")
    output_dict_test = classification_report(y_test, pred_test, output_dict=True)
    output_test_df = pd.DataFrame(output_dict_test)
    output_test_df = output_test_df.drop("accuracy",axis=1)
    output_test_df = output_test_df.drop("support",axis=0)
    print(output_test_df.round(3)) 
    print("\n\n\n\n")

    cm_train = confusion_matrix(y_train, pred_train)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)

    cm_test = confusion_matrix(y_test, pred_test)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    disp_train.plot(ax=ax1, colorbar=False)
    disp_train.ax_.set_title("k-fold Cross-validation")
    disp_test.plot(ax=ax2, colorbar=False)
    disp_test.ax_.set_title("Test")
    plt.show()

def plot_clf_performance(y_train, pred_train, y_test, pred_test, target):
    fig = make_subplots(rows=1, cols=2, start_cell="bottom-left")

    fig.add_trace(go.Scatter(x=y_train, y=pred_train, name='Training set', \
                             mode="markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_test, y=pred_test, name='Test set', \
                             mode="markers"), row=1, col=2)

    x = sm.add_constant(y_train)
    p = sm.OLS(pred_train, x).fit().params
    x = np.arange(y_train.min(), y_train.max())
    y = p.const + p[target] * x
    fig.add_trace(go.Scatter(x=x, y=y, name='', \
                             mode="lines", \
                             line=dict(dash='dash', color="black"),\
                             showlegend=False), \
                  row=1, col=1)
    x = sm.add_constant(y_test)
    p = sm.OLS(pred_test, x).fit().params
    x = np.arange(y_test.min(), y_test.max())
    y = p.const + p[target] * x
    fig.add_trace(go.Scatter(x=x, y=y, name='', \
                             mode="lines", \
                             line=dict(dash='dash', color="black"),\
                             showlegend=False), \
                  row=1, col=2)
    iplot(fig)

def gen_train_test_roc(y_train, pred_train, y_test, pred_test, classes):
    y_train_tmp = label_binarize(y_train, classes=classes)
    y_pred_train_tmp = label_binarize(pred_train, classes=classes)

    y_test_tmp = label_binarize(y_test, classes=classes)
    y_pred_test_tmp = label_binarize(pred_test, classes=classes)

    fig = make_subplots(rows=1, cols=2, start_cell="bottom-left")

    fpr, tpr, thresholds = roc_curve(y_train_tmp, y_pred_train_tmp)
    score_train = round(auc(fpr, tpr), 2)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name='Training set - AUC:{}'.format(score_train), \
                            stackgroup = 'one'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='', mode='lines', \
                             line=dict(dash='dash', color="gray"), \
                             showlegend=False), row=1, col=1)

    fpr, tpr, thresholds = roc_curve(y_test_tmp, y_pred_test_tmp)
    score_test = round(auc(fpr, tpr), 2)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name='Test set - AUC:{}'.format(score_test), \
                            stackgroup = 'one'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='', mode='lines', \
                             line=dict(dash='dash', color="gray"), \
                             showlegend=False), row=1, col=2)

    iplot(fig)

def plot_train_test_class(y_train, y_test):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])

    fig.add_trace(go.Pie(values=y_train.value_counts().values, \
                        labels=y_train.value_counts().index, \
                        title='Training set'), \
                 row=1, col=1)

    fig.add_trace(go.Pie(values=y_test.value_counts().values, \
                        labels=y_train.value_counts().index, \
                        title='Test set'),
                 row=1, col=2)

    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                     marker=dict(colors=['lightblue','lightcoral'], line=dict(color='#000000', width=2)))

    iplot(fig)
