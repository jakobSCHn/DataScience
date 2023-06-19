import matplotlib.pyplot as plt
import plotly.graph_objects as go
import kaleido
import os

from sklearn.metrics import roc_curve, confusion_matrix, auc


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def calculateMetrics(tn, fp, fn, tp, title_entry, subtitle_entry):
    balanced_accuracy = round(0.5 * ((tp/(tp + fn)) + (tn/(tn + fp))), 3)

    #plot the confusion matrix:
    extension = "png"
    filename = "ConfusionMatrix" + title_entry + subtitle_entry
    path = os.path.join("..\plots", f"{filename}.{extension}") #get path for saving
    tableConfMatrix = go.Figure(
        data=[go.Table(header=dict(values=["", "Predicted: 0", "Predicted: 1"]),
                       cells=dict(values=[["Observed: 0", "Observed: 1"],
                                          [tn, fn],
                                          [fp, tp]]),
                       columnwidth=[100])])
    tableConfMatrix.update_layout(title="ConfusionMatrix: " + title_entry + "_" + subtitle_entry, width=500, height=300)
    tableConfMatrix.write_image(path)
    
    return [balanced_accuracy]


def plotROCcurve(fp_ratesDef, tp_ratesDef, fp_ratesOpt, tp_ratesOpt, roc_aucDef, roc_aucOpt, title_entry):
    extension = "png"
    filename = "ROCcurves" + title_entry
    path = os.path.join("..\plots", f"{filename}.{extension}")  # get path for saving

    fig, ax = plt.subplots(ncols=2, figsize=(8, 6))
    ax[0].plot(fp_ratesDef, tp_ratesDef, label='default\nparameters')
    ax[0].set_xlabel('FPR')
    ax[0].set_ylabel('TPR')
    add_identity(ax[0], color="r", ls="--", label='random\nclassifier')
    ax[0].text(0.7, 0, f"AUC: {roc_aucDef}", fontsize=10, bbox=dict(facecolor="white", edgecolor= "black", boxstyle = "round"))
    ax[0].legend()
    ax[0].title.set_text('Default parameters')

    ax[1].plot(fp_ratesOpt, tp_ratesOpt, label='optimized\nparameters')
    ax[1].set_xlabel('FPR')
    ax[1].set_ylabel('TPR')
    add_identity(ax[1], color="r", ls="--", label='random\nclassifier')
    ax[1].text(0.7, 0, f"AUC: {roc_aucOpt}", fontsize=10, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"))
    ax[1].legend()
    ax[1].title.set_text('Optimized Parameters')
    fig.suptitle('ROC curves ' + title_entry)

    plt.tight_layout()
    plt.savefig(path)


def plotMetrics(metricsDef, metricsOpt, title_entry):
    a = metricsDef
    b = metricsOpt
    extension = "png"
    filename = "EvaluationMetrics" + title_entry
    path = os.path.join("..\plots", f"{filename}.{extension}")  #get path for saving
    tableConfMatrix = go.Figure(
        data=[go.Table(header=dict(values=["", "Default", "Optimized"]),
                       cells=dict(values=[["Balanced Accuracy", "ROC_AUC"],
                                          a,
                                          b]),
                       columnwidth=[150])])
    tableConfMatrix.update_layout(title="Evaluation Metrics: " + title_entry, width=500, height=400)
    tableConfMatrix.write_image(path)


def evaluation(y, y_predDefault, y_probaDefault, y_predOpt, y_probaOpt, title_entry='my_model'):
    # ---------- Get the evaluation metrics ----------
    #Get the confusion matrix
    tnDef, fpDef, fnDef, tpDef = confusion_matrix(y, y_predDefault).ravel()
    tnOpt, fpOpt, fnOpt, tpOpt = confusion_matrix(y, y_predOpt).ravel()
    #Calculate the evaluation metrics
    metricsDef = calculateMetrics(tnDef, fpDef, fnDef, tpDef, title_entry, "Default")
    print(metricsDef)
    metricsOpt = calculateMetrics(tnOpt, fpOpt, fnOpt, tpOpt, title_entry, "Optimized")

    #Get the roc curve
    fp_ratesDef, tp_ratesDef, _ = roc_curve(y, y_probaDefault)
    fp_ratesOpt, tp_ratesOpt, _ = roc_curve(y, y_probaOpt)
    # Calculate the area under the roc curve
    roc_aucDef = round(auc(fp_ratesDef, tp_ratesDef), 3)
    roc_aucOpt = round(auc(fp_ratesOpt, tp_ratesOpt), 3)
    metricsDef.append(roc_aucDef)
    metricsOpt.append(roc_aucOpt)

    # ---------- Get the evaluation plots ----------
    plotMetrics(metricsDef, metricsOpt, title_entry)
    plotROCcurve(fp_ratesDef, tp_ratesDef, fp_ratesOpt, tp_ratesOpt, roc_aucDef, roc_aucOpt, title_entry)


    return metricsDef, metricsOpt

def evaluationTypes(y, y_pred, imbalance, model, cancerType):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    balanced_accuracy = round(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))), 3)

    # plot the confusion matrix:
    extension = "png"
    filename = "ConfusionMatrix" + model + "_" + cancerType
    path = os.path.join("..\plots", f"{filename}.{extension}")  # get path for saving
    tableConfMatrix = go.Figure(
        data=[go.Table(header=dict(values=["", "Predicted: 0", "Predicted: 1"]),
                       cells=dict(values=[["Observed: 0", "Observed: 1", "Imb./Bal_acc"],
                                          [tn, fn, imbalance],
                                          [fp, tp, balanced_accuracy]]),
                       columnwidth=[200])])
    tableConfMatrix.update_layout(title="ConfusionMatrix: " + model + " " + cancerType, width=500, height=300)
    tableConfMatrix.write_image(path)

    return balanced_accuracy
