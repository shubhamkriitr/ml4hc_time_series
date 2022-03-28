from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
import os

def plot_auroc(y_test,y_predictions, figure_save_location_prefix=None, 
        plot_tag=""):
    # num_thresholds when discretizing under the curve
    # curve = 'ROC' OR 'PR'
    false_pos , true_pos , thresholds = roc_curve (y_test,y_predictions)
    auc_keras = auc(false_pos, true_pos)
    
    ##plotting it 
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_pos, true_pos, label='(area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve {plot_tag}')
    plt.legend(loc='best')
    if figure_save_location_prefix is None:
        plt.show()
    else:
        save_path = figure_save_location_prefix + "ROC.png"
        plt.savefig(save_path)

    plt.figure()
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_predictions)
    lr_f1, lr_auc = f1_score(y_test, y_predictions>0.5, average="macro"), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    pyplot.plot(lr_recall, lr_precision, label='(area = {:.3f})'.format(lr_auc))
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    plt.title(f"ROC {plot_tag   }")
    # show the legend
    pyplot.legend()

    if figure_save_location_prefix is None:
        # show the plot
        plt.show()
    else:
        # save the plot
        save_path = figure_save_location_prefix + "AUPRC.png"
        plt.savefig(save_path)