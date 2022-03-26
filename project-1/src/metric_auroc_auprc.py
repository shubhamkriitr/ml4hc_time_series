from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
def plot_auroc(y_test,y_predictions):
    # num_thresholds when discretizing under the curve
    # curve = 'ROC' OR 'PR'
    false_pos , true_pos , thresholds = roc_curve (y_test,y_predictions)
    auc_keras = auc(false_pos, true_pos)
    
    ##plotting it 
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_pos, true_pos, label='(area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_predictions)
    lr_f1, lr_auc = f1_score(y_test, y_predictions>0.5), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    pyplot.plot(lr_recall, lr_precision, label='(area = {:.3f})'.format(lr_auc))
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

'''    tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC' ,
        summation_method='interpolation', name=None, dtype=None,
        thresholds=None, multi_label=False, num_labels=None, label_weights=None,
        from_logits=False
    )
'''