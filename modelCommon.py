import gc
from datetime import datetime
import scipy.stats as stats 
import scipy
import matplotlib.pyplot as plt
import numpy as np
try:
    from sklearn.externals import joblib
except:
    import joblib
from sklearn import metrics
import sys
def cleanKeras():
    try:
        del data
    except:
        print("no data")
    try:
        del model
    except:
        print("no model")
    try:
        keras_model.clear_session()
    except:
        print("no keras model")
        
    gc.collect()

def loadModel(modelpath):
    return keras.models.load_model(modelpath, compile=False)
    
def saveModel(modelname, model):
    """
    There are two formats you can use to save an entire model to disk: the TensorFlow SavedModel format, and the older Keras H5 format. The recommended format is SavedModel. It is the default when you use model.save().
    This is the savedModel format.
    """
    modelname = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + modelname
    model.save(modelname)
    
def fitAnomalyScore(y_pred):
    # gamma distribution
    shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
    gamma_params = [shape_hat, loc_hat, scale_hat]
    print(gamma_params)
    joblib.dump(gamma_params, datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + "anomalyscroeDistribution")
    
    # ref: https://www.statology.org/gamma-distribution-in-python/
    x = np.linspace (0, np.max(y_pred) + loc_hat, 200) 

    #calculate pdf of Gamma distribution for each x-value
    y = stats.gamma.pdf(x, a=shape_hat, scale=scale_hat, loc=loc_hat)

    #create plot of Gamma distribution
    plt.figure(figsize=(5, 5))
    plt.plot(x, y)
    plt.title("anomaly score gamma fit result")
    #display plot
    plt.show()
    
    return gamma_params


def getModelPerformance(yTrue, yPred, decision, max_fpr):
    # reference: https://github.com/Kota-Dohi/dcase2022_task2_baseline_ae/blob/main/01_test.py 
    # calculate scores
    auc = metrics.roc_auc_score(yTrue, yPred)
    max_fpr = 0.1
    p_auc = metrics.roc_auc_score(yTrue, yPred, max_fpr=max_fpr)
    tn, fp, fn, tp = metrics.confusion_matrix(yTrue, decision).ravel()
    prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
    recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
    f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)

    # print scores
    print(f"AUC       : {auc}")
    print(f"pAUC      : {p_auc}")
    print(f"precision : {prec}")
    print(f"recall    : {recall}")
    print(f"F1 score  : {f1}")
    
    return [auc, p_auc, prec, recall, f1]