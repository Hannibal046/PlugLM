from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

def accuracy(preds, labels):
    return float((preds == labels).mean())

def binary_f1(preds, labels):   
    return float(f1_score(y_true=labels, y_pred=preds))

def spearman(preds,labels):
    return float(spearmanr(preds, labels)[0])

def pearson(preds,labels):
    return float(pearsonr(preds, labels)[0])

def macro_f1(preds,labels):
    return f1_score(y_true=labels,y_pred=preds,average='macro')

def micro_f1(preds,labels): 
    return f1_score(y_true=labels,y_pred=preds,average='micro')