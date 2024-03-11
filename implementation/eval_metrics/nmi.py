# Author: WANG Xiaoyu

from sklearn import metrics

def nmi(labels_true, labels_pred):
    return metrics.normalized_mutual_info_score(labels_true, labels_pred)