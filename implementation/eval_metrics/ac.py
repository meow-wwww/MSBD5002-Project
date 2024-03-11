# Author: WANG Xiaoyu

from coclust.evaluation.external import accuracy

def ac(labels_true, labels_pred):
    return accuracy(labels_true, labels_pred)
    # 确实比sklearn.metrics.rand_score和sklearn.metrics.adjusted_rand_score更准确