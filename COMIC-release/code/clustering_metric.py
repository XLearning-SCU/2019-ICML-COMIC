import numpy as np
from sklearn import metrics


def clustering_evaluate(gtlabels, labels):
    
    nmi = metrics.normalized_mutual_info_score(gtlabels, labels)
    v_mea = metrics.v_measure_score(gtlabels, labels)
    
    return nmi, v_mea