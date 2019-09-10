# Some codes (e.g. edgeConstruction.py) are referred to the RCC code [1], please approximately use and refer to the related works.
# [1] Shah, S. A. & Koltun, V. Robust continuous clustering. PNAS 114, 98149819 (2017). 


import comic as comic

from data import load_data
from clustering_metric import clustering_evaluate

if __name__ == '__main__':

    data_name = 'Caltech101-20'

    X_list, Y = load_data(data_name)
    view_size = len(X_list)
    data_size = Y.shape[0]
    print 'View Number: %d, Data Size: %d' % (view_size, data_size)
    
    gamma = 1
    pair_rate = 0.9
    
    import time
    start = time.time()

    COMIC = comic.COMIC(view_size = view_size, data_size=data_size, measure='cosine',
                        pair_rate=pair_rate, gamma=gamma, max_iter=1000)
    labels = COMIC.fit(X_list)

    elapsed = (time.time() - start)
    print 'Time used: %.2f' %elapsed
    
    label = labels['vote']
    print 'Evaluation:'
    nmi, v_mea = clustering_evaluate(Y, label)
    print 'nmi: %.4f, v_mea: %.4f' % (nmi, v_mea)