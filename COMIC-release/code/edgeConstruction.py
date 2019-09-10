import numpy as np

def kNN(X, k, measure='euclidean'):
    """
    Construct pairwise weights by finding the k nearest neighbors to each point
    and assigning a Gaussian-based distance.

    Parameters
    ----------
    X : [n_samples, n_dim] array
    k : int
        number of neighbors for each sample in X
    """
    from scipy.spatial import distance

    weights = []
    w = distance.cdist(X, X, measure)
    y = np.argsort(w, axis=1)

    for i, x in enumerate(X):
        distances, indices = w[i, y[i, 1:k + 1]], y[i, 1:k + 1]
        for (d, j) in zip(distances, indices):
            if i < j:
                weights.append((i, j, d*d))
            else:
                weights.append((j, i, d*d))
    weights = sorted(weights, key=lambda r: (r[0], r[1]))
    return unique_rows(np.asarray(weights))


def mkNN(X, k, measure='euclidean'):
    """
    Construct mutual_kNN for large scale dataset

    If j is one of i's closest neighbors and i is also one of j's closest members,
    the edge will appear once with (i,j) where i < j.

    Parameters
    ----------
    X : [n_samples, n_dim] array
    k : int
      number of neighbors for each sample in X
    """
    from scipy.spatial import distance
    from scipy.sparse import csr_matrix, triu, find
    from scipy.sparse.csgraph import minimum_spanning_tree

    samples = X.shape[0]
    batchsize = 10000
    b = np.arange(k+1)
    b = tuple(b[1:].ravel())

    z=np.zeros((samples,k))
    weigh=np.zeros_like(z)

    # This loop speeds up the computation by operating in batches
    # This can be parallelized to further utilize CPU/GPU resource
    for x in np.arange(0, samples, batchsize):
        start = x
        end = min(x+batchsize,samples)

        w = distance.cdist(X[start:end], X, measure)

        y = np.argpartition(w, b, axis=1)

        z[start:end,:] = y[:, 1:k + 1]
        weigh[start:end,:] = np.reshape(w[tuple(np.repeat(np.arange(end-start), k)), tuple(y[:, 1:k+1].ravel())], (end-start, k))
        del(w)

    ind = np.repeat(np.arange(samples), k)

    P = csr_matrix((np.ones((samples*k)), (ind.ravel(), z.ravel())), shape=(samples,samples))
    Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples,samples))

    Tcsr = minimum_spanning_tree(Q)
    P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
    P = triu(P, k=1)

    return np.asarray(find(P)).T