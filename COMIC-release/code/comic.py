import math
import itertools

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from scipy.sparse import csr_matrix, triu, find
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial import distance

class COMIC:

    def __init__(self, view_size, data_size, k=10, measure='cosine',
                 clustering_threshold=1., eps=1e-5, pair_rate = 0.01, gamma=1,
                 max_iter = 200, verbose=True):

        self.n_samples = data_size
        self.view_size = view_size
        self.k = k
        self.measure = measure
        self.clustering_threshold = clustering_threshold
        self.eps = eps
        self.verbose = verbose
        self.pair_rate = pair_rate
        self.gamma = gamma
        self.max_iter = max_iter


        self.labels_ = None
        self.Z = None
        self.i = None
        self.j = None

    def fit(self, X_list):
        """
        Computes the clustering and returns the labels
        Parameters
        ----------
        X_list (list) list of data to cluster with shape (view_size)
        """

        assert type(X_list) == list
        assert len(X_list) == self.view_size

        print '\n*** Compute m-knn graph ***\n'
        mknn_list = []
        for view in range(self.view_size):
            print 'compute m-knn graph of view', view+1
            X = np.squeeze(X_list[view])
            m_knn_matrix = self.m_knn(X, self.k, measure=self.measure)
            print 'm_knn_matrix', m_knn_matrix.shape
            mknn_list.append(m_knn_matrix)

        # perform the COMIC clustering
        self.labels = self.run_COMIC(X_list, mknn_list)

        # return the computed labels
        return self.labels

    def pretrain(self, X_list, w_list):
        # preprocess
        # list
        self.i_list = []
        self.j_list = []
        xi_list = []
        weights_list = []
        Z_list = []
        S_list = []
        epsilon_list = []
        lamb_list = []
        epsilon_mean_list = []
        mu_list = []
        max_iter = self.max_iter

        print "\n*** Initiation ***\n"
        for view in range(self.view_size):
            X = X_list[view]
            w = w_list[view]
            X = X.astype(np.float32)  # features stacked as N x D (D is the dimension)
            w = w.astype(np.int32)  # list of edges represented by start and end nodes
            # make sure w as size () * 2
            assert w.shape[1] == 2
            
            # initialization
            n_samples, n_features = X.shape
            n_pairs = w.shape[0]
            
            # slice w for convenience
            i = w[:, 0]
            j = w[:, 1]

            # precomputing xi
            xi = np.linalg.norm(X, 2)

            # sparsity
            R = scipy.sparse.coo_matrix((np.ones((i.shape[0] * 2,)),
                                            (np.concatenate([i, j], axis=0),
                                            np.concatenate([j, i], axis=0))), shape=[n_samples, n_samples])

            # number of connections
            n_conn = np.sum(R, axis=1)

            # make sure to convert back to a numpy array from a numpy matrix, since the output of the sum() operation on a
            # sparse matrix is a numpy matrix
            n_conn = np.asarray(n_conn)

            # equation [S1]
            weights = np.mean(n_conn) / np.sqrt(n_conn[i] * n_conn[j])
            weights = np.squeeze(weights)
            

            S = np.ones((i.shape[0],))
            Z = X.copy()

            # compute epsilon_mean, see SI for details
            epsilon = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2 + self.eps, axis=1))
            # Note: suppress low values. This hard coded threshold could lead to issues with very poorly normalized data.
            epsilon[epsilon / np.sqrt(n_features) < 1e-2] = np.max(epsilon)
            epsilon = np.sort(epsilon)
            
            # compute mu, see section Graduated Nonconvexity in the SI
             
            mu = epsilon[-1] ** 2

            # take the top 90% of the closest neighbours as a heuristic
            top_samples = np.minimum(250.0, math.ceil(n_pairs * self.pair_rate))

            epsilon_mean = np.mean(epsilon[:int(top_samples)])
            epsilon = np.mean(epsilon[:int(math.ceil(n_pairs * self.pair_rate))])

            # computation of matrix A = D-R (here D is the diagonal matrix and R is the symmetric matrix), see equation (8)
            
            R = scipy.sparse.coo_matrix((np.concatenate([weights * S, weights * S], axis=0),
                                            (np.concatenate([i, j], axis=0), np.concatenate([j, i], axis=0))),
                                        shape=[n_samples, n_samples])

            D = scipy.sparse.coo_matrix((np.squeeze(np.asarray(np.sum(R, axis=1))),
                                            ((range(n_samples), range(n_samples)))),
                                        (n_samples, n_samples))

            # initial computation of lamb
            # note: compute the largest magnitude eigenvalue instead of the matrix norm as it is faster to compute
            eigval = scipy.sparse.linalg.eigs(D - R, k=1, return_eigenvectors=False).real

            # Calculate lambda as per equation 9.
            lamb = xi / eigval[0]

            if self.verbose:
                print 'View', view
                print'lambda = %.6f, epsilon = %.6f, mu = %.6f' %(lamb, epsilon, mu)
            # save to list
            self.i_list.append(i)
            self.j_list.append(j)
            Z_list.append(Z)
            S = self.to_matrix(S, i, j, (self.n_samples, self.n_samples))
            S_list.append(S)
            epsilon_list.append(epsilon)
            mu_list.append(mu)
            xi_list.append(xi)
            epsilon_mean_list.append(epsilon_mean)
            lamb_list.append(lamb)
            weights = self.to_matrix(weights, i, j, (self.n_samples, self.n_samples))
            weights_list.append(weights)
        return S_list, Z_list, weights_list, lamb_list, epsilon_list, mu_list, xi_list, epsilon_mean_list

    def run_COMIC(self, X_list, w_list):
        """
        Main function for computing the clustering.

        Parameters
        ----------
        X (array) 2d array of data of shape (n_samples, n_dim).
        w (array) weights for each edge, as computed by the mutual knn clustering.
        max_iter (int) maximum number of iterations to run the algorithm.
        S_list: cluster assignement graph
        Z_list: representation
        weights_list:
        lamb_list: parameter in the objective
        epsilon_list: connected edges based on X
        epsilon_mean_list: the mean of the length of the shortest such as 90% of the edge in epsilon_list
        mu_list: parameter for the estimetor
        xi_list: spectral norm of X
        """

        max_iter = self.max_iter

        # preprocess S, Z, and so on
        S_list, Z_list, weights_list, lamb_list, epsilon_list, mu_list, xi_list, epsilon_mean_list = self.pretrain(X_list=X_list, w_list=w_list)
        Z_final_concat = np.concatenate((Z_list[:]), axis=1)

        # pre-allocate memory for the values of the objective function
        obj = np.zeros((max_iter,))

        hist_obj = []
        hist_nmi = []
        n_samples = X_list[0].shape[0]

        print '\n*** Training ***\n'
        # start of optimization phase
        for iter_num in range(1, max_iter):
            S_list_old = S_list[:]
            # compute objective. 
            obj[iter_num] = self.compute_obj(X_list, Z_list, S_list, lamb_list, mu_list, weights_list, iter_num)
            for view in range(self.view_size):
                X = X_list[view]
                w = w_list[view]
                X = X.astype(np.float32)  # features stacked as N x D (D is the dimension)
                w = w.astype(np.int32)  # list of edges represented by start and end nodes
                n_samples, n_features = X.shape
                n_pairs = w.shape[0]

                i = self.i_list[view]
                j = self.j_list[view]
                
                # update S.
                dist = self.to_matrix(np.sum((Z_list[view][i, :]-Z_list[view][j, :])**2, axis=1), i, j, (self.n_samples, self.n_samples))
                S_list[view] = self.update_S(S_list_old, view, lamb_list[view], mu_list[view], weights_list[view], dist)
                
                # update Z.
                R = weights_list[view] * (S_list[view]**2)
                R = scipy.sparse.coo_matrix(R)
                D = scipy.sparse.coo_matrix((np.asarray(np.sum(R, axis=1))[:, 0], ((range(n_samples), range(n_samples)))),
                                            shape=(n_samples, n_samples))
                L = D-R
                
                M = scipy.sparse.eye(n_samples) + lamb_list[view] * L
                
                # Solve for Z. This could be further optimised through appropriate preconditioning.
                Z_list[view] = scipy.sparse.linalg.spsolve(M, X)

                # update lamb
                eigval = scipy.sparse.linalg.eigs(L, k=1, return_eigenvectors=False).real
                # Calculate lambda as per equation 9.
                lamb = xi_list[view] / eigval[0]
                lamb_list[view] = lamb

            if (abs(obj[iter_num - 1] - obj[iter_num]) < 1e-8) and iter_num > 99:
                print 'Early stop'
                break

        # at the end of the run, assign values to the class members.
        self.Z_list = Z_list
        self.S_list = S_list
        labels = self.compute_assignment(epsilon_list)

        return labels
     
    def update_S(self, S_list, view_, lamb, mu, weights, dist):
        S = 0
        for view in range(self.view_size):
            if view_ == view:
                continue
            S += self.gamma * S_list[view]
        div = self.gamma*(self.view_size - 1) + lamb * weights * dist
        S = (S+mu) / (div+mu)
        return S

    def compute_assignment(self, epsilon_list):
        ret = {}
        is_conn_list = []
        for view in range(self.view_size):
            # computing connected components.
            diff = self.EuclideanDistances(self.Z_list[view], self.Z_list[view])
            is_conn = np.sqrt(diff) <= self.clustering_threshold*epsilon_list[view]
            is_conn = is_conn + 0
            is_conn_list.append(is_conn)
        
        conn = 0
        for is_conn in is_conn_list:
            conn = conn + is_conn
        conn = conn > self.view_size/2

        G = scipy.sparse.coo_matrix(conn)
        num_components, labels = connected_components(G, directed=False)
        ret['vote'] = labels

        return ret

    def compute_obj(self, X_list, Z_list, S_list, lamb_list, mu_list, weights_list, iter_num):
        # L_1
        L1 = 0
        l1 = 0
        l2 = 0
        for view in range(self.view_size):
            l1 += 0.5 * np.mean(np.sum((X_list[view] - Z_list[view])**2, axis=1))
            i = self.i_list[view]
            j = self.j_list[view]
            dist = self.to_matrix(np.sum((Z_list[view][i, :]-Z_list[view][j, :])**2, axis=1), i, j, (self.n_samples, self.n_samples))
            dot = weights_list[view] * (S_list[view]**2) * dist
            l2 += 0.5 * lamb_list[view] * (np.mean(dot)+ mu_list[view] * np.mean((S_list[view]-1)**2))
        L1 = (l1+l2) / self.view_size

        # L_2
        L2 = 0
        # generate permutation of different views
        ls = itertools.permutations(range(self.view_size), 2)
        
        for (view_i, view_j) in ls:
            L2 += 0.5 * np.mean((S_list[view_i]-S_list[view_j])**2)
        L2 = L2 * 0.5
        # final objective
        loss = L1 + self.gamma * L2

        if self.verbose:
            print 'iter: %d,  loss: %.6f' %(iter_num, loss)
        return loss

    @staticmethod
    def m_knn(X, k, measure='cosine'):
       
        samples = X.shape[0]
        batch_size = 10000
        b = np.arange(k+1)
        b = tuple(b[1:].ravel())

        z = np.zeros((samples, k))
        weigh = np.zeros_like(z)

        for x in np.arange(0, samples, batch_size):
            start = x
            end = min(x+batch_size, samples)

            w = distance.cdist(X[start:end], X, measure)

            y = np.argpartition(w, b, axis=1)

            z[start:end, :] = y[:, 1:k + 1]
            weigh[start:end, :] = np.reshape(w[tuple(np.repeat(np.arange(end-start), k)),
                                               tuple(y[:, 1:k+1].ravel())], (end-start, k))
            del w

        ind = np.repeat(np.arange(samples), k)

        P = csr_matrix((np.ones((samples*k)), (ind.ravel(), z.ravel())), shape=(samples, samples))
        Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))

        Tcsr = minimum_spanning_tree(Q)
        P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
        P = triu(P, k=1)

        V = np.asarray(find(P)).T
        return V[:, :2].astype(np.int32)

    def EuclideanDistances(self, A, B):
        BT = B.transpose()
        vecProd = np.dot(A,BT)
        SqA =  A**2
     
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

        SqB = B**2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
        SqED = sumSqBEx + sumSqAEx - 2*vecProd
        SqED[SqED<0]=0.0   
        ED = np.sqrt(SqED)
        return ED

    def to_matrix(self, S_, i, j, shape, is_symmetric=True):
        S = np.zeros(shape)
        for cnt in range(len(S_)):
            if is_symmetric:
                S[i[cnt], j[cnt]] = S_[cnt]
                S[j[cnt], i[cnt]] = S_[cnt]
            else:
                S[i[cnt], j[cnt]] = S_[cnt]
        return S
