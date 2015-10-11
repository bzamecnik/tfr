from scipy.sparse import coo_matrix, csr_matrix

def reduce(X, percentile=99.9, sparse=False):
    '''
    Select values from an array (eg. a 2D spectrum) greater than given
    percentile. The result is 1D.
    '''
    idxs = X > np.percentile(X, percentile)
    if not sparse:
        return X[idxs]
    else:
        return to_sparse(X, idxs)

def to_sparse(X, indexes):
    return csr_matrix(coo_matrix((X[indexes], np.where(indexes))))

def energy_ratio(X, X_reduced):
    '''
    how much energy was retained after reduction
    '''
    return energy(X_reduced) / energy(X.reshape(-1))
