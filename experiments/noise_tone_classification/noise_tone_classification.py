def generate_noise_frames(shape):
    'fake normalized white noise magnitude spectrum'
    count, size = shape
    return np.ones(shape) / size

def generate_tone_frames(shape):
    'fake spectrum frames with aligned pure sine tones'
    frames = np.zeros(shape)
    count, size = shape
    idxs = np.random.random_integers(0, size-1, size=count)
    frames[np.arange(count), idxs] = 1
    return frames

def gen_dataset(counts_by_category, dimensions):
    def fun_by_cat(cat):
        if cat == 0:
            return generate_tone_frames
        elif cat == 1:
            return generate_noise_frames
    
    X = np.vstack([
        fun_by_cat(cat)((counts_by_category[cat], dim)) \
        for cat in counts_by_category])
    Y = np.hstack([
        np.zeros(counts_by_category[cat]) + cat \
        for cat in counts_by_category])
    count = sum([counts_by_category[cat] for cat in counts_by_category])
    random_idxs = np.arange(count)
    np.random.shuffle(random_idxs)
    X = X[random_idxs, :]
    Y = Y[random_idxs]
    return X, Y

dim = 100
X, Y = gen_dataset({0: 1000, 1: 1000}, dim)
X_test, Y_test = gen_dataset({0: 1000, 1: 1000}, dim)

logistic = LogisticRegression(penalty='l1')
kpca = KernelPCA(kernel="rbf", n_components=dim)
Pipeline(steps=[('kpca', kpca), ('logistic', logistic)]) \
    .fit(X,Y) \
    .score(X_test, Y_test)

#Pipeline(steps=[('sgd', SGDClassifier(loss='perceptron', penalty='l1'))]) \
#    .fit(X,Y).score(X_test, Y_test)

# Notes:
# - dimensionality reduction:
#   - linear:
#     - MiniBatchDictionaryLearning works as well, but it is slow
#     - FastICA is good, fast, but does not give best results
#     - NMF with init='nndsvd'|'nndsvda' - quite good, slow
#   - manifold:
#     - Isomap(n_components=10) - slow, works with low n_components
# - classification:
#   - LogisticRegression - with penalty='l1' - works good, quite fast
#   - SGDClassifier - loss='perceptron', penalty='l1' - not that bad
#     - not that general for dimensionality-reduction preprocessings
