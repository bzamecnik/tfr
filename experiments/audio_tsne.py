from sklearn.manifold import TSNE

X_tsne = TSNE().fit_transform(X)

colors=np.linspace(0,1,len(X))
scatter(X_tsne[:,0], X_tsne[:,1], c=colors, alpha=0.1)
