import numpy as np
from plotly import graph_objects as go
from sklearn.datasets import make_blobs
import pfls

# You can modify the number of data points, pivots, and queried neighbors here.
n, k, n_neighbors = 1000, 1, 150

# Generating two 2D blobs and mean centering.
X = make_blobs(n, centers=2)[0]
X -= np.mean(X,axis=0)

# Chose what index to use by changing the index at the end of this list.
# This list only contains generating lambdas to not instantiate all
# indices prior to choosing.
idx_gen, smallests = [
	[lambda: pfls.EucDistancePFLSF64(X, num_pivots=k), True],
	[lambda: pfls.DotProductPFLSF64(X, num_pivots=k), False],
	[lambda: pfls.MahalanobisDistancePFLSF64(X, num_pivots=k, inv_cov=np.linalg.pinv(np.cov(X,rowvar=False))), True],
	[lambda: pfls.MahalanobisKernelPFLSF64(X, num_pivots=k, inv_cov=np.linalg.pinv(np.cov(X,rowvar=False))), False],
	[lambda: pfls.RBFDistancePFLSF64(X, num_pivots=k, bandwidth=1), True],
	[lambda: pfls.RBFKernelPFLSF64(X, num_pivots=k, bandwidth=1), False],
][0]
idx = idx_gen()
# "Smallests" is chosen to give you the most similar results.
# For products, most similar means largests measure, for distances smallests measure.
# Invert to get the most dissimilar examples.
# smallests = not smallests

# Selecting a "random" sample for querying and performing a knn query.
q = X[0]
knns = idx.query(q, n_neighbors+1, smallests=smallests)[1][0]
knns = [i for i in knns if i != 0]
# Computing fitlering masks for visualization
mask1 = ~np.array([i==0 or i in knns for i in range(X.shape[0])])
mask2 = np.array([i in knns for i in range(X.shape[0])])

# Visualizing both the query and the results.
go.Figure([
	go.Scatter(x=X[mask1,0],y=X[mask1,1],mode="markers",name="data"),
	go.Scatter(x=X[mask2,0],y=X[mask2,1],mode="markers",name="neighbors"),
	go.Scatter(x=X[:1,0],y=X[:1,1],mode="markers",name="query"),
], layout=dict(yaxis=dict(scaleanchor="x"))).show()
