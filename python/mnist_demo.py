import numpy as np
from sklearn.datasets import fetch_openml
from plotly import graph_objects as go
import pfls

# Plotting function for MNIST images
def show_mnist(*images, **layout_args):
	images = np.array(images)
	if len(images.shape) > 2: images = images[0]
	if len(images.shape) == 1: images = images[None,:]
	images = images.reshape((images.shape[0], 28, 28))[:,::-1,:]
	full_img = np.full((28, 29*images.shape[0]-1), np.nan)
	for i,img in enumerate(images):
		full_img[:,29*i:29*(i+1)-1] = img
	go.Figure(go.Heatmap(
		x=np.arange(full_img.shape[0]),
		y=np.arange(full_img.shape[1]),
		z=full_img,
		showscale=False,
		colorscale="gray",
	),layout=dict(yaxis=dict(scaleanchor="x"),**layout_args)).show()


# You can modify the number of pivots and queried neighbors here.
k, n_neighbors = 120, 10

# Loading MNIST from openml and mean centering.
# The original images are stored for visualization.
X = fetch_openml("mnist_784", as_frame=False, return_X_y=True)[0]
X_original = X.copy()
X -= np.mean(X,axis=0)
X = np.array([[x for x in row] for row in X])

# Chose what index to use by changing the index at the end of this list.
# This list only contains generating lambdas to not instantiate all
# indices prior to choosing.
idx_gen, smallests = [
	[lambda: pfls.EucDistancePFLSF64(X, k), True],
	[lambda: pfls.DotProductPFLSF64(X, k), False],
	[lambda: pfls.MahalanobisDistancePFLSF64(np.linalg.pinv(np.cov(X,rowvar=False)), X, k), True],
	[lambda: pfls.MahalanobisKernelPFLSF64(np.linalg.pinv(np.cov(X,rowvar=False)), X, k), False],
	[lambda: pfls.RBFDistancePFLSF64(1, X, k), True],
	[lambda: pfls.RBFKernelPFLSF64(1, X, k), False],
][1]
idx = idx_gen()
# "Smallests" is chosen to give you the most similar results.
# For products, most similar means largests measure, for distances smallests measure.
# Invert to get the most dissimilar examples.
# smallests = not smallests

# Selecting a random sample for querying and performing a knn query.
query_index = np.random.randint(X.shape[0])
q = X[query_index]
knns = idx.query(q, n_neighbors+1, smallests=smallests)[1][0]
knns = [i for i in knns if i != query_index]

# Visualizing both the query and the results.
show_mnist(
	X_original[query_index],
	title="Query",
	height=250,
	margin={s:10 for s in "blr"}
)
show_mnist(
	X_original[knns],
	title="\"Nearest neighbors\" (smallest measure)"
	if smallests else
	"\"Farthest neighbors\" (largest measure)",
	height=250,
	margin={s:10 for s in "blr"}
)
