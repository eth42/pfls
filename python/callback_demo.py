import numpy as np
from sklearn.datasets import fetch_openml
from plotly import graph_objects as go
from sklearn.metrics.pairwise import chi2_kernel
from skimage.feature import local_binary_pattern
import pfls

# Reshapes a vector into a square matrix if possible
def make_square(v):
	n_sq = len(v)
	n = int(np.round(n_sq**.5))
	assert n**2 == n_sq
	return v.reshape((n,n))
# Plotting function for images
def show_images(*images, **layout_args):
	images = np.array(images)
	if len(images.shape) > 2: images = images[0]
	if len(images.shape) == 1: images = images[None,:]
	images = np.array([make_square(img) for img in images])
	# Flipping axis for visualization
	images = images[:,::-1,:]
	n,d = images.shape[:2]
	full_img = np.full((d, (d+1)*n-1), np.nan)
	for i,img in enumerate(images):
		full_img[:,(d+1)*i:(d+1)*(i+1)-1] = img
	go.Figure(go.Heatmap(
		x=np.arange(full_img.shape[0]),
		y=np.arange(full_img.shape[1]),
		z=full_img,
		showscale=False,
		colorscale="gray",
	),layout=dict(
    plot_bgcolor='rgba(0,0,0,0)',
		xaxis=dict(showticklabels=False, showgrid=False),
		yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x", range=[0,d-1]),
		**layout_args
	)).show()


# In this demo we build an index based on an inner product defined
# in a Python function. Here we use the chi squared kernel for which
# we need to write a function that accepts two vectors and returns
# the kernel value.
# The passed function *must* be a real function and not just a lambda
# and the return value *must* be a floating point number!
def chi2_pairwise(x,y):
	return float(chi2_kernel(np.array([x,y]))[0,1])
# We will be considering LBP Histograms for each image
def make_feature_vec(x):
	global lbp_points, lbp_radius, n_hist_bins
	xmat = make_square(x)
	lbp = local_binary_pattern(xmat, lbp_points, lbp_radius)
	return np.histogram(lbp, bins=n_hist_bins, range=(0, 2**lbp_points))[0]


# You can modify the number of pivots and queried neighbors here.
k, n_neighbors = 150, 5
# Parameters for local binary pattern histograms.
lbp_points, lbp_radius, n_hist_bins = 8, 1, 300
# The entire STL-10 data set is pretty huge considering that
# the callbacks have quite some cost. This parameters controls
# how many random samples should be used in this demo.
max_samples = 4000


# Loading STL-10 from openml.
# The original images are stored for visualization.
X_original = fetch_openml("STL-10", as_frame=False, return_X_y=True)[0]
# Transforming the color images to grayscale
X_original = (
	X_original[:,0::3] +
	X_original[:,1::3] +
	X_original[:,2::3]
) / 3

# Selecting max_samples random images and computing LBP histograms for each
# image based on lbp_points points in an lbp_radius radius.
X = np.array([
	make_feature_vec(x)
	for x in X_original[np.random.choice(
		X_original.shape[0],
		size=max_samples,
		replace=False
	)]
], dtype=float)
X /= np.sum(X,axis=1)[:,None]
idx = pfls.PyProductPFLSF64(X, num_pivots=k, pyprod=chi2_pairwise)
smallests = False
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
show_images(
	X_original[query_index],
	title="Query",
	height=250,
	margin={s:10 for s in "blr"}
)
show_images(
	X_original[knns],
	title=(
		"\"Nearest neighbors\" (smallest measure) in {:}"
		if smallests else
		"\"Farthest neighbors\" (largest measure) in {:}"
	).format(type(idx).__name__),
	height=250,
	margin={s:10 for s in "blr"}
)
