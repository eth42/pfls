import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import euclidean_distances
import time
import pfls

# Benchmarking function which calls a function with given arguments
# and prints statistics on computation time.
def benchmark(call, *args, n_repetitions=1, **kwargs):
	times = []
	for _ in range(n_repetitions):
		start = time.time()
		result = call(*args, **kwargs)
		end = time.time()
		times.append((end-start)*1000)
	if n_repetitions > 1:
		print("{:} repetitions mean time: {:d} ± {:.2f} ms; min: {:d} ms; max: {:d} ms".format(
			n_repetitions,
			int(np.mean(times)),
			np.std(times),
			int(np.min(times)),
			int(np.max(times))
		))
	else: print("time: {:d} ms".format(int(times[0])))
	return result
# Evaluates if two given lists contain similar results.
def similar_lists(a,b,eps=1e-10):
	for x,y in zip(np.sort(a),np.sort(b)):
		if abs(x-y) > eps: return False
	return True

# You can modify the number of data points, dimensions, pivots, and queries here.
n,d,k,q = 10000, 200, 60, 100
# You can modify how many repetitions per benchmark are executed.
n_repetitions = 2

# Generating normally random data with a lowered intrinsic dimensionality
# by scaling down some dimensions and mean centering.
X = np.random.normal(0,1,(n,d)) * np.random.sample(d)**4
X -= np.mean(X,axis=0)

# Instantiating two indices for comparison.
idx1 = pfls.EucDistancePFLSF64(X, k)
idx2 = cKDTree(X)

# Benchmarking both indices on knn queries and comparing results
print("Benchmarking PFLS k-nearest neighbors query")
r1 = benchmark(idx1.query, X[:q], 50, n_repetitions=n_repetitions)
print("Benchmarking KD-Tree k-nearest neighbors query")
r2 = benchmark(idx2.query, X[:q], 50, n_repetitions=n_repetitions)
for i in range(q):
	assert similar_lists(r1[0][i], r2[0][i])
	assert similar_lists(r1[1][i], r2[1][i])

# Benchmarking both indices on range quers and comparing results
eps = np.percentile(euclidean_distances(X,X), 5)
print("Benchmarking PFLS range query")
r1 = benchmark(idx1.query_ball_point, X[:q], eps, n_repetitions=n_repetitions)
print("Benchmarking KD-Tree range query")
r2 = benchmark(idx2.query_ball_point, X[:q], eps, n_repetitions=n_repetitions)
for i in range(q):
	assert similar_lists(r1[i], r2[i])

