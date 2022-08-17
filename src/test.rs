use rand::prelude::*;
use std::vec::{Vec};
use ndarray::{Axis,ArrayBase,Array2};
use crate::spatialpruning::*;
use crate::measures::*;

macro_rules! assert_ge {
	($a: expr, $b: expr) => {
		assert!($a >= $b, "The following inequation failed: {} >= {}", $a, $b);
	};
	($a: expr, $b: expr, $tol: expr) => {
		assert!($a >= $b - $tol, "The following inequation failed despite tolerance {}: {} >= {}", $tol, $a, $b);
	};
}
macro_rules! assert_le {
	($a: expr, $b: expr) => {
		assert!($a <= $b, "The following inequation failed: {} <= {}", $a, $b);
	};
	($a: expr, $b: expr, $tol: expr) => {
		assert!($a <= $b + $tol, "The following inequation failed despite tolerance {}: {} <= {}", $tol, $a, $b);
	};
}

fn random_data_arr(n: usize, d: usize, seed: Option<u64>) -> Array2<f64> {
	let mut rng = if seed.is_some() { StdRng::seed_from_u64(seed.unwrap()) } else { StdRng::from_rng(thread_rng()).unwrap() };
	ArrayBase::from_iter(
		(0..n*d).map(|_| rng.gen_range::<f64,_>(0.0..1.0))
	).into_shape((n,d)).unwrap()
}

fn in_list<N: Eq>(l: &Vec<N>, v: &N) -> bool {
	for u in l {
		if u.eq(v) { return true; }
	}
	false
}


#[test]
fn test_bounds_and_index() {
	let dim = 50;
	let n_data = 1000;
	let n_pivots = 20;
	let n_queries = 100;
	let k = 50;
	let eps = 1e-8;
	let data = random_data_arr(n_data,dim,None);
	let prod = DotProduct::new();
	// let dist = EuclideanDistance::new();
	let bounds = InnerProductBounder::new(prod.clone(), &data, Some(n_pivots), None);
	let queries = random_data_arr(n_queries, dim, None);
	/* Test if distance bounds are valid */
	let lower1 = bounds.distance_bounds(&queries, true);
	let upper1 = bounds.distance_bounds(&queries, false);
	let (lower2, upper2) = bounds.distance_bounds_both(&queries);
	let true_dists: Array2<f64> = ArrayBase::from_iter(
		(0..n_queries).flat_map(|i|
			(0..n_data).map(|j|
				prod.induced_dist(
					&queries.index_axis(Axis(0), i),
					&data.index_axis(Axis(0), j)
				)
			).collect::<Vec<f64>>()
		)
	).into_shape((n_queries,n_data)).unwrap();
	for i in 0..n_queries {
		for j in 0..n_data {
			assert_le!(lower1[[i,j]], true_dists[[i,j]], eps);
			assert_ge!(upper1[[i,j]], true_dists[[i,j]], eps);
			assert_le!(lower2[[i,j]], true_dists[[i,j]], eps);
			assert_ge!(upper2[[i,j]], true_dists[[i,j]], eps);
		}
	}
	/* Test if product bounds are valid */
	let lower1 = bounds.prod_bounds(&queries, true);
	let upper1 = bounds.prod_bounds(&queries, false);
	let (lower2, upper2) = bounds.prod_bounds_both(&queries);
	let true_prods: Array2<f64> = ArrayBase::from_iter(
		(0..n_queries).flat_map(|i|
			(0..n_data).map(|j|
				prod.prod(
					&queries.index_axis(Axis(0), i),
					&data.index_axis(Axis(0), j)
				)
			).collect::<Vec<f64>>()
		)
	).into_shape((n_queries,n_data)).unwrap();
	for i in 0..n_queries {
		for j in 0..n_data {
			assert_le!(lower1[[i,j]], true_prods[[i,j]], eps);
			assert_ge!(upper1[[i,j]], true_prods[[i,j]], eps);
			assert_le!(lower2[[i,j]], true_prods[[i,j]], eps);
			assert_ge!(upper2[[i,j]], true_prods[[i,j]], eps);
		}
	}

	/* Test if brute force index gives correct results */
	let index = PFLS::new(prod.clone(), data.clone(), Some(n_pivots), None);
	let k_smallest_dist1 = index.query_k_smallest_distance_direct(&queries, k);
	for i in 0..n_queries {
		for v in &k_smallest_dist1.0[i] {
			for k in 0..n_data {
				if !in_list(&k_smallest_dist1.1[i], &k) {
					assert_ge!(true_dists[[i,k]], v, eps);
				}
			}
		}
	}
	let k_smallest_dist2 = index.query_k_smallest_distance_sorting(&queries, k);
	for i in 0..n_queries {
		for v in &k_smallest_dist2.0[i] {
			for k in 0..n_data {
				if !in_list(&k_smallest_dist2.1[i], &k) {
					assert_ge!(true_dists[[i,k]], v, eps);
				}
			}
		}
	}
	let k_largest_dist1 = index.query_k_largest_distance_direct(&queries, k);
	for i in 0..n_queries {
		for v in &k_largest_dist1.0[i] {
			for k in 0..n_data {
				if !in_list(&k_largest_dist1.1[i], &k) {
					assert_le!(true_dists[[i,k]], v, eps);
				}
			}
		}
	}
	let k_largest_dist2 = index.query_k_largest_distance_sorting(&queries, k);
	for i in 0..n_queries {
		for v in &k_largest_dist2.0[i] {
			for k in 0..n_data {
				if !in_list(&k_largest_dist2.1[i], &k) {
					assert_le!(true_dists[[i,k]], v, eps);
				}
			}
		}
	}
	let k_smallest_prod1 = index.query_k_smallest_product_direct(&queries, k);
	for i in 0..n_queries {
		for v in &k_smallest_prod1.0[i] {
			for k in 0..n_data {
				if !in_list(&k_smallest_prod1.1[i], &k) {
					assert_ge!(true_prods[[i,k]], v, eps);
				}
			}
		}
	}
	let k_smallest_prod2 = index.query_k_smallest_product_sorting(&queries, k);
	for i in 0..n_queries {
		for v in &k_smallest_prod2.0[i] {
			for k in 0..n_data {
				if !in_list(&k_smallest_prod2.1[i], &k) {
					assert_ge!(true_prods[[i,k]], v, eps);
				}
			}
		}
	}
	let k_largest_prod1 = index.query_k_largest_product_direct(&queries, k);
	for i in 0..n_queries {
		for v in &k_largest_prod1.0[i] {
			for k in 0..n_data {
				if !in_list(&k_largest_prod1.1[i], &k) {
					assert_le!(true_prods[[i,k]], v, eps);
				}
			}
		}
	}
	let k_largest_prod2 = index.query_k_largest_product_sorting(&queries, k);
	for i in 0..n_queries {
		for v in &k_largest_prod2.0[i] {
			for k in 0..n_data {
				if !in_list(&k_largest_prod2.1[i], &k) {
					assert_le!(true_prods[[i,k]], v, eps);
				}
			}
		}
	}
	let below_dist = index.query_distance_below(&queries, true_dists[[0,0]]);
	for i in 0..n_queries {
		for j in 0..n_data {
			if in_list(&below_dist[i], &j) {
				assert_le!(true_dists[[i,j]], true_dists[[0,0]], eps);
			} else {
				assert_ge!(true_dists[[i,j]], true_dists[[0,0]], eps);
			}
		}
	}
	let above_dist = index.query_distance_above(&queries, true_dists[[0,0]]);
	for i in 0..n_queries {
		for j in 0..n_data {
			if in_list(&above_dist[i], &j) {
				assert_ge!(true_dists[[i,j]], true_dists[[0,0]], eps);
			} else {
				assert_le!(true_dists[[i,j]], true_dists[[0,0]], eps);
			}
		}
	}
	let below_prod = index.query_product_below(&queries, true_prods[[0,0]]);
	for i in 0..n_queries {
		for j in 0..n_data {
			if in_list(&below_prod[i], &j) {
				assert_le!(true_prods[[i,j]], true_prods[[0,0]], eps);
			} else {
				assert_ge!(true_prods[[i,j]], true_prods[[0,0]], eps);
			}
		}
	}
	let above_prod = index.query_product_above(&queries, true_prods[[0,0]]);
	for i in 0..n_queries {
		for j in 0..n_data {
			if in_list(&above_prod[i], &j) {
				assert_ge!(true_prods[[i,j]], true_prods[[0,0]], eps);
			} else {
				assert_le!(true_prods[[i,j]], true_prods[[0,0]], eps);
			}
		}
	}

}

