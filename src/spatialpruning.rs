use rand::seq::index::sample;
use std::vec::Vec;
use num::traits::{Float,FromPrimitive};
use std::iter::Sum;
use std::collections::BinaryHeap;
use ndarray::{s, Axis, Array, Array1, Array2, ArrayBase, Data, Ix1, Ix2, ScalarOperand};
use std::cmp::Reverse;
use std::fmt::Debug;


use crate::measures::{InnerProduct};
use crate::primitives::{MeasurePair};





pub struct InnerProductBounder<P,N>
where
	P: InnerProduct<N>,
	N: Float + Debug,
{
	/* Hacking the Rust compiler. This works as intended by Rust. */
	product: P,
	// Self-products of all known data points
	data_squares: Array1<N>,
	// Original reference points as objects of type V; will be copied
	// into memory here because Rust sucks fucking balls
	pub reference_points: Array2<N>,
	// Inner products of original reference points with
	// the generated orthonormal basis.
	ref_ortho_prods: Array2<N>,
	// The divisors for dot products with the orthonormal
	// basis, which is independent of the other vector.
	ortho_dot_divisors: Array1<N>,
	// Inner products of all known data points with
	// the generated orthonormal basis.
	// Shape is (n_data, n_refs)
	data_ortho_prods: Array2<N>,
	// The part of the flexible bound term which is independent of y
	// (<x,x> - sum <x,~r_i>^2)**.5
	data_remainder_squares: Array1<N>,
}

impl<P,N> InnerProductBounder<P,N>
where
	P: InnerProduct<N>,
	N: Float + FromPrimitive + Sum + ScalarOperand + Debug
{
	pub fn new<D1: Data<Elem=N>>(
		product: P,
		data: &ArrayBase<D1, Ix2>,
		n_refs: Option<usize>,
		refs: Option<Array2<N>>
	) -> InnerProductBounder<P,N> {
		let zero: N = num::Zero::zero();
		/* This is a workaround because rust sucks. */
		let data_length = data.len_of(Axis(0));
		let data_dim = data.len_of(Axis(1));
		let n_refs = if refs.is_some() { refs.as_ref().unwrap().len_of(Axis(0)) } else { n_refs.unwrap().min(data_length).min(data_dim) };
		/* Precompute "squares" of all data points, i.e. <x,x> for all x in data */
		let data_squares = product.self_prods(data);
		/* Copy the reference points for later reference
		 * so we do not need to copy the entire dataset */
		/* TODO: Selecting the first n_refs points is easy but possibly a bad choice */
		let ref_points: Array2<N> = if refs.is_some() {
			refs.unwrap()
		} else {
			Array::from_shape_vec((n_refs, data.len_of(Axis(1))),
				sample(&mut rand::thread_rng(), data_length, n_refs).iter()
				.flat_map(|i| data.index_axis(Axis(0), i).into_owned())
				.collect::<Vec<N>>()
			).unwrap()
		};
		// data.slice(s![sample(&mut rng, data_length, n_refs).iter(),..]).into_owned();
		// let ref_points: Array2<N> = data.slice(s![..n_refs,..]).into_owned();
		/* Compute all products between original data and reference points, <x,r_i> */
		let data_ref_prods = product.cross_prods(data, &ref_points);
		/* Compute the pairwise products between all reference points, <r_i,r_j> */
		let ref_ref_prods =  product.cross_prods(&ref_points, &ref_points);
		/* Compute the divisors for orthonormal basis products
		 * which depend on the products between the i-th reference point
		 * and the first (i-1) orthonormal basis vectors.
		 * To obtain these products, we simultaneously 
		 */
		/* This corresponds to: (<r_i,r_i> - sum_{j=1}^{i-1} <r_i,\hat{r}_j>**2)**.5 */
		let mut ortho_dot_divisors = Array::zeros(n_refs);
		/* This corresponds to: <r_i,\hat{r}_j> */
		let mut ref_ortho_prods = Array::zeros((n_refs, n_refs));
		for k in 0..n_refs {
			for i in 0..k {
				if ortho_dot_divisors[i] > N::zero() {
					ref_ortho_prods[[i,k]] = (
						ref_ref_prods[[i,k]] /* <r_k,r_i> */
						- (&ref_ortho_prods.slice(s![k,..i]) * &ref_ortho_prods.slice(s![i,..i])).sum() /* Sum over j of <r_k,\hat{r}_j><r_i,\hat{r}_j> */
					) / ortho_dot_divisors[i];
				} else {
					ref_ortho_prods[[i,k]] = N::zero();
				}
				/* Copy to the other side of the symmetric matrix
				* to avoid errors due to order of arguments. */
				ref_ortho_prods[[k,i]] = ref_ortho_prods[[i,k]];
			}
			ortho_dot_divisors[k] = (
				ref_ref_prods[[k,k]] /* <r_i,r_i> */
				- (&ref_ortho_prods.slice(s![k,..k]) * &ref_ortho_prods.slice(s![k,..k])).sum() /* Sum over j of <r_k,\hat{r}_j>^2 */
			).max(N::zero()).sqrt();
		}
		/* This corresponds to: <x,\hat{r}_i> */
		let mut data_ortho_prods = Array::zeros((data_length, n_refs));
		for k in 0..data_length {
			for i in 0..n_refs {
				if ortho_dot_divisors[i] > N::zero() {
					data_ortho_prods[[k,i]] = (
						data_ref_prods[[k,i]] /* <x,r_i> */
						- (&data_ortho_prods.slice(s![k,..i]) * &ref_ortho_prods.slice(s![i,..i])).sum() /* Sum over j of <x,\hat{r}_j><r_i,\hat{r}_j> */
					) / ortho_dot_divisors[i];
				} else {
					data_ortho_prods[[k,i]] = N::zero();
				}
			}
		}
		/* This corresponds to: <x,x> - sum_{i=1}^n_refs <x,\hat{r}_i>**2 */
		let data_remainder_squares: Array1<N> = (
			&data_squares
			- (0..data_length)
			.map(|i| data_ortho_prods.index_axis(Axis(0), i).mapv(|v| v*v).sum())
			.collect::<Array1<N>>()
		).mapv(|v| v.max(zero)); /* Explicitly enforce >= 0 due to numerical errors. */
		/* Push the precomputed stuff into the instance and return it */
		InnerProductBounder {
			product: product,
			/* The precomputed data */
			data_squares: data_squares,
			reference_points: ref_points,
			ref_ortho_prods: ref_ortho_prods,
			ortho_dot_divisors: ortho_dot_divisors,
			data_ortho_prods: data_ortho_prods,
			data_remainder_squares: data_remainder_squares,
		}
	}

	/* Computes a matrix of <y_k,\hat{r}_i> for given vectors y_k.
	 * First index is k, second index is i. */
	fn ortho_prods<D1: Data<Elem=N>>(&self, vecs: &ArrayBase<D1, Ix2>) -> Array2<N> {
		let n_vecs = vecs.len_of(Axis(0));
		let n_refs = self.reference_points.len_of(Axis(0));
		let products = self.product.cross_prods(vecs, &self.reference_points);
		let mut ortho_prods = Array::zeros((n_vecs, n_refs));
		for k in 0..n_vecs { /* For each y_k */
			for i in 0..n_refs { /* For each \hat{r}_i */
				if self.ortho_dot_divisors[i] > N::zero() {
					ortho_prods[[k,i]] = (
						products[[k,i]] /* <y_k,r_i> */
						- (&ortho_prods.slice(s![k,..i]) * &self.ref_ortho_prods.slice(s![i,..i])).sum() /* Sum over j of <y_k,\hat{r}_j><r_i,\hat{r}_j> */
					) / self.ortho_dot_divisors[i];
				} else {
					ortho_prods[[k,i]] = N::zero();
				}
			}
		}
		ortho_prods
	}
	/* Computes the part of the squares not explained by the orthonormal projection */
	fn remainder_squares<D1: Data<Elem=N>, D2: Data<Elem=N>>(&self, vecs_squares: &ArrayBase<D1, Ix1>, ortho_prods: &ArrayBase<D2, Ix2>) -> Array1<N> {
		let zero: N = num::Zero::zero();
		(
			vecs_squares
			- ortho_prods.axis_iter(Axis(0))
			.map(|v| (&v * &v).sum())
			.collect::<Array1<N>>()
		).mapv(|v| v.max(zero))
	}
	/* Computes the fixed part of the dot product bounds */
	fn prod_bounds_fixed<D1: Data<Elem=N>>(&self, ortho_prods: &ArrayBase<D1, Ix2>) -> Array2<N> {
		Array::from_shape_vec(
			(ortho_prods.len_of(Axis(0)), self.data_ortho_prods.len_of(Axis(0))),
			ortho_prods.axis_iter(Axis(0))
			.flat_map(|v1|
				self.data_ortho_prods.axis_iter(Axis(0))
				.map(|v2| (&v1 * &v2).sum()).collect::<Vec<N>>()
			)
			.collect()
		).unwrap()
	}
	/* Computes the flexible part of the dot product bounds */
	fn prod_bounds_flex<D1: Data<Elem=N>>(&self, vecs_remainder_squares: &ArrayBase<D1, Ix1>) -> Array2<N> {
		Array::from_shape_vec(
			(vecs_remainder_squares.len(), self.data_remainder_squares.len()),
			vecs_remainder_squares.iter()
			.flat_map(|&a| self.data_remainder_squares.mapv(|b| (a*b).max(N::zero()).sqrt()))
			.collect()
		).unwrap()
	}

	pub fn prod_bounds<D1: Data<Elem=N>>(&self, vecs: &ArrayBase<D1, Ix2>, lower: bool) -> Array2<N> {
		/* This corresponds to: <y,y> */
		let vecs_squares = self.product.self_prods(vecs);
		let ortho_prods = self.ortho_prods(vecs);
		let vecs_remainder_squares = self.remainder_squares(&vecs_squares, &ortho_prods);
		/* This corresponds to: sum_{i=1}^n_refs <x,\hat{r}_i><y,\hat{r}_i> */
		let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
		/* This corresponds to: .... */
		let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
		if lower {
			prod_bounds_fixed - prod_bounds_flex
		} else {
			prod_bounds_fixed + prod_bounds_flex
		}
	}

	pub fn prod_bounds_both<D1: Data<Elem=N>>(&self, vecs: &ArrayBase<D1, Ix2>) -> (Array2<N>,Array2<N>) {
		/* This corresponds to: <y,y> */
		let vecs_squares = self.product.self_prods(vecs);
		let ortho_prods = self.ortho_prods(vecs);
		let vecs_remainder_squares = self.remainder_squares(&vecs_squares, &ortho_prods);
		/* This corresponds to: sum_{i=1}^n_refs <x,\hat{r}_i><y,\hat{r}_i> */
		let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
		/* This corresponds to: .... */
		let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
		(&prod_bounds_fixed - &prod_bounds_flex, &prod_bounds_fixed + &prod_bounds_flex)
	}

	pub fn distance_bounds<D1: Data<Elem=N>>(&self, vecs: &ArrayBase<D1, Ix2>, lower: bool) -> Array2<N> {
		let two = N::from_u8(2).unwrap();
		/* This corresponds to: <y,y> */
		let vecs_squares = self.product.self_prods(vecs);
		let ortho_prods = self.ortho_prods(vecs);
		let vecs_remainder_squares = self.remainder_squares(&vecs_squares, &ortho_prods);
		/* This corresponds to: sum_{i=1}^n_refs <x,\hat{r}_i><y,\hat{r}_i> */
		let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
		/* This corresponds to: .... */
		let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
		/* <x-y,x-y>
		 * = <x,x> + <y,y> - 2*<x,y>
		 * = <x,x> + <y,y> - 2*(sum <x,\hat{r}_i><y,\hat{r}_i>) +- 2*<x_R,y_R>
		 */
		let offset = Array::from_shape_vec(
			(vecs.len_of(Axis(0)), self.data_squares.len()),
			(0..vecs.len_of(Axis(0))).map(|i| &self.data_squares + vecs_squares[i])
			.flatten()
			.collect()
		).unwrap();
		if lower {
			(offset - (prod_bounds_fixed + prod_bounds_flex) * two).mapv(|v| v.max(N::zero()).sqrt())
		} else {
			(offset - (prod_bounds_fixed - prod_bounds_flex) * two).mapv(|v| v.max(N::zero()).sqrt())
		}
	}

	pub fn distance_bounds_both<D1: Data<Elem=N>>(&self, vecs: &ArrayBase<D1, Ix2>) -> (Array2<N>,Array2<N>) {
		let two = N::from_u8(2).unwrap();
		/* This corresponds to: <y,y> */
		let vecs_squares = self.product.self_prods(vecs);
		let ortho_prods = self.ortho_prods(vecs);
		let vecs_remainder_squares = self.remainder_squares(&vecs_squares, &ortho_prods);
		/* This corresponds to: sum_{i=1}^n_refs <x,\hat{r}_i><y,\hat{r}_i> */
		let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
		/* This corresponds to: .... */
		let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
		/* <x-y,x-y>
		 * = <x,x> + <y,y> - 2*<x,y>
		 * = <x,x> + <y,y> - 2*(sum <x,\hat{r}_i><y,\hat{r}_i>) +- 2*<x_R,y_R>
		 */
		let offset = Array::from_shape_vec(
			(vecs.len_of(Axis(0)), self.data_squares.len()),
			vecs_squares.iter().flat_map(|&v| &self.data_squares + v).collect()
		).unwrap();
		(
			(&offset - (&prod_bounds_fixed + &prod_bounds_flex) * two).mapv(|v| v.max(N::zero()).sqrt()),
			(&offset - (&prod_bounds_fixed - &prod_bounds_flex) * two).mapv(|v| v.max(N::zero()).sqrt())
		)
	}
}



pub struct PFLS<P, N, D1>
where
	P: InnerProduct<N>,
	D1: Data<Elem=N>,
	N: Float + Debug,
{
	product: P,
	data: ArrayBase<D1, Ix2>,
	bounder: InnerProductBounder<P, N>
}



macro_rules! query_below_gen {
	($fun_name: ident, $bound_fun: ident, $measure_fun: ident) => {
		pub fn $fun_name<D2: Data<Elem=N>>(
			&self,
			vecs: &ArrayBase<D2, Ix2>,
			threshold: N
		) -> Vec<Vec<usize>> {
			let (lower, upper) = self.bounder.$bound_fun(vecs);
			let mut indices = vec![vec![0 as usize;0]; vecs.len_of(Axis(0))];
			for (i, vec) in vecs.axis_iter(Axis(0)).enumerate() {
				let l = lower.index_axis(Axis(0), i);
				let h = upper.index_axis(Axis(0), i);
				for j in 0..l.len() {
					if l[j] <= threshold {
						if h[j] <= threshold {
							indices[i].push(j);
						} else {
							let v = self.product.$measure_fun(&vec, &self.data.index_axis(Axis(0), j));
							if v <= threshold {
								indices[i].push(j);
							}
						}
					}
				}
			}
			indices
		}
	};
}
macro_rules! query_above_gen {
	($fun_name: ident, $bound_fun: ident, $measure_fun: ident) => {
		pub fn $fun_name<D2: Data<Elem=N>>(
			&self,
			vecs: &ArrayBase<D2, Ix2>,
			threshold: N
		) -> Vec<Vec<usize>> {
			let (lower, upper) = self.bounder.$bound_fun(vecs);
			let mut indices = vec![vec![0 as usize;0]; vecs.len_of(Axis(0))];
			for (i, vec) in vecs.axis_iter(Axis(0)).enumerate() {
				let l = lower.index_axis(Axis(0), i);
				let h = upper.index_axis(Axis(0), i);
				for j in 0..l.len() {
					if h[j] >= threshold {
						if l[j] >= threshold {
							indices[i].push(j);
						} else {
							let v = self.product.$measure_fun(&vec, &self.data.index_axis(Axis(0), j));
							if v >= threshold {
								indices[i].push(j);
							}
						}
					}
				}
			}
			indices
		}
	};
}


macro_rules! query_k_smallest_sorting_gen {
	($fun_name: ident, $bound_fun: ident, $measure_fun: ident) => {
		pub fn $fun_name<D2: Data<Elem=N>>(
			&self,
			vecs: &ArrayBase<D2, Ix2>,
			k: usize
		) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
			/* Compute lower bounds for pruning */
			let lower = self.bounder.$bound_fun(vecs, true);
			let vecs_length = vecs.len_of(Axis(0));
			/* Initialize return value */
			let mut values = vec![vec![N::zero();0]; vecs_length];
			let mut indices = vec![vec![0 as usize;0]; vecs_length];
			/* Compute k nearest neighbors for every q in qs */
			for (i,(l,q)) in lower.axis_iter(Axis(0)).zip(vecs.axis_iter(Axis(0))).enumerate() {
				/* Join lower distance bounds with indices for arg sorting */
				let bounds_to_results: Vec<Reverse<MeasurePair<N>>> = l.iter().enumerate()
				.map(|(j,&v)| Reverse(MeasurePair{index:j,value:v}))
				.collect();
				/* Build min heap for all values */
				let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::from(bounds_to_results);
				/* Build max heap for closests */
				let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::with_capacity(k+1);
				/* Insert by increasing lower bounds */
				while min_heap.len() > 0 {
					let mut next_entry = min_heap.pop().unwrap().0;
					/* If lower bound exceeds k-smallests distance, we are done */
					if max_heap.len() >= k && next_entry.value >= max_heap.peek().unwrap().value {
						break;
					}
					/* Compute true distance */
					next_entry.value = self.product.$measure_fun(&q, &self.data.index_axis(Axis(0),next_entry.index));
					/* Reduce to k values */
					max_heap.push(next_entry);
					if max_heap.len() > k {
						max_heap.pop();
					}
				}
				for entry in max_heap {
					values[i].push(entry.value);
					indices[i].push(entry.index);
				}
			}
			(values, indices)
		}
	};
}
macro_rules! query_k_largest_sorting_gen {
	($fun_name: ident, $bound_fun: ident, $measure_fun: ident) => {
		pub fn $fun_name<D2: Data<Elem=N>>(
			&self,
			vecs: &ArrayBase<D2, Ix2>,
			k: usize
		) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
			/* Compute upper bounds for pruning */
			let upper = self.bounder.$bound_fun(vecs, false);
			let vecs_length = vecs.len_of(Axis(0));
			/* Initialize return value */
			let mut values = vec![vec![N::zero();0]; vecs_length];
			let mut indices = vec![vec![0 as usize;0]; vecs_length];
			/* Compute k nearest neighbors for every q in qs */
			for (i,(l,q)) in upper.axis_iter(Axis(0)).zip(vecs.axis_iter(Axis(0))).enumerate() {
				/* Join upper distance bounds with indices for arg sorting */
				let bounds_to_results: Vec<MeasurePair<N>> = l.iter().enumerate()
				.map(|(j,&v)| MeasurePair{index:j,value:v})
				.collect();
				/* Build max heap for all values */
				let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::from(bounds_to_results);
				/* Build min heap for closests */
				let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::with_capacity(k+1);
				/* Insert by increasing upper bounds */
				while max_heap.len() > 0 {
					let mut next_entry = max_heap.pop().unwrap();
					/* If upper bound exceeds k-largest distance, we are done */
					if min_heap.len() >= k && next_entry.value <= min_heap.peek().unwrap().0.value {
						break;
					}
					/* Compute true distance */
					next_entry.value = self.product.$measure_fun(&q, &self.data.index_axis(Axis(0),next_entry.index));
					/* Reduce to k values */
					min_heap.push(Reverse(next_entry));
					if min_heap.len() > k {
						min_heap.pop();
					}
				}
				for entry in min_heap {
					values[i].push(entry.0.value);
					indices[i].push(entry.0.index);
				}
			}
			(values, indices)
		}
	};
}

macro_rules! query_k_smallest_direct_gen {
	($fun_name: ident, $bound_fun: ident, $measure_fun: ident) => {
		pub fn $fun_name<D2: Data<Elem=N>>(
			&self,
			vecs: &ArrayBase<D2, Ix2>,
			k: usize
		) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
			/* Compute lower bounds for pruning */
			let lower = self.bounder.$bound_fun(vecs, true);
			let vecs_length = vecs.len_of(Axis(0));
			/* Initialize return value */
			let mut values = vec![vec![N::zero();0]; vecs_length];
			let mut indices = vec![vec![0 as usize;0]; vecs_length];
			/* Compute k nearest neighbors for every q in qs */
			for (i,(l,q)) in lower.axis_iter(Axis(0)).zip(vecs.axis_iter(Axis(0))).enumerate() {
				/* Build max heap for closests */
				let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::with_capacity(k+1);
				let mut heap_largest = N::max_value();
				for (j,(v,x)) in l.iter().zip(self.data.axis_iter(Axis(0))).enumerate() {
					/* Only consider points while less than k found and lower bound in considerable range */
					if max_heap.len() < k || *v <= heap_largest {
						/* Compute true distance */
						let true_v = self.product.$measure_fun(&q, &x);
						if true_v <= heap_largest {
							/* Reduce to k values */
							max_heap.push(MeasurePair{index: j, value: true_v});
							if max_heap.len() > k {
								max_heap.pop();
							}
							heap_largest = max_heap.peek().unwrap().value;
						}
					}
				}
				for entry in max_heap {
					values[i].push(entry.value);
					indices[i].push(entry.index);
				}
			}
			(values, indices)
		}
	};
}
macro_rules! query_k_largest_direct_gen {
	($fun_name: ident, $bound_fun: ident, $measure_fun: ident) => {
		pub fn $fun_name<D2: Data<Elem=N>>(
			&self,
			vecs: &ArrayBase<D2, Ix2>,
			k: usize
		) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
			/* Compute upper bounds for pruning */
			let upper = self.bounder.$bound_fun(vecs, false);
			let vecs_length = vecs.len_of(Axis(0));
			/* Initialize return value */
			let mut values = vec![vec![N::zero();0]; vecs_length];
			let mut indices = vec![vec![0 as usize;0]; vecs_length];
			/* Compute k nearest neighbors for every q in qs */
			for (i,(u,q)) in upper.axis_iter(Axis(0)).zip(vecs.axis_iter(Axis(0))).enumerate() {
				/* Build min heap for farthest */
				let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::with_capacity(k+1);
				let mut heap_smallest = N::zero();
				for (j,(v,x)) in u.iter().zip(self.data.axis_iter(Axis(0))).enumerate() {
					/* Only consider points while less than k found or upper bound in considerable range */
					if min_heap.len() < k || *v >= heap_smallest {
						/* Compute true distance */
						let true_v = self.product.$measure_fun(&q, &x);
						if true_v >= heap_smallest {
							/* Reduce to k values */
							min_heap.push(Reverse(MeasurePair{index: j, value: true_v}));
							if min_heap.len() > k {
								min_heap.pop();
							}
							heap_smallest = min_heap.peek().unwrap().0.value;
						}
					}
				}
				for entry in min_heap {
					values[i].push(entry.0.value);
					indices[i].push(entry.0.index);
				}
			}
			(values, indices)
		}
	};
}

impl <P, N, D1> PFLS<P, N, D1>
where
	P: InnerProduct<N>,
	D1: Data<Elem=N>,
	N: Float + FromPrimitive + Sum + ScalarOperand + Debug
{
	pub fn new(
		product: P,
		data: ArrayBase<D1, Ix2>,
		num_pivots: Option<usize>,
		refs: Option<Array2<N>>
	) -> PFLS<P, N, D1> {
		let bounder = InnerProductBounder::new(
			product.clone(),
			&data,
			num_pivots,
			refs
		);
		PFLS {
			product: product,
			data: data,
			bounder: bounder
		}
	}

	query_below_gen!(query_distance_below, distance_bounds_both, induced_dist);
	query_above_gen!(query_distance_above, distance_bounds_both, induced_dist);
	query_below_gen!(query_product_below, prod_bounds_both, prod);
	query_above_gen!(query_product_above, prod_bounds_both, prod);

	query_k_smallest_sorting_gen!(query_k_smallest_distance_sorting, distance_bounds, induced_dist);
	query_k_largest_sorting_gen!(query_k_largest_distance_sorting, distance_bounds, induced_dist);
	query_k_smallest_sorting_gen!(query_k_smallest_product_sorting, prod_bounds, prod);
	query_k_largest_sorting_gen!(query_k_largest_product_sorting, prod_bounds, prod);

	query_k_smallest_direct_gen!(query_k_smallest_distance_direct, distance_bounds, induced_dist);
	query_k_largest_direct_gen!(query_k_largest_distance_direct, distance_bounds, induced_dist);
	query_k_smallest_direct_gen!(query_k_smallest_product_direct, prod_bounds, prod);
	query_k_largest_direct_gen!(query_k_largest_product_direct, prod_bounds, prod);

}



