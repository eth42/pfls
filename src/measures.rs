use std::vec::Vec;
use std::marker::PhantomData;
use num::{Float};
use ndarray::{Data, Axis, ArrayBase, Array, Array1, Array2, Ix1, Ix2};


pub trait InnerProduct<N: Float>: Clone {
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N;
	#[inline(always)]
	fn prods
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj: &ArrayBase<D1, Ix1>, objs: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs.outer_iter()
		.map(|obj2| self.prod(obj, &obj2))
		.collect()
	}
	#[inline(always)]
	fn zip_prods
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs1.outer_iter().zip(objs2.outer_iter())
		.map(|(obj1,obj2)| self.prod(&obj1, &obj2))
		.collect()
	}
	#[inline(always)]
	fn cross_prods
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array2<N> {
		Array::from_shape_vec(
			(objs1.len_of(Axis(0)), objs2.len_of(Axis(0))),
			objs1.outer_iter()
			.flat_map(|obj1|
				objs2.outer_iter()
				.map(|obj2| self.prod(&obj1, &obj2))
				.collect::<Vec<N>>()
			)
			.collect()
		).unwrap()
	}
	#[inline(always)]
	fn self_prod
	<D1: Data<Elem=N>>
	(&self, obj: &ArrayBase<D1, Ix1>) -> N {
		self.prod(obj, obj)
	}
	#[inline(always)]
	fn self_prods
	<D1: Data<Elem=N>>
	(&self, objs: &ArrayBase<D1, Ix2>) -> Array1<N> {
		self.zip_prods(objs, objs)
	}
	
	#[inline(always)]
	fn induced_dist
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {
			PROD_COUNTER -= 1;
			DIST_COUNTER += 1;
		}
		self.self_prod(&(obj1-obj2)).sqrt()
	}
	#[inline(always)]
	fn induced_dists
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj: &ArrayBase<D1, Ix1>, objs: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs.outer_iter()
		.map(|obj2| self.induced_dist(obj, &obj2))
		.collect()
	}
	#[inline(always)]
	fn zip_induced_dists
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array1<N> {
		objs1.outer_iter().zip(objs2.outer_iter())
		.map(|(obj1,obj2)| self.induced_dist(&obj1, &obj2))
		.collect()
	}
	#[inline(always)]
	fn cross_induced_dists
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, objs1: &ArrayBase<D1, Ix2>, objs2: &ArrayBase<D2, Ix2>) -> Array2<N> {
		Array::from_shape_vec(
			(objs1.len_of(Axis(0)), objs2.len_of(Axis(0))),
			objs1.outer_iter()
			.flat_map(|obj1|
				objs2.outer_iter()
				.map(|obj2| self.induced_dist(&obj1, &obj2))
				.collect::<Vec<N>>()
			)
			.collect()
		).unwrap()
	}
}


/* These functions are optional and solely useful for benchmarking purposes.
 * They allow to keep track of the number of distance and product computations
 * after any operations performed after the last reset.
 * To use these functions you must set the "count_operations" feature during
 * compilation with "--features count_operations". */
#[cfg(feature="count_operations")]
pub static mut PROD_COUNTER: isize = 0;
#[cfg(feature="count_operations")]
pub static mut DIST_COUNTER: isize = 0;


/* Standard dot product for real vectors. */
#[derive(Debug,Clone)]
pub struct DotProduct<N: Float> { _marker: PhantomData<N> }
impl<N: Float> DotProduct<N> {
	pub fn new() -> Self { DotProduct{_marker: PhantomData} }
}
impl<N: Float> InnerProduct<N> for DotProduct<N> {
	fn prod
	<D1: Data<Elem=N>, D2: Data<Elem=N>>
	(&self, obj1: &ArrayBase<D1, Ix1>, obj2: &ArrayBase<D2, Ix1>) -> N {
		#[cfg(feature="count_operations")]
		unsafe {PROD_COUNTER += 1;}
		obj1.into_iter().zip(obj2.into_iter())
		.map(|(&a,&b)| a * b)
		.reduce(|a, b| a+b)
		.unwrap_or(num::Zero::zero())
	}
}

