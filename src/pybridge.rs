use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};
use paste::paste;
use ndarray::{OwnedRepr};

pub mod measures;
pub mod spatialpruning;
pub mod primitives;

#[cfg(test)]
mod test;

use spatialpruning::{PFLS, InnerProductBounder};
// #[cfg(not(feature="count_operations"))]
// use measures::{InnerProduct, DotProduct, RBFKernel, MahalanobisKernel, PolyKernel};
// #[cfg(feature="count_operations")]
// use measures::{InnerProduct, DotProduct, RBFKernel, MahalanobisKernel, PolyKernel, PROD_COUNTER, DIST_COUNTER};
use measures::{*};




/* These functions are optional and solely useful for benchmarking purposes.
 * They allow to keep track of the number of distance and product computations
 * after any operations performed after the last reset.
 * To use these functions you must set the "count_operations" feature during
 * compilation with "--features count_operations". */
#[pyfunction]
#[cfg(feature="count_operations")]
fn get_prod_counter() -> isize { unsafe {PROD_COUNTER} }
#[pyfunction]
#[cfg(feature="count_operations")]
fn reset_prod_counter() { unsafe {PROD_COUNTER = 0}; }
#[pyfunction]
#[cfg(feature="count_operations")]
fn get_dist_counter() -> isize { unsafe {DIST_COUNTER} }
#[pyfunction]
#[cfg(feature="count_operations")]
fn reset_dist_counter() { unsafe {DIST_COUNTER = 0}; }




/* Helper macro to transform dynamically shaped numpy arrays into 2d arrays.
 * Internally we always expect multiple queries at once, but sklearn syntax
 * allows for single vectors to be queries, so we need to reshape inputs
 * to force 2d inputs. Sadly the same trick is not as easily applicable for
 * outputs, since e.g. range queries do not guarantee equal result lengths. */
macro_rules! into_2d_array {
	($pydynarr: ident) => {
		if $pydynarr.shape().len() == 1 {
			$pydynarr.as_array().into_shape((1,$pydynarr.shape()[0])).unwrap()
		} else {
			$pydynarr.as_array().into_shape(($pydynarr.shape()[0],$pydynarr.shape()[1])).unwrap()
		}
	}
}

/* Macro to help expose all functions of inner products to python */
macro_rules! forward_inner_product_function {
	($struct_name: ident, $fun: ident, $type: ty) => {
		#[pymethods]
		impl $struct_name {
			pub fn $fun(&self, a: PyReadonlyArray1<$type>, b: PyReadonlyArray1<$type>) -> $type {
				self.product.$fun(&a.as_array(),&b.as_array())
			}
		}
	};
	($struct_name: ident, $fun: ident, $type: ty, single) => {
		#[pymethods]
		impl $struct_name {
			pub fn $fun(&self, a: PyReadonlyArray1<$type>) -> $type {
				self.product.$fun(&a.as_array())
			}
		}
	};
	($struct_name: ident, $fun: ident, $type: ty, $in1: ident, $out: ident) => {
		#[pymethods]
		impl $struct_name {
			pub fn $fun<'py>(&self, py: Python<'py>, a: $in1<$type>) -> &'py $out<$type> {
				self.product.$fun(&a.as_array()).into_pyarray(py)
			}
		}
	};
	($struct_name: ident, $fun: ident, $type: ty, $in1: ident, $in2: ident, $out: ident) => {
		#[pymethods]
		impl $struct_name {
			pub fn $fun<'py>(&self, py: Python<'py>, a: $in1<$type>, b: $in2<$type>) -> &'py $out<$type> {
				self.product.$fun(&a.as_array(),&b.as_array()).into_pyarray(py)
			}
		}
	};
}
/* Python wrapper for inner product types */
macro_rules! inner_product_wrapper_gen {
	($product_name: ident, $type_appendix: ident, $type: ty) => {
		inner_product_wrapper_gen!($product_name(){}, $type_appendix, $type);
	};
	($product_name: ident ($($args: ident : $arg_types: ty),*), $type_appendix: ident, $type: ty) => {
		inner_product_wrapper_gen!($product_name($($args: $arg_types),*){}, $type_appendix, $type);
	};
	($product_name: ident {$($prod_arg: expr),*}, $type_appendix: ident, $type: ty) => {
		inner_product_wrapper_gen!($product_name(){$($prod_arg),*}, $type_appendix, $type);
	};
	($product_name: ident ($($args: ident : $arg_types: ty),*) {$($prod_arg: expr),*}, $type_appendix: ident, $type: ty) => {
		paste! {
			#[pyclass]
			pub struct [<$product_name $type_appendix>] {
				product: $product_name<$type>
			}
			#[pymethods]
			impl [<$product_name $type_appendix>] {
				#[new]
				pub fn new($($args: $arg_types),*) -> [<$product_name $type_appendix>] {
					[<$product_name $type_appendix>] {
						product: $product_name::new($($prod_arg),*)
					}
				}
			}

			forward_inner_product_function!([<$product_name $type_appendix>], prod, $type);
			forward_inner_product_function!([<$product_name $type_appendix>], self_prod, $type, single);
			forward_inner_product_function!([<$product_name $type_appendix>], prods, $type, PyReadonlyArray1, PyReadonlyArray2, PyArray1);
			forward_inner_product_function!([<$product_name $type_appendix>], self_prods, $type, PyReadonlyArray2, PyArray1);
			forward_inner_product_function!([<$product_name $type_appendix>], zip_prods, $type, PyReadonlyArray2, PyReadonlyArray2, PyArray1);
			forward_inner_product_function!([<$product_name $type_appendix>], cross_prods, $type, PyReadonlyArray2, PyReadonlyArray2, PyArray2);
			forward_inner_product_function!([<$product_name $type_appendix>], induced_dist, $type);
			forward_inner_product_function!([<$product_name $type_appendix>], induced_dists, $type, PyReadonlyArray1, PyReadonlyArray2, PyArray1);
			forward_inner_product_function!([<$product_name $type_appendix>], zip_induced_dists, $type, PyReadonlyArray2, PyReadonlyArray2, PyArray1);
			forward_inner_product_function!([<$product_name $type_appendix>], cross_induced_dists, $type, PyReadonlyArray2, PyReadonlyArray2, PyArray2);
		}
	};
}

/* Macro to generate instantiated inner product bounder classes for python */
macro_rules! inner_product_bounds_struct_gen {
	/* Arguments:
	 * product_name - Name of the inner product type to use (e.g. DotProduct)
	 * (...) - input arguments to the python wrapper for inner product arguments
	 * {...} - inner product constructor arguments based on input arguments
	 * type_appendix - Suffix for the name of the type to generate which indicates the accepted coefficient types
	 * type - Explicit data type of input data, should be explained by type_appendix
	 */
	($product_name: ident ($($args: ident : $arg_types: ty),*) {$($prod_arg: expr),*}, $type_appendix: ident, $type: ty) => {
		paste! {
			#[pyclass]
			pub struct [<$product_name Bounds $type_appendix>] {
				bounds: InnerProductBounder<$product_name<$type>,$type>
			}
			#[pymethods]
			impl [<$product_name Bounds $type_appendix>] {
				#[new]
				pub fn new(data: PyReadonlyArray2<$type> $(, $args: $arg_types)*, num_pivots: Option<usize>, refs: Option<PyReadonlyArray2<$type>>) -> [<$product_name Bounds $type_appendix>] {
					[<$product_name Bounds $type_appendix>] {
						bounds: InnerProductBounder::new(
							$product_name::new($($prod_arg),*),
							&data.as_array(),
							num_pivots,
							if refs.is_some() { Some(refs.unwrap().as_array().to_owned()) } else { None }
						)
					}
				}
				pub fn product_lower_bounds<'py>(&self, py: Python<'py>, queries: PyReadonlyArrayDyn<$type>) -> &'py PyArray2<$type> {
					let queries_array = into_2d_array!(queries);
					self.bounds.prod_bounds(&queries_array, true).into_pyarray(py)
				}
				pub fn product_upper_bounds<'py>(&self, py: Python<'py>, queries: PyReadonlyArrayDyn<$type>) -> &'py PyArray2<$type> {
					let queries_array = into_2d_array!(queries);
					self.bounds.prod_bounds(&queries_array, false).into_pyarray(py)
				}
				pub fn product_bounds<'py>(&self, py: Python<'py>, queries: PyReadonlyArrayDyn<$type>) -> (&'py PyArray2<$type>, &'py PyArray2<$type>) {
					let queries_array = into_2d_array!(queries);
					let (lower, upper) = self.bounds.prod_bounds_both(&queries_array);
					(lower.into_pyarray(py), upper.into_pyarray(py))
				}
				pub fn distance_lower_bounds<'py>(&self, py: Python<'py>, queries: PyReadonlyArrayDyn<$type>) -> &'py PyArray2<$type> {
					let queries_array = into_2d_array!(queries);
					self.bounds.distance_bounds(&queries_array, true).into_pyarray(py)
				}
				pub fn distance_upper_bounds<'py>(&self, py: Python<'py>, queries: PyReadonlyArrayDyn<$type>) -> &'py PyArray2<$type> {
					let queries_array = into_2d_array!(queries);
					self.bounds.distance_bounds(&queries_array, false).into_pyarray(py)
				}
				pub fn distance_bounds<'py>(&self, py: Python<'py>, queries: PyReadonlyArrayDyn<$type>) -> (&'py PyArray2<$type>, &'py PyArray2<$type>) {
					let queries_array = into_2d_array!(queries);
					let (lower, upper) = self.bounds.distance_bounds_both(&queries_array);
					(lower.into_pyarray(py), upper.into_pyarray(py))
				}
				pub fn get_pivots<'py>(&self, py: Python<'py>) -> &'py PyArray2<$type> {
					self.bounds.reference_points.clone().into_pyarray(py)
				}
			}
		}
	}
}

/* Macro to generate instantiated PFLS classes for python */
macro_rules! pfls_struct_gen {
	/* Arguments:
	 * product_name - Name of the inner product type to use (e.g. DotProduct)
	 * (...) - input arguments to the python wrapper for inner product arguments
	 * {...} - inner product constructor arguments based on input arguments
	 * name_prefix - Prefix for the name of the type to generate
	 * fun_infix - Declares whether this index uses the product or induced distance values, either "product" or "distance".
	 * type_appendix - Suffix for the name of the type to generate which indicates the accepted coefficient types
	 * type - Explicit data type of input data, should be explained by type_appendix
	 */
	($product_name: ident ($($args: ident : $arg_types: ty),*) {$($prod_arg: expr),*}, $name_prefix: ident, $fun_infix: ident, $type_appendix: ident, $type: ty) => {
		paste! {
			#[pyclass]
			pub struct [<$name_prefix PFLS $type_appendix>] {
				index: PFLS<$product_name<$type>,$type,OwnedRepr<$type>>
			}
			#[pymethods]
			impl [<$name_prefix PFLS $type_appendix>] {
				#[new]
				pub fn new(data: PyReadonlyArray2<$type> $(, $args: $arg_types)*, num_pivots: Option<usize>, refs: Option<PyReadonlyArray2<$type>>) -> [<$name_prefix PFLS $type_appendix>] {
					[<$name_prefix PFLS $type_appendix>] {
						index: PFLS::new(
							$product_name::new($($prod_arg),*),
							data.as_array().into_owned(),
							num_pivots,
							if refs.is_some() { Some(refs.unwrap().as_array().into_owned()) } else { None }
						)
					}
				}
				pub fn query(&self, queries: PyReadonlyArrayDyn<$type>, k: usize, sorting: Option<bool>, smallests: Option<bool>) -> (Vec<Vec<$type>>, Vec<Vec<usize>>) {
					let queries_array = into_2d_array!(queries);
					if sorting.unwrap_or(true) {
						if smallests.unwrap_or(true) {
							self.index.[<query_k_smallest_ $fun_infix _sorting>](&queries_array, k)
						} else {
							self.index.[<query_k_largest_ $fun_infix _sorting>](&queries_array, k)
						}
					} else {
						if smallests.unwrap_or(true) {
							self.index.[<query_k_smallest_ $fun_infix _direct>](&queries_array, k)
						} else {
							self.index.[<query_k_largest_ $fun_infix _direct>](&queries_array, k)
						}
					}
				}
				pub fn query_ball_point(&self, queries: PyReadonlyArrayDyn<$type>, threshold: $type, smallests: Option<bool>) -> Vec<Vec<usize>> {
					let queries_array = into_2d_array!(queries);
					if smallests.unwrap_or(true) {
						self.index.[<query_ $fun_infix _below>](&queries_array, threshold)
					} else {
						self.index.[<query_ $fun_infix _above>](&queries_array, threshold)
					}
				}
			}
		}
	}
}


/* Instantiates the python wrappers for inner products and the
 * corresponding bounders and PFLS indices. */
macro_rules! build_structs_for_product {
	/* Arguments:
	 * product_name - Name of the inner product type to use (e.g. DotProduct)
	 * (...) - input arguments to the python wrapper for inner product arguments
	 * {...} - inner product constructor arguments based on input arguments
	 * dist_name - Name of the induced distance of the used inner product
	 * type_appendix - Suffix for the name of the type to generate which indicates the accepted coefficient types
	 * type - Explicit data type of input data, should be explained by type_appendix
	 * 
	 * If type_appendix and type are missing, this will create structs for f32 and f64.
	 * (...) and {...} are optional but the must both be present or absent.
	 */
	($product_name: ident, $dist_name: ident) => {
		build_structs_for_product!($product_name(){}, $dist_name);
	};
	($product_name: ident, $dist_name: ident, $type_appendix: ident, $type: ty) => {
		build_structs_for_product!($product_name(){}, $dist_name, $type_appendix, $type);
	};
	($product_name: ident ($($args: ident : $arg_types: ty),*) {$($prod_arg: expr),*}, $dist_name: ident) => {
		build_structs_for_product!($product_name($($args: $arg_types),*){$($prod_arg),*}, $dist_name, F32, f32);
		build_structs_for_product!($product_name($($args: $arg_types),*){$($prod_arg),*}, $dist_name, F64, f64);
	};
	($product_name: ident ($($args: ident : $arg_types: ty),*) {$($prod_arg: expr),*}, $dist_name: ident, $type_appendix: ident, $type: ty) => {
		inner_product_wrapper_gen!($product_name($($args: $arg_types),*){$($prod_arg),*}, $type_appendix, $type);
		inner_product_bounds_struct_gen!($product_name($($args: $arg_types),*){$($prod_arg),*}, $type_appendix, $type);
		pfls_struct_gen!($product_name($($args: $arg_types),*){$($prod_arg),*}, $product_name, product, $type_appendix, $type);
		pfls_struct_gen!($product_name($($args: $arg_types),*){$($prod_arg),*}, $dist_name, distance, $type_appendix, $type);
	};
}
build_structs_for_product!(DotProduct, EucDistance);
build_structs_for_product!(CosSim, SqrtCosDist);
build_structs_for_product!(RBFKernel(bandwidth: f32){bandwidth}, RBFDistance, F32, f32);
build_structs_for_product!(RBFKernel(bandwidth: f64){bandwidth}, RBFDistance, F64, f64);
build_structs_for_product!(MahalanobisKernel(inv_cov: PyReadonlyArray2<f32>){inv_cov.as_array()}, MahalanobisDistance, F32, f32);
build_structs_for_product!(MahalanobisKernel(inv_cov: PyReadonlyArray2<f64>){inv_cov.as_array()}, MahalanobisDistance, F64, f64);
build_structs_for_product!(PolyKernel(scale: f32, bias: f32, degree: f32){scale, bias, degree}, PolyDistance, F32, f32);
build_structs_for_product!(PolyKernel(scale: f64, bias: f64, degree: f64){scale, bias, degree}, PolyDistance, F64, f64);
build_structs_for_product!(SigmoidKernel(scale: f32, bias: f32){scale, bias}, SigmoidDistance, F32, f32);
build_structs_for_product!(SigmoidKernel(scale: f64, bias: f64){scale, bias}, SigmoidDistance, F64, f64);



/* Collection of exposable structs for each inner product. */
macro_rules! python_export {
	($module: ident, $product_name: ident, $distance_name: ident) => {
		paste!{
			$module.add_class::<[<$product_name F32>]>()?;
			$module.add_class::<[<$product_name F64>]>()?;
			$module.add_class::<[<$product_name BoundsF32>]>()?;
			$module.add_class::<[<$product_name BoundsF64>]>()?;
			$module.add_class::<[<$product_name PFLSF32>]>()?;
			$module.add_class::<[<$product_name PFLSF64>]>()?;
			$module.add_class::<[<$distance_name PFLSF32>]>()?;
			$module.add_class::<[<$distance_name PFLSF64>]>()?;
		}
	};
}

/* Declaration of the python package generated by maturin. */
#[pymodule]
fn pfls(_py: Python, m: &PyModule) -> PyResult<()> {
	python_export!(m, DotProduct, EucDistance);
	python_export!(m, CosSim, SqrtCosDist);
	python_export!(m, RBFKernel, RBFDistance);
	python_export!(m, MahalanobisKernel, MahalanobisDistance);
	python_export!(m, PolyKernel, PolyDistance);
	python_export!(m, SigmoidKernel, SigmoidDistance);
	#[cfg(feature="count_operations")]
	{
		m.add_function(wrap_pyfunction!(get_prod_counter, m)?)?;
		m.add_function(wrap_pyfunction!(reset_prod_counter, m)?)?;
		m.add_function(wrap_pyfunction!(get_dist_counter, m)?)?;
		m.add_function(wrap_pyfunction!(reset_dist_counter, m)?)?;
	}
	Ok(())
}
