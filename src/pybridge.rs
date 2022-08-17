use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn, PyReadonlyArray2};
use paste::paste;
use ndarray::{OwnedRepr};

pub mod measures;
pub mod spatialpruning;
pub mod primitives;

#[cfg(test)]
mod test;

use spatialpruning::{PFLS, InnerProductBounder};
#[cfg(not(feature="count_operations"))]
use measures::{DotProduct};
#[cfg(feature="count_operations")]
use measures::{DotProduct, PROD_COUNTER, DIST_COUNTER};


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


/* Macro to generate instantiated inner product bounder classes for python */
macro_rules! inner_product_bounds_struct_gen {
	/* Arguments:
	 * product_name - Name of the inner product type to use (e.g. DotProduct)
	 * type_appendix - Suffix for the name of the type to generate which indicates the accepted coefficient types
	 * type - Explicit data type of input data, should be explained by type_appendix
	 */
	($product_name: ident, $type_appendix: ident, $type: ty) => {
		paste! {
			#[pyclass]
			pub struct [<$product_name Bounds $type_appendix>] {
				bounds: InnerProductBounder<$product_name<$type>,$type>
			}
			#[pymethods]
			impl [<$product_name Bounds $type_appendix>] {
				#[new]
				pub fn new(data: PyReadonlyArray2<$type>, num_pivots: Option<usize>, refs: Option<PyReadonlyArray2<$type>>) -> [<$product_name Bounds $type_appendix>] {
					[<$product_name Bounds $type_appendix>] {
						bounds: InnerProductBounder::new(
							$product_name::new(),
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

/* Explicit instantiations of inner product bounders */
inner_product_bounds_struct_gen!(DotProduct, F32, f32);
inner_product_bounds_struct_gen!(DotProduct, F64, f64);



/* Macro to generate instantiated PFLS classes for python */
macro_rules! pfls_struct_gen {
	/* Arguments:
	 * product_name - Name of the inner product type to use (e.g. DotProduct)
	 * name_prefix - Prefix for the name of the type to generate
	 * fun_infix - Declares whether this index uses the product or induced distance values, either "product" or "distance".
	 * type_appendix - Suffix for the name of the type to generate which indicates the accepted coefficient types
	 * type - Explicit data type of input data, should be explained by type_appendix
	 */
	($product_name: ident, $name_prefix: ident, $fun_infix: ident, $type_appendix: ident, $type: ty) => {
		paste! {
			#[pyclass]
			pub struct [<$name_prefix PFLS $type_appendix>] {
				index: PFLS<$product_name<$type>,$type,OwnedRepr<$type>>
			}
			#[pymethods]
			impl [<$name_prefix PFLS $type_appendix>] {
				#[new]
				pub fn new(data: PyReadonlyArray2<$type>, num_pivots: Option<usize>, refs: Option<PyReadonlyArray2<$type>>) -> [<$name_prefix PFLS $type_appendix>] {
					[<$name_prefix PFLS $type_appendix>] {
						index: PFLS::new(
							$product_name::new(),
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

/* Explicit instantiations of PFLS indices */
pfls_struct_gen!(DotProduct, DotProduct, product, F32, f32);
pfls_struct_gen!(DotProduct, DotProduct, product, F64, f64);
pfls_struct_gen!(DotProduct, EucDistance, distance, F32, f32);
pfls_struct_gen!(DotProduct, EucDistance, distance, F64, f64);


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


/* Declaration of the python package generated by maturin. */
#[pymodule]
fn pfls(_py: Python, m: &PyModule) -> PyResult<()> {
	m.add_class::<DotProductBoundsF32>()?;
	m.add_class::<DotProductBoundsF64>()?;
	m.add_class::<DotProductPFLSF32>()?;
	m.add_class::<DotProductPFLSF64>()?;
	m.add_class::<EucDistancePFLSF32>()?;
	m.add_class::<EucDistancePFLSF64>()?;
	#[cfg(feature="count_operations")]
	{
		m.add_function(wrap_pyfunction!(get_prod_counter, m)?)?;
		m.add_function(wrap_pyfunction!(reset_prod_counter, m)?)?;
		m.add_function(wrap_pyfunction!(get_dist_counter, m)?)?;
		m.add_function(wrap_pyfunction!(reset_dist_counter, m)?)?;
	}
	Ok(())
}
