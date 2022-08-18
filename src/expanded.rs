#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
use pyo3::prelude::*;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn, PyReadonlyArray1,
    PyReadonlyArray2,
};
use paste::paste;
use ndarray::OwnedRepr;
pub mod measures {
    use std::vec::Vec;
    use std::marker::PhantomData;
    use num::Float;
    use ndarray::{Data, Axis, ArrayBase, Array, Array1, Array2, Ix1, Ix2};
    pub trait InnerProduct<N: Float>: Clone {
        fn prod<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            obj1: &ArrayBase<D1, Ix1>,
            obj2: &ArrayBase<D2, Ix1>,
        ) -> N;
        #[inline(always)]
        fn prods<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            obj: &ArrayBase<D1, Ix1>,
            objs: &ArrayBase<D2, Ix2>,
        ) -> Array1<N> {
            objs.outer_iter().map(|obj2| self.prod(obj, &obj2)).collect()
        }
        #[inline(always)]
        fn zip_prods<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            objs1: &ArrayBase<D1, Ix2>,
            objs2: &ArrayBase<D2, Ix2>,
        ) -> Array1<N> {
            objs1
                .outer_iter()
                .zip(objs2.outer_iter())
                .map(|(obj1, obj2)| self.prod(&obj1, &obj2))
                .collect()
        }
        #[inline(always)]
        fn cross_prods<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            objs1: &ArrayBase<D1, Ix2>,
            objs2: &ArrayBase<D2, Ix2>,
        ) -> Array2<N> {
            Array::from_shape_vec(
                    (objs1.len_of(Axis(0)), objs2.len_of(Axis(0))),
                    objs1
                        .outer_iter()
                        .flat_map(|obj1| {
                            objs2
                                .outer_iter()
                                .map(|obj2| self.prod(&obj1, &obj2))
                                .collect::<Vec<N>>()
                        })
                        .collect(),
                )
                .unwrap()
        }
        #[inline(always)]
        fn self_prod<D1: Data<Elem = N>>(&self, obj: &ArrayBase<D1, Ix1>) -> N {
            self.prod(obj, obj)
        }
        #[inline(always)]
        fn self_prods<D1: Data<Elem = N>>(
            &self,
            objs: &ArrayBase<D1, Ix2>,
        ) -> Array1<N> {
            self.zip_prods(objs, objs)
        }
        #[inline(always)]
        fn induced_dist<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            obj1: &ArrayBase<D1, Ix1>,
            obj2: &ArrayBase<D2, Ix1>,
        ) -> N {
            self.self_prod(&(obj1 - obj2)).sqrt()
        }
        #[inline(always)]
        fn induced_dists<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            obj: &ArrayBase<D1, Ix1>,
            objs: &ArrayBase<D2, Ix2>,
        ) -> Array1<N> {
            objs.outer_iter().map(|obj2| self.induced_dist(obj, &obj2)).collect()
        }
        #[inline(always)]
        fn zip_induced_dists<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            objs1: &ArrayBase<D1, Ix2>,
            objs2: &ArrayBase<D2, Ix2>,
        ) -> Array1<N> {
            objs1
                .outer_iter()
                .zip(objs2.outer_iter())
                .map(|(obj1, obj2)| self.induced_dist(&obj1, &obj2))
                .collect()
        }
        #[inline(always)]
        fn cross_induced_dists<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            objs1: &ArrayBase<D1, Ix2>,
            objs2: &ArrayBase<D2, Ix2>,
        ) -> Array2<N> {
            Array::from_shape_vec(
                    (objs1.len_of(Axis(0)), objs2.len_of(Axis(0))),
                    objs1
                        .outer_iter()
                        .flat_map(|obj1| {
                            objs2
                                .outer_iter()
                                .map(|obj2| self.induced_dist(&obj1, &obj2))
                                .collect::<Vec<N>>()
                        })
                        .collect(),
                )
                .unwrap()
        }
    }
    pub struct DotProduct<N: Float> {
        _marker: PhantomData<N>,
    }
    #[automatically_derived]
    impl<N: ::core::fmt::Debug + Float> ::core::fmt::Debug for DotProduct<N> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "DotProduct",
                "_marker",
                &&self._marker,
            )
        }
    }
    #[automatically_derived]
    impl<N: ::core::clone::Clone + Float> ::core::clone::Clone for DotProduct<N> {
        #[inline]
        fn clone(&self) -> DotProduct<N> {
            DotProduct {
                _marker: ::core::clone::Clone::clone(&self._marker),
            }
        }
    }
    impl<N: Float> DotProduct<N> {
        pub fn new() -> Self {
            DotProduct { _marker: PhantomData }
        }
    }
    impl<N: Float> InnerProduct<N> for DotProduct<N> {
        fn prod<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            obj1: &ArrayBase<D1, Ix1>,
            obj2: &ArrayBase<D2, Ix1>,
        ) -> N {
            obj1.into_iter()
                .zip(obj2.into_iter())
                .map(|(&a, &b)| a * b)
                .reduce(|a, b| a + b)
                .unwrap_or(num::Zero::zero())
        }
    }
    pub struct RBFKernel<N: Float> {
        bandwidth: N,
    }
    #[automatically_derived]
    impl<N: ::core::fmt::Debug + Float> ::core::fmt::Debug for RBFKernel<N> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "RBFKernel",
                "bandwidth",
                &&self.bandwidth,
            )
        }
    }
    #[automatically_derived]
    impl<N: ::core::clone::Clone + Float> ::core::clone::Clone for RBFKernel<N> {
        #[inline]
        fn clone(&self) -> RBFKernel<N> {
            RBFKernel {
                bandwidth: ::core::clone::Clone::clone(&self.bandwidth),
            }
        }
    }
    impl<N: Float> RBFKernel<N> {
        pub fn new(bandwidth: N) -> Self {
            RBFKernel { bandwidth: bandwidth }
        }
    }
    impl<N: Float> InnerProduct<N> for RBFKernel<N> {
        fn prod<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            obj1: &ArrayBase<D1, Ix1>,
            obj2: &ArrayBase<D2, Ix1>,
        ) -> N {
            let d2 = obj1
                .into_iter()
                .zip(obj2.into_iter())
                .map(|(&a, &b)| a - b)
                .map(|a| a * a)
                .reduce(|a, b| a + b)
                .unwrap_or(num::Zero::zero());
            N::exp(-d2 / self.bandwidth)
        }
    }
    pub struct MahalanobisKernel<N: Float> {
        inv_cov: Array2<N>,
    }
    #[automatically_derived]
    impl<N: ::core::fmt::Debug + Float> ::core::fmt::Debug for MahalanobisKernel<N> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "MahalanobisKernel",
                "inv_cov",
                &&self.inv_cov,
            )
        }
    }
    #[automatically_derived]
    impl<N: ::core::clone::Clone + Float> ::core::clone::Clone for MahalanobisKernel<N> {
        #[inline]
        fn clone(&self) -> MahalanobisKernel<N> {
            MahalanobisKernel {
                inv_cov: ::core::clone::Clone::clone(&self.inv_cov),
            }
        }
    }
    impl<N: Float> MahalanobisKernel<N> {
        pub fn new<D: Data<Elem = N>>(inv_cov: ArrayBase<D, Ix2>) -> Self {
            MahalanobisKernel {
                inv_cov: inv_cov.to_owned(),
            }
        }
    }
    impl<N: Float> InnerProduct<N> for MahalanobisKernel<N> {
        fn prod<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            obj1: &ArrayBase<D1, Ix1>,
            obj2: &ArrayBase<D2, Ix1>,
        ) -> N {
            obj1.into_iter()
                .enumerate()
                .zip(obj2.into_iter().enumerate())
                .map(|((i, &a), (j, &b))| a * b * self.inv_cov[[i, j]])
                .reduce(|a, b| a + b)
                .unwrap_or(num::Zero::zero())
        }
    }
}
pub mod spatialpruning {
    use rand::seq::index::sample;
    use std::vec::Vec;
    use num::traits::{Float, FromPrimitive};
    use std::iter::Sum;
    use std::collections::BinaryHeap;
    use ndarray::{
        s, Axis, Array, Array1, Array2, ArrayBase, Data, Ix1, Ix2, ScalarOperand,
    };
    use std::cmp::Reverse;
    use std::fmt::Debug;
    use crate::measures::InnerProduct;
    use crate::primitives::MeasurePair;
    pub struct InnerProductBounder<P, N>
    where
        P: InnerProduct<N>,
        N: Float + Debug,
    {
        product: P,
        data_squares: Array1<N>,
        pub reference_points: Array2<N>,
        ref_ortho_prods: Array2<N>,
        ortho_dot_divisors: Array1<N>,
        data_ortho_prods: Array2<N>,
        data_remainder_squares: Array1<N>,
    }
    impl<P, N> InnerProductBounder<P, N>
    where
        P: InnerProduct<N>,
        N: Float + FromPrimitive + Sum + ScalarOperand + Debug,
    {
        pub fn new<D1: Data<Elem = N>>(
            product: P,
            data: &ArrayBase<D1, Ix2>,
            n_refs: Option<usize>,
            refs: Option<Array2<N>>,
        ) -> InnerProductBounder<P, N> {
            let zero: N = num::Zero::zero();
            let data_length = data.len_of(Axis(0));
            let data_dim = data.len_of(Axis(1));
            let n_refs = if refs.is_some() {
                refs.as_ref().unwrap().len_of(Axis(0))
            } else {
                n_refs.unwrap().min(data_length).min(data_dim)
            };
            let data_squares = product.self_prods(data);
            let ref_points: Array2<N> = if refs.is_some() {
                refs.unwrap()
            } else {
                Array::from_shape_vec(
                        (n_refs, data.len_of(Axis(1))),
                        sample(&mut rand::thread_rng(), data_length, n_refs)
                            .iter()
                            .flat_map(|i| data.index_axis(Axis(0), i).into_owned())
                            .collect::<Vec<N>>(),
                    )
                    .unwrap()
            };
            let data_ref_prods = product.cross_prods(data, &ref_points);
            let ref_ref_prods = product.cross_prods(&ref_points, &ref_points);
            let mut ortho_dot_divisors = Array::zeros(n_refs);
            let mut ref_ortho_prods = Array::zeros((n_refs, n_refs));
            for k in 0..n_refs {
                for i in 0..k {
                    if ortho_dot_divisors[i] > N::zero() {
                        ref_ortho_prods[[
                            i,
                            k,
                        ]] = (ref_ref_prods[[i, k]]
                            - (&ref_ortho_prods
                                .slice(
                                    match k {
                                        r => {
                                            match ..i {
                                                r => {
                                                    let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                        &r,
                                                        ::ndarray::SliceNextDim::next_in_dim(
                                                            &r,
                                                            ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                        ),
                                                    );
                                                    let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                        &r,
                                                        ::ndarray::SliceNextDim::next_out_dim(
                                                            &r,
                                                            ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                        ),
                                                    );
                                                    #[allow(unsafe_code)]
                                                    unsafe {
                                                        ::ndarray::SliceInfo::new_unchecked(
                                                            [
                                                                <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                    _,
                                                                >>::from(r),
                                                                <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                    _,
                                                                >>::from(r),
                                                            ],
                                                            in_dim,
                                                            out_dim,
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    },
                                )
                                * &ref_ortho_prods
                                    .slice(
                                        match i {
                                            r => {
                                                match ..i {
                                                    r => {
                                                        let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                            &r,
                                                            ::ndarray::SliceNextDim::next_in_dim(
                                                                &r,
                                                                ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                            ),
                                                        );
                                                        let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                            &r,
                                                            ::ndarray::SliceNextDim::next_out_dim(
                                                                &r,
                                                                ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                            ),
                                                        );
                                                        #[allow(unsafe_code)]
                                                        unsafe {
                                                            ::ndarray::SliceInfo::new_unchecked(
                                                                [
                                                                    <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                        _,
                                                                    >>::from(r),
                                                                    <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                        _,
                                                                    >>::from(r),
                                                                ],
                                                                in_dim,
                                                                out_dim,
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                    ))
                                .sum()) / ortho_dot_divisors[i];
                    } else {
                        ref_ortho_prods[[i, k]] = N::zero();
                    }
                    ref_ortho_prods[[k, i]] = ref_ortho_prods[[i, k]];
                }
                ortho_dot_divisors[k] = (ref_ref_prods[[k, k]]
                    - (&ref_ortho_prods
                        .slice(
                            match k {
                                r => {
                                    match ..k {
                                        r => {
                                            let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                &r,
                                                ::ndarray::SliceNextDim::next_in_dim(
                                                    &r,
                                                    ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                ),
                                            );
                                            let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                &r,
                                                ::ndarray::SliceNextDim::next_out_dim(
                                                    &r,
                                                    ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                ),
                                            );
                                            #[allow(unsafe_code)]
                                            unsafe {
                                                ::ndarray::SliceInfo::new_unchecked(
                                                    [
                                                        <::ndarray::SliceInfoElem as ::core::convert::From<
                                                            _,
                                                        >>::from(r),
                                                        <::ndarray::SliceInfoElem as ::core::convert::From<
                                                            _,
                                                        >>::from(r),
                                                    ],
                                                    in_dim,
                                                    out_dim,
                                                )
                                            }
                                        }
                                    }
                                }
                            },
                        )
                        * &ref_ortho_prods
                            .slice(
                                match k {
                                    r => {
                                        match ..k {
                                            r => {
                                                let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                    &r,
                                                    ::ndarray::SliceNextDim::next_in_dim(
                                                        &r,
                                                        ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                    ),
                                                );
                                                let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                    &r,
                                                    ::ndarray::SliceNextDim::next_out_dim(
                                                        &r,
                                                        ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                    ),
                                                );
                                                #[allow(unsafe_code)]
                                                unsafe {
                                                    ::ndarray::SliceInfo::new_unchecked(
                                                        [
                                                            <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                _,
                                                            >>::from(r),
                                                            <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                _,
                                                            >>::from(r),
                                                        ],
                                                        in_dim,
                                                        out_dim,
                                                    )
                                                }
                                            }
                                        }
                                    }
                                },
                            ))
                        .sum())
                    .max(N::zero())
                    .sqrt();
            }
            let mut data_ortho_prods = Array::zeros((data_length, n_refs));
            for k in 0..data_length {
                for i in 0..n_refs {
                    if ortho_dot_divisors[i] > N::zero() {
                        data_ortho_prods[[
                            k,
                            i,
                        ]] = (data_ref_prods[[k, i]]
                            - (&data_ortho_prods
                                .slice(
                                    match k {
                                        r => {
                                            match ..i {
                                                r => {
                                                    let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                        &r,
                                                        ::ndarray::SliceNextDim::next_in_dim(
                                                            &r,
                                                            ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                        ),
                                                    );
                                                    let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                        &r,
                                                        ::ndarray::SliceNextDim::next_out_dim(
                                                            &r,
                                                            ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                        ),
                                                    );
                                                    #[allow(unsafe_code)]
                                                    unsafe {
                                                        ::ndarray::SliceInfo::new_unchecked(
                                                            [
                                                                <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                    _,
                                                                >>::from(r),
                                                                <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                    _,
                                                                >>::from(r),
                                                            ],
                                                            in_dim,
                                                            out_dim,
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    },
                                )
                                * &ref_ortho_prods
                                    .slice(
                                        match i {
                                            r => {
                                                match ..i {
                                                    r => {
                                                        let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                            &r,
                                                            ::ndarray::SliceNextDim::next_in_dim(
                                                                &r,
                                                                ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                            ),
                                                        );
                                                        let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                            &r,
                                                            ::ndarray::SliceNextDim::next_out_dim(
                                                                &r,
                                                                ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                            ),
                                                        );
                                                        #[allow(unsafe_code)]
                                                        unsafe {
                                                            ::ndarray::SliceInfo::new_unchecked(
                                                                [
                                                                    <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                        _,
                                                                    >>::from(r),
                                                                    <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                        _,
                                                                    >>::from(r),
                                                                ],
                                                                in_dim,
                                                                out_dim,
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                    ))
                                .sum()) / ortho_dot_divisors[i];
                    } else {
                        data_ortho_prods[[k, i]] = N::zero();
                    }
                }
            }
            let data_remainder_squares: Array1<N> = (&data_squares
                - (0..data_length)
                    .map(|i| {
                        data_ortho_prods.index_axis(Axis(0), i).mapv(|v| v * v).sum()
                    })
                    .collect::<Array1<N>>())
                .mapv(|v| v.max(zero));
            InnerProductBounder {
                product: product,
                data_squares: data_squares,
                reference_points: ref_points,
                ref_ortho_prods: ref_ortho_prods,
                ortho_dot_divisors: ortho_dot_divisors,
                data_ortho_prods: data_ortho_prods,
                data_remainder_squares: data_remainder_squares,
            }
        }
        fn ortho_prods<D1: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D1, Ix2>,
        ) -> Array2<N> {
            let n_vecs = vecs.len_of(Axis(0));
            let n_refs = self.reference_points.len_of(Axis(0));
            let products = self.product.cross_prods(vecs, &self.reference_points);
            let mut ortho_prods = Array::zeros((n_vecs, n_refs));
            for k in 0..n_vecs {
                for i in 0..n_refs {
                    if self.ortho_dot_divisors[i] > N::zero() {
                        ortho_prods[[
                            k,
                            i,
                        ]] = (products[[k, i]]
                            - (&ortho_prods
                                .slice(
                                    match k {
                                        r => {
                                            match ..i {
                                                r => {
                                                    let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                        &r,
                                                        ::ndarray::SliceNextDim::next_in_dim(
                                                            &r,
                                                            ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                        ),
                                                    );
                                                    let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                        &r,
                                                        ::ndarray::SliceNextDim::next_out_dim(
                                                            &r,
                                                            ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                        ),
                                                    );
                                                    #[allow(unsafe_code)]
                                                    unsafe {
                                                        ::ndarray::SliceInfo::new_unchecked(
                                                            [
                                                                <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                    _,
                                                                >>::from(r),
                                                                <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                    _,
                                                                >>::from(r),
                                                            ],
                                                            in_dim,
                                                            out_dim,
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    },
                                )
                                * &self
                                    .ref_ortho_prods
                                    .slice(
                                        match i {
                                            r => {
                                                match ..i {
                                                    r => {
                                                        let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                                                            &r,
                                                            ::ndarray::SliceNextDim::next_in_dim(
                                                                &r,
                                                                ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                            ),
                                                        );
                                                        let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                                                            &r,
                                                            ::ndarray::SliceNextDim::next_out_dim(
                                                                &r,
                                                                ::core::marker::PhantomData::<::ndarray::Ix0>,
                                                            ),
                                                        );
                                                        #[allow(unsafe_code)]
                                                        unsafe {
                                                            ::ndarray::SliceInfo::new_unchecked(
                                                                [
                                                                    <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                        _,
                                                                    >>::from(r),
                                                                    <::ndarray::SliceInfoElem as ::core::convert::From<
                                                                        _,
                                                                    >>::from(r),
                                                                ],
                                                                in_dim,
                                                                out_dim,
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                    ))
                                .sum()) / self.ortho_dot_divisors[i];
                    } else {
                        ortho_prods[[k, i]] = N::zero();
                    }
                }
            }
            ortho_prods
        }
        fn remainder_squares<D1: Data<Elem = N>, D2: Data<Elem = N>>(
            &self,
            vecs_squares: &ArrayBase<D1, Ix1>,
            ortho_prods: &ArrayBase<D2, Ix2>,
        ) -> Array1<N> {
            let zero: N = num::Zero::zero();
            (vecs_squares
                - ortho_prods
                    .axis_iter(Axis(0))
                    .map(|v| (&v * &v).sum())
                    .collect::<Array1<N>>())
                .mapv(|v| v.max(zero))
        }
        fn prod_bounds_fixed<D1: Data<Elem = N>>(
            &self,
            ortho_prods: &ArrayBase<D1, Ix2>,
        ) -> Array2<N> {
            Array::from_shape_vec(
                    (ortho_prods.len_of(Axis(0)), self.data_ortho_prods.len_of(Axis(0))),
                    ortho_prods
                        .axis_iter(Axis(0))
                        .flat_map(|v1| {
                            self
                                .data_ortho_prods
                                .axis_iter(Axis(0))
                                .map(|v2| (&v1 * &v2).sum())
                                .collect::<Vec<N>>()
                        })
                        .collect(),
                )
                .unwrap()
        }
        fn prod_bounds_flex<D1: Data<Elem = N>>(
            &self,
            vecs_remainder_squares: &ArrayBase<D1, Ix1>,
        ) -> Array2<N> {
            Array::from_shape_vec(
                    (vecs_remainder_squares.len(), self.data_remainder_squares.len()),
                    vecs_remainder_squares
                        .iter()
                        .flat_map(|&a| {
                            self
                                .data_remainder_squares
                                .mapv(|b| (a * b).max(N::zero()).sqrt())
                        })
                        .collect(),
                )
                .unwrap()
        }
        pub fn prod_bounds<D1: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D1, Ix2>,
            lower: bool,
        ) -> Array2<N> {
            let vecs_squares = self.product.self_prods(vecs);
            let ortho_prods = self.ortho_prods(vecs);
            let vecs_remainder_squares = self
                .remainder_squares(&vecs_squares, &ortho_prods);
            let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
            let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
            if lower {
                prod_bounds_fixed - prod_bounds_flex
            } else {
                prod_bounds_fixed + prod_bounds_flex
            }
        }
        pub fn prod_bounds_both<D1: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D1, Ix2>,
        ) -> (Array2<N>, Array2<N>) {
            let vecs_squares = self.product.self_prods(vecs);
            let ortho_prods = self.ortho_prods(vecs);
            let vecs_remainder_squares = self
                .remainder_squares(&vecs_squares, &ortho_prods);
            let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
            let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
            (
                &prod_bounds_fixed - &prod_bounds_flex,
                &prod_bounds_fixed + &prod_bounds_flex,
            )
        }
        pub fn distance_bounds<D1: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D1, Ix2>,
            lower: bool,
        ) -> Array2<N> {
            let two = N::from_u8(2).unwrap();
            let vecs_squares = self.product.self_prods(vecs);
            let ortho_prods = self.ortho_prods(vecs);
            let vecs_remainder_squares = self
                .remainder_squares(&vecs_squares, &ortho_prods);
            let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
            let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
            let offset = Array::from_shape_vec(
                    (vecs.len_of(Axis(0)), self.data_squares.len()),
                    (0..vecs.len_of(Axis(0)))
                        .map(|i| &self.data_squares + vecs_squares[i])
                        .flatten()
                        .collect(),
                )
                .unwrap();
            if lower {
                (offset - (prod_bounds_fixed + prod_bounds_flex) * two)
                    .mapv(|v| v.max(N::zero()).sqrt())
            } else {
                (offset - (prod_bounds_fixed - prod_bounds_flex) * two)
                    .mapv(|v| v.max(N::zero()).sqrt())
            }
        }
        pub fn distance_bounds_both<D1: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D1, Ix2>,
        ) -> (Array2<N>, Array2<N>) {
            let two = N::from_u8(2).unwrap();
            let vecs_squares = self.product.self_prods(vecs);
            let ortho_prods = self.ortho_prods(vecs);
            let vecs_remainder_squares = self
                .remainder_squares(&vecs_squares, &ortho_prods);
            let prod_bounds_fixed = self.prod_bounds_fixed(&ortho_prods);
            let prod_bounds_flex = self.prod_bounds_flex(&vecs_remainder_squares);
            let offset = Array::from_shape_vec(
                    (vecs.len_of(Axis(0)), self.data_squares.len()),
                    vecs_squares.iter().flat_map(|&v| &self.data_squares + v).collect(),
                )
                .unwrap();
            (
                (&offset - (&prod_bounds_fixed + &prod_bounds_flex) * two)
                    .mapv(|v| v.max(N::zero()).sqrt()),
                (&offset - (&prod_bounds_fixed - &prod_bounds_flex) * two)
                    .mapv(|v| v.max(N::zero()).sqrt()),
            )
        }
    }
    pub struct PFLS<P, N, D1>
    where
        P: InnerProduct<N>,
        D1: Data<Elem = N>,
        N: Float + Debug,
    {
        product: P,
        data: ArrayBase<D1, Ix2>,
        bounder: InnerProductBounder<P, N>,
    }
    impl<P, N, D1> PFLS<P, N, D1>
    where
        P: InnerProduct<N>,
        D1: Data<Elem = N>,
        N: Float + FromPrimitive + Sum + ScalarOperand + Debug,
    {
        pub fn new(
            product: P,
            data: ArrayBase<D1, Ix2>,
            num_pivots: Option<usize>,
            refs: Option<Array2<N>>,
        ) -> PFLS<P, N, D1> {
            let bounder = InnerProductBounder::new(
                product.clone(),
                &data,
                num_pivots,
                refs,
            );
            PFLS {
                product: product,
                data: data,
                bounder: bounder,
            }
        }
        pub fn query_distance_below<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            threshold: N,
        ) -> Vec<Vec<usize>> {
            let (lower, upper) = self.bounder.distance_bounds_both(vecs);
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs.len_of(Axis(0)),
            );
            for (i, vec) in vecs.axis_iter(Axis(0)).enumerate() {
                let l = lower.index_axis(Axis(0), i);
                let h = upper.index_axis(Axis(0), i);
                for j in 0..l.len() {
                    if l[j] <= threshold {
                        if h[j] <= threshold {
                            indices[i].push(j);
                        } else {
                            let v = self
                                .product
                                .induced_dist(&vec, &self.data.index_axis(Axis(0), j));
                            if v <= threshold {
                                indices[i].push(j);
                            }
                        }
                    }
                }
            }
            indices
        }
        pub fn query_distance_above<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            threshold: N,
        ) -> Vec<Vec<usize>> {
            let (lower, upper) = self.bounder.distance_bounds_both(vecs);
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs.len_of(Axis(0)),
            );
            for (i, vec) in vecs.axis_iter(Axis(0)).enumerate() {
                let l = lower.index_axis(Axis(0), i);
                let h = upper.index_axis(Axis(0), i);
                for j in 0..l.len() {
                    if h[j] >= threshold {
                        if l[j] >= threshold {
                            indices[i].push(j);
                        } else {
                            let v = self
                                .product
                                .induced_dist(&vec, &self.data.index_axis(Axis(0), j));
                            if v >= threshold {
                                indices[i].push(j);
                            }
                        }
                    }
                }
            }
            indices
        }
        pub fn query_product_below<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            threshold: N,
        ) -> Vec<Vec<usize>> {
            let (lower, upper) = self.bounder.prod_bounds_both(vecs);
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs.len_of(Axis(0)),
            );
            for (i, vec) in vecs.axis_iter(Axis(0)).enumerate() {
                let l = lower.index_axis(Axis(0), i);
                let h = upper.index_axis(Axis(0), i);
                for j in 0..l.len() {
                    if l[j] <= threshold {
                        if h[j] <= threshold {
                            indices[i].push(j);
                        } else {
                            let v = self
                                .product
                                .prod(&vec, &self.data.index_axis(Axis(0), j));
                            if v <= threshold {
                                indices[i].push(j);
                            }
                        }
                    }
                }
            }
            indices
        }
        pub fn query_product_above<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            threshold: N,
        ) -> Vec<Vec<usize>> {
            let (lower, upper) = self.bounder.prod_bounds_both(vecs);
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs.len_of(Axis(0)),
            );
            for (i, vec) in vecs.axis_iter(Axis(0)).enumerate() {
                let l = lower.index_axis(Axis(0), i);
                let h = upper.index_axis(Axis(0), i);
                for j in 0..l.len() {
                    if h[j] >= threshold {
                        if l[j] >= threshold {
                            indices[i].push(j);
                        } else {
                            let v = self
                                .product
                                .prod(&vec, &self.data.index_axis(Axis(0), j));
                            if v >= threshold {
                                indices[i].push(j);
                            }
                        }
                    }
                }
            }
            indices
        }
        pub fn query_k_smallest_distance_sorting<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let lower = self.bounder.distance_bounds(vecs, true);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (l, q)) in lower
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let bounds_to_results: Vec<Reverse<MeasurePair<N>>> = l
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| Reverse(MeasurePair { index: j, value: v }))
                    .collect();
                let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::from(
                    bounds_to_results,
                );
                let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                while min_heap.len() > 0 {
                    let mut next_entry = min_heap.pop().unwrap().0;
                    if max_heap.len() >= k
                        && next_entry.value >= max_heap.peek().unwrap().value
                    {
                        break;
                    }
                    next_entry
                        .value = self
                        .product
                        .induced_dist(
                            &q,
                            &self.data.index_axis(Axis(0), next_entry.index),
                        );
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
        pub fn query_k_largest_distance_sorting<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let upper = self.bounder.distance_bounds(vecs, false);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (l, q)) in upper
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let bounds_to_results: Vec<MeasurePair<N>> = l
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| MeasurePair { index: j, value: v })
                    .collect();
                let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::from(
                    bounds_to_results,
                );
                let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                while max_heap.len() > 0 {
                    let mut next_entry = max_heap.pop().unwrap();
                    if min_heap.len() >= k
                        && next_entry.value <= min_heap.peek().unwrap().0.value
                    {
                        break;
                    }
                    next_entry
                        .value = self
                        .product
                        .induced_dist(
                            &q,
                            &self.data.index_axis(Axis(0), next_entry.index),
                        );
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
        pub fn query_k_smallest_product_sorting<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let lower = self.bounder.prod_bounds(vecs, true);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (l, q)) in lower
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let bounds_to_results: Vec<Reverse<MeasurePair<N>>> = l
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| Reverse(MeasurePair { index: j, value: v }))
                    .collect();
                let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::from(
                    bounds_to_results,
                );
                let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                while min_heap.len() > 0 {
                    let mut next_entry = min_heap.pop().unwrap().0;
                    if max_heap.len() >= k
                        && next_entry.value >= max_heap.peek().unwrap().value
                    {
                        break;
                    }
                    next_entry
                        .value = self
                        .product
                        .prod(&q, &self.data.index_axis(Axis(0), next_entry.index));
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
        pub fn query_k_largest_product_sorting<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let upper = self.bounder.prod_bounds(vecs, false);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (l, q)) in upper
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let bounds_to_results: Vec<MeasurePair<N>> = l
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| MeasurePair { index: j, value: v })
                    .collect();
                let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::from(
                    bounds_to_results,
                );
                let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                while max_heap.len() > 0 {
                    let mut next_entry = max_heap.pop().unwrap();
                    if min_heap.len() >= k
                        && next_entry.value <= min_heap.peek().unwrap().0.value
                    {
                        break;
                    }
                    next_entry
                        .value = self
                        .product
                        .prod(&q, &self.data.index_axis(Axis(0), next_entry.index));
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
        pub fn query_k_smallest_distance_direct<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let lower = self.bounder.distance_bounds(vecs, true);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (l, q)) in lower
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                let mut heap_largest = N::max_value();
                for (j, (v, x)) in l.iter().zip(self.data.axis_iter(Axis(0))).enumerate()
                {
                    if max_heap.len() < k || *v <= heap_largest {
                        let true_v = self.product.induced_dist(&q, &x);
                        if true_v <= heap_largest {
                            max_heap
                                .push(MeasurePair {
                                    index: j,
                                    value: true_v,
                                });
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
        pub fn query_k_largest_distance_direct<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let upper = self.bounder.distance_bounds(vecs, false);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (u, q)) in upper
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                let mut heap_smallest = N::zero();
                for (j, (v, x)) in u.iter().zip(self.data.axis_iter(Axis(0))).enumerate()
                {
                    if min_heap.len() < k || *v >= heap_smallest {
                        let true_v = self.product.induced_dist(&q, &x);
                        if true_v >= heap_smallest {
                            min_heap
                                .push(
                                    Reverse(MeasurePair {
                                        index: j,
                                        value: true_v,
                                    }),
                                );
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
        pub fn query_k_smallest_product_direct<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let lower = self.bounder.prod_bounds(vecs, true);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (l, q)) in lower
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let mut max_heap: BinaryHeap<MeasurePair<N>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                let mut heap_largest = N::max_value();
                for (j, (v, x)) in l.iter().zip(self.data.axis_iter(Axis(0))).enumerate()
                {
                    if max_heap.len() < k || *v <= heap_largest {
                        let true_v = self.product.prod(&q, &x);
                        if true_v <= heap_largest {
                            max_heap
                                .push(MeasurePair {
                                    index: j,
                                    value: true_v,
                                });
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
        pub fn query_k_largest_product_direct<D2: Data<Elem = N>>(
            &self,
            vecs: &ArrayBase<D2, Ix2>,
            k: usize,
        ) -> (Vec<Vec<N>>, Vec<Vec<usize>>) {
            let upper = self.bounder.prod_bounds(vecs, false);
            let vecs_length = vecs.len_of(Axis(0));
            let mut values = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(N::zero(), 0),
                vecs_length,
            );
            let mut indices = ::alloc::vec::from_elem(
                ::alloc::vec::from_elem(0 as usize, 0),
                vecs_length,
            );
            for (i, (u, q)) in upper
                .axis_iter(Axis(0))
                .zip(vecs.axis_iter(Axis(0)))
                .enumerate()
            {
                let mut min_heap: BinaryHeap<Reverse<MeasurePair<N>>> = BinaryHeap::with_capacity(
                    k + 1,
                );
                let mut heap_smallest = N::zero();
                for (j, (v, x)) in u.iter().zip(self.data.axis_iter(Axis(0))).enumerate()
                {
                    if min_heap.len() < k || *v >= heap_smallest {
                        let true_v = self.product.prod(&q, &x);
                        if true_v >= heap_smallest {
                            min_heap
                                .push(
                                    Reverse(MeasurePair {
                                        index: j,
                                        value: true_v,
                                    }),
                                );
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
    }
}
pub mod primitives {
    use num::Float;
    use std::cmp::{Ord, PartialOrd, Eq, PartialEq, Ordering};
    pub struct MeasurePair<N> {
        pub index: usize,
        pub value: N,
    }
    #[automatically_derived]
    impl<N: ::core::fmt::Debug> ::core::fmt::Debug for MeasurePair<N> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "MeasurePair",
                "index",
                &&self.index,
                "value",
                &&self.value,
            )
        }
    }
    #[automatically_derived]
    impl<N: ::core::marker::Copy> ::core::marker::Copy for MeasurePair<N> {}
    #[automatically_derived]
    impl<N: ::core::clone::Clone> ::core::clone::Clone for MeasurePair<N> {
        #[inline]
        fn clone(&self) -> MeasurePair<N> {
            MeasurePair {
                index: ::core::clone::Clone::clone(&self.index),
                value: ::core::clone::Clone::clone(&self.value),
            }
        }
    }
    impl<N: Float> Eq for MeasurePair<N> {
        fn assert_receiver_is_total_eq(&self) {}
    }
    impl<N: Float> PartialEq for MeasurePair<N> {
        fn eq(&self, other: &Self) -> bool {
            self.value.eq(&other.value)
        }
    }
    impl<N: Float> PartialOrd for MeasurePair<N> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }
    impl<N: Float> Ord for MeasurePair<N> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.value.partial_cmp(&other.value).unwrap_or(Ordering::Equal)
        }
    }
}
use spatialpruning::{PFLS, InnerProductBounder};
#[cfg(not(feature = "count_operations"))]
use measures::{InnerProduct, DotProduct, RBFKernel, MahalanobisKernel};
pub struct DotProductF32 {
    product: DotProduct<f32>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for DotProductF32 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "DotProductF32";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for DotProductF32 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a DotProductF32 {
    type Target = ::pyo3::PyRef<'a, DotProductF32>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut DotProductF32 {
    type Target = ::pyo3::PyRefMut<'a, DotProductF32>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for DotProductF32 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForDotProductF32 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory for Pyo3MethodsInventoryForDotProductF32 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for DotProductF32 {
    type Methods = Pyo3MethodsInventoryForDotProductF32;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForDotProductF32 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<Pyo3MethodsInventoryForDotProductF32> = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for DotProductF32 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<DotProductF32>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<DotProductF32>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductF32> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl DotProductF32 {
    pub fn new() -> DotProductF32 {
        DotProductF32 {
            product: DotProduct::new(),
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<DotProductF32>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductF32> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = DotProductF32::new();
                    let initializer: ::pyo3::PyClassInitializer<DotProductF32> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init111995171933381579() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                ::alloc::vec::Vec::new(),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init111995171933381579___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init111995171933381579___rust_ctor___ctor() {
        __init111995171933381579()
    }
    __init111995171933381579___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn prod(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> f32 {
        self.product.prod(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init5709766327359551683() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prod",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::prod(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init5709766327359551683___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init5709766327359551683___rust_ctor___ctor() {
        __init5709766327359551683()
    }
    __init5709766327359551683___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn self_prod(&self, a: PyReadonlyArray1<f32>) -> f32 {
        self.product.self_prod(&a.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init12111922029467723324() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prod",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::self_prod(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init12111922029467723324___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init12111922029467723324___rust_ctor___ctor() {
        __init12111922029467723324()
    }
    __init12111922029467723324___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init7031301851698443608() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init7031301851698443608___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init7031301851698443608___rust_ctor___ctor() {
        __init7031301851698443608()
    }
    __init7031301851698443608___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn self_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.self_prods(&a.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init7763755067357875045() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prods",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::self_prods(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init7763755067357875045___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init7763755067357875045___rust_ctor___ctor() {
        __init7763755067357875045()
    }
    __init7763755067357875045___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn zip_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.zip_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init11051414991925199309() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::zip_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init11051414991925199309___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init11051414991925199309___rust_ctor___ctor() {
        __init11051414991925199309()
    }
    __init11051414991925199309___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn cross_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        self.product.cross_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init10508664255139033655() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::cross_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init10508664255139033655___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init10508664255139033655___rust_ctor___ctor() {
        __init10508664255139033655()
    }
    __init10508664255139033655___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn induced_dist(
        &self,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray1<f32>,
    ) -> f32 {
        self.product.induced_dist(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init18270752470055029325() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dist\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dist",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::induced_dist(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init18270752470055029325___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init18270752470055029325___rust_ctor___ctor() {
        __init18270752470055029325()
    }
    __init18270752470055029325___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init17354990529833808456() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init17354990529833808456___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init17354990529833808456___rust_ctor___ctor() {
        __init17354990529833808456()
    }
    __init17354990529833808456___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn zip_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.zip_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init17805367291320052558() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::zip_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init17805367291320052558___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init17805367291320052558___rust_ctor___ctor() {
        __init17805367291320052558()
    }
    __init17805367291320052558___rust_ctor___ctor
};
impl DotProductF32 {
    pub fn cross_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        self.product.cross_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3816318190715045371() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF32::cross_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3816318190715045371___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3816318190715045371___rust_ctor___ctor() {
        __init3816318190715045371()
    }
    __init3816318190715045371___rust_ctor___ctor
};
pub struct DotProductF64 {
    product: DotProduct<f64>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for DotProductF64 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "DotProductF64";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for DotProductF64 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a DotProductF64 {
    type Target = ::pyo3::PyRef<'a, DotProductF64>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut DotProductF64 {
    type Target = ::pyo3::PyRefMut<'a, DotProductF64>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for DotProductF64 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForDotProductF64 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory for Pyo3MethodsInventoryForDotProductF64 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for DotProductF64 {
    type Methods = Pyo3MethodsInventoryForDotProductF64;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForDotProductF64 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<Pyo3MethodsInventoryForDotProductF64> = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for DotProductF64 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<DotProductF64>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<DotProductF64>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductF64> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl DotProductF64 {
    pub fn new() -> DotProductF64 {
        DotProductF64 {
            product: DotProduct::new(),
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<DotProductF64>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductF64> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = DotProductF64::new();
                    let initializer: ::pyo3::PyClassInitializer<DotProductF64> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init8280523027597936587() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                ::alloc::vec::Vec::new(),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init8280523027597936587___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init8280523027597936587___rust_ctor___ctor() {
        __init8280523027597936587()
    }
    __init8280523027597936587___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn prod(&self, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
        self.product.prod(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init18002436895765956049() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prod",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::prod(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init18002436895765956049___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init18002436895765956049___rust_ctor___ctor() {
        __init18002436895765956049()
    }
    __init18002436895765956049___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn self_prod(&self, a: PyReadonlyArray1<f64>) -> f64 {
        self.product.self_prod(&a.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init15437193042514876200() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prod",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::self_prod(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init15437193042514876200___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init15437193042514876200___rust_ctor___ctor() {
        __init15437193042514876200()
    }
    __init15437193042514876200___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init14595267073437518880() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init14595267073437518880___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init14595267073437518880___rust_ctor___ctor() {
        __init14595267073437518880()
    }
    __init14595267073437518880___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn self_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.self_prods(&a.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3703055565605132473() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prods",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::self_prods(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3703055565605132473___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3703055565605132473___rust_ctor___ctor() {
        __init3703055565605132473()
    }
    __init3703055565605132473___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn zip_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.zip_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9450946176484312524() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::zip_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9450946176484312524___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9450946176484312524___rust_ctor___ctor() {
        __init9450946176484312524()
    }
    __init9450946176484312524___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn cross_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.product.cross_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init570509124540142306() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::cross_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init570509124540142306___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init570509124540142306___rust_ctor___ctor() {
        __init570509124540142306()
    }
    __init570509124540142306___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn induced_dist(
        &self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> f64 {
        self.product.induced_dist(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init16641315719792536700() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dist\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dist",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::induced_dist(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init16641315719792536700___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init16641315719792536700___rust_ctor___ctor() {
        __init16641315719792536700()
    }
    __init16641315719792536700___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9209642255308423982() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9209642255308423982___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9209642255308423982___rust_ctor___ctor() {
        __init9209642255308423982()
    }
    __init9209642255308423982___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn zip_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.zip_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init5164261189393069205() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::zip_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init5164261189393069205___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init5164261189393069205___rust_ctor___ctor() {
        __init5164261189393069205()
    }
    __init5164261189393069205___rust_ctor___ctor
};
impl DotProductF64 {
    pub fn cross_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.product.cross_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init4568873416103139006() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductF64::cross_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init4568873416103139006___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init4568873416103139006___rust_ctor___ctor() {
        __init4568873416103139006()
    }
    __init4568873416103139006___rust_ctor___ctor
};
pub struct RBFKernelF32 {
    product: RBFKernel<f32>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for RBFKernelF32 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "RBFKernelF32";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for RBFKernelF32 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a RBFKernelF32 {
    type Target = ::pyo3::PyRef<'a, RBFKernelF32>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut RBFKernelF32 {
    type Target = ::pyo3::PyRefMut<'a, RBFKernelF32>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for RBFKernelF32 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForRBFKernelF32 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory for Pyo3MethodsInventoryForRBFKernelF32 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for RBFKernelF32 {
    type Methods = Pyo3MethodsInventoryForRBFKernelF32;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForRBFKernelF32 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<Pyo3MethodsInventoryForRBFKernelF32> = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for RBFKernelF32 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<RBFKernelF32>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<RBFKernelF32>
for ::pyo3::class::impl_::PyClassImplCollector<RBFKernelF32> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl RBFKernelF32 {
    pub fn new(bandwidth: f32) -> RBFKernelF32 {
        RBFKernelF32 {
            product: RBFKernel::new(bandwidth),
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<RBFKernelF32>
for ::pyo3::class::impl_::PyClassImplCollector<RBFKernelF32> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["bandwidth"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 1usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "bandwidth",
                                    e,
                                ))?
                        };
                        RBFKernelF32::new(arg0)
                    };
                    let initializer: ::pyo3::PyClassInitializer<RBFKernelF32> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init13694003815069983100() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                ::alloc::vec::Vec::new(),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init13694003815069983100___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init13694003815069983100___rust_ctor___ctor() {
        __init13694003815069983100()
    }
    __init13694003815069983100___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn prod(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> f32 {
        self.product.prod(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3433618784732787083() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prod",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::prod(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3433618784732787083___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3433618784732787083___rust_ctor___ctor() {
        __init3433618784732787083()
    }
    __init3433618784732787083___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn self_prod(&self, a: PyReadonlyArray1<f32>) -> f32 {
        self.product.self_prod(&a.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init15789190342913561013() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prod",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::self_prod(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init15789190342913561013___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init15789190342913561013___rust_ctor___ctor() {
        __init15789190342913561013()
    }
    __init15789190342913561013___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9228329466008615351() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9228329466008615351___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9228329466008615351___rust_ctor___ctor() {
        __init9228329466008615351()
    }
    __init9228329466008615351___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn self_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.self_prods(&a.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init13416667645489843968() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prods",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::self_prods(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init13416667645489843968___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init13416667645489843968___rust_ctor___ctor() {
        __init13416667645489843968()
    }
    __init13416667645489843968___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn zip_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.zip_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9309069920721639284() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::zip_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9309069920721639284___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9309069920721639284___rust_ctor___ctor() {
        __init9309069920721639284()
    }
    __init9309069920721639284___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn cross_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        self.product.cross_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init17291101135573445926() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::cross_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init17291101135573445926___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init17291101135573445926___rust_ctor___ctor() {
        __init17291101135573445926()
    }
    __init17291101135573445926___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn induced_dist(
        &self,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray1<f32>,
    ) -> f32 {
        self.product.induced_dist(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init5979669342403551421() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dist\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dist",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::induced_dist(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init5979669342403551421___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init5979669342403551421___rust_ctor___ctor() {
        __init5979669342403551421()
    }
    __init5979669342403551421___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init4157236128068987018() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init4157236128068987018___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init4157236128068987018___rust_ctor___ctor() {
        __init4157236128068987018()
    }
    __init4157236128068987018___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn zip_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.zip_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init14995711678214662976() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::zip_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init14995711678214662976___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init14995711678214662976___rust_ctor___ctor() {
        __init14995711678214662976()
    }
    __init14995711678214662976___rust_ctor___ctor
};
impl RBFKernelF32 {
    pub fn cross_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        self.product.cross_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init10061517839440923601() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF32::cross_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init10061517839440923601___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init10061517839440923601___rust_ctor___ctor() {
        __init10061517839440923601()
    }
    __init10061517839440923601___rust_ctor___ctor
};
pub struct RBFKernelF64 {
    product: RBFKernel<f64>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for RBFKernelF64 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "RBFKernelF64";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for RBFKernelF64 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a RBFKernelF64 {
    type Target = ::pyo3::PyRef<'a, RBFKernelF64>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut RBFKernelF64 {
    type Target = ::pyo3::PyRefMut<'a, RBFKernelF64>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for RBFKernelF64 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForRBFKernelF64 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory for Pyo3MethodsInventoryForRBFKernelF64 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for RBFKernelF64 {
    type Methods = Pyo3MethodsInventoryForRBFKernelF64;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForRBFKernelF64 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<Pyo3MethodsInventoryForRBFKernelF64> = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for RBFKernelF64 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<RBFKernelF64>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<RBFKernelF64>
for ::pyo3::class::impl_::PyClassImplCollector<RBFKernelF64> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl RBFKernelF64 {
    pub fn new(bandwidth: f64) -> RBFKernelF64 {
        RBFKernelF64 {
            product: RBFKernel::new(bandwidth),
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<RBFKernelF64>
for ::pyo3::class::impl_::PyClassImplCollector<RBFKernelF64> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["bandwidth"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 1usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "bandwidth",
                                    e,
                                ))?
                        };
                        RBFKernelF64::new(arg0)
                    };
                    let initializer: ::pyo3::PyClassInitializer<RBFKernelF64> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init13559021504860397903() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                ::alloc::vec::Vec::new(),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init13559021504860397903___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init13559021504860397903___rust_ctor___ctor() {
        __init13559021504860397903()
    }
    __init13559021504860397903___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn prod(&self, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
        self.product.prod(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init15610216110809949049() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prod",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::prod(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init15610216110809949049___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init15610216110809949049___rust_ctor___ctor() {
        __init15610216110809949049()
    }
    __init15610216110809949049___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn self_prod(&self, a: PyReadonlyArray1<f64>) -> f64 {
        self.product.self_prod(&a.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init12170700598808104041() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prod",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::self_prod(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init12170700598808104041___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init12170700598808104041___rust_ctor___ctor() {
        __init12170700598808104041()
    }
    __init12170700598808104041___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3048318890097407124() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3048318890097407124___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3048318890097407124___rust_ctor___ctor() {
        __init3048318890097407124()
    }
    __init3048318890097407124___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn self_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.self_prods(&a.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init16504082169672795413() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prods",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::self_prods(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init16504082169672795413___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init16504082169672795413___rust_ctor___ctor() {
        __init16504082169672795413()
    }
    __init16504082169672795413___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn zip_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.zip_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9450443628014076902() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::zip_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9450443628014076902___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9450443628014076902___rust_ctor___ctor() {
        __init9450443628014076902()
    }
    __init9450443628014076902___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn cross_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.product.cross_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init17253271485615529415() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::cross_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init17253271485615529415___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init17253271485615529415___rust_ctor___ctor() {
        __init17253271485615529415()
    }
    __init17253271485615529415___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn induced_dist(
        &self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> f64 {
        self.product.induced_dist(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9425130340858254708() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dist\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dist",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::induced_dist(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9425130340858254708___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9425130340858254708___rust_ctor___ctor() {
        __init9425130340858254708()
    }
    __init9425130340858254708___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init2022185834166829050() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init2022185834166829050___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init2022185834166829050___rust_ctor___ctor() {
        __init2022185834166829050()
    }
    __init2022185834166829050___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn zip_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.zip_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init16975544442751403012() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::zip_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init16975544442751403012___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init16975544442751403012___rust_ctor___ctor() {
        __init16975544442751403012()
    }
    __init16975544442751403012___rust_ctor___ctor
};
impl RBFKernelF64 {
    pub fn cross_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.product.cross_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init13496548359119470928() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <RBFKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<RBFKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <RBFKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    RBFKernelF64::cross_induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init13496548359119470928___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init13496548359119470928___rust_ctor___ctor() {
        __init13496548359119470928()
    }
    __init13496548359119470928___rust_ctor___ctor
};
pub struct MahalanobisKernelF32 {
    product: MahalanobisKernel<f32>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for MahalanobisKernelF32 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "MahalanobisKernelF32";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for MahalanobisKernelF32 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a MahalanobisKernelF32 {
    type Target = ::pyo3::PyRef<'a, MahalanobisKernelF32>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut MahalanobisKernelF32 {
    type Target = ::pyo3::PyRefMut<'a, MahalanobisKernelF32>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for MahalanobisKernelF32 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForMahalanobisKernelF32 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForMahalanobisKernelF32 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for MahalanobisKernelF32 {
    type Methods = Pyo3MethodsInventoryForMahalanobisKernelF32;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForMahalanobisKernelF32 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForMahalanobisKernelF32,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for MahalanobisKernelF32 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<MahalanobisKernelF32>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<MahalanobisKernelF32>
for ::pyo3::class::impl_::PyClassImplCollector<MahalanobisKernelF32> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl MahalanobisKernelF32 {
    pub fn new(inv_cov: PyReadonlyArray2<f32>) -> MahalanobisKernelF32 {
        MahalanobisKernelF32 {
            product: MahalanobisKernel::new(inv_cov.as_array()),
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<MahalanobisKernelF32>
for ::pyo3::class::impl_::PyClassImplCollector<MahalanobisKernelF32> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["inv_cov"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 1usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "inv_cov",
                                    e,
                                ))?
                        };
                        MahalanobisKernelF32::new(arg0)
                    };
                    let initializer: ::pyo3::PyClassInitializer<MahalanobisKernelF32> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9866570818874512096() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                ::alloc::vec::Vec::new(),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9866570818874512096___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9866570818874512096___rust_ctor___ctor() {
        __init9866570818874512096()
    }
    __init9866570818874512096___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn prod(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> f32 {
        self.product.prod(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init6056985369997674855() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prod",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::prod(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init6056985369997674855___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init6056985369997674855___rust_ctor___ctor() {
        __init6056985369997674855()
    }
    __init6056985369997674855___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn self_prod(&self, a: PyReadonlyArray1<f32>) -> f32 {
        self.product.self_prod(&a.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init4721504208230699135() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prod",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::self_prod(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init4721504208230699135___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init4721504208230699135___rust_ctor___ctor() {
        __init4721504208230699135()
    }
    __init4721504208230699135___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3206374177798119661() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3206374177798119661___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3206374177798119661___rust_ctor___ctor() {
        __init3206374177798119661()
    }
    __init3206374177798119661___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn self_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.self_prods(&a.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init14245585039802128226() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prods",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::self_prods(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init14245585039802128226___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init14245585039802128226___rust_ctor___ctor() {
        __init14245585039802128226()
    }
    __init14245585039802128226___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn zip_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.zip_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init14225504353359240988() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::zip_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init14225504353359240988___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init14225504353359240988___rust_ctor___ctor() {
        __init14225504353359240988()
    }
    __init14225504353359240988___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn cross_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        self.product.cross_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init14811506591485693680() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::cross_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init14811506591485693680___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init14811506591485693680___rust_ctor___ctor() {
        __init14811506591485693680()
    }
    __init14811506591485693680___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn induced_dist(
        &self,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray1<f32>,
    ) -> f32 {
        self.product.induced_dist(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init764582982971157411() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dist\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dist",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::induced_dist(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init764582982971157411___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init764582982971157411___rust_ctor___ctor() {
        __init764582982971157411()
    }
    __init764582982971157411___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init18439400507531151067() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init18439400507531151067___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init18439400507531151067___rust_ctor___ctor() {
        __init18439400507531151067()
    }
    __init18439400507531151067___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn zip_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray1<f32> {
        self.product.zip_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3318783369988492089() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::zip_induced_dists(
                                                        _slf,
                                                        arg0,
                                                        arg1,
                                                        arg2,
                                                    ),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3318783369988492089___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3318783369988492089___rust_ctor___ctor() {
        __init3318783369988492089()
    }
    __init3318783369988492089___rust_ctor___ctor
};
impl MahalanobisKernelF32 {
    pub fn cross_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        self.product.cross_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init945459218856726358() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF32::cross_induced_dists(
                                                        _slf,
                                                        arg0,
                                                        arg1,
                                                        arg2,
                                                    ),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init945459218856726358___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init945459218856726358___rust_ctor___ctor() {
        __init945459218856726358()
    }
    __init945459218856726358___rust_ctor___ctor
};
pub struct MahalanobisKernelF64 {
    product: MahalanobisKernel<f64>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for MahalanobisKernelF64 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "MahalanobisKernelF64";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for MahalanobisKernelF64 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a MahalanobisKernelF64 {
    type Target = ::pyo3::PyRef<'a, MahalanobisKernelF64>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut MahalanobisKernelF64 {
    type Target = ::pyo3::PyRefMut<'a, MahalanobisKernelF64>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for MahalanobisKernelF64 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForMahalanobisKernelF64 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForMahalanobisKernelF64 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for MahalanobisKernelF64 {
    type Methods = Pyo3MethodsInventoryForMahalanobisKernelF64;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForMahalanobisKernelF64 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForMahalanobisKernelF64,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for MahalanobisKernelF64 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<MahalanobisKernelF64>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<MahalanobisKernelF64>
for ::pyo3::class::impl_::PyClassImplCollector<MahalanobisKernelF64> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl MahalanobisKernelF64 {
    pub fn new(inv_cov: PyReadonlyArray2<f64>) -> MahalanobisKernelF64 {
        MahalanobisKernelF64 {
            product: MahalanobisKernel::new(inv_cov.as_array()),
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<MahalanobisKernelF64>
for ::pyo3::class::impl_::PyClassImplCollector<MahalanobisKernelF64> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["inv_cov"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 1usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "inv_cov",
                                    e,
                                ))?
                        };
                        MahalanobisKernelF64::new(arg0)
                    };
                    let initializer: ::pyo3::PyClassInitializer<MahalanobisKernelF64> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init15611533184341321333() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                ::alloc::vec::Vec::new(),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init15611533184341321333___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init15611533184341321333___rust_ctor___ctor() {
        __init15611533184341321333()
    }
    __init15611533184341321333___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn prod(&self, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
        self.product.prod(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init4190270668773730809() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prod",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::prod(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init4190270668773730809___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init4190270668773730809___rust_ctor___ctor() {
        __init4190270668773730809()
    }
    __init4190270668773730809___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn self_prod(&self, a: PyReadonlyArray1<f64>) -> f64 {
        self.product.self_prod(&a.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init17032011872171695840() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prod\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prod",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::self_prod(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init17032011872171695840___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init17032011872171695840___rust_ctor___ctor() {
        __init17032011872171695840()
    }
    __init17032011872171695840___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init211204111134064820() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init211204111134064820___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init211204111134064820___rust_ctor___ctor() {
        __init211204111134064820()
    }
    __init211204111134064820___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn self_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.self_prods(&a.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init11886775042756808987() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "self_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "self_prods",
                                                    positional_parameter_names: &["a"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::self_prods(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init11886775042756808987___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init11886775042756808987___rust_ctor___ctor() {
        __init11886775042756808987()
    }
    __init11886775042756808987___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn zip_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.zip_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3092710939583827038() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::zip_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3092710939583827038___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3092710939583827038___rust_ctor___ctor() {
        __init3092710939583827038()
    }
    __init3092710939583827038___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn cross_prods<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.product.cross_prods(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init6767899596170320077() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_prods\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_prods",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::cross_prods(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init6767899596170320077___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init6767899596170320077___rust_ctor___ctor() {
        __init6767899596170320077()
    }
    __init6767899596170320077___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn induced_dist(
        &self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> f64 {
        self.product.induced_dist(&a.as_array(), &b.as_array())
    }
}
#[allow(non_upper_case_globals)]
extern fn __init9886054645762453643() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dist\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dist",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::induced_dist(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init9886054645762453643___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init9886054645762453643___rust_ctor___ctor() {
        __init9886054645762453643()
    }
    __init9886054645762453643___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init14841794150344396057() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::induced_dists(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init14841794150344396057___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init14841794150344396057___rust_ctor___ctor() {
        __init14841794150344396057()
    }
    __init14841794150344396057___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn zip_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray1<f64> {
        self.product.zip_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init15663950609021909551() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "zip_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "zip_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::zip_induced_dists(
                                                        _slf,
                                                        arg0,
                                                        arg1,
                                                        arg2,
                                                    ),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init15663950609021909551___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init15663950609021909551___rust_ctor___ctor() {
        __init15663950609021909551()
    }
    __init15663950609021909551___rust_ctor___ctor
};
impl MahalanobisKernelF64 {
    pub fn cross_induced_dists<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        self.product.cross_induced_dists(&a.as_array(), &b.as_array()).into_pyarray(py)
    }
}
#[allow(non_upper_case_globals)]
extern fn __init14066172293624276443() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <MahalanobisKernelF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "cross_induced_dists\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<MahalanobisKernelF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <MahalanobisKernelF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "cross_induced_dists",
                                                    positional_parameter_names: &["a", "b"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 2usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "a",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "b",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    MahalanobisKernelF64::cross_induced_dists(
                                                        _slf,
                                                        arg0,
                                                        arg1,
                                                        arg2,
                                                    ),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init14066172293624276443___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init14066172293624276443___rust_ctor___ctor() {
        __init14066172293624276443()
    }
    __init14066172293624276443___rust_ctor___ctor
};
pub struct DotProductBoundsF32 {
    bounds: InnerProductBounder<DotProduct<f32>, f32>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for DotProductBoundsF32 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "DotProductBoundsF32";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for DotProductBoundsF32 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a DotProductBoundsF32 {
    type Target = ::pyo3::PyRef<'a, DotProductBoundsF32>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut DotProductBoundsF32 {
    type Target = ::pyo3::PyRefMut<'a, DotProductBoundsF32>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for DotProductBoundsF32 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForDotProductBoundsF32 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForDotProductBoundsF32 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for DotProductBoundsF32 {
    type Methods = Pyo3MethodsInventoryForDotProductBoundsF32;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForDotProductBoundsF32 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForDotProductBoundsF32,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for DotProductBoundsF32 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<DotProductBoundsF32>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<DotProductBoundsF32>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductBoundsF32> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl DotProductBoundsF32 {
    pub fn new(
        data: PyReadonlyArray2<f32>,
        num_pivots: Option<usize>,
        refs: Option<PyReadonlyArray2<f32>>,
    ) -> DotProductBoundsF32 {
        DotProductBoundsF32 {
            bounds: InnerProductBounder::new(
                DotProduct::new(),
                &data.as_array(),
                num_pivots,
                if refs.is_some() {
                    Some(refs.unwrap().as_array().to_owned())
                } else {
                    None
                },
            ),
        }
    }
    pub fn product_lower_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f32>,
    ) -> &'py PyArray2<f32> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.prod_bounds(&queries_array, true).into_pyarray(py)
    }
    pub fn product_upper_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f32>,
    ) -> &'py PyArray2<f32> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.prod_bounds(&queries_array, false).into_pyarray(py)
    }
    pub fn product_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f32>,
    ) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        let (lower, upper) = self.bounds.prod_bounds_both(&queries_array);
        (lower.into_pyarray(py), upper.into_pyarray(py))
    }
    pub fn distance_lower_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f32>,
    ) -> &'py PyArray2<f32> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.distance_bounds(&queries_array, true).into_pyarray(py)
    }
    pub fn distance_upper_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f32>,
    ) -> &'py PyArray2<f32> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.distance_bounds(&queries_array, false).into_pyarray(py)
    }
    pub fn distance_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f32>,
    ) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        let (lower, upper) = self.bounds.distance_bounds_both(&queries_array);
        (lower.into_pyarray(py), upper.into_pyarray(py))
    }
    pub fn get_pivots<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        self.bounds.reference_points.clone().into_pyarray(py)
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<DotProductBoundsF32>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductBoundsF32> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["data", "num_pivots", "refs"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 3usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "data",
                                    e,
                                ))?
                        };
                        let arg1 = output[1usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "num_pivots",
                                            e,
                                        ))
                                },
                            )?;
                        let arg2 = output[2usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "refs",
                                            e,
                                        ))
                                },
                            )?;
                        DotProductBoundsF32::new(arg0, arg1, arg2)
                    };
                    let initializer: ::pyo3::PyClassInitializer<DotProductBoundsF32> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init11656476619234632484() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductBoundsF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "product_lower_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "product_lower_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF32::product_lower_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "product_upper_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "product_upper_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF32::product_upper_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "product_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "product_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF32::product_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "distance_lower_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "distance_lower_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF32::distance_lower_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "distance_upper_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "distance_upper_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF32::distance_upper_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "distance_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "distance_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF32::distance_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "get_pivots\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "get_pivots",
                                                    positional_parameter_names: &[],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 0usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 0usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF32::get_pivots(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init11656476619234632484___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init11656476619234632484___rust_ctor___ctor() {
        __init11656476619234632484()
    }
    __init11656476619234632484___rust_ctor___ctor
};
pub struct DotProductBoundsF64 {
    bounds: InnerProductBounder<DotProduct<f64>, f64>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for DotProductBoundsF64 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "DotProductBoundsF64";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for DotProductBoundsF64 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a DotProductBoundsF64 {
    type Target = ::pyo3::PyRef<'a, DotProductBoundsF64>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut DotProductBoundsF64 {
    type Target = ::pyo3::PyRefMut<'a, DotProductBoundsF64>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for DotProductBoundsF64 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForDotProductBoundsF64 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForDotProductBoundsF64 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for DotProductBoundsF64 {
    type Methods = Pyo3MethodsInventoryForDotProductBoundsF64;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForDotProductBoundsF64 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForDotProductBoundsF64,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for DotProductBoundsF64 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<DotProductBoundsF64>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<DotProductBoundsF64>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductBoundsF64> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl DotProductBoundsF64 {
    pub fn new(
        data: PyReadonlyArray2<f64>,
        num_pivots: Option<usize>,
        refs: Option<PyReadonlyArray2<f64>>,
    ) -> DotProductBoundsF64 {
        DotProductBoundsF64 {
            bounds: InnerProductBounder::new(
                DotProduct::new(),
                &data.as_array(),
                num_pivots,
                if refs.is_some() {
                    Some(refs.unwrap().as_array().to_owned())
                } else {
                    None
                },
            ),
        }
    }
    pub fn product_lower_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray2<f64> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.prod_bounds(&queries_array, true).into_pyarray(py)
    }
    pub fn product_upper_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray2<f64> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.prod_bounds(&queries_array, false).into_pyarray(py)
    }
    pub fn product_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f64>,
    ) -> (&'py PyArray2<f64>, &'py PyArray2<f64>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        let (lower, upper) = self.bounds.prod_bounds_both(&queries_array);
        (lower.into_pyarray(py), upper.into_pyarray(py))
    }
    pub fn distance_lower_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray2<f64> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.distance_bounds(&queries_array, true).into_pyarray(py)
    }
    pub fn distance_upper_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray2<f64> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        self.bounds.distance_bounds(&queries_array, false).into_pyarray(py)
    }
    pub fn distance_bounds<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArrayDyn<f64>,
    ) -> (&'py PyArray2<f64>, &'py PyArray2<f64>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        let (lower, upper) = self.bounds.distance_bounds_both(&queries_array);
        (lower.into_pyarray(py), upper.into_pyarray(py))
    }
    pub fn get_pivots<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.bounds.reference_points.clone().into_pyarray(py)
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<DotProductBoundsF64>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductBoundsF64> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["data", "num_pivots", "refs"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 3usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "data",
                                    e,
                                ))?
                        };
                        let arg1 = output[1usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "num_pivots",
                                            e,
                                        ))
                                },
                            )?;
                        let arg2 = output[2usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "refs",
                                            e,
                                        ))
                                },
                            )?;
                        DotProductBoundsF64::new(arg0, arg1, arg2)
                    };
                    let initializer: ::pyo3::PyClassInitializer<DotProductBoundsF64> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init3349356313390576027() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductBoundsF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "product_lower_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "product_lower_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF64::product_lower_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "product_upper_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "product_upper_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF64::product_upper_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "product_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "product_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF64::product_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "distance_lower_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "distance_lower_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF64::distance_lower_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "distance_upper_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "distance_upper_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF64::distance_upper_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "distance_bounds\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "distance_bounds",
                                                    positional_parameter_names: &["queries"],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 1usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 1usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                let arg1 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF64::distance_bounds(_slf, arg0, arg1),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "get_pivots\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductBoundsF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductBoundsF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "get_pivots",
                                                    positional_parameter_names: &[],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 0usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 0usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = _py;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductBoundsF64::get_pivots(_slf, arg0),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init3349356313390576027___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init3349356313390576027___rust_ctor___ctor() {
        __init3349356313390576027()
    }
    __init3349356313390576027___rust_ctor___ctor
};
pub struct DotProductPFLSF32 {
    index: PFLS<DotProduct<f32>, f32, OwnedRepr<f32>>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for DotProductPFLSF32 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "DotProductPFLSF32";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for DotProductPFLSF32 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a DotProductPFLSF32 {
    type Target = ::pyo3::PyRef<'a, DotProductPFLSF32>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut DotProductPFLSF32 {
    type Target = ::pyo3::PyRefMut<'a, DotProductPFLSF32>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for DotProductPFLSF32 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForDotProductPFLSF32 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForDotProductPFLSF32 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for DotProductPFLSF32 {
    type Methods = Pyo3MethodsInventoryForDotProductPFLSF32;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForDotProductPFLSF32 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForDotProductPFLSF32,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for DotProductPFLSF32 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<DotProductPFLSF32>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<DotProductPFLSF32>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductPFLSF32> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl DotProductPFLSF32 {
    pub fn new(
        data: PyReadonlyArray2<f32>,
        num_pivots: Option<usize>,
        refs: Option<PyReadonlyArray2<f32>>,
    ) -> DotProductPFLSF32 {
        DotProductPFLSF32 {
            index: PFLS::new(
                DotProduct::new(),
                data.as_array().into_owned(),
                num_pivots,
                if refs.is_some() {
                    Some(refs.unwrap().as_array().into_owned())
                } else {
                    None
                },
            ),
        }
    }
    pub fn query(
        &self,
        queries: PyReadonlyArrayDyn<f32>,
        k: usize,
        sorting: Option<bool>,
        smallests: Option<bool>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if sorting.unwrap_or(true) {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_product_sorting(&queries_array, k)
            } else {
                self.index.query_k_largest_product_sorting(&queries_array, k)
            }
        } else {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_product_direct(&queries_array, k)
            } else {
                self.index.query_k_largest_product_direct(&queries_array, k)
            }
        }
    }
    pub fn query_ball_point(
        &self,
        queries: PyReadonlyArrayDyn<f32>,
        threshold: f32,
        smallests: Option<bool>,
    ) -> Vec<Vec<usize>> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if smallests.unwrap_or(true) {
            self.index.query_product_below(&queries_array, threshold)
        } else {
            self.index.query_product_above(&queries_array, threshold)
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<DotProductPFLSF32>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductPFLSF32> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <DotProductPFLSF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["data", "num_pivots", "refs"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 3usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "data",
                                    e,
                                ))?
                        };
                        let arg1 = output[1usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "num_pivots",
                                            e,
                                        ))
                                },
                            )?;
                        let arg2 = output[2usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "refs",
                                            e,
                                        ))
                                },
                            )?;
                        DotProductPFLSF32::new(arg0, arg1, arg2)
                    };
                    let initializer: ::pyo3::PyClassInitializer<DotProductPFLSF32> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init5382419465995122353() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductPFLSF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductPFLSF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductPFLSF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "k",
                                                        "sorting",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 4usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "k",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "sorting",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                let arg3 = output[3usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductPFLSF32::query(_slf, arg0, arg1, arg2, arg3),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query_ball_point\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductPFLSF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductPFLSF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query_ball_point",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "threshold",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 3usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "threshold",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductPFLSF32::query_ball_point(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init5382419465995122353___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init5382419465995122353___rust_ctor___ctor() {
        __init5382419465995122353()
    }
    __init5382419465995122353___rust_ctor___ctor
};
pub struct DotProductPFLSF64 {
    index: PFLS<DotProduct<f64>, f64, OwnedRepr<f64>>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for DotProductPFLSF64 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "DotProductPFLSF64";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for DotProductPFLSF64 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a DotProductPFLSF64 {
    type Target = ::pyo3::PyRef<'a, DotProductPFLSF64>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut DotProductPFLSF64 {
    type Target = ::pyo3::PyRefMut<'a, DotProductPFLSF64>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for DotProductPFLSF64 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForDotProductPFLSF64 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForDotProductPFLSF64 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for DotProductPFLSF64 {
    type Methods = Pyo3MethodsInventoryForDotProductPFLSF64;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForDotProductPFLSF64 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForDotProductPFLSF64,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for DotProductPFLSF64 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<DotProductPFLSF64>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<DotProductPFLSF64>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductPFLSF64> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl DotProductPFLSF64 {
    pub fn new(
        data: PyReadonlyArray2<f64>,
        num_pivots: Option<usize>,
        refs: Option<PyReadonlyArray2<f64>>,
    ) -> DotProductPFLSF64 {
        DotProductPFLSF64 {
            index: PFLS::new(
                DotProduct::new(),
                data.as_array().into_owned(),
                num_pivots,
                if refs.is_some() {
                    Some(refs.unwrap().as_array().into_owned())
                } else {
                    None
                },
            ),
        }
    }
    pub fn query(
        &self,
        queries: PyReadonlyArrayDyn<f64>,
        k: usize,
        sorting: Option<bool>,
        smallests: Option<bool>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if sorting.unwrap_or(true) {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_product_sorting(&queries_array, k)
            } else {
                self.index.query_k_largest_product_sorting(&queries_array, k)
            }
        } else {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_product_direct(&queries_array, k)
            } else {
                self.index.query_k_largest_product_direct(&queries_array, k)
            }
        }
    }
    pub fn query_ball_point(
        &self,
        queries: PyReadonlyArrayDyn<f64>,
        threshold: f64,
        smallests: Option<bool>,
    ) -> Vec<Vec<usize>> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if smallests.unwrap_or(true) {
            self.index.query_product_below(&queries_array, threshold)
        } else {
            self.index.query_product_above(&queries_array, threshold)
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<DotProductPFLSF64>
for ::pyo3::class::impl_::PyClassImplCollector<DotProductPFLSF64> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <DotProductPFLSF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["data", "num_pivots", "refs"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 3usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "data",
                                    e,
                                ))?
                        };
                        let arg1 = output[1usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "num_pivots",
                                            e,
                                        ))
                                },
                            )?;
                        let arg2 = output[2usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "refs",
                                            e,
                                        ))
                                },
                            )?;
                        DotProductPFLSF64::new(arg0, arg1, arg2)
                    };
                    let initializer: ::pyo3::PyClassInitializer<DotProductPFLSF64> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init4945458787730927254() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <DotProductPFLSF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductPFLSF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductPFLSF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "k",
                                                        "sorting",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 4usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "k",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "sorting",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                let arg3 = output[3usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductPFLSF64::query(_slf, arg0, arg1, arg2, arg3),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query_ball_point\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<DotProductPFLSF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <DotProductPFLSF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query_ball_point",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "threshold",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 3usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "threshold",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    DotProductPFLSF64::query_ball_point(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init4945458787730927254___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init4945458787730927254___rust_ctor___ctor() {
        __init4945458787730927254()
    }
    __init4945458787730927254___rust_ctor___ctor
};
pub struct EucDistancePFLSF32 {
    index: PFLS<DotProduct<f32>, f32, OwnedRepr<f32>>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for EucDistancePFLSF32 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "EucDistancePFLSF32";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for EucDistancePFLSF32 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a EucDistancePFLSF32 {
    type Target = ::pyo3::PyRef<'a, EucDistancePFLSF32>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut EucDistancePFLSF32 {
    type Target = ::pyo3::PyRefMut<'a, EucDistancePFLSF32>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for EucDistancePFLSF32 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForEucDistancePFLSF32 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForEucDistancePFLSF32 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for EucDistancePFLSF32 {
    type Methods = Pyo3MethodsInventoryForEucDistancePFLSF32;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForEucDistancePFLSF32 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForEucDistancePFLSF32,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for EucDistancePFLSF32 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<EucDistancePFLSF32>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<EucDistancePFLSF32>
for ::pyo3::class::impl_::PyClassImplCollector<EucDistancePFLSF32> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl EucDistancePFLSF32 {
    pub fn new(
        data: PyReadonlyArray2<f32>,
        num_pivots: Option<usize>,
        refs: Option<PyReadonlyArray2<f32>>,
    ) -> EucDistancePFLSF32 {
        EucDistancePFLSF32 {
            index: PFLS::new(
                DotProduct::new(),
                data.as_array().into_owned(),
                num_pivots,
                if refs.is_some() {
                    Some(refs.unwrap().as_array().into_owned())
                } else {
                    None
                },
            ),
        }
    }
    pub fn query(
        &self,
        queries: PyReadonlyArrayDyn<f32>,
        k: usize,
        sorting: Option<bool>,
        smallests: Option<bool>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if sorting.unwrap_or(true) {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_distance_sorting(&queries_array, k)
            } else {
                self.index.query_k_largest_distance_sorting(&queries_array, k)
            }
        } else {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_distance_direct(&queries_array, k)
            } else {
                self.index.query_k_largest_distance_direct(&queries_array, k)
            }
        }
    }
    pub fn query_ball_point(
        &self,
        queries: PyReadonlyArrayDyn<f32>,
        threshold: f32,
        smallests: Option<bool>,
    ) -> Vec<Vec<usize>> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if smallests.unwrap_or(true) {
            self.index.query_distance_below(&queries_array, threshold)
        } else {
            self.index.query_distance_above(&queries_array, threshold)
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<EucDistancePFLSF32>
for ::pyo3::class::impl_::PyClassImplCollector<EucDistancePFLSF32> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <EucDistancePFLSF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["data", "num_pivots", "refs"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 3usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "data",
                                    e,
                                ))?
                        };
                        let arg1 = output[1usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "num_pivots",
                                            e,
                                        ))
                                },
                            )?;
                        let arg2 = output[2usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "refs",
                                            e,
                                        ))
                                },
                            )?;
                        EucDistancePFLSF32::new(arg0, arg1, arg2)
                    };
                    let initializer: ::pyo3::PyClassInitializer<EucDistancePFLSF32> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init6812675821524331921() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <EucDistancePFLSF32 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<EucDistancePFLSF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <EucDistancePFLSF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "k",
                                                        "sorting",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 4usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "k",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "sorting",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                let arg3 = output[3usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    EucDistancePFLSF32::query(_slf, arg0, arg1, arg2, arg3),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query_ball_point\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<EucDistancePFLSF32>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <EucDistancePFLSF32 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query_ball_point",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "threshold",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 3usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "threshold",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    EucDistancePFLSF32::query_ball_point(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init6812675821524331921___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init6812675821524331921___rust_ctor___ctor() {
        __init6812675821524331921()
    }
    __init6812675821524331921___rust_ctor___ctor
};
pub struct EucDistancePFLSF64 {
    index: PFLS<DotProduct<f64>, f64, OwnedRepr<f64>>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for EucDistancePFLSF64 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "EucDistancePFLSF64";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for EucDistancePFLSF64 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a EucDistancePFLSF64 {
    type Target = ::pyo3::PyRef<'a, EucDistancePFLSF64>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut EucDistancePFLSF64 {
    type Target = ::pyo3::PyRefMut<'a, EucDistancePFLSF64>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for EucDistancePFLSF64 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForEucDistancePFLSF64 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForEucDistancePFLSF64 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for EucDistancePFLSF64 {
    type Methods = Pyo3MethodsInventoryForEucDistancePFLSF64;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForEucDistancePFLSF64 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForEucDistancePFLSF64,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for EucDistancePFLSF64 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<EucDistancePFLSF64>;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<EucDistancePFLSF64>
for ::pyo3::class::impl_::PyClassImplCollector<EucDistancePFLSF64> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
impl EucDistancePFLSF64 {
    pub fn new(
        data: PyReadonlyArray2<f64>,
        num_pivots: Option<usize>,
        refs: Option<PyReadonlyArray2<f64>>,
    ) -> EucDistancePFLSF64 {
        EucDistancePFLSF64 {
            index: PFLS::new(
                DotProduct::new(),
                data.as_array().into_owned(),
                num_pivots,
                if refs.is_some() {
                    Some(refs.unwrap().as_array().into_owned())
                } else {
                    None
                },
            ),
        }
    }
    pub fn query(
        &self,
        queries: PyReadonlyArrayDyn<f64>,
        k: usize,
        sorting: Option<bool>,
        smallests: Option<bool>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if sorting.unwrap_or(true) {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_distance_sorting(&queries_array, k)
            } else {
                self.index.query_k_largest_distance_sorting(&queries_array, k)
            }
        } else {
            if smallests.unwrap_or(true) {
                self.index.query_k_smallest_distance_direct(&queries_array, k)
            } else {
                self.index.query_k_largest_distance_direct(&queries_array, k)
            }
        }
    }
    pub fn query_ball_point(
        &self,
        queries: PyReadonlyArrayDyn<f64>,
        threshold: f64,
        smallests: Option<bool>,
    ) -> Vec<Vec<usize>> {
        let queries_array = if queries.shape().len() == 1 {
            queries.as_array().into_shape((1, queries.shape()[0])).unwrap()
        } else {
            queries
                .as_array()
                .into_shape((queries.shape()[0], queries.shape()[1]))
                .unwrap()
        };
        if smallests.unwrap_or(true) {
            self.index.query_distance_below(&queries_array, threshold)
        } else {
            self.index.query_distance_above(&queries_array, threshold)
        }
    }
}
impl ::pyo3::class::impl_::PyClassNewImpl<EucDistancePFLSF64>
for ::pyo3::class::impl_::PyClassImplCollector<EucDistancePFLSF64> {
    fn new_impl(self) -> ::std::option::Option<::pyo3::ffi::newfunc> {
        ::std::option::Option::Some({
            unsafe extern "C" fn __wrap(
                subtype: *mut ::pyo3::ffi::PyTypeObject,
                _args: *mut ::pyo3::ffi::PyObject,
                _kwargs: *mut ::pyo3::ffi::PyObject,
            ) -> *mut ::pyo3::ffi::PyObject {
                use ::pyo3::callback::IntoPyCallbackOutput;
                ::pyo3::callback::handle_panic(|_py| {
                    let _args = _py.from_borrowed_ptr::<::pyo3::types::PyTuple>(_args);
                    let _kwargs: ::std::option::Option<&::pyo3::types::PyDict> = _py
                        .from_borrowed_ptr_or_opt(_kwargs);
                    let result = {
                        const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                            cls_name: ::std::option::Option::Some(
                                <EucDistancePFLSF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                            ),
                            func_name: "__new__",
                            positional_parameter_names: &["data", "num_pivots", "refs"],
                            positional_only_parameters: 0usize,
                            required_positional_parameters: 1usize,
                            keyword_only_parameters: &[],
                            accept_varargs: false,
                            accept_varkeywords: false,
                        };
                        let mut output = [::std::option::Option::None; 3usize];
                        let (_args, _kwargs) = DESCRIPTION
                            .extract_arguments(
                                _py,
                                _args.iter(),
                                _kwargs.map(|dict| dict.iter()),
                                &mut output,
                            )?;
                        let arg0 = {
                            let _obj = output[0usize]
                                .expect("Failed to extract required method argument");
                            _obj.extract()
                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                    _py,
                                    "data",
                                    e,
                                ))?
                        };
                        let arg1 = output[1usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "num_pivots",
                                            e,
                                        ))
                                },
                            )?;
                        let arg2 = output[2usize]
                            .map_or(
                                ::std::result::Result::Ok(::std::option::Option::None),
                                |_obj| {
                                    _obj
                                        .extract()
                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                            _py,
                                            "refs",
                                            e,
                                        ))
                                },
                            )?;
                        EucDistancePFLSF64::new(arg0, arg1, arg2)
                    };
                    let initializer: ::pyo3::PyClassInitializer<EucDistancePFLSF64> = result
                        .convert(_py)?;
                    let cell = initializer.create_cell_from_subtype(_py, subtype)?;
                    ::std::result::Result::Ok(cell as *mut ::pyo3::ffi::PyObject)
                })
            }
            __wrap
        })
    }
}
#[allow(non_upper_case_globals)]
extern fn __init1211763545877895310() {
    ::pyo3::inventory::submit({
        {
            type Inventory = <EucDistancePFLSF64 as ::pyo3::class::impl_::HasMethodsInventory>::Methods;
            <Inventory as ::pyo3::class::impl_::PyMethodsInventory>::new(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<EucDistancePFLSF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <EucDistancePFLSF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "k",
                                                        "sorting",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 4usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "k",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "sorting",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                let arg3 = output[3usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    EucDistancePFLSF64::query(_slf, arg0, arg1, arg2, arg3),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                        ::pyo3::class::PyMethodDefType::Method(
                            ::pyo3::class::methods::PyMethodDef::fastcall_cfunction_with_keywords(
                                "query_ball_point\0",
                                ::pyo3::class::methods::PyCFunctionFastWithKeywords({
                                    unsafe extern "C" fn __wrap(
                                        _slf: *mut ::pyo3::ffi::PyObject,
                                        _args: *const *mut ::pyo3::ffi::PyObject,
                                        _nargs: ::pyo3::ffi::Py_ssize_t,
                                        _kwnames: *mut ::pyo3::ffi::PyObject,
                                    ) -> *mut ::pyo3::ffi::PyObject {
                                        ::pyo3::callback::handle_panic(|_py| {
                                            let _cell = _py
                                                .from_borrowed_ptr::<::pyo3::PyAny>(_slf)
                                                .downcast::<::pyo3::PyCell<EucDistancePFLSF64>>()?;
                                            let _ref = _cell.try_borrow()?;
                                            let _slf = &_ref;
                                            let _kwnames: ::std::option::Option<
                                                &::pyo3::types::PyTuple,
                                            > = _py.from_borrowed_ptr_or_opt(_kwnames);
                                            let _args = _args as *const &::pyo3::PyAny;
                                            let _kwargs = if let ::std::option::Option::Some(kwnames)
                                                = _kwnames {
                                                ::std::slice::from_raw_parts(
                                                    _args.offset(_nargs),
                                                    kwnames.len(),
                                                )
                                            } else {
                                                &[]
                                            };
                                            let _args = ::std::slice::from_raw_parts(
                                                _args,
                                                _nargs as usize,
                                            );
                                            {
                                                const DESCRIPTION: ::pyo3::derive_utils::FunctionDescription = ::pyo3::derive_utils::FunctionDescription {
                                                    cls_name: ::std::option::Option::Some(
                                                        <EucDistancePFLSF64 as ::pyo3::type_object::PyTypeInfo>::NAME,
                                                    ),
                                                    func_name: "query_ball_point",
                                                    positional_parameter_names: &[
                                                        "queries",
                                                        "threshold",
                                                        "smallests",
                                                    ],
                                                    positional_only_parameters: 0usize,
                                                    required_positional_parameters: 2usize,
                                                    keyword_only_parameters: &[],
                                                    accept_varargs: false,
                                                    accept_varkeywords: false,
                                                };
                                                let mut output = [::std::option::Option::None; 3usize];
                                                let (_args, _kwargs) = DESCRIPTION
                                                    .extract_arguments(
                                                        _py,
                                                        ::std::iter::Iterator::copied(_args.iter()),
                                                        _kwnames
                                                            .map(|kwnames| {
                                                                use ::std::iter::Iterator;
                                                                kwnames
                                                                    .as_slice()
                                                                    .iter()
                                                                    .copied()
                                                                    .zip(_kwargs.iter().copied())
                                                            }),
                                                        &mut output,
                                                    )?;
                                                let arg0 = {
                                                    let _obj = output[0usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "queries",
                                                            e,
                                                        ))?
                                                };
                                                let arg1 = {
                                                    let _obj = output[1usize]
                                                        .expect("Failed to extract required method argument");
                                                    _obj.extract()
                                                        .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                            _py,
                                                            "threshold",
                                                            e,
                                                        ))?
                                                };
                                                let arg2 = output[2usize]
                                                    .map_or(
                                                        ::std::result::Result::Ok(::std::option::Option::None),
                                                        |_obj| {
                                                            _obj
                                                                .extract()
                                                                .map_err(|e| ::pyo3::derive_utils::argument_extraction_error(
                                                                    _py,
                                                                    "smallests",
                                                                    e,
                                                                ))
                                                        },
                                                    )?;
                                                ::pyo3::callback::convert(
                                                    _py,
                                                    EucDistancePFLSF64::query_ball_point(_slf, arg0, arg1, arg2),
                                                )
                                            }
                                        })
                                    }
                                    __wrap
                                }),
                                "\u{0}",
                            ),
                        ),
                    ]),
                ),
                ::alloc::vec::Vec::new(),
            )
        }
    });
}
#[used]
#[allow(non_upper_case_globals)]
#[doc(hidden)]
#[link_section = ".init_array"]
static __init1211763545877895310___rust_ctor___ctor: unsafe extern "C" fn() = {
    #[link_section = ".text.startup"]
    unsafe extern "C" fn __init1211763545877895310___rust_ctor___ctor() {
        __init1211763545877895310()
    }
    __init1211763545877895310___rust_ctor___ctor
};
pub struct MahalanobisDistancePFLSF32 {
    index: PFLS<MahalanobisKernel<f32>, f32, OwnedRepr<f32>>,
}
unsafe impl ::pyo3::type_object::PyTypeInfo for MahalanobisDistancePFLSF32 {
    type AsRefTarget = ::pyo3::PyCell<Self>;
    const NAME: &'static str = "MahalanobisDistancePFLSF32";
    const MODULE: ::std::option::Option<&'static str> = ::std::option::Option::None;
    #[inline]
    fn type_object_raw(py: ::pyo3::Python<'_>) -> *mut ::pyo3::ffi::PyTypeObject {
        use ::pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl ::pyo3::PyClass for MahalanobisDistancePFLSF32 {
    type Dict = ::pyo3::pyclass_slots::PyClassDummySlot;
    type WeakRef = ::pyo3::pyclass_slots::PyClassDummySlot;
    type BaseNativeType = ::pyo3::PyAny;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a MahalanobisDistancePFLSF32 {
    type Target = ::pyo3::PyRef<'a, MahalanobisDistancePFLSF32>;
}
impl<'a> ::pyo3::derive_utils::ExtractExt<'a> for &'a mut MahalanobisDistancePFLSF32 {
    type Target = ::pyo3::PyRefMut<'a, MahalanobisDistancePFLSF32>;
}
impl ::pyo3::IntoPy<::pyo3::PyObject> for MahalanobisDistancePFLSF32 {
    fn into_py(self, py: ::pyo3::Python) -> ::pyo3::PyObject {
        ::pyo3::IntoPy::into_py(::pyo3::Py::new(py, self).unwrap(), py)
    }
}
#[doc(hidden)]
pub struct Pyo3MethodsInventoryForMahalanobisDistancePFLSF32 {
    methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
    slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
}
impl ::pyo3::class::impl_::PyMethodsInventory
for Pyo3MethodsInventoryForMahalanobisDistancePFLSF32 {
    fn new(
        methods: ::std::vec::Vec<::pyo3::class::PyMethodDefType>,
        slots: ::std::vec::Vec<::pyo3::ffi::PyType_Slot>,
    ) -> Self {
        Self { methods, slots }
    }
    fn methods(&'static self) -> &'static [::pyo3::class::PyMethodDefType] {
        &self.methods
    }
    fn slots(&'static self) -> &'static [::pyo3::ffi::PyType_Slot] {
        &self.slots
    }
}
impl ::pyo3::class::impl_::HasMethodsInventory for MahalanobisDistancePFLSF32 {
    type Methods = Pyo3MethodsInventoryForMahalanobisDistancePFLSF32;
}
impl ::inventory::Collect for Pyo3MethodsInventoryForMahalanobisDistancePFLSF32 {
    #[inline]
    fn registry() -> &'static ::inventory::Registry<Self> {
        static REGISTRY: ::inventory::Registry<
            Pyo3MethodsInventoryForMahalanobisDistancePFLSF32,
        > = ::inventory::Registry::new();
        &REGISTRY
    }
}
impl ::pyo3::class::impl_::PyClassImpl for MahalanobisDistancePFLSF32 {
    const DOC: &'static str = "\u{0}";
    const IS_GC: bool = false;
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    type Layout = ::pyo3::PyCell<Self>;
    type BaseType = ::pyo3::PyAny;
    type ThreadChecker = ::pyo3::class::impl_::ThreadCheckerStub<
        MahalanobisDistancePFLSF32,
    >;
    fn for_each_method_def(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::class::PyMethodDefType]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::methods(inventory));
        }
        visitor(collector.py_class_descriptors());
        visitor(collector.object_protocol_methods());
        visitor(collector.async_protocol_methods());
        visitor(collector.context_protocol_methods());
        visitor(collector.descr_protocol_methods());
        visitor(collector.mapping_protocol_methods());
        visitor(collector.number_protocol_methods());
    }
    fn get_new() -> ::std::option::Option<::pyo3::ffi::newfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.new_impl()
    }
    fn get_alloc() -> ::std::option::Option<::pyo3::ffi::allocfunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.alloc_impl()
    }
    fn get_free() -> ::std::option::Option<::pyo3::ffi::freefunc> {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.free_impl()
    }
    fn for_each_proto_slot(
        visitor: &mut dyn ::std::ops::FnMut(&[::pyo3::ffi::PyType_Slot]),
    ) {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        visitor(collector.object_protocol_slots());
        visitor(collector.number_protocol_slots());
        visitor(collector.iter_protocol_slots());
        visitor(collector.gc_protocol_slots());
        visitor(collector.descr_protocol_slots());
        visitor(collector.mapping_protocol_slots());
        visitor(collector.sequence_protocol_slots());
        visitor(collector.async_protocol_slots());
        visitor(collector.buffer_protocol_slots());
        for inventory in ::pyo3::inventory::iter::<
            <Self as ::pyo3::class::impl_::HasMethodsInventory>::Methods,
        >() {
            visitor(::pyo3::class::impl_::PyMethodsInventory::slots(inventory));
        }
    }
    fn get_buffer() -> ::std::option::Option<
        &'static ::pyo3::class::impl_::PyBufferProcs,
    > {
        use ::pyo3::class::impl_::*;
        let collector = PyClassImplCollector::<Self>::new();
        collector.buffer_procs()
    }
}
impl ::pyo3::class::impl_::PyClassDescriptors<MahalanobisDistancePFLSF32>
for ::pyo3::class::impl_::PyClassImplCollector<MahalanobisDistancePFLSF32> {
    fn py_class_descriptors(self) -> &'static [::pyo3::class::methods::PyMethodDefType] {
        static METHODS: &[::pyo3::class::methods::PyMethodDefType] = &[];
        METHODS
    }
}
fn pfls(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DotProductF32>()?;
    m.add_class::<DotProductF64>()?;
    m.add_class::<RBFKernelF32>()?;
    m.add_class::<RBFKernelF64>()?;
    m.add_class::<MahalanobisKernelF32>()?;
    m.add_class::<MahalanobisKernelF64>()?;
    m.add_class::<DotProductBoundsF32>()?;
    m.add_class::<DotProductBoundsF64>()?;
    m.add_class::<DotProductPFLSF32>()?;
    m.add_class::<DotProductPFLSF64>()?;
    m.add_class::<EucDistancePFLSF32>()?;
    m.add_class::<EucDistancePFLSF64>()?;
    Ok(())
}
#[no_mangle]
#[allow(non_snake_case)]
/// This autogenerated function is called by the python interpreter when importing
/// the module.
pub unsafe extern "C" fn PyInit_pfls() -> *mut ::pyo3::ffi::PyObject {
    use ::pyo3::derive_utils::ModuleDef;
    static NAME: &str = "pfls\u{0}";
    static DOC: &str = "\u{0}";
    static MODULE_DEF: ModuleDef = unsafe { ModuleDef::new(NAME, DOC) };
    ::pyo3::callback::handle_panic(|_py| { MODULE_DEF.make_module(_py, pfls) })
}
