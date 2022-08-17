use num::Float;
use std::cmp::{Ord,PartialOrd,Eq,PartialEq,Ordering};

/* Small data storage to keep track of data indices when
 * sorting by smallest/largest distances. */
#[derive(Debug, Copy, Clone)]
pub struct MeasurePair<N> {
	pub index: usize,
	pub value: N,
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

