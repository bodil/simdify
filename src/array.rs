use std::arch::x86_64 as arch;
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use generic_array::{ArrayLength, GenericArray};

use crate::{DefaultZero, SimdArrayOps};

/// A fixed capacity stack allocated SIMD aligned vector.
///
/// The capacity `N` denotes the number of 32-byte chunks allocated, which means
/// the maximum capacity of the vector will be `N * (32 / size_of::<A>())`
/// elements. You can use `SimdArray::max_size()` to get the capacity.
pub struct SimdArray<A, N>
where
    N: ArrayLength<arch::__m256i>,
{
    phantom: PhantomData<(A, N)>,
    size: usize,
    data: GenericArray<arch::__m256i, N>,
}

impl<A, N> SimdArray<A, N>
where
    N: ArrayLength<arch::__m256i>,
    A: Copy + DefaultZero,
{
    /// Construct an empty vector.
    pub fn new() -> Self {
        SimdArray {
            size: 0,
            phantom: PhantomData,
            data: GenericArray::default_zero(),
        }
    }

    /// Push a value to the end of the vector.
    ///
    /// Returns `false` if the vector was at capacity.
    pub fn push(&mut self, value: A) -> bool {
        if self.at_capacity() {
            false
        } else {
            let index = self.size;
            self.size += 1;
            self[index] = value;
            true
        }
    }

    /// Pop a value off the end of the vector.
    ///
    /// Returns `None` if the vector was empty.
    pub fn pop(&mut self) -> Option<A> {
        if self.is_empty() {
            None
        } else {
            let result = self[self.size - 1];
            self.size -= 1;
            Some(result)
        }
    }
}

impl<A, N> Default for SimdArray<A, N>
where
    N: ArrayLength<arch::__m256i>,
    A: Copy + DefaultZero,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, N> SimdArray<A, N>
where
    N: ArrayLength<arch::__m256i>,
{
    /// Get the maximum capacity of the array.
    pub fn max_size() -> usize {
        std::mem::size_of::<GenericArray<arch::__m256i, N>>() / std::mem::size_of::<A>()
    }

    fn at_capacity(&self) -> bool {
        self.size == Self::max_size()
    }
}

impl<A, N> SimdArrayOps<A> for SimdArray<A, N>
where
    A: Ord,
    N: ArrayLength<arch::__m256i>,
{
    /// Get the current number of elements in the array.
    fn len(&self) -> usize {
        self.size
    }

    /// Test if the array is currently empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn data_m256(&self) -> &[arch::__m256i] {
        &self.data
    }
}

impl<A, N> Deref for SimdArray<A, N>
where
    N: ArrayLength<arch::__m256i>,
{
    type Target = [A];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const _, self.size) }
    }
}

impl<A, N> DerefMut for SimdArray<A, N>
where
    N: ArrayLength<arch::__m256i>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut _, self.size) }
    }
}

impl<A, N> Extend<A> for SimdArray<A, N>
where
    A: Copy + DefaultZero,
    N: ArrayLength<arch::__m256i>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = A>,
    {
        for item in iter {
            if !self.push(item) {
                panic!(
                    "SimdArray::extend: exceeded array capacity of {}",
                    Self::max_size()
                )
            }
        }
    }
}

impl<'a, A, N> From<&'a [A]> for SimdArray<A, N>
where
    N: ArrayLength<arch::__m256i>,
    A: Copy + DefaultZero,
{
    fn from(slice: &'a [A]) -> Self {
        if slice.len() > Self::max_size() {
            panic!(
                "SimdArray::from: slice has length {} but array capacity is {}",
                slice.len(),
                Self::max_size()
            )
        }
        let mut out = Self::new();
        let source = &slice[..std::cmp::min(Self::max_size(), slice.len())];
        out.size = source.len();
        out.deref_mut().copy_from_slice(source);
        out
    }
}

impl<A, N> Debug for SimdArray<A, N>
where
    A: Debug,
    N: ArrayLength<arch::__m256i>,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        self.deref().fmt(f)
    }
}

impl<A, N> Hash for SimdArray<A, N>
where
    A: Hash,
    N: ArrayLength<arch::__m256i>,
{
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        self.deref().hash(hasher)
    }
}

impl<A, N> PartialEq for SimdArray<A, N>
where
    A: PartialEq,
    N: ArrayLength<arch::__m256i>,
{
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}

impl<A, N> Eq for SimdArray<A, N>
where
    A: Eq,
    N: ArrayLength<arch::__m256i>,
{
}

impl<A, N> PartialOrd for SimdArray<A, N>
where
    A: PartialOrd,
    N: ArrayLength<arch::__m256i>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.deref().partial_cmp(other.deref())
    }
}

impl<A, N> Ord for SimdArray<A, N>
where
    A: Ord,
    N: ArrayLength<arch::__m256i>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.deref().cmp(other.deref())
    }
}
