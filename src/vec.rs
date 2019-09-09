use std::arch::x86_64 as arch;
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::{DefaultZero, SimdArrayOps};

/// A heap allocated SIMD aligned vector.
pub struct SimdVec<A> {
    phantom: PhantomData<A>,
    size: usize,
    vec: Vec<arch::__m256i>,
}

impl<A> SimdVec<A>
where
    A: Copy + DefaultZero,
{
    /// Construct an empty vector.
    pub fn new() -> Self {
        SimdVec {
            phantom: PhantomData,
            size: 0,
            vec: Vec::new(),
        }
    }

    /// Construct an empty vector with a given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        SimdVec {
            phantom: PhantomData,
            size: 0,
            vec: Vec::with_capacity((capacity + (Self::block_size() - 1)) / Self::block_size()),
        }
    }

    fn max_size(&self) -> usize {
        self.vec.len() * Self::block_size()
    }

    fn block_size() -> usize {
        std::mem::size_of::<arch::__m256i>() / std::mem::size_of::<A>()
    }

    fn at_capacity(&self) -> bool {
        self.size == self.max_size()
    }

    fn add_block(&mut self) {
        self.vec.push(arch::__m256i::default_zero())
    }

    fn trim_excess(&mut self) {
        while self.len() + Self::block_size() <= self.max_size() {
            self.vec.pop();
        }
    }

    /// Push a value to the end of the vector.
    pub fn push(&mut self, value: A) {
        if self.at_capacity() {
            self.add_block()
        };
        let index = self.size;
        self.size += 1;
        self[index] = value;
    }

    /// Pop a value off the end of the vector.
    ///
    /// Returns `None` if the vector was empty.
    pub fn pop(&mut self) -> Option<A> {
        if self.is_empty() {
            return None;
        }
        let result = self[self.size - 1];
        self.size -= 1;
        self.trim_excess();
        Some(result)
    }
}

impl<A> SimdArrayOps<A> for SimdVec<A>
where
    A: Ord,
{
    /// Get the current number of elements in the vector.
    fn len(&self) -> usize {
        self.size
    }

    /// Test if the vector is currently empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn data_m256(&self) -> &[arch::__m256i] {
        self.vec.as_slice()
    }
}

impl<A> Default for SimdVec<A>
where
    A: Copy + DefaultZero,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A> Deref for SimdVec<A> {
    type Target = [A];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.vec.as_ptr() as *const _, self.size) }
    }
}

impl<A> DerefMut for SimdVec<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.vec.as_mut_ptr() as *mut _, self.size) }
    }
}

impl<A> Extend<A> for SimdVec<A>
where
    A: Copy + DefaultZero,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = A>,
    {
        for item in iter {
            self.push(item);
        }
    }
}

impl<'a, A> From<&'a [A]> for SimdVec<A>
where
    A: Copy + DefaultZero,
{
    fn from(slice: &'a [A]) -> Self {
        let mut out = Self::with_capacity(slice.len());
        out.extend(slice.iter().copied());
        out
    }
}

impl<A> Debug for SimdVec<A>
where
    A: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        self.deref().fmt(f)
    }
}

impl<A> Hash for SimdVec<A>
where
    A: Hash,
{
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        self.deref().hash(hasher)
    }
}

impl<A> PartialEq for SimdVec<A>
where
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}

impl<A> Eq for SimdVec<A> where A: Eq {}

impl<A> PartialOrd for SimdVec<A>
where
    A: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.deref().partial_cmp(other.deref())
    }
}

impl<A> Ord for SimdVec<A>
where
    A: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.deref().cmp(other.deref())
    }
}
