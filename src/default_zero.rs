use std::arch::x86_64::{__m128i, __m256i};

use generic_array::{ArrayLength, GenericArray};

/// A marker trait for types whose default value is equal to zeroed memory.
pub trait DefaultZero {
    fn default_zero() -> Self
    where
        Self: Sized,
    {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}

impl DefaultZero for i8 {}
impl DefaultZero for i16 {}
impl DefaultZero for i32 {}
impl DefaultZero for i64 {}
impl DefaultZero for i128 {}
impl DefaultZero for isize {}
impl DefaultZero for u8 {}
impl DefaultZero for u16 {}
impl DefaultZero for u32 {}
impl DefaultZero for u64 {}
impl DefaultZero for u128 {}
impl DefaultZero for usize {}
impl DefaultZero for __m128i {}
impl DefaultZero for __m256i {}

impl<A> DefaultZero for *const A {}
impl<A> DefaultZero for *mut A {}

impl<A, N> DefaultZero for GenericArray<A, N>
where
    A: DefaultZero,
    N: ArrayLength<A>,
{
}
