use std::arch::x86_64::{self as arch, __m128i, __m256i};
use std::ops::Deref;

use bitmaps::Bitmap;

use crate::simd_ops::{SimdOps, SimdRegister};

/// SIMD optimised array operations.
pub trait SimdArrayOps<A>: Deref<Target = [A]>
where
    A: Ord,
{
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn data_m256(&self) -> &[arch::__m256i];

    fn search(&self, key: A) -> Result<usize, usize>
    where
        A: Copy + SimdOps<__m256i> + SimdOps<__m128i>,
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.k_ary_search::<__m256i>(key) }
        } else if is_x86_feature_detected!("sse2") {
            unsafe { self.k_ary_search::<__m128i>(key) }
        } else {
            self.deref().binary_search(&key)
        }
    }

    #[inline]
    unsafe fn load<R>(&self, index: usize) -> R
    where
        R: SimdRegister,
        A: SimdOps<R>,
    {
        debug_assert_eq!(
            0,
            index & (A::ALIGNMENT - 1),
            "load index must be divisible by {}",
            A::ALIGNMENT
        );
        debug_assert!(index < self.len(), "index out of range");
        R::load(&R::from_m256i(self.data_m256())[index / A::ALIGNMENT])
    }

    /// Fast k-ary search for a key in an array.
    ///
    /// The algorithm is described in ['k-Ary Search
    /// on Modern Processors,'
    /// 2009](https://event.cwi.nl/damon2009/DaMoN09-KarySearch.pdf).
    unsafe fn k_ary_search<R>(&self, key: A) -> Result<usize, usize>
    where
        R: SimdRegister,
        A: Copy + SimdOps<R>,
    {
        if self.is_empty() {
            return Err(0);
        }
        let keys = A::set(key);
        let mut middle = self.len() / (2 * A::ALIGNMENT);
        let mut pos = middle * A::ALIGNMENT;
        let mut low = 0;
        let mut high = self.len();
        loop {
            let data = self.load(pos);
            let mut eq = A::cmp_eq(data, keys);
            let mut cmp = A::cmp_gt(data, keys);
            let all_greater = if pos + A::ALIGNMENT > self.len() {
                // At last chunk, mask away out-of-bounds bits
                let mask = Bitmap::mask((self.len() - pos) * A::BITS_PER_CMP);
                eq &= mask;
                cmp &= mask;
                mask
            } else {
                !Bitmap::new()
            };
            if !eq.is_empty() {
                // We found a match
                return Ok(pos + eq.first_index().unwrap() / A::BITS_PER_CMP);
            }
            if cmp.is_empty() {
                // Everything was smaller, move up
                low = pos + A::ALIGNMENT;
                middle = std::cmp::max(middle / 2, 1);
                pos += middle * A::ALIGNMENT;
                if pos >= high {
                    // Nowhere to move up, we found the insertion point
                    return Err(high);
                }
                continue;
            }
            if cmp == all_greater {
                // Everything was greater, move down
                if pos <= low {
                    // Nowhere to move down, we found the insertion point
                    return Err(low);
                }
                high = pos;
                middle = std::cmp::max(middle / 2, 1);
                pos -= middle * A::ALIGNMENT;
                continue;
            }
            // We found a transition point
            let index = (cmp.first_index().unwrap() / A::BITS_PER_CMP) + pos;
            return Err(index);
        }
    }
}
