use std::arch::x86_64::{self as arch, __m128i, __m256i};
use std::mem::size_of;

use bitmaps::{Bitmap, Bits};
use typenum::{U16, U32};

/// Marker trait for SIMD registers.
pub trait SimdRegister: Copy + Sized {
    type MovemaskSize: Bits;

    #[inline]
    unsafe fn from_m256i(slice256: &[__m256i]) -> &[Self] {
        let len = slice256.len() * (size_of::<__m256i>() / size_of::<Self>());
        std::slice::from_raw_parts(slice256.as_ptr() as *const Self, len)
    }
    unsafe fn load(&self) -> Self;
}

impl SimdRegister for __m128i {
    type MovemaskSize = U16;

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load(&self) -> Self {
        arch::_mm_load_si128(self)
    }
}

impl SimdRegister for __m256i {
    type MovemaskSize = U32;

    #[inline]
    unsafe fn from_m256i(slice: &[__m256i]) -> &[Self] {
        slice
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(&self) -> Self {
        arch::_mm256_load_si256(self)
    }
}

/// Operations on datatypes stored in SIMD registers.
pub trait SimdOps<R: SimdRegister>: Sized {
    const ALIGNMENT: usize = size_of::<R>() / size_of::<Self>();
    const BITS_PER_CMP: usize = size_of::<Self>();

    unsafe fn set(value: Self) -> R;
    unsafe fn cmp_eq(left: R, right: R) -> Bitmap<R::MovemaskSize>;
    unsafe fn cmp_gt(left: R, right: R) -> Bitmap<R::MovemaskSize>;
}

impl SimdOps<__m128i> for i8 {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn set(value: i8) -> __m128i {
        arch::_mm_set1_epi8(value)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_eq(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpeq_epi8(left, right)) as u16)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_gt(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpgt_epi8(left, right)) as u16)
    }
}

impl SimdOps<__m256i> for i8 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set(value: i8) -> __m256i {
        arch::_mm256_set1_epi8(value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_eq(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpeq_epi8(left, right)) as u32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_gt(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpgt_epi8(left, right)) as u32)
    }
}

impl SimdOps<__m128i> for i16 {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn set(value: i16) -> __m128i {
        arch::_mm_set1_epi16(value)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_eq(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpeq_epi16(left, right)) as u16)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_gt(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpgt_epi16(left, right)) as u16)
    }
}

impl SimdOps<__m256i> for i16 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set(value: i16) -> __m256i {
        arch::_mm256_set1_epi16(value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_eq(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpeq_epi16(left, right)) as u32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_gt(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpgt_epi16(left, right)) as u32)
    }
}

impl SimdOps<__m128i> for i32 {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn set(value: i32) -> __m128i {
        arch::_mm_set1_epi32(value)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_eq(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpeq_epi32(left, right)) as u16)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_gt(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpgt_epi32(left, right)) as u16)
    }
}

impl SimdOps<__m256i> for i32 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set(value: i32) -> __m256i {
        arch::_mm256_set1_epi32(value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_eq(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpeq_epi32(left, right)) as u32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_gt(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpgt_epi32(left, right)) as u32)
    }
}

impl SimdOps<__m128i> for i64 {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn set(value: i64) -> __m128i {
        arch::_mm_set1_epi64x(value)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_eq(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpeq_epi64(left, right)) as u16)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_gt(left: __m128i, right: __m128i) -> Bitmap<U16> {
        Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpgt_epi64(left, right)) as u16)
    }
}

impl SimdOps<__m256i> for i64 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set(value: i64) -> __m256i {
        arch::_mm256_set1_epi64x(value)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_eq(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpeq_epi64(left, right)) as u32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_gt(left: __m256i, right: __m256i) -> Bitmap<U32> {
        Bitmap::from_value(arch::_mm256_movemask_epi8(arch::_mm256_cmpgt_epi64(left, right)) as u32)
    }
}

impl SimdOps<__m128i> for isize {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn set(value: isize) -> __m128i {
        if std::mem::size_of::<isize>() == 8 {
            arch::_mm_set1_epi64x(value as i64)
        } else if std::mem::size_of::<isize>() == 4 {
            arch::_mm_set1_epi32(value as i32)
        } else {
            panic!(
                "did not expect isize to be {} bytes long",
                std::mem::size_of::<isize>()
            )
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_eq(left: __m128i, right: __m128i) -> Bitmap<U16> {
        if std::mem::size_of::<isize>() == 8 {
            Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpeq_epi64(left, right)) as u16)
        } else if std::mem::size_of::<isize>() == 4 {
            Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpeq_epi32(left, right)) as u16)
        } else {
            panic!(
                "did not expect isize to be {} bytes long",
                std::mem::size_of::<isize>()
            )
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn cmp_gt(left: __m128i, right: __m128i) -> Bitmap<U16> {
        if std::mem::size_of::<isize>() == 8 {
            Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpgt_epi64(left, right)) as u16)
        } else if std::mem::size_of::<isize>() == 4 {
            Bitmap::from_value(arch::_mm_movemask_epi8(arch::_mm_cmpgt_epi32(left, right)) as u16)
        } else {
            panic!(
                "did not expect isize to be {} bytes long",
                std::mem::size_of::<isize>()
            )
        }
    }
}

impl SimdOps<__m256i> for isize {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set(value: isize) -> __m256i {
        if std::mem::size_of::<isize>() == 8 {
            arch::_mm256_set1_epi64x(value as i64)
        } else if std::mem::size_of::<isize>() == 4 {
            arch::_mm256_set1_epi32(value as i32)
        } else {
            panic!(
                "did not expect isize to be {} bytes long",
                std::mem::size_of::<isize>()
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_eq(left: __m256i, right: __m256i) -> Bitmap<U32> {
        if std::mem::size_of::<isize>() == 8 {
            Bitmap::from_value(
                arch::_mm256_movemask_epi8(arch::_mm256_cmpeq_epi64(left, right)) as u32,
            )
        } else if std::mem::size_of::<isize>() == 4 {
            Bitmap::from_value(
                arch::_mm256_movemask_epi8(arch::_mm256_cmpeq_epi32(left, right)) as u32,
            )
        } else {
            panic!(
                "did not expect isize to be {} bytes long",
                std::mem::size_of::<isize>()
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmp_gt(left: __m256i, right: __m256i) -> Bitmap<U32> {
        if std::mem::size_of::<isize>() == 8 {
            Bitmap::from_value(
                arch::_mm256_movemask_epi8(arch::_mm256_cmpgt_epi64(left, right)) as u32,
            )
        } else if std::mem::size_of::<isize>() == 4 {
            Bitmap::from_value(
                arch::_mm256_movemask_epi8(arch::_mm256_cmpgt_epi32(left, right)) as u32,
            )
        } else {
            panic!(
                "did not expect isize to be {} bytes long",
                std::mem::size_of::<isize>()
            )
        }
    }
}
