mod default_zero;
pub use crate::default_zero::DefaultZero;

mod simd_ops;
pub use crate::simd_ops::{SimdOps, SimdRegister};

mod array_ops;
pub use crate::array_ops::SimdArrayOps;

mod array;
pub use crate::array::SimdArray;

mod vec;
pub use crate::vec::SimdVec;

#[cfg(test)]
// FIXME: Clippy-in-rls is unhappy about something in the proptest! macro,
// remove the below when it stops being silly.
#[allow(clippy::unnecessary_operation)]
mod test {
    use super::*;
    use proptest::collection::{btree_set, SizeRange};
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use proptest::{num, proptest};
    use std::arch::x86_64::{__m128i, __m256i};
    use std::fmt::{Debug, Display};
    use typenum::U32;

    fn sorted_vec<T>(
        element: T,
        size: impl Into<SizeRange>,
    ) -> BoxedStrategy<Vec<<T::Tree as ValueTree>::Value>>
    where
        T: Strategy + 'static,
        <T::Tree as ValueTree>::Value: Ord,
    {
        btree_set(element, size)
            .prop_map(|h| {
                let mut v: Vec<_> = h.into_iter().collect();
                v.sort();
                v
            })
            .boxed()
    }

    #[test]
    fn big_k_ary_search_1() {
        let data: &[i8] = &[
            -127, -125, -124, -122, -121, -120, -119, -118, -117, -116, -115, -113, -112, -111,
            -110, -109, -108, -106, -105, -103, -102, -101, -100, -99, -98, -97, -96, -95, -93,
            -92, -91, -90, -88, -87, -86, -85, -84, -82, -81, -80, -77, -76, -75, -74, -73, -72,
            -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55,
            -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38,
            -37, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -22, -21, -19, -18,
            -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, 0, 1, 2, 3, 4,
            5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80,
            83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 102, 103, 105, 106,
            107, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126,
            127,
        ];
        let vector: SimdArray<i8, U32> = data.into();
        assert_eq!(vector[data.len() - 6], 121);
        assert_eq!(Ok(data.len() - 6), vector.search(121));
    }

    #[test]
    fn big_k_ary_search_2() {
        let data: &[i8] = &[
            -128, -127, -126, -125, -124, -122, -121, -120, -119, -118, -117, -116, -115, -114,
            -113, -112, -111, -110, -109, -107, -106, -104, -103, -102, -101, -100, -99, -98, -96,
            -95, -94, -93, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, -78,
            -77, -76, -75, -74, -73, -72, -70, -69, -68, -67, -66, -65, -64, -63, -61, -60, -59,
            -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -44, -43, -42, -41,
            -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24,
            -23, -21, -20, -19, -18, -17, -16, -15, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
            92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
            111, 113, 114, 115, 118, 120, 121, 122, 123, 124, 125, 126, 127,
        ];
        let vector: SimdArray<i8, U32> = data.into();
        assert_eq!(vector[16], -111);
        assert_eq!(Ok(16), vector.search(-111));
    }

    #[test]
    fn two_chunk_k_ary_search() {
        let data: &[i8] = &[
            -114, -111, -101, -81, -61, -56, -40, -35, -32, -31, -30, -29, -28, -27, -26, -22, -21,
            -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -9, -8, -7, -6, -5, -4, -3, -2, -1,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 32, 35,
            43, 44, 56, 57,
        ];
        let vector: SimdArray<i8, U32> = data.into();
        assert_eq!(vector[32], -4);
        assert_eq!(Ok(32), vector.search(-4));
    }

    #[test]
    fn one_item_k_ary_search() {
        let data: &[i8] = &[-1];
        let vector: SimdArray<i8, U32> = data.into();
        assert_eq!(Err(1), vector.search(0));
    }

    fn simdify_k_ary_search_present_128<A>(items: Vec<A>, index: usize)
    where
        A: Ord + Copy + DefaultZero + Debug + SimdOps<__m128i>,
        SimdArray<A, U32>: SimdArrayOps<A>,
    {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let index = index % items.len();
        let item = items[index];
        let vector: SimdVec<A> = items.as_slice().into();
        assert_eq!(Ok(index), unsafe { vector.k_ary_search::<__m128i>(item) });
    }

    fn simdify_k_ary_search_any_128<A>(items: Vec<A>, key: A)
    where
        A: Ord + Copy + DefaultZero + Debug + Display + SimdOps<__m128i>,
        SimdArray<A, U32>: SimdArrayOps<A>,
    {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let vector: SimdVec<A> = items.as_slice().into();
        match unsafe { vector.k_ary_search::<__m128i>(key) } {
            Ok(index) => assert_eq!(items[index], key),
            Err(index) => {
                if index < items.len() {
                    assert!(
                        items[index] > key,
                        "insert index value {} should be higher than search key {}",
                        items[index],
                        key
                    );
                }
                if index > 0 {
                    assert!(
                        items[index - 1] < key,
                        "pre-insert index value {} should be lower than search key {}",
                        items[index - 1],
                        key
                    );
                }
            }
        }
    }

    fn simdify_k_ary_search_present_256<A>(items: Vec<A>, index: usize)
    where
        A: Ord + Copy + DefaultZero + Debug + SimdOps<__m256i>,
        SimdArray<A, U32>: SimdArrayOps<A>,
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let index = index % items.len();
        let item = items[index];
        let vector: SimdVec<A> = items.as_slice().into();
        assert_eq!(Ok(index), unsafe { vector.k_ary_search::<__m256i>(item) });
    }

    fn simdify_k_ary_search_any_256<A>(items: Vec<A>, key: A)
    where
        A: Ord + Copy + DefaultZero + Debug + Display + SimdOps<__m256i>,
        SimdArray<A, U32>: SimdArrayOps<A>,
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let vector: SimdVec<A> = items.as_slice().into();
        match unsafe { vector.k_ary_search::<__m256i>(key) } {
            Ok(index) => assert_eq!(items[index], key),
            Err(index) => {
                if index < items.len() {
                    assert!(
                        items[index] > key,
                        "insert index value {} should be higher than search key {}",
                        items[index],
                        key
                    );
                }
                if index > 0 {
                    assert!(
                        items[index - 1] < key,
                        "pre-insert index value {} should be lower than search key {}",
                        items[index - 1],
                        key
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn plain_binary_search_present(items in sorted_vec(num::i8::ANY, 1..1024), index in num::usize::ANY) {
            let index = index % items.len();
            let item = items[index];
            assert_eq!(Ok(index), items.binary_search(&item))
        }

        #[test]
        fn plain_binary_search_any(items in sorted_vec(num::i8::ANY, 1..1024), key in num::i8::ANY) {
            match items.binary_search(&key) {
                Ok(index) => assert_eq!(items[index], key),
                Err(index) => {
                    if index < items.len() {
                        assert!(items[index] > key, "insert index value {} should be higher than search key {}", items[index], key);
                    }
                    if index > 0 {
                        assert!(items[index-1] < key, "pre-insert index value {} should be lower than search key {}", items[index-1], key);
                    }
                }
            }
        }

        #[test]
        fn simdify_k_ary_search_present_i8_128(items in sorted_vec(num::i8::ANY, 1..1024), index in num::usize::ANY) {
            simdify_k_ary_search_present_128(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i8_128(items in sorted_vec(num::i8::ANY, 1..1024), key in num::i8::ANY) {
            simdify_k_ary_search_any_128(items,key)
        }

        #[test]
        fn simdify_k_ary_search_present_i16_128(items in sorted_vec(num::i16::ANY, 1..512), index in num::usize::ANY) {
            simdify_k_ary_search_present_128(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i16_128(items in sorted_vec(num::i16::ANY, 1..512), key in num::i16::ANY) {
            simdify_k_ary_search_any_128(items,key)
        }

        #[test]
        fn simdify_k_ary_search_present_i32_128(items in sorted_vec(num::i32::ANY, 1..256), index in num::usize::ANY) {
            simdify_k_ary_search_present_128(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i32_128(items in sorted_vec(num::i32::ANY, 1..256), key in num::i32::ANY) {
            simdify_k_ary_search_any_128(items,key)
        }

        #[test]
        fn simdify_k_ary_search_present_i64_128(items in sorted_vec(num::i64::ANY, 1..128), index in num::usize::ANY) {
            simdify_k_ary_search_present_128(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i64_128(items in sorted_vec(num::i64::ANY, 1..128), key in num::i64::ANY) {
            simdify_k_ary_search_any_128(items,key)
        }

        #[test]
        fn simdify_k_ary_search_present_i8_256(items in sorted_vec(num::i8::ANY, 1..1024), index in num::usize::ANY) {
            simdify_k_ary_search_present_256(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i8_256(items in sorted_vec(num::i8::ANY, 1..1024), key in num::i8::ANY) {
            simdify_k_ary_search_any_256(items,key)
        }

        #[test]
        fn simdify_k_ary_search_present_i16_256(items in sorted_vec(num::i16::ANY, 1..512), index in num::usize::ANY) {
            simdify_k_ary_search_present_256(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i16_256(items in sorted_vec(num::i16::ANY, 1..512), key in num::i16::ANY) {
            simdify_k_ary_search_any_256(items,key)
        }

        #[test]
        fn simdify_k_ary_search_present_i32_256(items in sorted_vec(num::i32::ANY, 1..256), index in num::usize::ANY) {
            simdify_k_ary_search_present_256(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i32_256(items in sorted_vec(num::i32::ANY, 1..256), key in num::i32::ANY) {
            simdify_k_ary_search_any_256(items,key)
        }

        #[test]
        fn simdify_k_ary_search_present_i64_256(items in sorted_vec(num::i64::ANY, 1..128), index in num::usize::ANY) {
            simdify_k_ary_search_present_256(items, index)
        }

        #[test]
        fn simdify_k_ary_search_any_i64_256(items in sorted_vec(num::i64::ANY, 1..128), key in num::i64::ANY) {
            simdify_k_ary_search_any_256(items,key)
        }
    }
}
