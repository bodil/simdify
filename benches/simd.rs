#![feature(test)]

extern crate test;

use test::Bencher;

use std::arch::x86_64::{__m128i, __m256i};

use rand::{rngs::SmallRng, Rng, SeedableRng};

mod range;
use range::GenRange;

use simdify::{DefaultZero, SimdArrayOps, SimdOps, SimdVec};

fn simdify_k_ary_search<Int>(size: usize, b: &mut Bencher)
where
    Int: Ord + Copy + DefaultZero + GenRange + SimdOps<__m256i> + SimdOps<__m128i>,
    SimdVec<Int>: SimdArrayOps<Int>,
{
    let mut gen = SmallRng::from_entropy();
    let keys = Int::gen_range(size);
    let index = gen.gen_range(0, keys.len());
    let key = keys[index];
    b.iter(|| {
        assert_eq!(Ok(index), keys.search(key));
    })
}

#[bench]
fn simdify_k_ary_search_i8_16(b: &mut Bencher) {
    simdify_k_ary_search::<i8>(16, b)
}
#[bench]
fn simdify_k_ary_search_i8_256(b: &mut Bencher) {
    simdify_k_ary_search::<i8>(256, b)
}

#[bench]
fn simdify_k_ary_search_i32_10(b: &mut Bencher) {
    simdify_k_ary_search::<i32>(10, b)
}
#[bench]
fn simdify_k_ary_search_i32_1000(b: &mut Bencher) {
    simdify_k_ary_search::<i32>(1000, b)
}
#[bench]
fn simdify_k_ary_search_i32_100_000(b: &mut Bencher) {
    simdify_k_ary_search::<i32>(100_000, b)
}
#[bench]
fn simdify_k_ary_search_i32_10_000_000(b: &mut Bencher) {
    simdify_k_ary_search::<i32>(10_000_000, b)
}
#[bench]
fn simdify_k_ary_search_i32_1_000_000_000(b: &mut Bencher) {
    simdify_k_ary_search::<i32>(1_000_000_000, b)
}

#[bench]
fn simdify_k_ary_search_i64_10(b: &mut Bencher) {
    simdify_k_ary_search::<i64>(10, b)
}
#[bench]
fn simdify_k_ary_search_i64_1000(b: &mut Bencher) {
    simdify_k_ary_search::<i64>(1000, b)
}
#[bench]
fn simdify_k_ary_search_i64_100_000(b: &mut Bencher) {
    simdify_k_ary_search::<i64>(100_000, b)
}
#[bench]
fn simdify_k_ary_search_i64_10_000_000(b: &mut Bencher) {
    simdify_k_ary_search::<i64>(10_000_000, b)
}
#[bench]
fn simdify_k_ary_search_i64_1_000_000_000(b: &mut Bencher) {
    simdify_k_ary_search::<i64>(1_000_000_000, b)
}
