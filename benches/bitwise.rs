#![feature(test)]

extern crate test;

use test::Bencher;

use rand::{rngs::SmallRng, Rng, SeedableRng};

mod range;
use range::GenRange;

use simdify::DefaultZero;

fn search<A: Ord>(slice: &[A], key: A) -> Result<usize, usize> {
    if slice.is_empty() {
        return Err(0);
    }
    let mut i = 0;
    let mut b = (slice.len() - 1).next_power_of_two();
    while b > 1 {
        b >>= 1;
        let j = i | b;
        if slice.len() <= j {
            continue;
        }
        if slice[j] <= key {
            i = j;
        } else {
            b >>= 1;
            while b > 0 {
                if slice[i | b] <= key {
                    i |= b;
                }
                b >>= 1;
            }
            break;
        }
    }
    if slice[i] == key {
        Ok(i)
    } else {
        Err(i)
    }
}

fn bitwise_search<Int>(size: usize, b: &mut Bencher)
where
    Int: Ord + Copy + DefaultZero + GenRange,
{
    let mut gen = SmallRng::from_entropy();
    let keys = Int::gen_range(size);
    let index = gen.gen_range(0, keys.len());
    let key = keys[index];
    b.iter(|| {
        // assert_eq!(Ok(index), keys.binary_search(&key));
        assert_eq!(Ok(index), search(&keys, key));
    })
}

#[bench]
fn bitwise_search_i8_16(b: &mut Bencher) {
    bitwise_search::<i8>(16, b)
}
#[bench]
fn bitwise_search_i8_256(b: &mut Bencher) {
    bitwise_search::<i8>(256, b)
}

#[bench]
fn bitwise_search_i32_10(b: &mut Bencher) {
    bitwise_search::<i32>(10, b)
}
#[bench]
fn bitwise_search_i32_1000(b: &mut Bencher) {
    bitwise_search::<i32>(1000, b)
}
#[bench]
fn bitwise_search_i32_100_000(b: &mut Bencher) {
    bitwise_search::<i32>(100_000, b)
}
#[bench]
fn bitwise_search_i32_10_000_000(b: &mut Bencher) {
    bitwise_search::<i32>(10_000_000, b)
}
#[bench]
fn bitwise_search_i32_1_000_000_000(b: &mut Bencher) {
    bitwise_search::<i32>(1_000_000_000, b)
}

#[bench]
fn bitwise_search_i64_10(b: &mut Bencher) {
    bitwise_search::<i64>(10, b)
}
#[bench]
fn bitwise_search_i64_1000(b: &mut Bencher) {
    bitwise_search::<i64>(1000, b)
}
#[bench]
fn bitwise_search_i64_100_000(b: &mut Bencher) {
    bitwise_search::<i64>(100_000, b)
}
#[bench]
fn bitwise_search_i64_10_000_000(b: &mut Bencher) {
    bitwise_search::<i64>(10_000_000, b)
}
#[bench]
fn bitwise_search_i64_1_000_000_000(b: &mut Bencher) {
    bitwise_search::<i64>(1_000_000_000, b)
}
