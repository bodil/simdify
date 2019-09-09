use simdify::SimdVec;

pub trait GenRange: Sized {
    fn gen_range(size: usize) -> SimdVec<Self>;
}

impl GenRange for i8 {
    fn gen_range(size: usize) -> SimdVec<Self> {
        let mut i = i8::min_value();
        let mut vec = SimdVec::new();
        for _ in 0..size {
            vec.push(i);
            i += 1;
        }
        vec
    }
}

impl GenRange for i32 {
    fn gen_range(size: usize) -> SimdVec<Self> {
        let mut i = i32::min_value();
        let mut vec = SimdVec::new();
        for _ in 0..size {
            vec.push(i);
            i += 1;
        }
        vec
    }
}

impl GenRange for i64 {
    fn gen_range(size: usize) -> SimdVec<Self> {
        let mut i = i64::min_value();
        let mut vec = SimdVec::new();
        for _ in 0..size {
            vec.push(i);
            i += 1;
        }
        vec
    }
}
