(function() {var implementors = {};
implementors["bitmaps"] = [{text:"impl&lt;Size&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"bitmaps/struct.Bitmap.html\" title=\"struct bitmaps::Bitmap\">Bitmap</a>&lt;Size&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;Size as <a class=\"trait\" href=\"bitmaps/trait.Bits.html\" title=\"trait bitmaps::Bits\">Bits</a>&gt;::<a class=\"type\" href=\"bitmaps/trait.Bits.html#associatedtype.Store\" title=\"type bitmaps::Bits::Store\">Store</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["bitmaps::bitmap::Bitmap"]},{text:"impl&lt;'a, Size&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"bitmaps/struct.Iter.html\" title=\"struct bitmaps::Iter\">Iter</a>&lt;'a, Size&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;Size as <a class=\"trait\" href=\"bitmaps/trait.Bits.html\" title=\"trait bitmaps::Bits\">Bits</a>&gt;::<a class=\"type\" href=\"bitmaps/trait.Bits.html#associatedtype.Store\" title=\"type bitmaps::Bits::Store\">Store</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>,&nbsp;</span>",synthetic:true,types:["bitmaps::bitmap::Iter"]},];
implementors["generic_array"] = [{text:"impl&lt;T, N&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"generic_array/iter/struct.GenericArrayIter.html\" title=\"struct generic_array::iter::GenericArrayIter\">GenericArrayIter</a>&lt;T, N&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["generic_array::iter::GenericArrayIter"]},{text:"impl&lt;T:&nbsp;<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>, N:&nbsp;<a class=\"trait\" href=\"generic_array/trait.ArrayLength.html\" title=\"trait generic_array::ArrayLength\">ArrayLength</a>&lt;T&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"generic_array/struct.GenericArray.html\" title=\"struct generic_array::GenericArray\">GenericArray</a>&lt;T, N&gt;",synthetic:false,types:["generic_array::GenericArray"]},];
implementors["simdify"] = [{text:"impl&lt;A, N&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"simdify/struct.SimdArray.html\" title=\"struct simdify::SimdArray\">SimdArray</a>&lt;A, N&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;N: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["simdify::array::SimdArray"]},{text:"impl&lt;A&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"simdify/struct.SimdVec.html\" title=\"struct simdify::SimdVec\">SimdVec</a>&lt;A&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["simdify::vec::SimdVec"]},];
implementors["typenum"] = [{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/struct.Greater.html\" title=\"struct typenum::Greater\">Greater</a>",synthetic:true,types:["typenum::Greater"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/struct.Less.html\" title=\"struct typenum::Less\">Less</a>",synthetic:true,types:["typenum::Less"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/struct.Equal.html\" title=\"struct typenum::Equal\">Equal</a>",synthetic:true,types:["typenum::Equal"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/bit/struct.B0.html\" title=\"struct typenum::bit::B0\">B0</a>",synthetic:true,types:["typenum::bit::B0"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/bit/struct.B1.html\" title=\"struct typenum::bit::B1\">B1</a>",synthetic:true,types:["typenum::bit::B1"]},{text:"impl&lt;U&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/int/struct.PInt.html\" title=\"struct typenum::int::PInt\">PInt</a>&lt;U&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;U: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["typenum::int::PInt"]},{text:"impl&lt;U&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/int/struct.NInt.html\" title=\"struct typenum::int::NInt\">NInt</a>&lt;U&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;U: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["typenum::int::NInt"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/int/struct.Z0.html\" title=\"struct typenum::int::Z0\">Z0</a>",synthetic:true,types:["typenum::int::Z0"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/uint/struct.UTerm.html\" title=\"struct typenum::uint::UTerm\">UTerm</a>",synthetic:true,types:["typenum::uint::UTerm"]},{text:"impl&lt;U, B&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/uint/struct.UInt.html\" title=\"struct typenum::uint::UInt\">UInt</a>&lt;U, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;B: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;U: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["typenum::uint::UInt"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/array/struct.ATerm.html\" title=\"struct typenum::array::ATerm\">ATerm</a>",synthetic:true,types:["typenum::array::ATerm"]},{text:"impl&lt;V, A&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"typenum/array/struct.TArr.html\" title=\"struct typenum::array::TArr\">TArr</a>&lt;V, A&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>",synthetic:true,types:["typenum::array::TArr"]},];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        
})()