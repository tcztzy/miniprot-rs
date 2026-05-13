/// LSB radix sort for u64 slices. 8 passes with 8-bit (256-bucket) digits.
/// ~3-5x faster than comparison sort for arrays > 256 elements.
pub fn radix_sort_u64(data: &mut [u64]) {
    if data.len() < 128 {
        data.sort_unstable();
        return;
    }
    let mut buf = vec![0u64; data.len()];
    radix_pass(data, &mut buf, 0);
    radix_pass(&mut buf, data, 8);
    radix_pass(data, &mut buf, 16);
    radix_pass(&mut buf, data, 24);
    radix_pass(data, &mut buf, 32);
    radix_pass(&mut buf, data, 40);
    radix_pass(data, &mut buf, 48);
    radix_pass(&mut buf, data, 56);
}

#[inline(always)]
fn radix_pass(src: &[u64], dst: &mut [u64], shift: u32) {
    const BUCKETS: usize = 256;
    let mut counts = [0u32; BUCKETS];

    for &val in src.iter() {
        let digit = ((val >> shift) as u8) as usize;
        counts[digit] += 1;
    }

    // Skip pass if all elements in same bucket
    let n = src.len() as u32;
    for &c in &counts {
        if c == n {
            if !std::ptr::eq(src.as_ptr(), dst.as_ptr()) {
                dst.copy_from_slice(src);
            }
            return;
        }
    }

    let mut offsets = [0u32; BUCKETS];
    for i in 1..BUCKETS {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }

    for &val in src.iter() {
        let digit = ((val >> shift) as u8) as usize;
        let pos = offsets[digit] as usize;
        unsafe { *dst.get_unchecked_mut(pos) = val };
        offsets[digit] += 1;
    }
}

/// Sort a slice of Anchors (repr(transparent) over u64) in-place.
#[inline]
pub fn radix_sort_anchors(anchors: &mut [crate::types::Anchor]) {
    // SAFETY: Anchor is #[repr(transparent)] over u64
    let raw: &mut [u64] =
        unsafe { std::slice::from_raw_parts_mut(anchors.as_mut_ptr().cast(), anchors.len()) };
    radix_sort_u64(raw);
}
