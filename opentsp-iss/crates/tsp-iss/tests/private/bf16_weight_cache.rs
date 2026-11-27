use super::*;

#[test]
fn packed_bf16_cache_rejects_same_shape_mutated_allocation() {
    let mut weights = Arc::new(vec![[0u8; crate::geom::MAX_LANES]; 2]);
    Arc::make_mut(&mut weights)[1][0] = 0x3f;
    let cached = Bf16Cache::new(&weights, 1, 1);
    let activation = Bf16Act::new(&[0x80], &[0x3f]);
    assert!(cached.matches(&weights, 1, 1));
    assert_eq!(cached.packed.dot(0, &weights[0][..1], &weights[1][..1], &activation), 0x3f00_0000);
    assert!(!cached.matches(&weights, 1, 2));
    Arc::make_mut(&mut weights)[1][0] = 0x40;
    assert!(!cached.matches(&weights, 1, 1));
    let refreshed = Bf16Cache::new(&weights, 1, 1);
    assert_eq!(refreshed.packed.dot(0, &weights[0][..1], &weights[1][..1], &activation), 0x4000_0000);
}
