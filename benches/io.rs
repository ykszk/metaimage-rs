use criterion::{Criterion, criterion_group, criterion_main};
use metaimage::MetaImage;
use std::hint::black_box;

fn tmp_mhd_path(z: usize) -> std::path::PathBuf {
    let temp_dir = std::env::temp_dir();
    temp_dir.join(format!("bench_write_{}.mhd", z))
}

fn write_image(z: usize) {
    let arr = ndarray::Array3::<u16>::zeros((z, 512, 512));
    let image = MetaImage::from_array(arr.into_dyn());
    let mhd_path = tmp_mhd_path(z);
    image.write(&mhd_path).expect("Failed to write MHD file.");
}

fn read_image(z: usize) {
    write_image(z);
    let mhd_path = tmp_mhd_path(z);
    let _image = MetaImage::read(&mhd_path).expect("Failed to read MHD file.");
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("io");
    group.bench_function("write image", |b| b.iter(|| write_image(black_box(20))));
    group.bench_function("read image", |b| b.iter(|| read_image(black_box(20))));
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10);
    targets = bench
);
criterion_main!(benches);
