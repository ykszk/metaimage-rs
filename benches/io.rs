use criterion::{Criterion, criterion_group, criterion_main};
use metaimage::{MetaImage, WriteOption};
use std::hint::black_box;

fn tmp_mhd_path(z: usize) -> std::path::PathBuf {
    let temp_dir = std::env::temp_dir();
    temp_dir.join(format!("bench_write_{}.mha", z))
}

fn tmp_compressed_mhd_path(z: usize) -> std::path::PathBuf {
    let temp_dir = std::env::temp_dir();
    temp_dir.join(format!("bench_compressed_write_{}.mha", z))
}

fn write_image(z: usize) {
    let arr = ndarray::Array3::<u16>::zeros((z, 512, 512));
    let image = MetaImage::from_array(arr.into_dyn());
    let mhd_path = tmp_mhd_path(z);
    let option = WriteOption {
        data_file: None,
        compress: false,
    };
    image
        .write_with_option(&mhd_path, option)
        .expect("Failed to write MHD file.");
}

/// Write image first!
fn read_image(z: usize) {
    let mhd_path = tmp_mhd_path(z);
    let _image = MetaImage::read(&mhd_path).expect("Failed to read MHD file.");
}

fn compressed_write(z: usize) {
    let arr = ndarray::Array3::<u16>::zeros((z, 512, 512));
    let image = MetaImage::from_array(arr.into_dyn());
    let mhd_path = tmp_compressed_mhd_path(z);
    let option = WriteOption {
        data_file: None,
        compress: true,
    };
    image
        .write_with_option(&mhd_path, option)
        .expect("Failed to write compressed MHD file.");
}

fn compressed_read(z: usize) {
    let mhd_path = tmp_compressed_mhd_path(z);
    let _image = MetaImage::read(&mhd_path).expect("Failed to read compressed MHD file.");
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("io");
    group.bench_function("write", |b| b.iter(|| write_image(black_box(20))));
    group.bench_function("read", |b| b.iter(|| read_image(black_box(20))));
    group.bench_function("compressed write", |b| {
        b.iter(|| compressed_write(black_box(20)))
    });
    group.bench_function("compressed read", |b| {
        b.iter(|| compressed_read(black_box(20)))
    });
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10);
    targets = bench
);
criterion_main!(benches);
