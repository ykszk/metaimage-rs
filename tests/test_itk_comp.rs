use metaimage::{MetaImage, PixelData};
use ndarray::Array3;
use std::path::Path;

/// Env var PYTHON_BIN or "python" command
fn python_bin() -> String {
    std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python".to_string())
}

fn check_python() {
    // check if python is available
    let python_path =
        which::which(python_bin()).expect("Python is not available in the system PATH.");
    // check if SimpleITK is installed
    let output = std::process::Command::new(python_bin())
        .arg("-c")
        .arg("import SimpleITK")
        .output()
        .expect("Failed to execute Python command.");
    if !output.status.success() {
        panic!(
            "SimpleITK is not installed in the Python environment ({}).",
            python_path.display()
        );
    }
}

#[test]
#[ignore = "Testing the compatiblity with ITK is skipped. Run tests with `--include-ignored` to enable it."]
fn test_itk_write_compatibility() {
    check_python();
    let temp_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));

    let mut paths = Vec::new();
    macro_rules! save_mhd_mha {
        ($ty:ty, $basename:expr) => {{
            let arr = Array3::<$ty>::zeros((10, 10, 10));
            let mhd_path = temp_dir.join(format!("{}_{}.mhd", $basename, stringify!($ty)));
            let image = MetaImage::from_array(arr.clone().into_dyn());
            image.write(&mhd_path).expect("Failed to write MHD file.");
            paths.push(mhd_path);
            let mha_path = temp_dir.join(format!("{}_{}.mha", $basename, stringify!($ty),));
            let image = MetaImage::from_array(arr.into_dyn());
            image.write(&mha_path).expect("Failed to write MHA file.");
            paths.push(mha_path);
        }};
    }

    save_mhd_mha!(u8, "test_image");
    save_mhd_mha!(u16, "test_image");
    save_mhd_mha!(f32, "test_image");
    save_mhd_mha!(f64, "test_image");

    let output = std::process::Command::new(python_bin())
        .arg("tests/read_mhd.py")
        .args(&paths)
        .arg("--shape")
        .args(["10", "10", "10"])
        .output()
        .expect("Failed to execute Python script to read the image.");
    if !output.status.success() {
        panic!(
            "Failed to read the image using SimpleITK. stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "Testing the compatiblity with ITK is skipped. Run tests with `--include-ignored` to enable it."]
fn test_itk_read_compatibility() {
    check_python();
    let temp_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    // raw uncompressed float32 image
    let mhd_path = temp_dir.join("itk_test_image.mhd");
    let output = std::process::Command::new(python_bin())
        .arg("tests/write_mhd.py")
        .arg("--shape")
        .args(["10", "10", "10"])
        .arg("--value")
        .arg("42")
        .arg("--dtype")
        .arg("float32")
        .arg("--")
        .arg(&mhd_path)
        .output()
        .expect("Failed to execute Python script to write the image.");
    if !output.status.success() {
        panic!(
            "Failed to write the image using SimpleITK. stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let image = MetaImage::read(&mhd_path).expect("Failed to read MHD file.");
    let PixelData::F32(arr) = image.data else {
        panic!("expected f32 data");
    };
    arr.iter().for_each(|&v| {
        assert_eq!(v, 42.0);
    });

    // compressed uint16 image
    let mhd_path = temp_dir.join("itk_test_image.mha");
    let output = std::process::Command::new(python_bin())
        .arg("tests/write_mhd.py")
        .arg("--shape")
        .args(["10", "10", "10"])
        .arg("--value")
        .arg("42")
        .arg("--dtype")
        .arg("uint16")
        .arg("--compress")
        .arg("--")
        .arg(&mhd_path)
        .output()
        .expect("Failed to execute Python script to write the image.");
    if !output.status.success() {
        panic!(
            "Failed to write the image using SimpleITK. stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let image = MetaImage::read(&mhd_path).expect("Failed to read MHD file.");
    let PixelData::U16(arr) = image.data else {
        panic!("expected uint16 data");
    };
    arr.iter().for_each(|&v| {
        assert_eq!(v, 42);
    });
}
