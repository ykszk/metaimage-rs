fn check_python() {
    // check if python is available
    let python_path = which::which("python").expect("Python is not available in the system PATH.");
    // check if SimpleITK is installed
    let output = std::process::Command::new("python")
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
#[ignore = "Testing the compatiblity with ITk is skipped. Run tests with `--include-ignored` to enable it."]
fn test_itk_compatibility() {
    use ndarray::Array3;
    use std::path::Path;
    check_python();
    let temp_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));

    let mut paths = Vec::new();
    macro_rules! save_mhd_mha {
        ($ty:ty, $basename:expr) => {{
            let arr = Array3::<$ty>::zeros((10, 10, 10));
            let mhd_path = temp_dir.join(format!("{}_{}.mhd", $basename, stringify!($ty)));
            let image = metaimage::MetaImage::from_array(arr.clone().into_dyn());
            image
                .write_mhd(&mhd_path)
                .expect("Failed to write MHD file.");
            paths.push(mhd_path);
            let mha_path = temp_dir.join(format!("{}_{}.mha", $basename, stringify!($ty)));
            let image = metaimage::MetaImage::from_array(arr.into_dyn());
            image
                .write_mha(&mha_path)
                .expect("Failed to write MHA file.");
            paths.push(mha_path);
        }};
    }

    save_mhd_mha!(u8, "test_image");
    save_mhd_mha!(u16, "test_image");
    save_mhd_mha!(f32, "test_image");
    save_mhd_mha!(f64, "test_image");

    let output = std::process::Command::new("python")
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
