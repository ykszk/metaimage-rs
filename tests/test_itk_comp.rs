use metaimage::MetaImage;
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
        ($ty:ty, $basename:expr, $value:expr) => {{
            let arr = Array3::<$ty>::from_elem((10, 10, 10), $value);
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

    save_mhd_mha!(u8, "test_image", 42u8);
    save_mhd_mha!(u16, "test_image", 42u16);
    save_mhd_mha!(u32, "test_image", 42u32);
    save_mhd_mha!(f32, "test_image", 42f32);
    save_mhd_mha!(f64, "test_image", 42f64);

    let output = std::process::Command::new(python_bin())
        .arg("tests/read_mhd.py")
        .args(&paths)
        .arg("--shape")
        .args(["10", "10", "10"])
        .arg("--value")
        .arg("42")
        .output()
        .expect("Failed to execute Python script to read the image.");
    if !output.status.success() {
        panic!(
            "Failed to read the image using SimpleITK. stderr: {}\n stdout: {}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
    }

    paths.clear();
    save_mhd_mha!(i8, "test_image", -42i8);
    save_mhd_mha!(i16, "test_image", -42i16);
    save_mhd_mha!(i32, "test_image", -42i32);

    let output = std::process::Command::new(python_bin())
        .arg("tests/read_mhd.py")
        .args(&paths)
        .arg("--shape")
        .args(["10", "10", "10"])
        .arg("--value=-42")
        .output()
        .expect("Failed to execute Python script to read the image.");
    if !output.status.success() {
        panic!(
            "Failed to read the image using SimpleITK. stderr: {}\n stdout: {}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

#[test]
fn test_uncontiguous_write() {
    let temp_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));

    let value_vec: Vec<_> = (0u16..100).collect();
    let arr = ndarray::Array2::<u16>::from_shape_vec((10, 10), value_vec).unwrap();
    let arr = arr.permuted_axes([1, 0]); // make it uncontiguous
    let image = MetaImage::from_array(arr.view().into_dyn());
    assert!(!image.data.as_u16_array().unwrap().is_standard_layout());
    let mhd_path = temp_dir.join("uncontiguous_image.mha");
    image.write(&mhd_path).expect("Failed to write MHD file.");

    // read back
    let image = MetaImage::read(&mhd_path).expect("Failed to read MHD file.");
    let read_vec = image
        .data
        .into_u16_array()
        .expect("expected u16 data")
        .into_raw_vec_and_offset()
        .0;
    let permuted_vec = arr
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    assert_eq!(read_vec, permuted_vec);

    // test for i8
    let value_vec: Vec<i8> = (-50..50).collect();
    let arr = ndarray::Array2::<i8>::from_shape_vec((10, 10), value_vec).unwrap();
    let arr = arr.permuted_axes([1, 0]); // make it uncontiguous
    let image = MetaImage::from_array(arr.clone().into_dyn());
    assert!(!image.data.as_i8_array().unwrap().is_standard_layout());
    let mhd_path = temp_dir.join("uncontiguous_image.mha");
    image.write(&mhd_path).expect("Failed to write MHD file.");

    // read back
    let image = MetaImage::read(&mhd_path).expect("Failed to read MHD file.");
    let read_vec = image
        .data
        .into_i8_array()
        .expect("expected i8 data")
        .into_raw_vec_and_offset()
        .0;
    let permuted_vec = arr
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    assert_eq!(read_vec, permuted_vec);
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
    let arr = image.data.as_f32_array().expect("expected f32 data");
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
    let arr = image.data.as_u16_array().expect("expected u16 data");
    arr.iter().for_each(|&v| {
        assert_eq!(v, 42);
    });
}

#[test]
#[ignore = "Testing the compatiblity with ITK is skipped. Run tests with `--include-ignored` to enable it."]
fn test_itk_vector_image_compatibility() {
    let temp_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));

    // test read
    let pixel_value = 42u8;
    let mhd_path = temp_dir.join("itk_test_vector_image.mhd");
    let output = std::process::Command::new(python_bin())
        .arg("tests/write_mhd.py")
        .arg("--shape")
        .args(["10", "10", "10", "3"])
        .arg("--value")
        .arg(pixel_value.to_string())
        .arg("--dtype")
        .arg("uint8")
        .arg("--vector")
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
    let shape = Vec::from(image.data.shape());
    assert_eq!(shape, vec![10, 10, 10, 3]);
    let arr = image.data.as_u8_array().expect("expected u8 data");
    arr.iter().for_each(|&v| {
        assert_eq!(v, pixel_value);
    });

    // test write
    let pixel_value = 128u8;
    let arr = ndarray::Array4::<u8>::from_elem((10, 10, 10, 3), pixel_value);
    let image = MetaImage::with_channels(arr.into_dyn());
    let mhd_path = temp_dir.join("itk_test_vector_image_write.mhd");
    image.write(&mhd_path).expect("Failed to write MHD file.");

    // read back
    let image = MetaImage::read(&mhd_path).expect("Failed to read MHD file.");
    let shape = Vec::from(image.data.shape());
    assert_eq!(shape, vec![10, 10, 10, 3]);
    let arr = image.data.as_u8_array().expect("expected u8 data");
    arr.iter().for_each(|&v| {
        assert_eq!(v, pixel_value);
    });

    // verify with SimpleITK
    let output = std::process::Command::new(python_bin())
        .arg("tests/read_mhd.py")
        .args([&mhd_path])
        .arg("--shape")
        .args(["10", "10", "10", "3"])
        .arg("--value")
        .arg(pixel_value.to_string())
        .output()
        .expect("Failed to execute Python script to read the image.");
    if !output.status.success() {
        panic!(
            "Failed to read the image using SimpleITK. stderr: {}\n stdout: {}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

#[test]
fn test_endian() {
    let temp_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));

    let arr = ndarray::Array2::<u16>::from_elem((10, 10), 1);
    let image: MetaImage = MetaImage::from_array(arr.into_dyn());

    // Write 1u16 in non-native endian
    let non_native_path = temp_dir.join("non_native_endian_image.mhd");
    // manually flip the endianness flag
    let mut non_native_image = image;
    non_native_image.metadata.element_byte_order_msb =
        !non_native_image.metadata.element_byte_order_msb;
    non_native_image
        .write(&non_native_path)
        .expect("Failed to write MHD file.");

    // Read non-native endian file normally
    let result = MetaImage::read(&non_native_path).unwrap();
    let arr = result.data.as_u16_array().expect("expected u16 data");
    arr.iter().for_each(|&v| {
        assert_eq!(v, 1);
    });

    // Read image with incorrect endianness flag
    let header = std::fs::read_to_string(&non_native_path).unwrap();
    #[cfg(target_endian = "big")]
    let incorrect_header =
        header.replace("ElementByteOrderMSB = False", "ElementByteOrderMSB = True");
    #[cfg(target_endian = "little")]
    let incorrect_header =
        header.replace("ElementByteOrderMSB = True", "ElementByteOrderMSB = False");
    std::fs::write(&non_native_path, incorrect_header).unwrap();
    let result = MetaImage::read(&non_native_path).unwrap();
    let arr = result.data.as_u16_array().expect("expected u16 data");
    arr.iter().for_each(|&v| {
        assert_eq!(v, 1 << 8); // byte swapped
    });

    // test for u32
    let arr = ndarray::Array2::<u32>::from_elem((10, 10), 1);
    let image: MetaImage = MetaImage::from_array(arr.into_dyn());

    // Write
    let non_native_path = temp_dir.join("non_native_endian_image.mhd");
    // manually flip the endianness flag
    let mut non_native_image = image;
    non_native_image.metadata.element_byte_order_msb =
        !non_native_image.metadata.element_byte_order_msb;
    non_native_image
        .write(&non_native_path)
        .expect("Failed to write MHD file.");

    // Read non-native endian file normally
    let result = MetaImage::read(&non_native_path).unwrap();
    let arr = result.data.as_u32_array().expect("expected u32 data");
    arr.iter().for_each(|&v| {
        assert_eq!(v, 1);
    });

    // Read image with incorrect endianness flag
    let header = std::fs::read_to_string(&non_native_path).unwrap();
    #[cfg(target_endian = "big")]
    let incorrect_header =
        header.replace("ElementByteOrderMSB = False", "ElementByteOrderMSB = True");
    #[cfg(target_endian = "little")]
    let incorrect_header =
        header.replace("ElementByteOrderMSB = True", "ElementByteOrderMSB = False");
    std::fs::write(&non_native_path, incorrect_header).unwrap();
    let result = MetaImage::read(&non_native_path).unwrap();
    let arr = result.data.as_u32_array().expect("expected u32 data");
    arr.iter().for_each(|&v| {
        assert_eq!(v, 1 << 24); // byte reversed
    });
}
