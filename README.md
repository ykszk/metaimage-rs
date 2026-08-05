MetaImage (mhd/mha) file IO library

# Examples

## Write
[`MetaImage::write`]
```rust
use metaimage::{MetaImage, WriteMhd};

let temp_path = std::env::temp_dir().join("image.mhd");
let arr = ndarray::Array3::<u8>::from_elem((10, 10, 10), 42u8);

// Simplest write
MetaImage::write_mhd(arr.view(), &temp_path).unwrap();
assert!(temp_path.exists());

// Create a MetaImage from an array
let image = MetaImage::from(arr);

// Write
assert!(image.write(&temp_path).is_ok());
assert!(temp_path.exists());

// Commpression is automatically enabled (thus .zraw suffix) because pixel type is `u8`
let data_path = std::env::temp_dir().join("image.zraw");
assert!(data_path.exists());

// Write as an single file
let temp_mha_path = std::env::temp_dir().join("single_file.mha");
assert!(image.write(&temp_mha_path).is_ok());
assert!(temp_mha_path.exists());
```

## Read
[`MetaImage::read`]
```rust
use metaimage::{MetaImage, WriteMhd};

let temp_path = std::env::temp_dir().join("image_read_example.mhd");
let arr = ndarray::Array3::<u8>::from_elem((10, 10, 10), 42u8);
MetaImage::write_mhd(arr.view(), &temp_path).unwrap();

// Read
let image = MetaImage::read(&temp_path).unwrap();
// Extract pixel values as an u8 array
let read_arr = image.data.into_u8_array().unwrap();
```

## RGB image
[`MetaImage::with_channels`]
```rust
use metaimage::{MetaImage, WithChannels};

let arr = ndarray::Array4::<u8>::from_elem((10, 10, 10, 3), 42u8);
let rgb_image = MetaImage::with_channels(arr);
assert_eq!(rgb_image.metadata.element_no_of_channels, 3);
```

# References
- [Specification](https://insightsoftwareconsortium.github.io/ITKWikiArchive/Wiki/ITK/MetaIO/Documentation/#MetaImage)
