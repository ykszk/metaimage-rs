MetaImage (mhd/mha) file IO library

# References
- [Specification](https://insightsoftwareconsortium.github.io/ITKWikiArchive/Wiki/ITK/MetaIO/Documentation/#MetaImage)

# Example
```rust
use metaimage::MetaImage;

let temp_path = std::env::temp_dir().join("image.mhd");
let arr = ndarray::Array3::<u8>::from_elem((10, 10, 10), 42u8);

// Create a MetaImage from an array
let image = MetaImage::from_array(arr.into_dyn());

// Write
assert!(image.write(&temp_path).is_ok());

// Read back the written image
let read_image = MetaImage::read(&temp_path).unwrap();
// Check the shape
assert_eq!(image.data.shape(), read_image.data.shape());
// Extract pixel values as Array3<u8>
let read_arr = read_image.data.into_u8_array().unwrap();
let written_arr = image.data.into_u8_array().unwrap();
// Check the pixel values
assert_eq!(written_arr.into_raw_vec_and_offset().0, read_arr.into_raw_vec_and_offset().0);
```