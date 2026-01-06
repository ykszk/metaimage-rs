#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
use ndarray::{ArrayD, IxDyn};
use std::borrow::Cow;
use std::error::Error;
use std::fmt::{self, Display};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Pixel storage type supported by MetaImage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    UChar,
    Char,
    UShort,
    Short,
    UInt,
    Int,
    Float,
    Double,
    ULong,
    Long,
}

impl ElementType {
    fn from_tag(raw: &str) -> Option<Self> {
        let upper = raw.trim().to_ascii_uppercase();
        match upper.as_str() {
            "MET_UCHAR" | "MET_UCHAR_ARRAY" => Some(Self::UChar),
            "MET_CHAR" | "MET_CHAR_ARRAY" => Some(Self::Char),
            "MET_USHORT" | "MET_USHORT_ARRAY" => Some(Self::UShort),
            "MET_SHORT" | "MET_SHORT_ARRAY" => Some(Self::Short),
            "MET_UINT" | "MET_UINT_ARRAY" => Some(Self::UInt),
            "MET_INT" | "MET_INT_ARRAY" => Some(Self::Int),
            "MET_FLOAT" | "MET_FLOAT_ARRAY" => Some(Self::Float),
            "MET_DOUBLE" | "MET_DOUBLE_ARRAY" => Some(Self::Double),
            "MET_ULONG" | "MET_ULONG_ARRAY" => Some(Self::ULong),
            "MET_LONG" | "MET_LONG_ARRAY" => Some(Self::Long),
            _ => None,
        }
    }

    fn byte_len(self) -> usize {
        match self {
            Self::UChar | Self::Char => 1,
            Self::UShort | Self::Short => 2,
            Self::UInt | Self::Int | Self::Float => 4,
            Self::ULong | Self::Long | Self::Double => 8,
        }
    }
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::UChar => "MET_UCHAR",
            Self::Char => "MET_CHAR",
            Self::UShort => "MET_USHORT",
            Self::Short => "MET_SHORT",
            Self::UInt => "MET_UINT",
            Self::Int => "MET_INT",
            Self::Float => "MET_FLOAT",
            Self::Double => "MET_DOUBLE",
            Self::ULong => "MET_ULONG",
            Self::Long => "MET_LONG",
        };
        write!(f, "{s}")
    }
}

/// Error type for MetaImage operations.
#[derive(Debug)]
pub enum MetaImageError {
    Io(io::Error),
    Parse(String),
    Unsupported(String),
    Shape(String),
}

impl Display for MetaImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::Shape(msg) => write!(f, "shape error: {msg}"),
        }
    }
}

impl Error for MetaImageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for MetaImageError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Trait implemented for element types that can be encoded in MetaImage files.
pub trait MetaElement: Clone + Send + Sync + 'static {
    const ELEMENT_TYPE: ElementType;
}

macro_rules! impl_meta_element {
    ($ty:ty, $elem:expr) => {
        impl MetaElement for $ty {
            const ELEMENT_TYPE: ElementType = $elem;
        }
    };
}

// One-byte integers do not depend on endianness.
impl MetaElement for u8 {
    const ELEMENT_TYPE: ElementType = ElementType::UChar;
}

impl MetaElement for i8 {
    const ELEMENT_TYPE: ElementType = ElementType::Char;
}

impl_meta_element!(u16, ElementType::UShort);
impl_meta_element!(i16, ElementType::Short);
impl_meta_element!(u32, ElementType::UInt);
impl_meta_element!(i32, ElementType::Int);
impl_meta_element!(u64, ElementType::ULong);
impl_meta_element!(i64, ElementType::Long);
impl_meta_element!(f32, ElementType::Float);
impl_meta_element!(f64, ElementType::Double);

/// Generic pixel container backed by ndarray.
#[derive(Debug, Clone)]
pub enum PixelData {
    U8(ArrayD<u8>),
    I8(ArrayD<i8>),
    U16(ArrayD<u16>),
    I16(ArrayD<i16>),
    U32(ArrayD<u32>),
    I32(ArrayD<i32>),
    U64(ArrayD<u64>),
    I64(ArrayD<i64>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

impl PixelData {
    pub fn element_type(&self) -> ElementType {
        match self {
            Self::U8(_) => ElementType::UChar,
            Self::I8(_) => ElementType::Char,
            Self::U16(_) => ElementType::UShort,
            Self::I16(_) => ElementType::Short,
            Self::U32(_) => ElementType::UInt,
            Self::I32(_) => ElementType::Int,
            Self::U64(_) => ElementType::ULong,
            Self::I64(_) => ElementType::Long,
            Self::F32(_) => ElementType::Float,
            Self::F64(_) => ElementType::Double,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::U8(arr) => arr.shape(),
            Self::I8(arr) => arr.shape(),
            Self::U16(arr) => arr.shape(),
            Self::I16(arr) => arr.shape(),
            Self::U32(arr) => arr.shape(),
            Self::I32(arr) => arr.shape(),
            Self::U64(arr) => arr.shape(),
            Self::I64(arr) => arr.shape(),
            Self::F32(arr) => arr.shape(),
            Self::F64(arr) => arr.shape(),
        }
    }

    fn _to_bytes<T: MetaElement>(arr: &'_ ArrayD<T>) -> Cow<'_, [u8]> {
        if let Some(slice) = arr.as_slice() {
            let byte_len = T::ELEMENT_TYPE.byte_len() * slice.len();
            let bytes =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
            Cow::Borrowed(bytes)
        } else {
            // convert to contiguous vec
            let raw: Vec<_> = arr.iter().collect();
            let len = raw.len() / T::ELEMENT_TYPE.byte_len();
            let boxed = raw.into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut u8;
            let values = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) };
            let values: Vec<u8> = values.into_vec();
            Cow::Owned(values)
        }
    }

    fn to_bytes(&'_ self, msb: bool) -> Cow<'_, [u8]> {
        if cfg!(target_endian = "big") {
            unimplemented!("big endian environment not supported yet");
        }
        if msb {
            unimplemented!("writing big endian data not supported yet");
        }

        match self {
            Self::U8(arr) => {
                if let Some(slice) = arr.as_slice() {
                    Cow::Borrowed(slice)
                } else {
                    Cow::Owned(arr.iter().cloned().collect())
                }
            }
            Self::I8(arr) => {
                if let Some(slice) = arr.as_slice() {
                    Cow::Borrowed(unsafe {
                        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len())
                    })
                } else {
                    Cow::Owned(arr.iter().map(|&v| v as u8).collect())
                }
            }
            Self::U16(arr) => Self::_to_bytes(arr),
            Self::I16(arr) => Self::_to_bytes(arr),
            Self::U32(arr) => Self::_to_bytes(arr),
            Self::I32(arr) => Self::_to_bytes(arr),
            Self::U64(arr) => Self::_to_bytes(arr),
            Self::I64(arr) => Self::_to_bytes(arr),
            Self::F32(arr) => Self::_to_bytes(arr),
            Self::F64(arr) => Self::_to_bytes(arr),
        }
    }

    fn from_bytes<T: MetaElement>(
        raw: Vec<u8>,
        shape: &[usize],
        msb: bool,
    ) -> Result<Self, MetaImageError>
    where
        Self: From<ArrayD<T>>,
    {
        let bytes_per_elem = std::mem::size_of::<T>();
        if raw.len() != bytes_per_elem * shape.iter().product::<usize>() {
            return Err(MetaImageError::Shape(
                "raw byte length does not match shape".into(),
            ));
        }

        #[cfg(target_endian = "little")]
        let values = if msb {
            unimplemented!("endianness conversion not implemented yet");
        } else {
            // TODO: Use into_raw_parts when 1.93.0 is released
            let len = raw.len() / bytes_per_elem;
            // SAFETY: Vec<u8> is properly sized, and T is Pod (MetaElement is only implemented for Pod types)
            let boxed = raw.into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut T;
            let values = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) };
            let values: Vec<T> = values.into_vec();
            values
        };

        #[cfg(target_endian = "big")]
        unimplemented!("endianness conversion not implemented yet");

        let array = ArrayD::from_shape_vec(IxDyn(shape), values)
            .map_err(|err| MetaImageError::Shape(format!("{err}")))?;
        Ok(Self::from(array))
    }
}

macro_rules! impl_from_arrayd {
    ($ty:ty, $variant:ident) => {
        impl From<ArrayD<$ty>> for PixelData {
            fn from(value: ArrayD<$ty>) -> Self {
                Self::$variant(value)
            }
        }
    };
}

impl_from_arrayd!(u8, U8);
impl_from_arrayd!(i8, I8);
impl_from_arrayd!(u16, U16);
impl_from_arrayd!(i16, I16);
impl_from_arrayd!(u32, U32);
impl_from_arrayd!(i32, I32);
impl_from_arrayd!(u64, U64);
impl_from_arrayd!(i64, I64);
impl_from_arrayd!(f32, F32);
impl_from_arrayd!(f64, F64);

#[derive(Debug, Clone)]
pub struct MetaData {
    pub dims: usize,
    pub dim_size: Vec<usize>,
    pub element_spacing: Vec<f64>,
    pub element_type: ElementType,
    pub element_byte_order_msb: bool,
    pub element_no_of_channels: usize,
    /// Number of Bytes to skip at the head of each data file
    pub header_size: isize,
    pub optional_tags: Vec<(String, String)>,
}

impl MetaData {
    fn into_compressed(mut self, compressed_size: usize) -> Self {
        self.optional_tags
            .push(("CompressedData".to_string(), "True".to_string()));
        self.optional_tags.push((
            "CompressedDataSize".to_string(),
            compressed_size.to_string(),
        ));
        self
    }
}

/// In-memory MetaImage representation.
#[derive(Debug, Clone)]
pub struct MetaImage {
    pub metadata: MetaData,
    pub data: PixelData,
}

#[derive(Debug, Clone)]
pub struct WriteOption {
    pub data_file: Option<String>,
    pub compress: bool,
}

impl WriteOption {
    /// Automatically determine write option based on path and pixel data.
    /// - If the pixel data is floating-point type, compression is disabled.
    /// - If the path has `.mhd` extension, data_file is set to the corresponding raw/zraw file name. Otherwise (`.mha`), data_file is None (inline).
    pub fn new_auto(path: &Path, data: &PixelData) -> Self {
        let is_float = matches!(
            data.element_type(),
            ElementType::Float | ElementType::Double
        );
        let compress = !is_float;
        let data_file = match path.extension().and_then(|s| s.to_str()) {
            Some("mha") | Some("MHA") => None,
            Some("mhd") | Some("MHD") => {
                let mut name = path
                    .file_stem()
                    .unwrap_or_else(|| std::ffi::OsStr::new("data"))
                    .to_os_string();
                if compress {
                    name.push(".zraw");
                } else {
                    name.push(".raw");
                }
                Some(name)
            }
            _ => None, // Fallback to None
        }
        .map(|s| s.to_string_lossy().into_owned());
        Self {
            data_file,
            compress,
        }
    }
}

impl MetaImage {
    /// Build a MetaImage from an ndarray of a supported element type.
    pub fn from_array<T: MetaElement>(array: ArrayD<T>) -> Self
    where
        PixelData: From<ArrayD<T>>,
    {
        let dim_size = array.shape().to_vec();
        let dims = dim_size.len();
        let element_spacing = vec![1.0; dims];
        let element_no_of_channels = 1;
        Self {
            metadata: MetaData {
                dims,
                dim_size,
                element_spacing,
                element_type: T::ELEMENT_TYPE,
                element_byte_order_msb: cfg!(target_endian = "big"),
                element_no_of_channels,
                header_size: 0,
                optional_tags: Vec::new(),
            },
            data: PixelData::from(array),
        }
    }

    /// [`MetaImage::from_array`] variant for vector/rgb images, where the last dimension is treated as channels.
    pub fn with_channels<T: MetaElement>(array: ArrayD<T>) -> Self
    where
        PixelData: From<ArrayD<T>>,
    {
        let dim_size = array.shape().to_vec();
        let dims = dim_size.len();
        let element_no_of_channels = if dims >= 1 { dim_size[dims - 1] } else { 1 };
        let dim_size = if dims >= 1 {
            dim_size[..dims - 1].to_vec()
        } else {
            vec![]
        };
        let dims = dim_size.len();
        let element_spacing = vec![1.0; dims];
        Self {
            metadata: MetaData {
                dims,
                dim_size,
                element_spacing,
                element_type: T::ELEMENT_TYPE,
                element_byte_order_msb: cfg!(target_endian = "big"),
                element_no_of_channels,
                header_size: 0,
                optional_tags: Vec::new(),
            },
            data: PixelData::from(array),
        }
    }

    /// Read a MetaImage from a header (.mhd) or combined (.mha) file.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, MetaImageError> {
        let path = path.as_ref();
        let file_bytes = fs::read(path)?;
        let (header, inline_offset) = parse_header(&file_bytes)?;

        let shape = if header.element_no_of_channels > 1 {
            let mut s = header.dim_size.clone();
            s.push(header.element_no_of_channels);
            s
        } else {
            header.dim_size.clone()
        };

        let element_count = shape
            .iter()
            .try_fold(1usize, |acc, v| acc.checked_mul(*v).ok_or("overflow"))
            .map_err(|_| MetaImageError::Shape("dim_size product overflow".into()))?;
        let expected_bytes = if let Some(compressed_size) = header.compressed_size {
            compressed_size
        } else {
            element_count
                .checked_mul(header.element_type.byte_len())
                .ok_or_else(|| MetaImageError::Shape("byte size overflow".into()))?
        };

        let mut raw_data = if header.element_data_file.eq_ignore_ascii_case("LOCAL") {
            let start = inline_offset
                .ok_or_else(|| MetaImageError::Parse("inline data offset missing".into()))?;
            if file_bytes.len() < start + expected_bytes {
                return Err(MetaImageError::Parse(
                    "inline data shorter than expected".into(),
                ));
            }
            file_bytes[start..start + expected_bytes].to_vec()
        } else {
            let data_path = resolve_data_path(path, &header.element_data_file);
            read_raw_data(&data_path, header.header_size, expected_bytes)?
        };
        if header.compressed_size.is_some() {
            let mut decoder = flate2::read::ZlibDecoder::new(&raw_data[..]);
            let mut decompressed_data = Vec::new();
            decoder.read_to_end(&mut decompressed_data)?;
            raw_data = decompressed_data;
        }

        let data = match header.element_type {
            ElementType::UChar => {
                PixelData::from_bytes::<u8>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::Char => {
                PixelData::from_bytes::<i8>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::UShort => {
                PixelData::from_bytes::<u16>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::Short => {
                PixelData::from_bytes::<i16>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::UInt => {
                PixelData::from_bytes::<u32>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::Int => {
                PixelData::from_bytes::<i32>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::ULong => {
                PixelData::from_bytes::<u64>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::Long => {
                PixelData::from_bytes::<i64>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::Float => {
                PixelData::from_bytes::<f32>(raw_data, &shape, header.element_byte_order_msb)?
            }
            ElementType::Double => {
                PixelData::from_bytes::<f64>(raw_data, &shape, header.element_byte_order_msb)?
            }
        };

        Ok(Self {
            metadata: MetaData {
                dims: header.dim_size.len(),
                dim_size: header.dim_size,
                element_spacing: header.element_spacing,
                element_type: header.element_type,
                element_byte_order_msb: header.element_byte_order_msb,
                element_no_of_channels: header.element_no_of_channels,
                header_size: header.header_size,
                optional_tags: header.optional_tags,
            },
            data,
        })
    }

    /// Write the MetaImage to a file, automatically choosing between the format (MHA or MHD) and compression options.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), MetaImageError> {
        let option = WriteOption::new_auto(path.as_ref(), &self.data);
        self.write_with_option(path, option)
    }

    /// Write with the specified write options.
    pub fn write_with_option(
        &self,
        path: impl AsRef<Path>,
        option: WriteOption,
    ) -> Result<(), MetaImageError> {
        if let Some(ref data_file_name) = option.data_file {
            self.write_mhd_with_option(path, data_file_name, &option)
        } else {
            self.write_mha_with_option(path, &option)
        }
    }

    fn write_mha_with_option(
        &self,
        path: impl AsRef<Path>,
        option: &WriteOption,
    ) -> Result<(), MetaImageError> {
        if option.compress {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&self.data.to_bytes(self.metadata.element_byte_order_msb))?;
            let compressed_data = encoder.finish()?;

            let mut header_file = File::create(path)?;
            let metadata = self.metadata.clone().into_compressed(compressed_data.len());

            write_header(&mut header_file, &metadata, "LOCAL")?;
            header_file.write_all(&compressed_data)?;
        } else {
            let path = path.as_ref();
            let mut file = File::create(path)?;
            write_header(&mut file, &self.metadata, "LOCAL")?;
            file.write_all(&self.data.to_bytes(self.metadata.element_byte_order_msb))?;
        }
        Ok(())
    }

    fn write_mhd_with_option(
        &self,
        header_path: impl AsRef<Path>,
        data_file_name: impl AsRef<Path>,
        option: &WriteOption,
    ) -> Result<(), MetaImageError> {
        use std::borrow::Cow;
        let header_path = header_path.as_ref();
        let data_file_name = data_file_name.as_ref();
        let bytes = self.data.to_bytes(self.metadata.element_byte_order_msb);
        let (metadata, data): (MetaData, Cow<[u8]>) = if option.compress {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&bytes)?;
            let compressed_data = encoder.finish()?;
            (
                self.metadata.clone().into_compressed(compressed_data.len()),
                Cow::Owned(compressed_data),
            )
        } else {
            (self.metadata.clone(), Cow::Borrowed(&bytes))
        };
        let mut header_file = File::create(header_path)?;
        write_header(
            &mut header_file,
            &metadata,
            &data_file_name.to_string_lossy(),
        )?;
        let mut data_file = File::create(
            header_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(data_file_name),
        )?;
        data_file.write_all(&data)?;
        Ok(())
    }
}

struct ParsedHeader {
    dim_size: Vec<usize>,
    element_spacing: Vec<f64>,
    element_type: ElementType,
    element_byte_order_msb: bool,
    element_no_of_channels: usize,
    header_size: isize,
    optional_tags: Vec<(String, String)>,
    compressed_size: Option<usize>,
    element_data_file: String,
}

fn parse_header(file_bytes: &[u8]) -> Result<(ParsedHeader, Option<usize>), MetaImageError> {
    let mut dim_size: Option<Vec<usize>> = None;
    let mut spacing: Option<Vec<f64>> = None;
    let mut element_type: Option<ElementType> = None;
    let mut header_size: isize = 0;
    let mut element_byte_order_msb = false;
    let mut element_no_of_channels: usize = 1;
    let mut optional_tags: Vec<(String, String)> = Vec::new();
    let mut compressed_size: Option<usize> = None;
    let mut element_data_file: Option<String> = None;
    let mut inline_offset: Option<usize> = None;

    let mut line_start = 0usize;
    for (idx, &byte) in file_bytes.iter().enumerate() {
        if byte != b'\n' {
            continue;
        }

        let line_bytes = &file_bytes[line_start..idx];
        line_start = idx + 1;
        let line = String::from_utf8_lossy(line_bytes).trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(MetaImageError::Parse(format!("invalid line: {line}")));
        };
        let key = key.trim();
        let value = value.trim();

        match key {
            "NDims" => {
                let dims: usize = value
                    .parse()
                    .map_err(|_| MetaImageError::Parse("NDims must be an integer".into()))?;
                if dims == 0 {
                    return Err(MetaImageError::Parse("NDims must be positive".into()));
                }
            }
            "DimSize" => {
                let parsed: Vec<usize> = parse_list(value)?;
                if parsed.is_empty() {
                    return Err(MetaImageError::Parse("DimSize cannot be empty".into()));
                }
                dim_size = Some(parsed);
            }
            "ElementSpacing" | "ElementSize" => {
                spacing = Some(parse_list(value)?);
            }
            "ElementType" => {
                element_type = ElementType::from_tag(value);
            }
            "ElementByteOrderMSB" => {
                element_byte_order_msb = parse_bool(value)?;
            }
            "ElementNumberOfChannels" => {
                element_no_of_channels = value.parse::<usize>().map_err(|_| {
                    MetaImageError::Parse("ElementNumberOfChannels must be an integer".into())
                })?;
                if element_no_of_channels == 0 {
                    return Err(MetaImageError::Parse(
                        "ElementNumberOfChannels must be positive".into(),
                    ));
                }
            }
            "BinaryDataByteOrderMSB" => {
                element_byte_order_msb = parse_bool(value)?;
            }
            "HeaderSize" => {
                header_size = value
                    .parse::<isize>()
                    .map_err(|_| MetaImageError::Parse("HeaderSize must be integer".into()))?;
            }
            "CompressedDataSize" => {
                compressed_size = Some(value.parse::<usize>().map_err(|_| {
                    MetaImageError::Parse("CompressedDataSize must be an integer".into())
                })?);
            }
            "ElementDataFile" => {
                element_data_file = Some(value.to_string());
                inline_offset = Some(line_start);
                break; // ElementDataFile is specified to be last.
            }
            _ => {
                optional_tags.push((key.to_string(), value.to_string()));
            }
        }
    }

    let dim_size = dim_size.ok_or_else(|| MetaImageError::Parse("DimSize missing".into()))?;
    let element_type =
        element_type.ok_or_else(|| MetaImageError::Parse("ElementType missing".into()))?;
    let element_data_file =
        element_data_file.ok_or_else(|| MetaImageError::Parse("ElementDataFile missing".into()))?;
    let element_spacing = spacing.unwrap_or_else(|| vec![1.0; dim_size.len()]);
    if element_spacing.len() != dim_size.len() {
        return Err(MetaImageError::Shape(
            "ElementSpacing length must match DimSize".into(),
        ));
    }

    Ok((
        ParsedHeader {
            dim_size,
            element_spacing,
            element_type,
            element_byte_order_msb,
            element_no_of_channels,
            header_size,
            optional_tags,
            compressed_size,
            element_data_file,
        },
        inline_offset,
    ))
}

fn parse_list<T>(raw: &str) -> Result<Vec<T>, MetaImageError>
where
    T: FromStr,
{
    raw.split_whitespace()
        .map(|part| {
            part.parse::<T>()
                .map_err(|_| MetaImageError::Parse(format!("failed to parse {part}")))
        })
        .collect()
}

fn parse_bool(raw: &str) -> Result<bool, MetaImageError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(MetaImageError::Parse(format!("invalid boolean: {raw}"))),
    }
}

fn resolve_data_path(header_path: &Path, data_file: &str) -> PathBuf {
    let data_path = Path::new(data_file);
    if data_path.is_absolute() {
        data_path.to_path_buf()
    } else {
        header_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(data_path)
    }
}

fn read_raw_data(
    path: &Path,
    header_size: isize,
    expected_bytes: usize,
) -> Result<Vec<u8>, MetaImageError> {
    let mut file = File::open(path)?;
    let meta = file.metadata()?;
    let file_len = meta.len();

    let skip = if header_size >= 0 {
        header_size as u64
    } else {
        let total_needed = expected_bytes as u64;
        if file_len < total_needed {
            return Err(MetaImageError::Parse(
                "data file smaller than expected".into(),
            ));
        }
        file_len - total_needed
    };

    file.seek(SeekFrom::Start(skip))?;
    let mut buf = vec![0u8; expected_bytes];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_header(
    writer: &mut impl Write,
    metadata: &MetaData,
    data_file: &str,
) -> Result<(), MetaImageError> {
    let dim_size_str = metadata
        .dim_size
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let spacing_str = metadata
        .element_spacing
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join(" ");

    writeln!(writer, "ObjectType = Image")?;
    writeln!(writer, "NDims = {}", metadata.dims)?;
    writeln!(writer, "DimSize = {dim_size_str}")?;
    writeln!(writer, "ElementType = {}", metadata.element_type)?;
    writeln!(writer, "ElementSpacing = {spacing_str}")?;
    writeln!(
        writer,
        "ElementByteOrderMSB = {}",
        if metadata.element_byte_order_msb {
            "True"
        } else {
            "False"
        }
    )?;
    writeln!(
        writer,
        "ElementNumberOfChannels = {}",
        metadata.element_no_of_channels
    )?;
    writeln!(writer, "BinaryData = True")?;
    writeln!(writer, "HeaderSize = {}", metadata.header_size)?;
    for (key, value) in &metadata.optional_tags {
        writeln!(writer, "{} = {}", key, value)?;
    }
    writeln!(writer, "ElementDataFile = {data_file}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_auto_option() {
        let data = PixelData::U16(ArrayD::zeros(IxDyn(&[10, 10])));
        let option = WriteOption::new_auto(Path::new("image.mha"), &data);
        assert!(option.data_file.is_none());
        assert!(option.compress);

        let option = WriteOption::new_auto(Path::new("image.mhd"), &data);
        assert_eq!(option.data_file.as_deref(), Some("image.zraw"));
        assert!(option.compress);

        let data = PixelData::F32(ArrayD::zeros(IxDyn(&[10, 10])));
        let option = WriteOption::new_auto(Path::new("image.mhd"), &data);
        assert_eq!(option.data_file.as_deref(), Some("image.raw"));
        assert!(!option.compress);
    }

    #[test]
    fn test_image_with_channels() {
        let array = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0u8..24).collect()).unwrap();
        let image = MetaImage::with_channels(array.clone());
        assert_eq!(image.metadata.dims, 2);
        assert_eq!(image.metadata.dim_size, vec![2, 3]);
        assert_eq!(image.metadata.element_no_of_channels, 4);

        match image.data {
            PixelData::U8(ref arr) => {
                assert_eq!(arr.shape(), &[2, 3, 4]);
                for (a, b) in arr.iter().zip(array.iter()) {
                    assert_eq!(*a, *b);
                }
            }
            _ => panic!("unexpected pixel data type"),
        }
    }

    #[test]
    fn test_optional_tags() {
        let image = MetaImage::from_array(ArrayD::<u8>::zeros(IxDyn(&[2, 2])));
        let mut metadata = image.metadata;
        metadata
            .optional_tags
            .push(("TestTag".to_string(), "TestValue".to_string()));
        metadata
            .optional_tags
            .push(("ListTag".to_string(), "1 2 3 4".to_string()));

        let mut buf: Vec<u8> = Vec::new();
        write_header(&mut buf, &metadata, "data.raw").unwrap();

        let parsed_header = parse_header(&buf).unwrap().0;
        let tag_entry = ("TestTag".to_string(), "TestValue".to_string());
        assert!(parsed_header.optional_tags.contains(&tag_entry));
        let list_entry = ("ListTag".to_string(), "1 2 3 4".to_string());
        assert!(parsed_header.optional_tags.contains(&list_entry));
    }
}
