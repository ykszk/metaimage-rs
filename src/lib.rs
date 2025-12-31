use ndarray::{ArrayD, IxDyn};
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
    fn from_le(bytes: &[u8]) -> Self;
    fn from_be(bytes: &[u8]) -> Self;
    fn to_le_bytes(value: &Self) -> Vec<u8>;
    fn to_be_bytes(value: &Self) -> Vec<u8>;
}

macro_rules! impl_meta_element {
    ($ty:ty, $elem:expr) => {
        impl MetaElement for $ty {
            const ELEMENT_TYPE: ElementType = $elem;
            fn from_le(bytes: &[u8]) -> Self {
                <$ty>::from_le_bytes(bytes.try_into().expect("slice length checked"))
            }
            fn from_be(bytes: &[u8]) -> Self {
                <$ty>::from_be_bytes(bytes.try_into().expect("slice length checked"))
            }
            fn to_le_bytes(value: &Self) -> Vec<u8> {
                <$ty>::to_le_bytes(*value).to_vec()
            }
            fn to_be_bytes(value: &Self) -> Vec<u8> {
                <$ty>::to_be_bytes(*value).to_vec()
            }
        }
    };
}

// One-byte integers do not depend on endianness.
impl MetaElement for u8 {
    const ELEMENT_TYPE: ElementType = ElementType::UChar;
    fn from_le(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn from_be(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn to_le_bytes(value: &Self) -> Vec<u8> {
        vec![*value]
    }
    fn to_be_bytes(value: &Self) -> Vec<u8> {
        vec![*value]
    }
}

impl MetaElement for i8 {
    const ELEMENT_TYPE: ElementType = ElementType::Char;
    fn from_le(bytes: &[u8]) -> Self {
        bytes[0] as i8
    }
    fn from_be(bytes: &[u8]) -> Self {
        bytes[0] as i8
    }
    fn to_le_bytes(value: &Self) -> Vec<u8> {
        vec![*value as u8]
    }
    fn to_be_bytes(value: &Self) -> Vec<u8> {
        vec![*value as u8]
    }
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
    #[allow(dead_code)]
    fn element_type(&self) -> ElementType {
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

    #[allow(dead_code)]
    fn shape(&self) -> &[usize] {
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

    #[allow(dead_code)]
    fn element_count(&self) -> usize {
        match self {
            Self::U8(arr) => arr.len(),
            Self::I8(arr) => arr.len(),
            Self::U16(arr) => arr.len(),
            Self::I16(arr) => arr.len(),
            Self::U32(arr) => arr.len(),
            Self::I32(arr) => arr.len(),
            Self::U64(arr) => arr.len(),
            Self::I64(arr) => arr.len(),
            Self::F32(arr) => arr.len(),
            Self::F64(arr) => arr.len(),
        }
    }

    fn to_bytes(&self, msb: bool) -> Vec<u8> {
        match self {
            Self::U8(arr) => arr.iter().copied().collect(),
            Self::I8(arr) => arr.iter().map(|v| *v as u8).collect(),
            Self::U16(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
            Self::I16(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
            Self::U32(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
            Self::I32(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
            Self::U64(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
            Self::I64(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
            Self::F32(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
            Self::F64(arr) => arr
                .iter()
                .flat_map(|v| {
                    if msb {
                        v.to_be_bytes()
                    } else {
                        v.to_le_bytes()
                    }
                })
                .collect(),
        }
    }

    pub fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u16>> {
        let mut file = File::open(path)?;

        let len = file.metadata()?.len();
        let len = if len % 2 == 0 {
            usize::try_from(len / 2)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "File is too large"))?
        } else {
            return Err(io::Error::new(io::ErrorKind::Other, "Length is odd"));
        };

        let mut vec = vec![0u16; len];

        let slice: &mut [u8] = Self::to_u8_slice(&mut vec);

        file.read_exact(slice)?;
        Ok(vec)
    }

    fn to_u8_slice(slice: &mut [u16]) -> &mut [u8] {
        let byte_len = 2 * slice.len();
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), byte_len) }
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

/// In-memory MetaImage representation.
#[derive(Debug, Clone)]
pub struct MetaImage {
    pub dims: usize,
    pub dim_size: Vec<usize>,
    pub element_spacing: Vec<f64>,
    pub element_type: ElementType,
    pub element_byte_order_msb: bool,
    pub header_size: isize,
    pub optional_tags: Vec<(String, String)>,
    pub data: PixelData,
}

#[derive(Debug, Clone)]
pub struct WriteOption {
    pub use_compression: bool,
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
        Self {
            dims,
            dim_size,
            element_spacing,
            element_type: T::ELEMENT_TYPE,
            element_byte_order_msb: cfg!(target_endian = "big"),
            header_size: 0,
            optional_tags: Vec::new(),
            data: PixelData::from(array),
        }
    }

    /// Read a MetaImage from a header (.mhd) or combined (.mha) file.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, MetaImageError> {
        let path = path.as_ref();
        let file_bytes = fs::read(path)?;
        let (header, inline_offset) = parse_header(&file_bytes)?;

        let element_count = header
            .dim_size
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
            ElementType::UChar => PixelData::from_bytes::<u8>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::Char => PixelData::from_bytes::<i8>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::UShort => PixelData::from_bytes::<u16>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::Short => PixelData::from_bytes::<i16>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::UInt => PixelData::from_bytes::<u32>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::Int => PixelData::from_bytes::<i32>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::ULong => PixelData::from_bytes::<u64>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::Long => PixelData::from_bytes::<i64>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::Float => PixelData::from_bytes::<f32>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
            ElementType::Double => PixelData::from_bytes::<f64>(
                raw_data,
                &header.dim_size,
                header.element_byte_order_msb,
            )?,
        };

        Ok(Self {
            dims: header.dim_size.len(),
            dim_size: header.dim_size,
            element_spacing: header.element_spacing,
            element_type: header.element_type,
            element_byte_order_msb: header.element_byte_order_msb,
            header_size: header.header_size,
            optional_tags: header.optional_tags,
            data,
        })
    }

    /// Write a combined MetaImage file (.mha).
    pub fn write_mha(&self, path: impl AsRef<Path>) -> Result<(), MetaImageError> {
        let path = path.as_ref();
        let mut file = File::create(path)?;
        write_header(&mut file, self, "LOCAL")?;
        file.write_all(&self.data.to_bytes(self.element_byte_order_msb))?;
        Ok(())
    }

    /// Write a separate header (.mhd) and raw data file.
    pub fn write_mhd_with_data_file_path(
        &self,
        header_path: impl AsRef<Path>,
        data_file_name: impl AsRef<Path>,
    ) -> Result<(), MetaImageError> {
        let header_path = header_path.as_ref();
        let data_file_name = data_file_name.as_ref();
        let mut header_file = File::create(header_path)?;
        write_header(&mut header_file, self, &data_file_name.to_string_lossy())?;

        let mut data_file = File::create(
            header_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(data_file_name),
        )?;
        data_file.write_all(&self.data.to_bytes(self.element_byte_order_msb))?;
        Ok(())
    }

    pub fn write_mhd(&self, header_path: impl AsRef<Path>) -> Result<(), MetaImageError> {
        let mut name = header_path
            .as_ref()
            .file_stem()
            .unwrap_or_else(|| std::ffi::OsStr::new("data"))
            .to_os_string();
        name.push(".raw");
        let data_file_path = PathBuf::from(name);
        self.write_mhd_with_data_file_path(header_path, data_file_path)
    }

    pub fn write_with_option(
        &self,
        path: impl AsRef<Path>,
        option: WriteOption,
    ) -> Result<(), MetaImageError> {
        if option.use_compression {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&self.data.to_bytes(self.element_byte_order_msb))?;
            let compressed_data = encoder.finish()?;

            let mut header_file = File::create("compressed_image.mha")?;
            write_header(&mut header_file, self, "LOCAL")?;
            header_file.write_all(&compressed_data)?;
            Ok(())
        } else {
            self.write_mha(path)
        }
    }
}

struct ParsedHeader {
    dim_size: Vec<usize>,
    element_spacing: Vec<f64>,
    element_type: ElementType,
    element_byte_order_msb: bool,
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
    image: &MetaImage,
    data_file: &str,
) -> Result<(), MetaImageError> {
    let dim_size_str = image
        .dim_size
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let spacing_str = image
        .element_spacing
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join(" ");

    writeln!(writer, "ObjectType = Image")?;
    writeln!(writer, "NDims = {}", image.dims)?;
    writeln!(writer, "DimSize = {dim_size_str}")?;
    writeln!(
        writer,
        "ElementType = {}",
        format_element_type(image.element_type)
    )?;
    writeln!(writer, "ElementSpacing = {spacing_str}")?;
    writeln!(
        writer,
        "ElementByteOrderMSB = {}",
        if image.element_byte_order_msb {
            "True"
        } else {
            "False"
        }
    )?;
    writeln!(writer, "BinaryData = True")?;
    writeln!(writer, "HeaderSize = {}", image.header_size)?;
    writeln!(writer, "ElementDataFile = {data_file}")?;
    Ok(())
}

fn format_element_type(element_type: ElementType) -> &'static str {
    match element_type {
        ElementType::UChar => "MET_UCHAR",
        ElementType::Char => "MET_CHAR",
        ElementType::UShort => "MET_USHORT",
        ElementType::Short => "MET_SHORT",
        ElementType::UInt => "MET_UINT",
        ElementType::Int => "MET_INT",
        ElementType::ULong => "MET_ULONG",
        ElementType::Long => "MET_LONG",
        ElementType::Float => "MET_FLOAT",
        ElementType::Double => "MET_DOUBLE",
    }
}
