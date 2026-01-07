#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
use bytemuck::NoUninit;
use ndarray::{ArrayD, IxDyn};
use std::borrow::Cow;
use std::error::Error;
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::vec;

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
    // Ommit ULong and Long because they are not u64/i64
    // ULong,
    // Long,
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
            _ => None,
        }
    }

    fn byte_len(self) -> usize {
        match self {
            Self::UChar | Self::Char => 1,
            Self::UShort | Self::Short => 2,
            Self::UInt | Self::Int | Self::Float => 4,
            Self::Double => 8,
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

impl_meta_element!(u8, ElementType::UChar);
impl_meta_element!(i8, ElementType::Char);
impl_meta_element!(u16, ElementType::UShort);
impl_meta_element!(i16, ElementType::Short);
impl_meta_element!(u32, ElementType::UInt);
impl_meta_element!(i32, ElementType::Int);
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
            Self::F32(arr) => arr.shape(),
            Self::F64(arr) => arr.shape(),
        }
    }

    fn _to_bytes<T: MetaElement + NoUninit>(arr: &'_ ArrayD<T>) -> Cow<'_, [u8]> {
        if let Some(slice) = arr.as_slice() {
            let bytes = bytemuck::must_cast_slice(slice);
            Cow::Borrowed(bytes)
        } else {
            // convert to contiguous vec
            let raw = arr
                .as_standard_layout()
                .to_owned()
                .into_raw_vec_and_offset()
                .0;
            let len = std::mem::size_of::<T>() * raw.len();
            let boxed = raw.into_boxed_slice();
            let ptr = Box::into_raw(boxed) as *mut u8;
            let values = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) };
            let values: Vec<u8> = values.into_vec();
            Cow::Owned(values)
        }
    }

    fn to_bytes(&'_ self, msb: bool) -> Cow<'_, [u8]> {
        let mut bytes = match self {
            Self::U8(arr) => {
                if let Some(slice) = arr.as_slice() {
                    Cow::Borrowed(slice)
                } else {
                    Cow::Owned(
                        arr.as_standard_layout()
                            .to_owned()
                            .into_raw_vec_and_offset()
                            .0,
                    )
                }
            }
            Self::I8(arr) => Self::_to_bytes(arr),
            Self::U16(arr) => Self::_to_bytes(arr),
            Self::I16(arr) => Self::_to_bytes(arr),
            Self::U32(arr) => Self::_to_bytes(arr),
            Self::I32(arr) => Self::_to_bytes(arr),
            Self::F32(arr) => Self::_to_bytes(arr),
            Self::F64(arr) => Self::_to_bytes(arr),
        };
        if !is_native_endianness(msb) {
            let byte_len = match self {
                Self::U8(_) | Self::I8(_) => 1,
                Self::U16(_) | Self::I16(_) => 2,
                Self::U32(_) | Self::I32(_) | Self::F32(_) => 4,
                Self::F64(_) => 8,
            };
            match &mut bytes {
                Cow::Borrowed(slice) => {
                    let mut owned = slice.to_vec();
                    for chunk in owned.chunks_exact_mut(byte_len) {
                        chunk.reverse();
                    }
                    bytes = Cow::Owned(owned);
                }
                Cow::Owned(vec) => {
                    for chunk in vec.chunks_exact_mut(byte_len) {
                        chunk.reverse();
                    }
                }
            }
        }
        bytes
    }
}

macro_rules! impl_into_array {
    ($into_name:ident, $as_name:ident,$ty:ty, $variant:ident) => {
        impl PixelData {
            /// Extract inner array if the variant matches, otherwise returns None.
            pub fn $into_name(self) -> Option<ArrayD<$ty>> {
                match self {
                    Self::$variant(arr) => Some(arr),
                    _ => None,
                }
            }
            pub fn $as_name(&self) -> Option<&ArrayD<$ty>> {
                match self {
                    Self::$variant(arr) => Some(arr),
                    _ => None,
                }
            }
        }
        impl From<PixelData> for Option<ArrayD<$ty>> {
            /// Extract inner array if the variant matches, otherwise returns None.
            fn from(val: PixelData) -> Self {
                match val {
                    PixelData::$variant(arr) => Some(arr),
                    _ => None,
                }
            }
        }
    };
}

impl_into_array!(into_u8_array, as_u8_array, u8, U8);
impl_into_array!(into_i8_array, as_i8_array, i8, I8);
impl_into_array!(into_u16_array, as_u16_array, u16, U16);
impl_into_array!(into_i16_array, as_i16_array, i16, I16);
impl_into_array!(into_u32_array, as_u32_array, u32, U32);
impl_into_array!(into_i32_array, as_i32_array, i32, I32);
impl_into_array!(into_f32_array, as_f32_array, f32, F32);
impl_into_array!(into_f64_array, as_f64_array, f64, F64);

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
        fn update_or_add(tags: &mut Vec<(String, String)>, key: &str, value: String) {
            if let Some(idx) = tags.iter().position(|(k, _)| k == key) {
                tags[idx].1 = value;
            } else {
                tags.push((key.to_string(), value));
            }
        }
        update_or_add(
            &mut self.optional_tags,
            "CompressedData",
            "True".to_string(),
        );
        update_or_add(
            &mut self.optional_tags,
            "CompressedDataSize",
            compressed_size.to_string(),
        );
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
pub struct WriteOptions {
    pub data_file: Option<String>,
    pub compress: Option<u32>,
}

impl WriteOptions {
    /// Automatically determine write option based on path and pixel data.
    /// - If the pixel data is floating-point type, compression is disabled.
    /// - If the path has `.mhd` extension, data_file is set to the corresponding raw/zraw file name. Otherwise(`.mha` and others), data_file is None and the image is written as a single file.
    pub fn new_auto(path: &Path, data: &PixelData) -> Self {
        let is_float = matches!(
            data.element_type(),
            ElementType::Float | ElementType::Double
        );
        let compress = if is_float { None } else { Some(6) };
        let data_file = match path.extension().and_then(|s| s.to_str()) {
            Some("mha") | Some("MHA") => None,
            Some("mhd") | Some("MHD") => {
                let mut name = path
                    .file_stem()
                    .unwrap_or_else(|| std::ffi::OsStr::new("data"))
                    .to_os_string();
                if compress.is_some() {
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

fn to_u8_slice<T>(slice: &mut [T]) -> &mut [u8] {
    let byte_len = std::mem::size_of_val(slice);
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), byte_len) }
}

fn is_native_endianness(msb: bool) -> bool {
    #[allow(clippy::needless_bool)]
    if msb {
        cfg!(target_endian = "big")
    } else {
        cfg!(target_endian = "little")
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

    fn typed_read<T: Default + Clone>(
        mut reader: BufReader<File>,
        len_to_read: usize,
    ) -> io::Result<Vec<T>> {
        // https://users.rust-lang.org/t/how-best-to-convert-u8-to-u16/57551/2
        let len_to_read = len_to_read as u64;
        let len = if len_to_read.is_multiple_of(std::mem::size_of::<T>() as u64) {
            usize::try_from(len_to_read / std::mem::size_of::<T>() as u64)
                .map_err(|_| io::Error::other("File is too large"))?
        } else {
            return Err(io::Error::other("Length is odd"));
        };

        let mut vec: Vec<T> = vec![T::default(); len];

        let slice: &mut [u8] = to_u8_slice(&mut vec);

        reader.read_exact(slice)?;
        Ok(vec)
    }

    /// Read a MetaImage from a header (.mhd) or combined (.mha) file.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, MetaImageError> {
        let path = path.as_ref();
        let mut reader = BufReader::new(File::open(path)?);
        let (header, inline_offset) = parse_header(&mut reader)?;

        fn read_pixel_data<T: MetaElement + Default + Clone>(
            header: &ParsedHeader,
            reader: BufReader<File>,
            path: &Path,
            inline_offset: Option<usize>,
        ) -> Result<PixelData, MetaImageError>
        where
            PixelData: From<ArrayD<T>>,
        {
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
            let mut raw_data = if let Some(compressed_size) = header.compressed_size {
                let mut buf = vec![0u8; compressed_size];
                let mut reader = reader;
                if header.element_data_file.eq_ignore_ascii_case("LOCAL") {
                    inline_offset.ok_or_else(|| {
                        MetaImageError::Parse("inline data offset missing".into())
                    })?;
                    reader.read_exact(&mut buf)?;
                } else {
                    let data_path = resolve_data_path(path, &header.element_data_file);
                    let mut data_file = File::open(data_path)?;
                    if header.header_size > 0 {
                        data_file.seek(SeekFrom::Start(header.header_size as u64))?;
                    }
                    let mut data_reader = BufReader::new(data_file);
                    data_reader.read_exact(&mut buf)?;
                }
                let mut decoder = flate2::read::ZlibDecoder::new(&buf[..]);
                let mut decompressed_data: Vec<T> = vec![T::default(); element_count];
                decoder.read_exact(to_u8_slice(&mut decompressed_data))?;
                decompressed_data
            } else {
                let expected_bytes = element_count
                    .checked_mul(header.element_type.byte_len())
                    .ok_or_else(|| MetaImageError::Shape("byte size overflow".into()))?;

                if header.element_data_file.eq_ignore_ascii_case("LOCAL") {
                    inline_offset.ok_or_else(|| {
                        MetaImageError::Parse("inline data offset missing".into())
                    })?;
                    MetaImage::typed_read::<T>(reader, expected_bytes)?
                } else {
                    let data_path = resolve_data_path(path, &header.element_data_file);
                    let data_file = File::open(data_path)?;
                    let mut data_reader = BufReader::new(data_file);
                    if header.header_size > 0 {
                        data_reader.seek(SeekFrom::Start(header.header_size as u64))?;
                    }
                    MetaImage::typed_read::<T>(data_reader, expected_bytes)?
                }
            };
            if !is_native_endianness(header.element_byte_order_msb) {
                change_endian(&mut raw_data);
            }
            let array = ArrayD::from_shape_vec(IxDyn(&shape), raw_data)
                .map_err(|err| MetaImageError::Shape(format!("{err}")))?;
            Ok(PixelData::from(array))
        }

        let data = match header.element_type {
            ElementType::UChar => read_pixel_data::<u8>(&header, reader, path, inline_offset)?,
            ElementType::Char => read_pixel_data::<i8>(&header, reader, path, inline_offset)?,
            ElementType::UShort => read_pixel_data::<u16>(&header, reader, path, inline_offset)?,
            ElementType::Short => read_pixel_data::<i16>(&header, reader, path, inline_offset)?,
            ElementType::UInt => read_pixel_data::<u32>(&header, reader, path, inline_offset)?,
            ElementType::Int => read_pixel_data::<i32>(&header, reader, path, inline_offset)?,
            ElementType::Float => read_pixel_data::<f32>(&header, reader, path, inline_offset)?,
            ElementType::Double => read_pixel_data::<f64>(&header, reader, path, inline_offset)?,
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
    /// See [`WriteOptions::new_auto`] for the logic.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), MetaImageError> {
        let option = WriteOptions::new_auto(path.as_ref(), &self.data);
        self.write_with_option(path, option)
    }

    /// Write with the specified write options.
    pub fn write_with_option(
        &self,
        path: impl AsRef<Path>,
        option: WriteOptions,
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
        option: &WriteOptions,
    ) -> Result<(), MetaImageError> {
        if let Some(level) = option.compress {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
            encoder.write_all(&self.data.to_bytes(self.metadata.element_byte_order_msb))?;
            let compressed_data = encoder.finish()?;

            let metadata = self.metadata.clone().into_compressed(compressed_data.len());

            let mut writer = BufWriter::new(File::create(path)?);
            write_header(&mut writer, &metadata, "LOCAL")?;
            writer.write_all(&compressed_data)?;
        } else {
            let path = path.as_ref();
            let mut writer = BufWriter::new(File::create(path)?);
            write_header(&mut writer, &self.metadata, "LOCAL")?;
            writer.write_all(&self.data.to_bytes(self.metadata.element_byte_order_msb))?;
        }
        Ok(())
    }

    fn write_mhd_with_option(
        &self,
        header_path: impl AsRef<Path>,
        data_file_name: impl AsRef<Path>,
        option: &WriteOptions,
    ) -> Result<(), MetaImageError> {
        use std::borrow::Cow;
        let header_path = header_path.as_ref();
        let data_file_name = data_file_name.as_ref();
        let bytes = self.data.to_bytes(self.metadata.element_byte_order_msb);
        let (metadata, data): (MetaData, Cow<[u8]>) = if let Some(level) = option.compress {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
            encoder.write_all(&bytes)?;
            let compressed_data = encoder.finish()?;
            (
                self.metadata.clone().into_compressed(compressed_data.len()),
                Cow::Owned(compressed_data),
            )
        } else {
            (self.metadata.clone(), Cow::Borrowed(&bytes))
        };
        let mut writer = BufWriter::new(File::create(header_path)?);
        write_header(&mut writer, &metadata, &data_file_name.to_string_lossy())?;
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

fn change_endian<T>(slice: &mut [T]) {
    let byte_len = std::mem::size_of::<T>();
    let u8_slice = to_u8_slice(slice);
    for chunk in u8_slice.chunks_exact_mut(byte_len) {
        chunk.reverse();
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

fn parse_header<R: BufRead>(
    reader: &mut R,
) -> Result<(ParsedHeader, Option<usize>), MetaImageError> {
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

    let mut bytes_read = 0usize;
    let mut line_buffer = String::new();

    loop {
        line_buffer.clear();
        let bytes = reader.read_line(&mut line_buffer)?;
        if bytes == 0 {
            break; // EOF
        }
        bytes_read += bytes;

        let line = line_buffer.trim();
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
                inline_offset = Some(bytes_read);
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
        let option = WriteOptions::new_auto(Path::new("image.mha"), &data);
        assert!(option.data_file.is_none());
        assert!(option.compress.is_some());

        let option = WriteOptions::new_auto(Path::new("image.mhd"), &data);
        assert_eq!(option.data_file.as_deref(), Some("image.zraw"));
        assert!(option.compress.is_some());

        let data = PixelData::F32(ArrayD::zeros(IxDyn(&[10, 10])));
        let option = WriteOptions::new_auto(Path::new("image.mhd"), &data);
        assert_eq!(option.data_file.as_deref(), Some("image.raw"));
        assert!(option.compress.is_none());
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
    fn test_change_endian() {
        // u16
        let mut data: Vec<u16> = vec![0x1234, 0xABCD, 0x0F0F];
        change_endian(&mut data);
        assert_eq!(data, vec![0x3412, 0xCDAB, 0x0F0F]);
        // u32
        let mut data: Vec<u32> = vec![0x12345678, 0xABCDEF01];
        change_endian(&mut data);
        assert_eq!(data, vec![0x78563412, 0x01EFCDAB]);
        // f32
        let mut data: Vec<f32> = vec![1.0, -2.0, 3.0];
        let expected: Vec<f32> = data
            .iter()
            .map(|v| f32::from_bits(v.to_bits().swap_bytes()))
            .collect();
        change_endian(&mut data);
        assert_eq!(data, expected);
        // f64
        let mut data: Vec<f64> = vec![1.0, -2.0, 3.0];
        let expected: Vec<f64> = data
            .iter()
            .map(|v| f64::from_bits(v.to_bits().swap_bytes()))
            .collect();
        change_endian(&mut data);
        assert_eq!(data, expected);
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

        let cursor = io::Cursor::new(buf.clone());
        let mut reader = BufReader::new(cursor);
        let parsed_header = parse_header(&mut reader).unwrap().0;
        let tag_entry = ("TestTag".to_string(), "TestValue".to_string());
        assert!(parsed_header.optional_tags.contains(&tag_entry));
        let list_entry = ("ListTag".to_string(), "1 2 3 4".to_string());
        assert!(parsed_header.optional_tags.contains(&list_entry));
    }
}
