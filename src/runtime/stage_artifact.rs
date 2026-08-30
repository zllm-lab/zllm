//! 分段运行时在节点之间传递 hidden state 的顺序二进制格式。

use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const MAGIC: &[u8; 8] = b"ZLLMSTG1";
const VERSION: u32 = 1;
const DTYPE_BF16: u32 = 2;

pub struct StageArtifactHeader {
    model: [u8; 16],
    pub layer_start: usize,
    pub layer_end: usize,
    pub token_count: usize,
    pub hidden_size: usize,
}

impl StageArtifactHeader {
    pub fn new(model: &str, layer_start: usize, layer_end: usize, token_count: usize, hidden_size: usize) -> io::Result<Self> {
        if model.len() > 16 || layer_start >= layer_end || token_count == 0 || hidden_size == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "stage artifact header 参数非法"));
        }
        let mut model_id = [0; 16];
        model_id[..model.len()].copy_from_slice(model.as_bytes());
        Ok(Self { model: model_id, layer_start, layer_end, token_count, hidden_size })
    }
}

pub struct StageArtifactWriter {
    output_path: PathBuf,
    temporary_path: PathBuf,
    output: BufWriter<File>,
    header: StageArtifactHeader,
    next_position: usize,
}

impl StageArtifactWriter {
    pub fn create(path: &Path, header: StageArtifactHeader) -> io::Result<Self> {
        let temporary_path = path.with_extension(path.extension().map(|extension| format!("{}.part", extension.to_string_lossy())).unwrap_or_else(|| "part".to_owned()));
        let mut output = BufWriter::new(File::create(&temporary_path)?);
        let payload_bytes = header.token_count.checked_mul(header.hidden_size).and_then(|elements| elements.checked_mul(size_of::<u16>())).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "stage artifact payload 大小溢出"))?;
        output.write_all(MAGIC)?;
        output.write_all(&VERSION.to_le_bytes())?;
        output.write_all(&DTYPE_BF16.to_le_bytes())?;
        output.write_all(&(header.layer_start as u32).to_le_bytes())?;
        output.write_all(&(header.layer_end as u32).to_le_bytes())?;
        output.write_all(&(header.hidden_size as u32).to_le_bytes())?;
        output.write_all(&0_u32.to_le_bytes())?;
        output.write_all(&(header.token_count as u64).to_le_bytes())?;
        output.write_all(&(payload_bytes as u64).to_le_bytes())?;
        output.write_all(&header.model)?;
        Ok(Self { output_path: path.to_owned(), temporary_path, output, header, next_position: 0 })
    }

    pub fn write_bf16_chunk(&mut self, position: usize, rows: usize, values: &[u16]) -> io::Result<()> {
        if position != self.next_position || values.len() != rows * self.header.hidden_size {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("stage artifact chunk 非连续或形状错误: position={position} expected={} rows={rows} values={}", self.next_position, values.len())));
        }
        #[cfg(target_endian = "little")]
        {
            // BF16 payload 固定为 little-endian；当前 ROCm 服务节点均为 little-endian。
            let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
            self.output.write_all(bytes)?;
        }
        #[cfg(target_endian = "big")]
        for value in values {
            self.output.write_all(&value.to_le_bytes())?;
        }
        self.next_position += rows;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<PathBuf> {
        if self.next_position != self.header.token_count {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("stage artifact token 数不足: written={} expected={}", self.next_position, self.header.token_count)));
        }
        self.output.flush()?;
        drop(self.output);
        std::fs::rename(&self.temporary_path, &self.output_path)?;
        Ok(self.output_path)
    }
}

pub struct StageArtifactReader {
    input: BufReader<File>,
    pub header: StageArtifactHeader,
    next_position: usize,
}

impl StageArtifactReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_bytes = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0_u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "stage artifact magic 非法"));
        }
        let mut raw = [0_u8; 56];
        input.read_exact(&mut raw)?;
        let u32_at = |offset| u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
        let u64_at = |offset| u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
        if u32_at(0) != VERSION || u32_at(4) != DTYPE_BF16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "stage artifact version/dtype 不受支持"));
        }
        let layer_start = u32_at(8) as usize;
        let layer_end = u32_at(12) as usize;
        let hidden_size = u32_at(16) as usize;
        let token_count = usize::try_from(u64_at(24)).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "stage token_count 超出 usize"))?;
        let payload_bytes = usize::try_from(u64_at(32)).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "stage payload_bytes 超出 usize"))?;
        let expected_payload = token_count.checked_mul(hidden_size).and_then(|elements| elements.checked_mul(size_of::<u16>())).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "stage payload 大小溢出"))?;
        if payload_bytes != expected_payload || file_bytes != 64 + payload_bytes as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("stage artifact 大小非法: file={file_bytes} payload={payload_bytes} expected={expected_payload}",)));
        }
        let mut model = [0_u8; 16];
        model.copy_from_slice(&raw[40..56]);
        let header = StageArtifactHeader { model, layer_start, layer_end, token_count, hidden_size };
        Ok(Self { input, header, next_position: 0 })
    }

    pub fn read_all_bf16(&mut self) -> io::Result<Vec<u16>> {
        if self.next_position != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "stage artifact 已读取"));
        }
        let elements = self.header.token_count * self.header.hidden_size;
        let mut values = vec![0_u16; elements];
        #[cfg(target_endian = "little")]
        {
            let bytes = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), elements * size_of::<u16>()) };
            self.input.read_exact(bytes)?;
        }
        #[cfg(target_endian = "big")]
        {
            let mut bytes = [0_u8; 2];
            for value in &mut values {
                self.input.read_exact(&mut bytes)?;
                *value = u16::from_le_bytes(bytes);
            }
        }
        self.next_position = self.header.token_count;
        Ok(values)
    }
}
