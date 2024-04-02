use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn discard_read(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    mut bytes: u64,
) -> Result<(), std::io::Error> {
    let mut buf = vec![0; 1024.min(bytes) as usize];
    while bytes > 0 {
        let bytes_read = reader.read(&mut buf).await?;
        bytes -= bytes_read as u64;
    }
    Ok(())
}

async fn discard_remaining_segments(
    mut source: impl tokio::io::AsyncRead + Unpin,
    did_read_length: Option<u64>,
    did_read_bytes: Option<u64>,
    remaining_segments: u32,
) -> Result<(), tokio::io::Error> {
    if let Some(length) = did_read_length {
        if let Some(bytes) = did_read_bytes {
            discard_read(&mut source, length - bytes).await?;
        } else {
            discard_read(&mut source, length).await?;
        }
    }
    for _ in 0..(remaining_segments - 1) {
        let length = source.read_u64().await?;
        discard_read(&mut source, length).await?;
    }
    Ok(())
}

pub async fn discard_additional_segments(
    source: impl tokio::io::AsyncRead + Unpin,
    num_additional_segments: u32,
) -> Result<(), tokio::io::Error> {
    discard_remaining_segments(source, None, None, num_additional_segments).await
}

pub async fn write_additional_segments<
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
>(
    num_additional_segments: u32,
    mut reader: R,
    mut writer: W,
) -> Result<(), tokio::io::Error> {
    for i in 0..num_additional_segments {
        let length = reader.read_u64().await?;

        if let Err(e) = writer.write_u64(length).await {
            discard_remaining_segments(reader, Some(length), None, num_additional_segments - i)
                .await?;
            return Err(e);
        }

        let mut buf = vec![0; (4096 * 4).min(length as usize)];

        let mut pos = 0;
        while pos < length {
            let bytes_read = reader.read(&mut buf).await?;
            pos += bytes_read as u64;
            if let Err(e) = writer.write_all(&buf[0..bytes_read]).await {
                discard_remaining_segments(
                    reader,
                    Some(length),
                    Some(pos),
                    num_additional_segments - i,
                )
                .await?;
                return Err(e);
            }
        }
    }
    Ok(())
}
