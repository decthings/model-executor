use futures::future::BoxFuture;
use std::{
    pin::Pin,
    sync::Arc,
    task::{ready, Poll},
};
use tokio::io::AsyncReadExt;

#[auto_impl::auto_impl(Box, &mut)]
pub trait AdditionalSegments {
    fn amount(&self) -> u32;

    fn remaining(&self) -> u32;

    #[allow(clippy::type_complexity)]
    fn next<'a>(
        &'a mut self,
    ) -> BoxFuture<'a, tokio::io::Result<Option<(u64, Pin<Box<dyn tokio::io::AsyncRead + 'a>>)>>>;
}

pub struct VecAdditionalSegments<S: AsRef<[u8]> + Send> {
    data: Vec<S>,
    index: usize,
}

impl<S: AsRef<[u8]> + Send> VecAdditionalSegments<S> {
    pub fn new(data: Vec<S>) -> Self {
        Self { data, index: 0 }
    }
}

impl<S: AsRef<[u8]> + Send> AdditionalSegments for VecAdditionalSegments<S> {
    fn amount(&self) -> u32 {
        self.data.len().try_into().unwrap()
    }

    fn remaining(&self) -> u32 {
        (self.data.len() - self.index).try_into().unwrap()
    }

    fn next<'a>(
        &'a mut self,
    ) -> BoxFuture<'a, tokio::io::Result<Option<(u64, Pin<Box<dyn tokio::io::AsyncRead + 'a>>)>>>
    {
        Box::pin(async move {
            let Some(data) = self.data.get(self.index) else {
                return Ok(None);
            };
            self.index += 1;
            Ok(Some((
                data.as_ref().len() as u64,
                Box::pin(data.as_ref()) as Pin<Box<dyn tokio::io::AsyncRead>>,
            )))
        })
    }
}

struct Segment {
    size: [u8; 8],
    size_bytes_read: usize,
    remaining: Option<u64>,
}

#[pin_project::pin_project]
struct SegmentFromSocketReader<'a, R: tokio::io::AsyncRead + Unpin> {
    #[pin]
    reader: R,
    index: usize,
    segments: tokio::sync::MutexGuard<'a, Vec<Segment>>,
    did_err: Arc<std::sync::Mutex<bool>>,
}

impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for SegmentFromSocketReader<'_, R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let remaining = self.segments[self.index].remaining.unwrap();
        if remaining == 0 {
            return std::task::Poll::Ready(Ok(()));
        }

        let me = self.project();
        let mut b = buf.take(usize::try_from(remaining).unwrap_or(usize::MAX));

        match ready!(me.reader.poll_read(cx, &mut b)) {
            Ok(()) => {}
            Err(e) => {
                *me.did_err.lock().unwrap() = true;
                return Poll::Ready(Err(e));
            }
        }

        let n = b.filled().len();

        // We need to update the original ReadBuf
        unsafe {
            buf.assume_init(n);
        }
        buf.advance(n);
        me.segments[*me.index].remaining = Some(remaining - n as u64);
        Poll::Ready(Ok(()))
    }
}

pub struct AdditionalSegmentsFromSocket<R: tokio::io::AsyncRead + Send + Unpin> {
    amount: u32,
    reader: R,
    index: Arc<std::sync::Mutex<usize>>,
    segments: Arc<tokio::sync::Mutex<Vec<Segment>>>,
    did_err: Arc<std::sync::Mutex<bool>>,
}

impl<R: tokio::io::AsyncRead + Send + Unpin> AdditionalSegmentsFromSocket<R> {
    pub fn new(reader: R, amount: u32) -> Self {
        Self {
            amount,
            reader,
            index: Arc::new(std::sync::Mutex::new(0)),
            segments: Arc::new(tokio::sync::Mutex::new(
                (0..amount as usize)
                    .map(|_| Segment {
                        size: [0; 8],
                        size_bytes_read: 0,
                        remaining: None,
                    })
                    .collect(),
            )),
            did_err: Arc::new(std::sync::Mutex::new(false)),
        }
    }

    pub fn did_err(&self) -> bool {
        *self.did_err.lock().unwrap()
    }

    #[allow(clippy::await_holding_lock)]
    pub async fn skip_until_index(&mut self, index: usize) -> tokio::io::Result<()> {
        for i in 0..index {
            let mut segments = self.segments.lock().await;
            let segment = &mut segments[i];

            while segment.size_bytes_read < 8 {
                segment.size_bytes_read += self
                    .reader
                    .read(&mut segment.size[segment.size_bytes_read..])
                    .await?;
            }
            let size = u64::from_be_bytes(segment.size);
            if segment.remaining.is_none() {
                segment.remaining = Some(size);
            }
            let mut reader = SegmentFromSocketReader {
                reader: &mut self.reader,
                segments,
                index: i,
                did_err: Arc::clone(&self.did_err),
            };
            tokio::io::copy(&mut reader, &mut tokio::io::empty()).await?;
        }

        Ok(())
    }

    pub async fn skip_all(mut self) -> tokio::io::Result<()> {
        let amount = self.amount();
        self.skip_until_index(amount as usize).await
    }
}

impl<R: tokio::io::AsyncRead + Send + Unpin> AdditionalSegments
    for AdditionalSegmentsFromSocket<R>
{
    fn amount(&self) -> u32 {
        self.amount
    }

    fn remaining(&self) -> u32 {
        self.amount - *self.index.lock().unwrap() as u32
    }

    fn next<'a>(
        &'a mut self,
    ) -> BoxFuture<'a, tokio::io::Result<Option<(u64, Pin<Box<dyn tokio::io::AsyncRead + 'a>>)>>>
    {
        if self.remaining() == 0 {
            return Box::pin(async { Ok(None) });
        }

        let index = {
            let mut stored_index = self.index.lock().unwrap();
            let index = *stored_index;
            *stored_index += 1;
            index
        };

        Box::pin(async move {
            self.skip_until_index(index).await?;

            let segments = self.segments.lock().await;
            let size = u64::from_be_bytes(segments[index].size);

            Ok(Some((
                size,
                Box::pin(SegmentFromSocketReader {
                    reader: &mut self.reader,
                    segments,
                    index,
                    did_err: Arc::clone(&self.did_err),
                }) as Pin<Box<dyn tokio::io::AsyncRead>>,
            )))
        })
    }
}
