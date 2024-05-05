use futures::future::BoxFuture;
use std::{
    marker::PhantomData,
    pin::Pin,
    sync::Arc,
    task::{ready, Poll},
};
use tokio::io::AsyncReadExt;

#[auto_impl::auto_impl(Box, &mut)]
pub trait Blobs {
    fn amount(&self) -> u32;

    fn remaining(&self) -> u32;

    #[allow(clippy::type_complexity)]
    fn next<'a>(
        &'a mut self,
    ) -> BoxFuture<
        'a,
        tokio::io::Result<Option<(u64, Pin<Box<dyn tokio::io::AsyncRead + Send + 'a>>)>>,
    >;
}

pub struct ListBlobs<L: AsRef<[S]> + Send, S: AsRef<[u8]> + Send> {
    phantom: PhantomData<S>,
    data: L,
    index: usize,
}

impl<L: AsRef<[S]> + Send, S: AsRef<[u8]> + Send> From<L> for ListBlobs<L, S> {
    fn from(value: L) -> Self {
        Self {
            phantom: PhantomData,
            data: value,
            index: 0,
        }
    }
}

impl<L: AsRef<[S]> + Send, S: AsRef<[u8]> + Send> Blobs for ListBlobs<L, S> {
    fn amount(&self) -> u32 {
        self.data.as_ref().len().try_into().unwrap()
    }

    fn remaining(&self) -> u32 {
        (self.data.as_ref().len() - self.index).try_into().unwrap()
    }

    fn next<'a>(
        &'a mut self,
    ) -> BoxFuture<
        'a,
        tokio::io::Result<Option<(u64, Pin<Box<dyn tokio::io::AsyncRead + Send + 'a>>)>>,
    > {
        Box::pin(async move {
            let Some(data) = self.data.as_ref().get(self.index) else {
                return Ok(None);
            };
            self.index += 1;
            Ok(Some((
                data.as_ref().len() as u64,
                Box::pin(data.as_ref()) as Pin<Box<dyn tokio::io::AsyncRead + Send>>,
            )))
        })
    }
}

struct Blob {
    size: [u8; 8],
    size_bytes_read: usize,
    remaining: Option<u64>,
}

#[pin_project::pin_project]
struct BlobFromSocketReader<'a, R: tokio::io::AsyncRead + Unpin> {
    #[pin]
    reader: R,
    index: usize,
    blobs: tokio::sync::MutexGuard<'a, Vec<Blob>>,
    did_err: Arc<std::sync::Mutex<bool>>,
}

impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for BlobFromSocketReader<'_, R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let remaining = self.blobs[self.index].remaining.unwrap();
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
        me.blobs[*me.index].remaining = Some(remaining - n as u64);
        Poll::Ready(Ok(()))
    }
}

pub struct BlobsFromSocket<R: tokio::io::AsyncRead + Send + Unpin> {
    amount: u32,
    reader: R,
    index: Arc<std::sync::Mutex<usize>>,
    blobs: Arc<tokio::sync::Mutex<Vec<Blob>>>,
    did_err: Arc<std::sync::Mutex<bool>>,
}

impl<R: tokio::io::AsyncRead + Send + Unpin> BlobsFromSocket<R> {
    pub fn new(reader: R, amount: u32) -> Self {
        Self {
            amount,
            reader,
            index: Arc::new(std::sync::Mutex::new(0)),
            blobs: Arc::new(tokio::sync::Mutex::new(
                (0..amount as usize)
                    .map(|_| Blob {
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
            let mut blobs = self.blobs.lock().await;
            let blob = &mut blobs[i];

            while blob.size_bytes_read < 8 {
                blob.size_bytes_read += self
                    .reader
                    .read(&mut blob.size[blob.size_bytes_read..])
                    .await?;
            }
            let size = u64::from_be_bytes(blob.size);
            if blob.remaining.is_none() {
                blob.remaining = Some(size);
            }
            let mut reader = BlobFromSocketReader {
                reader: &mut self.reader,
                blobs,
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

impl<R: tokio::io::AsyncRead + Send + Unpin> Blobs for BlobsFromSocket<R> {
    fn amount(&self) -> u32 {
        self.amount
    }

    fn remaining(&self) -> u32 {
        self.amount - *self.index.lock().unwrap() as u32
    }

    fn next<'a>(
        &'a mut self,
    ) -> BoxFuture<
        'a,
        tokio::io::Result<Option<(u64, Pin<Box<dyn tokio::io::AsyncRead + Send + 'a>>)>>,
    > {
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

            let blobs = self.blobs.lock().await;
            let size = u64::from_be_bytes(blobs[index].size);

            Ok(Some((
                size,
                Box::pin(BlobFromSocketReader {
                    reader: &mut self.reader,
                    blobs,
                    index,
                    did_err: Arc::clone(&self.did_err),
                }) as Pin<Box<dyn tokio::io::AsyncRead + Send>>,
            )))
        })
    }
}
