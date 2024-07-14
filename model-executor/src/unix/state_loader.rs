use std::{future::Future, pin::Pin};

use tokio::io::AsyncRead;

pub struct DataLoaderFromState {
    pub byte_size: u64,
    pub inner: Box<dyn super::StateLoader>,
}

impl super::DataLoader for DataLoaderFromState {
    fn read(
        &self,
        start_index: u32,
        amount: u32,
    ) -> Pin<Box<dyn Future<Output = Box<dyn blob_stream::Blobs + Send + '_>> + Send + '_>> {
        Box::pin(async move {
            if amount < 1 || start_index > 0 {
                return Box::new(blob_stream::ListBlobs::<Vec<Vec<u8>>, Vec<u8>>::from(
                    vec![],
                )) as Box<dyn blob_stream::Blobs + Send>;
            }

            let reader = self.inner.read().await;

            struct ReaderBlobs<'a>(u64, Option<Pin<Box<dyn AsyncRead + Send + 'a>>>);

            impl blob_stream::Blobs for ReaderBlobs<'_> {
                fn amount(&self) -> u32 {
                    1
                }

                fn next<'a>(
                    &'a mut self,
                ) -> futures::future::BoxFuture<
                    'a,
                    tokio::io::Result<
                        Option<(u64, Pin<Box<dyn tokio::io::AsyncRead + Send + 'a>>)>,
                    >,
                > {
                    Box::pin(async move { Ok(self.1.take().map(|x| (self.0, x))) })
                }

                fn remaining(&self) -> u32 {
                    if self.1.is_some() {
                        1
                    } else {
                        0
                    }
                }
            }

            Box::new(ReaderBlobs(self.byte_size, Some(reader)))
                as Box<dyn blob_stream::Blobs + Send>
        })
    }

    fn id(&self) -> u32 {
        0
    }

    fn shuffle(&self, _others: &[u32]) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}
