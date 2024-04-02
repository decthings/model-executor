mod segments;
pub mod types;

use atomic_counter::AtomicCounter;
use std::{collections::HashMap, future::Future, path::Path, pin::Pin, sync::Arc};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const MESSAGE_BYTE: u8 = 0;
pub const DATA_BYTE: u8 = 1;

pub struct AdditionalSegments<'a> {
    pub amount: u32,
    pub reader: &'a mut tokio::net::unix::OwnedReadHalf,
}

#[auto_impl::auto_impl(&, Rc, Box)]
pub trait ChildEventCallbacks {
    #[must_use]
    fn on_event<'a>(
        &'a self,
        event: types::EventMessage,
        additional_segments: Option<AdditionalSegments<'a>>,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>>;

    #[must_use]
    fn on_data_event<'a>(
        &'a self,
        event: types::DataEvent,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

#[derive(Debug)]
pub enum CallMethodOnChildError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Cancelled,
}

type ResponseCallback = Box<
    dyn for<'b> FnOnce(
            String,
            Result<(serde_json::Value, Option<AdditionalSegments<'b>>), CallMethodOnChildError>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ()>> + Send + 'b>>
        + Send
        + 'static,
>;

pub struct ChildRpcListener {
    model_count: usize,
    unix_reader: tokio::net::unix::OwnedReadHalf,
    awaiting_responses: Arc<std::sync::Mutex<HashMap<String, ResponseCallback>>>,
}

impl ChildRpcListener {
    pub async fn listen(mut self, on_event: impl ChildEventCallbacks) {
        loop {
            let res = async {
                let first_byte = self.unix_reader.read_u8().await?;
                if first_byte == MESSAGE_BYTE {
                    let num_additional_segments = self.unix_reader.read_u32().await?;
                    let first_segment_length = self.unix_reader.read_u64().await? as usize;

                    #[derive(serde::Deserialize)]
                    #[serde(untagged)]
                    enum ResultOrEvent {
                        Event(types::EventMessage),
                        Result(types::ResultMessage),
                    }

                    let mut first_segment = vec![0; first_segment_length];
                    self.unix_reader.read_exact(&mut first_segment).await?;

                    {
                        let additional_segments = if num_additional_segments == 0 {
                            None
                        } else {
                            Some(AdditionalSegments {
                                amount: num_additional_segments,
                                reader: &mut self.unix_reader
                            })
                        };

                        match serde_json::from_slice::<ResultOrEvent>(&first_segment).unwrap() {
                            ResultOrEvent::Result(val) => {
                                log::trace!(
                                    "Got response for method with id {} from spawned model #{}: {}",
                                    val.id,
                                    self.model_count,
                                    String::from_utf8_lossy(&first_segment),
                                );

                                let resolve = {
                                    let mut awaiting = self.awaiting_responses.lock().unwrap();
                                    awaiting.remove(&val.id)
                                };

                                if let Some(resolve) = resolve {
                                    resolve(
                                        val.id,
                                        Ok((val.result, additional_segments)),
                                    )
                                    .await
                                    .map_err(|_| None)?;
                                }
                            },
                            ResultOrEvent::Event(ev) => {
                                log::trace!("Got event from spawned model #{}: {ev:?}", self.model_count);

                                if !on_event.on_event(ev, additional_segments).await {
                                    log::info!(
                                        "Closing RPC to spawned model #{} because it seems we got an invalid event message.",
                                        self.model_count,
                                    );
                                    return Ok(true);
                                }
                            }
                        }
                    }

                    // Read one extra byte to make the API compatible with host. This is the
                    // "success" byte, which the vmlauncher uses but we can ignore.
                    self.unix_reader.read_u8().await?;
                    Ok::<_, Option<std::io::Error>>(false)
                } else {
                    let data_event_length = self.unix_reader.read_u64().await? as usize;

                    let mut data_event = vec![0; data_event_length];
                    self.unix_reader.read_exact(&mut data_event).await?;

                    let parsed: types::DataEvent = serde_json::from_slice(&data_event).unwrap();

                    log::trace!("Got data event from spawned model #{}: {parsed:?}", self.model_count);

                    on_event.on_data_event(parsed).await;
                    Ok::<_, Option<std::io::Error>>(false)
                }
            }.await;
            match res {
                Ok(should_exit) => {
                    if should_exit {
                        return;
                    }
                }
                Err(err) => {
                    log::info!(
                        "Read from spawned model #{} Unix socket failed with error: {err:?}",
                        self.model_count,
                    );
                    return;
                }
            }
        }
    }
}

pub struct ChildRpc {
    model_count: usize,
    unix_socket: tokio::sync::Mutex<tokio::io::BufWriter<tokio::net::unix::OwnedWriteHalf>>,
    id_counter: atomic_counter::ConsistentCounter,
    awaiting_responses: Arc<std::sync::Mutex<HashMap<String, ResponseCallback>>>,
}

impl ChildRpc {
    fn call_method<'a>(
        &'a self,
        params: types::CommandMessageWithResponse<'_>,
        result_cb: ResponseCallback,
    ) -> (String, impl Future<Output = ()> + 'a) {
        let id = self.id_counter.inc();
        let id = id.to_string();

        let with_id = params.with_id(&id);

        log::trace!(
            "Sending command {with_id:?} with id {id} to spawned model #{}",
            self.model_count,
        );

        let buf = serde_json::to_vec(&with_id).unwrap();

        {
            let mut m = self.awaiting_responses.lock().unwrap();
            m.insert(id.clone(), result_cb);
        }

        (id.clone(), async move {
            let mut writer = self.unix_socket.lock().await;
            let writeres = async move {
                writer.write_u8(MESSAGE_BYTE).await?;
                writer.write_u64(buf.len() as u64).await?;
                writer.write_all(&buf).await?;
                writer.flush().await?;
                Ok(())
            }
            .await;

            if let Err(err) = writeres {
                let response_cb = {
                    let mut m = self.awaiting_responses.lock().unwrap();
                    m.remove(&id)
                };
                if let Some(f) = response_cb {
                    // This should be the case, otherwise we really quickly got a response
                    (f)(id, Err(CallMethodOnChildError::Io(err))).await.unwrap();
                }
            }
        })
    }

    pub fn call_create_model_state<'a>(
        &'a self,
        params: &types::CreateModelStateCommand,
    ) -> (
        String,
        impl Future<Output = Result<types::CreateModelStateResult, CallMethodOnChildError>> + 'a,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (id, fut) = self.call_method(
            types::CommandMessageWithResponse::CallCreateModelState(params),
            Box::new(|_, res| {
                Box::pin(async move {
                    let res = res.and_then(|(res, _)| {
                        serde_json::from_value(res).map_err(CallMethodOnChildError::Json)
                    });
                    tx.send(res).ok();
                    Ok(())
                })
            }),
        );
        (id, async move {
            fut.await;
            rx.await.unwrap()
        })
    }

    pub fn call_instantiate_model<'a>(
        &'a self,
        params: &types::InstantiateModelCommand,
    ) -> (
        String,
        impl Future<Output = Result<types::InstantiateModelResult, CallMethodOnChildError>> + 'a,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (id, fut) = self.call_method(
            types::CommandMessageWithResponse::CallInstantiateModel(params),
            Box::new(|_, res| {
                Box::pin(async move {
                    let res = res.and_then(|(res, _)| {
                        serde_json::from_value(res).map_err(CallMethodOnChildError::Json)
                    });
                    tx.send(res).ok();
                    Ok(())
                })
            }),
        );
        (id, async move {
            fut.await;
            rx.await.unwrap()
        })
    }

    pub fn call_train<'a>(
        &'a self,
        params: &types::TrainCommand,
    ) -> (
        String,
        impl Future<Output = Result<types::TrainResult, CallMethodOnChildError>> + 'a,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (id, fut) = self.call_method(
            types::CommandMessageWithResponse::CallTrain(params),
            Box::new(|_, res| {
                Box::pin(async move {
                    let res = res.and_then(|(res, _)| {
                        serde_json::from_value(res).map_err(CallMethodOnChildError::Json)
                    });
                    tx.send(res).ok();
                    Ok(())
                })
            }),
        );
        (id, async move {
            fut.await;
            rx.await.unwrap()
        })
    }

    pub fn call_evaluate<'a>(
        &'a self,
        params: &types::EvaluateCommand,
        result_cb: impl for<'b> FnOnce(
                String,
                Result<
                    (types::EvaluateResult, Option<AdditionalSegments<'b>>),
                    CallMethodOnChildError,
                >,
            ) -> Pin<Box<dyn Future<Output = Result<(), ()>> + Send + 'b>>
            + Send
            + 'static,
    ) -> (String, impl Future<Output = ()> + 'a) {
        self.call_method(
            types::CommandMessageWithResponse::CallEvaluate(params),
            Box::new(|id, res| {
                Box::pin(async move {
                    match res {
                        Ok((res, additional_segments)) => {
                            let res = serde_json::from_value(res).map_err(|e| {
                                log::warn!("Invalid response from child for evaluate. JSON parse failed: {e:?}")
                            })?;
                            result_cb(id, Ok((res, additional_segments))).await
                        }
                        Err(e) => result_cb(id, Err(e)).await,
                    }
                })
            }),
        )
    }

    pub fn call_get_model_state<'a>(
        &'a self,
        params: &types::GetModelStateCommand,
    ) -> (
        String,
        impl Future<Output = Result<types::GetModelStateResult, CallMethodOnChildError>> + 'a,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (id, fut) = self.call_method(
            types::CommandMessageWithResponse::CallGetModelState(params),
            Box::new(|_, res| {
                Box::pin(async move {
                    let res = res.and_then(|(res, _)| {
                        serde_json::from_value(res).map_err(CallMethodOnChildError::Json)
                    });
                    tx.send(res).ok();
                    Ok(())
                })
            }),
        );
        (id, async move {
            fut.await;
            rx.await.unwrap()
        })
    }

    fn call_method_without_response<'a>(
        &'a self,
        params: types::CommandMessageWithoutResponse<'_>,
    ) -> impl Future<Output = Result<(), tokio::io::Error>> + 'a {
        log::trace!(
            "Sending command {params:?} to spawned model #{}",
            self.model_count,
        );

        let buf = serde_json::to_vec(&params).unwrap();

        async move {
            let mut writer = self.unix_socket.lock().await;
            writer.write_u8(MESSAGE_BYTE).await?;
            writer.write_u64(buf.len() as u64).await?;
            writer.write_all(&buf).await?;
            writer.flush().await?;
            Ok(())
        }
    }

    pub fn call_initialize<'a>(
        &'a self,
        params: &types::InitializeCommand,
    ) -> impl Future<Output = Result<(), tokio::io::Error>> + 'a {
        self.call_method_without_response(types::CommandMessageWithoutResponse::Initialize(params))
    }

    pub fn call_dispose_instantiated_model<'a>(
        &'a self,
        params: &types::DisposeInstantiatedModelCommand,
    ) -> impl Future<Output = Result<(), tokio::io::Error>> + 'a {
        self.call_method_without_response(
            types::CommandMessageWithoutResponse::CallDisposeInstantiatedModel(params),
        )
    }

    pub fn call_cancel_train<'a>(
        &'a self,
        params: &types::CancelTrainCommand,
    ) -> impl Future<Output = Result<(), tokio::io::Error>> + 'a {
        self.call_method_without_response(types::CommandMessageWithoutResponse::CallCancelTrain(
            params,
        ))
    }

    pub async fn cancel_all_calling_functions(&self) {
        let awaiting_responses = {
            let mut awaiting_responses = self.awaiting_responses.lock().unwrap();
            std::mem::take(&mut *awaiting_responses)
        };
        for (id, f) in awaiting_responses {
            f(id, Err(CallMethodOnChildError::Cancelled)).await.unwrap();
        }
    }

    pub async fn provide_data(
        &self,
        request_id: u32,
        num_segments: u32,
        segments_reader: impl tokio::io::AsyncRead + Unpin,
    ) -> Result<(), tokio::io::Error> {
        log::trace!(
            "Providing {num_segments} data segments for data request {request_id} to spawned model #{}",
            self.model_count,
        );

        let mut writer = self.unix_socket.lock().await;
        let maybe_err = async {
            writer.write_u8(DATA_BYTE).await?;
            writer.write_u32(request_id).await?;
            writer.write_u32(num_segments).await?;
            Ok::<_, tokio::io::Error>(())
        }
        .await;
        if let Err(e) = maybe_err {
            let _ = segments::discard_additional_segments(segments_reader, num_segments).await;
            return Err(e);
        }
        segments::write_additional_segments(num_segments, segments_reader, &mut *writer).await?;
        writer.flush().await?;
        Ok(())
    }
}

pub(crate) async fn rpc(
    unixsocket_path: impl AsRef<Path>,
    model_count: usize,
) -> (ChildRpc, ChildRpcListener) {
    let unix_socket = tokio::net::UnixListener::bind(unixsocket_path).unwrap();
    let unix_stream = unix_socket.accept().await.unwrap().0;
    let (reader, writer) = unix_stream.into_split();
    let awaiting_responses = Arc::new(std::sync::Mutex::new(HashMap::new()));
    (
        ChildRpc {
            model_count,
            unix_socket: tokio::sync::Mutex::new(tokio::io::BufWriter::new(writer)),
            id_counter: atomic_counter::ConsistentCounter::new(0),
            awaiting_responses: Arc::clone(&awaiting_responses),
        },
        ChildRpcListener {
            model_count,
            unix_reader: reader,
            awaiting_responses,
        },
    )
}