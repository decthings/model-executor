use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

pub struct EventCallbacks {
    pub rpc: Arc<super::spawn::rpc::ChildRpc>,
    pub on_initialized: Mutex<
        Option<
            tokio::sync::oneshot::Sender<
                Result<(), super::spawn::rpc::types::ModelSessionInitializedError>,
            >,
        >,
    >,
    pub datasets: Arc<Mutex<HashMap<String, Arc<Box<dyn super::DataLoader>>>>>,
    pub state_providers: Arc<Mutex<HashMap<String, Arc<Box<dyn super::StateProvider>>>>>,
    pub training_sessions: Arc<Mutex<HashMap<String, Arc<Box<dyn super::TrainTracker>>>>>,
}

impl super::spawn::rpc::ChildEventCallbacks for EventCallbacks {
    fn on_event<'a>(
        &'a self,
        event: super::spawn::rpc::types::EventMessage,
        blobs: Box<dyn super::Blobs + Send + 'a>,
    ) -> std::pin::Pin<Box<dyn futures::Future<Output = ()> + Send + 'a>> {
        match event {
            super::spawn::rpc::types::EventMessage::ModelSessionInitialized { error } => {
                Box::pin(async move {
                    let Some(on_initialized) = self.on_initialized.lock().await.take() else {
                        return;
                    };
                    on_initialized
                        .send(match error {
                            Some(e) => Err(e),
                            None => Ok(()),
                        })
                        .ok();
                })
            }
            super::spawn::rpc::types::EventMessage::ProvideStateData { command_id, names } => {
                Box::pin(async move {
                    let state_providers = self.state_providers.lock().await;
                    let Some(state_provider) = state_providers.get(&command_id) else {
                        return;
                    };
                    let state_provider = Arc::clone(state_provider);
                    drop(state_providers);
                    state_provider.provide(names, blobs).await;
                })
            }
            super::spawn::rpc::types::EventMessage::TrainingProgress {
                training_session_id,
                progress,
            } => Box::pin(async move {
                let training_sessions = self.training_sessions.lock().await;
                let Some(training_session) = training_sessions.get(&training_session_id) else {
                    return;
                };
                let training_session = Arc::clone(training_session);
                drop(training_sessions);

                training_session.progress(progress).await;
            }),
            super::spawn::rpc::types::EventMessage::TrainingMetrics {
                training_session_id,
                names,
            } => Box::pin(async move {
                let training_sessions = self.training_sessions.lock().await;
                let Some(training_session) = training_sessions.get(&training_session_id) else {
                    return;
                };
                let training_session = Arc::clone(training_session);
                drop(training_sessions);

                training_session.metrics(names, blobs).await;
            }),
        }
    }

    fn on_data_event<'a>(
        &'a self,
        event: super::spawn::rpc::types::DataEvent,
    ) -> std::pin::Pin<Box<dyn futures::Future<Output = ()> + Send + 'a>> {
        match event {
            super::spawn::rpc::types::DataEvent::RequestData {
                dataset,
                request_id,
                start_index,
                amount,
            } => Box::pin(async move {
                let datasets = self.datasets.lock().await;
                let Some(data_loader) = datasets.get(&dataset) else {
                    return;
                };
                let data_loader = Arc::clone(data_loader);
                drop(datasets);

                let data = data_loader.read(start_index, amount).await;
                self.rpc.provide_data(request_id, data).await.unwrap();
            }),
            super::spawn::rpc::types::DataEvent::Shuffle { datasets } => Box::pin(async move {
                let datasets_locked = self.datasets.lock().await;
                let mut data_loaders: Vec<_> = datasets
                    .into_iter()
                    .filter_map(|id| datasets_locked.get(&id).map(Arc::clone))
                    .collect();
                drop(datasets_locked);

                let Some(last) = data_loaders.pop() else {
                    return;
                };

                last.shuffle(&data_loaders.into_iter().map(|x| x.id()).collect::<Vec<_>>())
                    .await;
            }),
        }
    }
}
