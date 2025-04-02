mod events;
mod spawn;
mod types;
mod weights_loader;

pub use blob_stream;

use std::{collections::HashMap, future::Future, path::Path, pin::Pin, sync::Arc};

use atomic_counter::AtomicCounter;

pub use spawn::ModelExecutor;
pub use spawn::*;
use tokio::sync::Mutex;
pub use types::*;

impl ModelExecutor {
    async fn inner_run(
        &self,
        to_execute: spawn::ModelToExecute<'_, impl AsRef<str>>,
        current_dir: impl AsRef<Path>,
        inherit_env: bool,
        with_command: Option<impl FnOnce(&mut tokio::process::Command)>,
        initialize_path: Option<String>,
    ) -> Result<(tokio::process::Child, RunningUnixModel), RunError> {
        let (mut cmd, rpc_fut) = self.run(to_execute);
        cmd.current_dir(current_dir);
        if inherit_env {
            for (key, val) in std::env::vars() {
                if key == "IPC_PATH" {
                    continue;
                }
                cmd.env(key, val);
            }
        }
        if let Some(with_command) = with_command {
            with_command(&mut cmd);
        }

        let mut child = cmd.spawn().map_err(RunError::Std)?;

        let (rpc, listener) = rpc_fut.await;

        log::trace!("Model connected over Unix socket");

        let rpc = Arc::new(rpc);
        let rpc_clone = Arc::clone(&rpc);

        let (initialized_tx, initialized_rx) = tokio::sync::oneshot::channel();
        let datasets = Arc::new(Mutex::new(HashMap::new()));
        let datasets_clone = Arc::clone(&datasets);
        let weights_providers = Arc::new(Mutex::new(HashMap::new()));
        let weights_providers_clone = Arc::clone(&weights_providers);
        let training_sessions = Arc::new(Mutex::new(HashMap::new()));
        let training_sessions_clone = Arc::clone(&training_sessions);

        tokio::spawn(async move {
            listener
                .listen(events::EventCallbacks {
                    rpc,
                    on_initialized: Mutex::new(Some(initialized_tx)),
                    datasets,
                    weights_providers,
                    training_sessions,
                })
                .await;
        });

        if let Some(initialize_path) = initialize_path {
            log::trace!("Calling initialize on model, with path = {initialize_path}");
            if let Err(e) = rpc_clone
                .call_initialize(&spawn::rpc::types::InitializeCommand {
                    path: initialize_path,
                })
                .await
            {
                child.kill().await.ok();
                return Err(RunError::Std(e));
            }
        }

        log::trace!("Waiting for model initialization to complete");
        if let Err(spawn::rpc::types::ModelSessionInitializedError::Exception { details }) =
            initialized_rx.await.unwrap()
        {
            log::trace!("Model initialization completed with error: Exception");
            child.kill().await.ok();
            return Err(RunError::Exception { details });
        }

        log::trace!("Model initialization completed");

        Ok((
            child,
            RunningUnixModel {
                rpc: rpc_clone,
                dataset_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
                instantiated_model_id_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
                datasets: datasets_clone,
                weights_providers: weights_providers_clone,
                training_session_id_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
                training_sessions: training_sessions_clone,
            },
        ))
    }

    /// Execute a Decthings model using a custom commmand. The *program* argument should contain
    /// the name of the command to execute, without any arguments. To pass arguments and environment
    /// variables, or to change the current directory of the child, use the *options* parameter.
    ///
    /// Returns a tokio child which can be used to wait for or kill the child, and a
    /// *RunningUnixModel*, which allows you to call functions such as train or evaluate on the
    /// model.
    pub async fn run_custom(
        &self,
        program: &str,
        options: &RunBinOptions<'_>,
    ) -> Result<(tokio::process::Child, RunningUnixModel), RunError> {
        log::trace!("Executing model using program {program}");

        let to_execute: spawn::ModelToExecute<String> =
            spawn::ModelToExecute::Bin { path: program };
        self.inner_run(
            to_execute,
            std::env::current_dir().unwrap(),
            options.inherit_env,
            options.with_command.as_ref(),
            None,
        )
        .await
    }

    /// Execute a Decthings model which is defined by a binary executable, which is the case for
    /// compiled languages like Rust or Go.
    ///
    /// Will execute the file "model" inside the model directory.
    ///
    /// Returns a tokio child which can be used to wait for or kill the child, and a
    /// *RunningUnixModel*, which allows you to call functions such as train or evaluate on the
    /// model.
    pub async fn run_bin(
        &self,
        model_dir: &Path,
        options: &RunBinOptions<'_>,
    ) -> Result<(tokio::process::Child, RunningUnixModel), RunError> {
        let model_file = model_dir.join("model");

        log::trace!(
            "Executing model using binary executable at {}",
            model_file.display()
        );

        let to_execute: spawn::ModelToExecute<String> = spawn::ModelToExecute::Bin {
            path: &model_file.to_string_lossy(),
        };
        self.inner_run(
            to_execute,
            model_dir,
            options.inherit_env,
            options.with_command.as_ref(),
            None,
        )
        .await
    }

    /// Execute a Node.js Decthings model.
    ///
    /// Will execute the file "index.js" inside the model directory, using *node*.
    ///
    /// Returns a tokio child which can be used to wait for or kill the child, and a
    /// *RunningUnixModel*, which allows you to call functions such as train or evaluate on the
    /// model.
    pub async fn run_node_js(
        &self,
        model_dir: &Path,
        options: &RunNodeJsOptions<'_>,
    ) -> Result<(tokio::process::Child, RunningUnixModel), RunError> {
        let index_js = model_dir.join("index.js").to_string_lossy().to_string();
        log::trace!("Executing model using Node.js at {index_js}");
        self.inner_run(
            spawn::ModelToExecute::NodeJs {
                flags: &options.flags,
            },
            model_dir,
            options.inherit_env,
            options.with_command.as_ref(),
            Some(index_js),
        )
        .await
    }

    /// Execute a Python Decthings model.
    ///
    /// Will execute the file "main.py" inside the model directory, using *python3*.
    ///
    /// Returns a tokio child which can be used to wait for or kill the child, and a
    /// *RunningUnixModel*, which allows you to call functions such as train or evaluate on the
    /// model.
    pub async fn run_python(
        &self,
        model_dir: &Path,
        options: &RunPythonOptions<'_>,
    ) -> Result<(tokio::process::Child, RunningUnixModel), RunError> {
        let main_py = model_dir.join("main.py").to_string_lossy().to_string();
        log::trace!("Executing model using Python at {main_py}");
        self.inner_run(
            spawn::ModelToExecute::Python {
                flags: &options.flags,
            },
            model_dir,
            options.inherit_env,
            options.with_command.as_ref(),
            Some(main_py),
        )
        .await
    }
}

#[derive(Clone)]
pub struct RunningUnixModel {
    rpc: Arc<spawn::rpc::ChildRpc>,

    dataset_counter: Arc<atomic_counter::ConsistentCounter>,
    instantiated_model_id_counter: Arc<atomic_counter::ConsistentCounter>,
    datasets: Arc<Mutex<HashMap<String, Arc<Box<dyn DataLoader>>>>>,
    weights_providers: Arc<Mutex<HashMap<String, Arc<Box<dyn WeightsProvider>>>>>,
    training_session_id_counter: Arc<atomic_counter::ConsistentCounter>,
    training_sessions: Arc<Mutex<HashMap<String, Arc<Box<dyn TrainTracker>>>>>,
}

impl RunningUnixModel {
    fn add_dataset(
        &self,
        datasets: &mut HashMap<String, Arc<Box<dyn DataLoader>>>,
        name: String,
        param: Param,
    ) -> spawn::rpc::types::Param {
        let dataset = self.dataset_counter.inc().to_string();

        datasets.insert(dataset.clone(), Arc::new(param.data_loader));

        spawn::rpc::types::Param {
            name,
            dataset,
            amount: param.amount,
            total_byte_size: param.total_byte_size,
        }
    }

    /// Call the *initialize_weights* function on the running model.
    ///
    /// The function takes a list of parameters and outputs new model weights. Model weights can
    /// be some arbitrary binary data which contains the trained model. For a neural network,
    /// the weights would contain the weights and biases of the neurons.
    pub async fn initialize_weights(
        &self,
        options: InitializeWeightsOptions,
    ) -> Result<(), types::CallFunctionError> {
        let (child_params, child_other_models) = {
            let mut datasets = self.datasets.lock().await;
            let child_params = options
                .params
                .into_iter()
                .map(|(name, param)| self.add_dataset(&mut datasets, name, param))
                .collect();
            let child_other_models = options
                .other_models
                .into_iter()
                .map(
                    |(model_id, other_model)| spawn::rpc::types::OtherModelWithWeights {
                        id: model_id,
                        mount_path: other_model.mount_path,
                        weights: other_model
                            .weights
                            .into_iter()
                            .map(|(key, weight)| {
                                self.add_dataset(
                                    &mut datasets,
                                    key,
                                    Param {
                                        amount: 1,
                                        total_byte_size: weight.byte_size,
                                        data_loader: Box::new(
                                            weights_loader::DataLoaderFromWeights {
                                                byte_size: weight.byte_size,
                                                inner: weight.weights_loader,
                                            },
                                        ),
                                    },
                                )
                            })
                            .collect(),
                    },
                )
                .collect();
            (child_params, child_other_models)
        };

        let cmd = spawn::rpc::types::InitializeWeightsCommand {
            params: child_params,
            other_models: child_other_models,
        };

        let mut weights_providers = self.weights_providers.lock().await;

        let (cmd_id, fut) = self.rpc.call_initialize_weights(&cmd);

        weights_providers.insert(cmd_id.clone(), Arc::new(options.weights_provider));
        drop(weights_providers);

        let res = fut.await;

        {
            let mut datasets = self.datasets.lock().await;
            for param in cmd.params {
                datasets.remove(&param.dataset);
            }
            for other_model in cmd.other_models {
                for param in other_model.weights {
                    datasets.remove(&param.dataset);
                }
            }
        }
        {
            let mut weights_providers = self.weights_providers.lock().await;
            weights_providers.remove(&cmd_id);
        }

        match res {
            Ok(spawn::rpc::types::InitializeWeightsResult { error: None }) => Ok(()),
            Ok(spawn::rpc::types::InitializeWeightsResult {
                error: Some(spawn::rpc::types::InitializeWeightsError::Exception { details }),
            }) => Err(types::CallFunctionError::Exception { details }),
            Err(e) => Err(types::CallFunctionError::Rpc(e)),
        }
    }

    /// Call the *instantiate_model* function on the running model.
    ///
    /// The function loads previously created or trained weights and prepares it for execution. The
    /// returned struct then allows you to call the functions evaluate and train.
    ///
    /// After you are done with the instantiate model, make sure to call *dispose*. Simply dropping
    /// it is not enough to free up all resources.
    pub async fn instantiate_model(
        &self,
        options: InstantiateModelOptions,
    ) -> Result<UnixInstantiated, types::CallFunctionError> {
        let child_params = {
            let mut datasets = self.datasets.lock().await;
            options
                .weights
                .into_iter()
                .map(|(key, weight)| {
                    self.add_dataset(
                        &mut datasets,
                        key,
                        Param {
                            amount: 1,
                            total_byte_size: weight.byte_size,
                            data_loader: Box::new(weights_loader::DataLoaderFromWeights {
                                byte_size: weight.byte_size,
                                inner: weight.weights_loader,
                            }),
                        },
                    )
                })
                .collect()
        };

        let instantiated_model_id = self.instantiated_model_id_counter.inc().to_string();

        let cmd = spawn::rpc::types::InstantiateModelCommand {
            instantiated_model_id: instantiated_model_id.clone(),
            weights: child_params,
            other_models: options
                .other_models
                .into_iter()
                .map(|(model_id, other_model)| spawn::rpc::types::OtherModel {
                    id: model_id,
                    mount_path: other_model.mount_path,
                })
                .collect(),
        };

        let (_, fut) = self.rpc.call_instantiate_model(&cmd);

        let res = fut.await;

        {
            let mut datasets = self.datasets.lock().await;
            for param in cmd.weights {
                datasets.remove(&param.dataset);
            }
        }

        match res {
            Ok(spawn::rpc::types::InstantiateModelResult { error: None }) => Ok(UnixInstantiated {
                model: self.clone(),
                instantiated_model_id,
            }),
            Ok(spawn::rpc::types::InstantiateModelResult {
                error: Some(spawn::rpc::types::InstantiateModelError::Exception { details }),
            }) => Err(types::CallFunctionError::Exception { details }),
            Ok(spawn::rpc::types::InstantiateModelResult {
                error: Some(spawn::rpc::types::InstantiateModelError::Disposed),
            }) => unreachable!(),
            Err(e) => Err(types::CallFunctionError::Rpc(e)),
        }
    }
}

pub struct UnixInstantiated {
    model: RunningUnixModel,
    instantiated_model_id: String,
}

impl UnixInstantiated {
    /// Call the *evaluate* function on the running model.
    ///
    /// The function takes a set of input parameters and outputs some data.
    pub async fn evaluate(
        &self,
        options: EvaluateOptions<
            impl for<'b> FnOnce(
                Result<
                    (
                        Vec<spawn::rpc::types::EvaluateOutput>,
                        Box<dyn blob_stream::Blobs + Send + 'b>,
                    ),
                    types::CallFunctionError,
                >,
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'b>>
            + Send
            + 'static,
        >,
    ) {
        let child_params = {
            let mut datasets = self.model.datasets.lock().await;
            options
                .params
                .into_iter()
                .map(|(name, param)| self.model.add_dataset(&mut datasets, name, param))
                .collect()
        };

        let cmd = spawn::rpc::types::EvaluateCommand {
            instantiated_model_id: self.instantiated_model_id.clone(),
            params: child_params,
            expected_output_types: options.expected_output_types,
        };

        let (res_tx, res_rx) = tokio::sync::oneshot::channel();
        let (_, fut) = self.model.rpc.call_evaluate(&cmd, |_, res| {
            Box::pin(async move {
                let res2 = match res {
                    Ok((
                        spawn::rpc::types::EvaluateResult {
                            outputs: Some(outputs),
                            error: None,
                        },
                        blobs,
                    )) => Ok((outputs, blobs)),
                    Ok((
                        spawn::rpc::types::EvaluateResult {
                            outputs: _,
                            error: Some(spawn::rpc::types::EvaluateError::Exception { details }),
                        },
                        _,
                    )) => Err(types::CallFunctionError::Exception { details }),
                    Ok((
                        spawn::rpc::types::EvaluateResult {
                            outputs: _,
                            error: Some(spawn::rpc::types::EvaluateError::InstantiatedModelNotFound),
                        },
                        _,
                    )) => {
                        unreachable!()
                    }
                    Ok((
                        spawn::rpc::types::EvaluateResult {
                            outputs: None,
                            error: None,
                        },
                        _,
                    )) => {
                        unreachable!()
                    }
                    Err(e) => Err(types::CallFunctionError::Rpc(e)),
                };
                (options.result_cb)(res2).await;
                res_tx.send(()).ok();
            })
        });

        fut.await;
        res_rx.await.unwrap();

        {
            let mut datasets = self.model.datasets.lock().await;
            for param in cmd.params {
                datasets.remove(&param.dataset);
            }
        }
    }

    /// Call the *train* function on the running model.
    ///
    /// The function takes a set of parameters and trains the model. The *tracker* option is used
    /// to listen for events, such as progress and metrics.
    ///
    /// After training, the *evaluate* function will use the new weights. To save the trained model,
    /// call the *get_weights* function, which will output the binary weights. The returned weights
    /// can then be loaded again using *instantiate_model*, which allows you to use the trained
    /// weights after the model is restarted.
    pub async fn train(&self, options: TrainOptions) -> Result<(), types::CallFunctionError> {
        let mut datasets = self.model.datasets.lock().await;
        let child_params = options
            .params
            .into_iter()
            .map(|(name, param)| self.model.add_dataset(&mut datasets, name, param))
            .collect();
        drop(datasets);

        let training_session_id = self.model.training_session_id_counter.inc().to_string();
        {
            let mut training_sessions = self.model.training_sessions.lock().await;
            training_sessions.insert(training_session_id.clone(), Arc::new(options.tracker));
        }

        let cmd = spawn::rpc::types::TrainCommand {
            instantiated_model_id: self.instantiated_model_id.clone(),
            training_session_id: training_session_id.clone(),
            params: child_params,
        };

        let (_, fut) = self.model.rpc.call_train(&cmd);

        let res = fut.await;

        {
            let mut datasets = self.model.datasets.lock().await;
            for param in cmd.params {
                datasets.remove(&param.dataset);
            }
        }
        {
            let mut training_sessions = self.model.training_sessions.lock().await;
            training_sessions.remove(&training_session_id);
        }

        match res {
            Ok(spawn::rpc::types::TrainResult { error: None }) => Ok(()),
            Ok(spawn::rpc::types::TrainResult {
                error: Some(spawn::rpc::types::TrainError::Exception { details }),
            }) => Err(types::CallFunctionError::Exception { details }),
            Ok(spawn::rpc::types::TrainResult {
                error: Some(spawn::rpc::types::TrainError::InstantiatedModelNotFound),
            }) => {
                unreachable!()
            }
            Err(e) => Err(types::CallFunctionError::Rpc(e)),
        }
    }

    /// Call the *get_weights* function on the running model.
    ///
    /// The function outputs the model weights. If the *train* function was called on this
    /// instantiated model, the function will output the new, trained weights.
    pub async fn get_weights(
        &self,
        options: GetWeightsOptions,
    ) -> Result<(), types::CallFunctionError> {
        let cmd = spawn::rpc::types::GetWeightsCommand {
            instantiated_model_id: self.instantiated_model_id.clone(),
        };

        let mut weights_providers = self.model.weights_providers.lock().await;

        let (cmd_id, fut) = self.model.rpc.call_get_weights(&cmd);

        weights_providers.insert(cmd_id.clone(), Arc::new(options.weights_provider));
        drop(weights_providers);

        let res = fut.await;

        match res {
            Ok(spawn::rpc::types::GetWeightsResult { error: None }) => Ok(()),
            Ok(spawn::rpc::types::GetWeightsResult {
                error: Some(spawn::rpc::types::GetWeightsError::Exception { details }),
            }) => Err(types::CallFunctionError::Exception { details }),
            Ok(spawn::rpc::types::GetWeightsResult {
                error: Some(spawn::rpc::types::GetWeightsError::InstantiatedModelNotFound),
            }) => {
                unreachable!()
            }
            Err(e) => Err(types::CallFunctionError::Rpc(e)),
        }
    }

    /// Call the *dispose_instantiated_model* function on the running model.
    ///
    /// The function will deallocate any resources used by the instantiated model. To avoid a
    /// memory leak, this function must be called when you are done with the instantiated model.
    /// Simply dropping the instantiated model struct will not free up all resources.
    pub async fn dispose(self) -> Result<(), tokio::io::Error> {
        self.model
            .rpc
            .call_dispose_instantiated_model(&spawn::rpc::types::DisposeInstantiatedModelCommand {
                instantiated_model_id: self.instantiated_model_id,
            })
            .await
    }
}
