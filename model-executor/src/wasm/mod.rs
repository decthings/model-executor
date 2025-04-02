mod bindings;

use std::{
    collections::HashMap,
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use wasmtime::component::ResourceAny;

pub trait DataLoader: Send + 'static {
    fn id(&self) -> u32;
    fn read(&mut self, start_index: u32, amount: u32) -> Vec<Vec<u8>>;
    fn shuffle(&mut self, others: &[u32]);
}

pub trait WeightsProvider: Send + 'static {
    fn provide(&self, data: Vec<(String, Vec<u8>)>);
}

pub trait WeightsLoader: Send + 'static {
    fn read(&self) -> Vec<u8>;
}

pub trait TrainTracker: Send + 'static {
    fn progress(&self, progress: f32);
    fn metrics(&self, metrics: Vec<(String, Vec<u8>)>);
    fn is_cancelled(&self) -> bool;
}

#[derive(Debug)]
pub enum CallFunctionError {
    Wasmtime(wasmtime::Error),
    Function(String),
}

impl From<wasmtime::Error> for CallFunctionError {
    fn from(value: wasmtime::Error) -> Self {
        Self::Wasmtime(value)
    }
}

impl From<wasmtime::component::ResourceTableError> for CallFunctionError {
    fn from(value: wasmtime::component::ResourceTableError) -> Self {
        Self::Wasmtime(value.into())
    }
}

impl From<String> for CallFunctionError {
    fn from(value: String) -> Self {
        Self::Function(value)
    }
}

pub struct Param {
    pub amount: u32,
    pub total_byte_size: u64,
    pub data_loader: Box<dyn DataLoader>,
}

pub struct WeightKey {
    pub byte_size: u64,
    pub weights_loader: Box<dyn WeightsLoader>,
}

pub struct OtherModelWithWeights {
    pub model_id: String,
    pub mount_path: String,
    pub weights: HashMap<String, WeightKey>,
}

pub struct InitializeWeightsOptions {
    pub params: HashMap<String, Param>,
    pub weights_provider: Box<dyn WeightsProvider>,
    pub other_models: Vec<OtherModelWithWeights>,
}

pub struct OtherModel {
    pub model_id: String,
    pub mount_path: String,
}

pub struct InstantiateModelOptions {
    pub weights: HashMap<String, WeightKey>,
    pub other_models: Vec<OtherModel>,
}

#[derive(Clone)]
pub struct RunningWasmModel {
    store: Arc<Mutex<wasmtime::Store<bindings::Host>>>,
    bindings: Arc<bindings::ModelRunner>,
}

impl RunningWasmModel {
    pub fn run(
        wasi: wasmtime_wasi::WasiCtx,
        engine: &wasmtime::Engine,
        component: &wasmtime::component::Component,
    ) -> wasmtime::Result<Self> {
        let mut store = wasmtime::Store::new(
            engine,
            bindings::Host {
                table: wasmtime::component::ResourceTable::new(),
                wasi,
            },
        );

        let mut linker = wasmtime::component::Linker::new(store.engine());
        bindings::ModelRunner::add_to_linker(&mut linker, |state| state).unwrap();
        wasmtime_wasi::add_to_linker_sync(&mut linker).unwrap();

        let (bindings, _) =
            bindings::ModelRunner::instantiate(&mut store, component, &linker).unwrap();

        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            bindings: Arc::new(bindings),
        })
    }

    pub fn store(&self) -> impl DerefMut<Target = wasmtime::Store<bindings::Host>> + '_ {
        self.store.lock().unwrap()
    }

    /// Call the *initialize_weights* function on the running model.
    ///
    /// The function takes a list of parameters and outputs new model weights. Model weights can
    /// be some arbitrary binary data which contains the trained model. For a neural network,
    /// the weights would contain the weights and biases of the neurons.
    pub fn initialize_weights(
        &self,
        options: InitializeWeightsOptions,
    ) -> Result<(), CallFunctionError> {
        let mut store = self.store.lock().unwrap();

        let options = bindings::exports::decthings::model::model::InitializeWeightsOptions {
            params: options
                .params
                .into_iter()
                .map(|(name, param)| {
                    Ok(bindings::exports::decthings::model::model::Param {
                        name,
                        amount: param.amount,
                        total_byte_size: param.total_byte_size,
                        data_loader: store.data_mut().table.push(param.data_loader)?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
            weights_provider: store.data_mut().table.push(options.weights_provider)?,
            other_models: options
                .other_models
                .into_iter()
                .map(|other_model| {
                    Ok(
                        bindings::exports::decthings::model::model::OtherModelWithWeights {
                            model_id: other_model.model_id,
                            mount_path: other_model.mount_path,
                            weights: other_model
                                .weights
                                .into_iter()
                                .map(|(key, weight)| {
                                    Ok(bindings::exports::decthings::model::model::WeightKey {
                                        key,
                                        byte_size: weight.byte_size,
                                        weights_loader: store
                                            .data_mut()
                                            .table
                                            .push(weight.weights_loader)?,
                                    })
                                })
                                .collect::<wasmtime::Result<_>>()?,
                        },
                    )
                })
                .collect::<wasmtime::Result<_>>()?,
        };

        self.bindings
            .decthings_model_model()
            .call_initialize_weights(&mut *store, &options)??;

        Ok(())
    }

    /// Call the *instantiate_model* function on the running model.
    ///
    /// The function loads previously created or trained weights and prepares it for execution. The
    /// returned struct then allows you to call the functions evaluate and train.
    ///
    /// After you are done with the instantiate model, make sure to call *dispose*. Simply dropping
    /// it is not enough to free up all resources.
    pub fn instantiate_model(
        &self,
        options: InstantiateModelOptions,
    ) -> Result<WasmInstantiated, CallFunctionError> {
        let mut store = self.store.lock().unwrap();

        let options = bindings::exports::decthings::model::model::InstantiateModelOptions {
            weights: options
                .weights
                .into_iter()
                .map(|(key, weight)| {
                    Ok(bindings::exports::decthings::model::model::WeightKey {
                        key,
                        byte_size: weight.byte_size,
                        weights_loader: store.data_mut().table.push(weight.weights_loader)?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
            other_models: options
                .other_models
                .into_iter()
                .map(|other_model| {
                    Ok(bindings::exports::decthings::model::model::OtherModel {
                        model_id: other_model.model_id,
                        mount_path: other_model.mount_path,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
        };

        let instantiated = self
            .bindings
            .decthings_model_model()
            .call_instantiate_model(&mut *store, &options)??;

        Ok(WasmInstantiated {
            instantiated,
            model: self.clone(),
        })
    }
}

pub struct EvaluateOptions {
    pub params: HashMap<String, Param>,
    pub expected_output_types: Vec<decthings_api::tensor::DecthingsParameterDefinition>,
}

pub struct EvaluateOutput {
    pub name: String,
    pub data: Vec<Vec<u8>>,
}

pub struct TrainOptions {
    pub params: HashMap<String, Param>,
    pub tracker: Box<dyn TrainTracker>,
}

pub struct GetWeightsOptions {
    pub weights_provider: Box<dyn WeightsProvider>,
}

pub struct WasmInstantiated {
    instantiated: ResourceAny,
    model: RunningWasmModel,
}

impl WasmInstantiated {
    /// Call the *evaluate* function on the running model.
    ///
    /// The function takes a set of input parameters and outputs some data.
    pub fn evaluate(
        &self,
        options: EvaluateOptions,
    ) -> Result<Vec<EvaluateOutput>, CallFunctionError> {
        let mut store = self.model.store.lock().unwrap();

        let options = bindings::exports::decthings::model::model::EvaluateOptions {
            params: options
                .params
                .into_iter()
                .map(|(name, param)| {
                    Ok(bindings::exports::decthings::model::model::Param {
                        name,
                        amount: param.amount,
                        total_byte_size: param.total_byte_size,
                        data_loader: store.data_mut().table.push(param.data_loader)?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
            expected_output_types: options
                .expected_output_types
                .into_iter()
                .map(
                    |x| bindings::exports::decthings::model::model::DecthingsParameterDefinition {
                        name: x.name,
                        required: x.required,
                        rules: bindings::exports::decthings::model::model::DecthingsTensorRules {
                            shape: x.rules.shape,
                            allowed_types: x
                                .rules
                                .allowed_types
                                .into_iter()
                                .map(|y| match y {
                                    decthings_api::tensor::DecthingsElementType::F32 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::F32
                                    }
                                    decthings_api::tensor::DecthingsElementType::F64 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::F64
                                    }
                                    decthings_api::tensor::DecthingsElementType::I8 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::I8
                                    }
                                    decthings_api::tensor::DecthingsElementType::I16 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::I16
                                    }
                                    decthings_api::tensor::DecthingsElementType::I32 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::I32
                                    }
                                    decthings_api::tensor::DecthingsElementType::I64 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::I64
                                    }
                                    decthings_api::tensor::DecthingsElementType::U8 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::U8
                                    }
                                    decthings_api::tensor::DecthingsElementType::U16 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::U16
                                    }
                                    decthings_api::tensor::DecthingsElementType::U32 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::U32
                                    }
                                    decthings_api::tensor::DecthingsElementType::U64 => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::U64
                                    }
                                    decthings_api::tensor::DecthingsElementType::String => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::String
                                    }
                                    decthings_api::tensor::DecthingsElementType::Boolean => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::Boolean
                                    }
                                    decthings_api::tensor::DecthingsElementType::Binary => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::Binary
                                    }
                                    decthings_api::tensor::DecthingsElementType::Image => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::Image
                                    }
                                    decthings_api::tensor::DecthingsElementType::Audio => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::Audio
                                    }
                                    decthings_api::tensor::DecthingsElementType::Video => {
                                        bindings::exports::decthings::model::model::DecthingsElementType::Video
                                    }
                                })
                                .collect(),
                        },
                    },
                )
                .collect(),
        };

        let res = self
            .model
            .bindings
            .decthings_model_model()
            .instantiated()
            .call_evaluate(&mut *store, self.instantiated, &options)??;

        Ok(res
            .into_iter()
            .map(|output| EvaluateOutput {
                name: output.name,
                data: output.data,
            })
            .collect())
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
    pub fn train(&self, options: TrainOptions) -> Result<(), CallFunctionError> {
        let mut store = self.model.store.lock().unwrap();

        let options = bindings::exports::decthings::model::model::TrainOptions {
            params: options
                .params
                .into_iter()
                .map(|(name, param)| {
                    Ok(bindings::exports::decthings::model::model::Param {
                        name,
                        amount: param.amount,
                        total_byte_size: param.total_byte_size,
                        data_loader: store.data_mut().table.push(param.data_loader)?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
            tracker: store.data_mut().table.push(options.tracker)?,
        };

        self.model
            .bindings
            .decthings_model_model()
            .instantiated()
            .call_train(&mut *store, self.instantiated, &options)??;

        Ok(())
    }

    /// Call the *get_weights* function on the running model.
    ///
    /// The function outputs the model weights. If the *train* function was called on this
    /// instantiated model, the function will output the new, trained weights.
    pub fn get_weights(&self, options: GetWeightsOptions) -> Result<(), CallFunctionError> {
        let mut store = self.model.store.lock().unwrap();

        let options = bindings::exports::decthings::model::model::GetWeightsOptions {
            weights_provider: store.data_mut().table.push(options.weights_provider)?,
        };

        self.model
            .bindings
            .decthings_model_model()
            .instantiated()
            .call_get_weights(&mut *store, self.instantiated, options)??;

        Ok(())
    }

    /// Dispose the instantiated model.
    ///
    /// The function will deallocate any resources used by the instantiated model. To avoid a
    /// memory leak, this function must be called when you are done with the instantiated model.
    /// Simply dropping the instantiated model struct will not free up all resources.
    pub fn dispose(self) -> wasmtime::Result<()> {
        let mut store = self.model.store.lock().unwrap();
        self.instantiated.resource_drop(&mut *store)
    }
}
