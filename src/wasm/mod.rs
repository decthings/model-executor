mod bindings;
pub use bindings::*;

use std::{collections::HashMap, ops::DerefMut, sync::Mutex};

use wasmtime::component::ResourceAny;

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
    pub data_loader: Box<dyn HostDataLoader>,
}

pub struct StateKey {
    pub byte_size: u64,
    pub state_loader: Box<dyn HostStateLoader>,
}

pub struct OtherModelWithState {
    pub model_id: String,
    pub mount_path: String,
    pub state: HashMap<String, StateKey>,
}

pub struct CreateModelStateOptions {
    pub params: HashMap<String, Param>,
    pub state_provider: Box<dyn HostStateProvider>,
    pub other_models: Vec<OtherModelWithState>,
}

pub struct OtherModel {
    pub model_id: String,
    pub mount_path: String,
}

pub struct InstantiateModelOptions {
    pub state: HashMap<String, StateKey>,
    pub other_models: Vec<OtherModel>,
}

pub struct RunningWasmModel {
    store: Mutex<wasmtime::Store<Host>>,
    bindings: ModelRunner,
}

impl RunningWasmModel {
    pub fn store(&self) -> impl DerefMut<Target = wasmtime::Store<Host>> + '_ {
        self.store.lock().unwrap()
    }

    pub fn call_create_model_state(
        &self,
        options: CreateModelStateOptions,
    ) -> Result<(), CallFunctionError> {
        let mut store = self.store.lock().unwrap();

        let options = exports::decthings::model::model::CreateModelStateOptions {
            params: options
                .params
                .into_iter()
                .map(|(name, param)| {
                    Ok(exports::decthings::model::model::Param {
                        name,
                        amount: param.amount,
                        total_byte_size: param.total_byte_size,
                        data_loader: store.data_mut().table.push(param.data_loader)?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
            state_provider: store.data_mut().table.push(options.state_provider)?,
            other_models: options
                .other_models
                .into_iter()
                .map(|other_model| {
                    Ok(exports::decthings::model::model::OtherModelWithState {
                        model_id: other_model.model_id,
                        mount_path: other_model.mount_path,
                        state: other_model
                            .state
                            .into_iter()
                            .map(|(key, state)| {
                                Ok(exports::decthings::model::model::StateKey {
                                    key,
                                    byte_size: state.byte_size,
                                    state_loader: store
                                        .data_mut()
                                        .table
                                        .push(state.state_loader)?,
                                })
                            })
                            .collect::<wasmtime::Result<_>>()?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
        };

        self.bindings
            .decthings_model_model()
            .call_create_model_state(&mut *store, &options)??;

        Ok(())
    }

    pub fn call_instantiate_model(
        &self,
        options: InstantiateModelOptions,
    ) -> Result<WasmInstantiated, CallFunctionError> {
        let mut store = self.store.lock().unwrap();

        let options = exports::decthings::model::model::InstantiateModelOptions {
            state: options
                .state
                .into_iter()
                .map(|(key, state)| {
                    Ok(exports::decthings::model::model::StateKey {
                        key,
                        byte_size: state.byte_size,
                        state_loader: store.data_mut().table.push(state.state_loader)?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
            other_models: options
                .other_models
                .into_iter()
                .map(|other_model| {
                    Ok(exports::decthings::model::model::OtherModel {
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
            model: self,
        })
    }
}

pub struct EvaluateOptions {
    pub params: HashMap<String, Param>,
}

pub struct EvaluateOutput {
    pub name: String,
    pub data: Vec<Vec<u8>>,
}

pub struct TrainOptions {
    pub params: HashMap<String, Param>,
    pub tracker: Box<dyn HostTrainTracker>,
}

pub struct GetModelStateOptions {
    pub state_provider: Box<dyn HostStateProvider>,
}

pub struct WasmInstantiated<'a> {
    instantiated: ResourceAny,
    model: &'a RunningWasmModel,
}

impl WasmInstantiated<'_> {
    pub fn call_evaluate(
        &self,
        options: EvaluateOptions,
    ) -> Result<Vec<EvaluateOutput>, CallFunctionError> {
        let mut store = self.model.store.lock().unwrap();

        let options = exports::decthings::model::model::EvaluateOptions {
            params: options
                .params
                .into_iter()
                .map(|(name, param)| {
                    Ok(exports::decthings::model::model::Param {
                        name,
                        amount: param.amount,
                        total_byte_size: param.total_byte_size,
                        data_loader: store.data_mut().table.push(param.data_loader)?,
                    })
                })
                .collect::<wasmtime::Result<_>>()?,
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

    pub fn call_train(&self, options: TrainOptions) -> Result<(), CallFunctionError> {
        let mut store = self.model.store.lock().unwrap();

        let options = exports::decthings::model::model::TrainOptions {
            params: options
                .params
                .into_iter()
                .map(|(name, param)| {
                    Ok(exports::decthings::model::model::Param {
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

    pub fn call_get_model_state(
        &self,
        options: GetModelStateOptions,
    ) -> Result<(), CallFunctionError> {
        let mut store = self.model.store.lock().unwrap();

        let options = exports::decthings::model::model::GetModelStateOptions {
            state_provider: store.data_mut().table.push(options.state_provider)?,
        };

        self.model
            .bindings
            .decthings_model_model()
            .instantiated()
            .call_get_model_state(&mut *store, self.instantiated, options)??;

        Ok(())
    }

    pub fn dispose(self) -> wasmtime::Result<()> {
        let mut store = self.model.store.lock().unwrap();
        self.instantiated.resource_drop(&mut *store)
    }
}

pub fn run_wasm(
    wasi: wasmtime_wasi::WasiCtx,
    engine: &wasmtime::Engine,
    component: &wasmtime::component::Component,
) -> wasmtime::Result<RunningWasmModel> {
    let mut store = wasmtime::Store::new(
        engine,
        Host {
            table: wasmtime::component::ResourceTable::new(),
            wasi,
        },
    );

    let mut linker = wasmtime::component::Linker::new(store.engine());
    ModelRunner::add_to_linker(&mut linker, |state| state).unwrap();
    wasmtime_wasi::add_to_linker_sync(&mut linker).unwrap();

    let (bindings, _) = ModelRunner::instantiate(&mut store, &component, &linker).unwrap();

    Ok(RunningWasmModel {
        store: Mutex::new(store),
        bindings,
    })
}
