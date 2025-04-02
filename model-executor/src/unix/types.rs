use std::{collections::HashMap, future::Future, pin::Pin};

use tokio::io::AsyncRead;

#[derive(Default)]
pub struct RunBinOptions<'a> {
    pub(crate) inherit_env: bool,
    pub(crate) with_command: Option<Box<dyn Fn(&mut tokio::process::Command) + Send + Sync + 'a>>,
}

impl<'a> RunBinOptions<'a> {
    pub fn inherit_env(&mut self, inherit_env: bool) -> &mut Self {
        self.inherit_env = inherit_env;
        self
    }

    pub fn with_command(
        &mut self,
        cb: impl Fn(&mut tokio::process::Command) + Send + Sync + 'a,
    ) -> &mut Self {
        self.with_command = Some(Box::new(cb));
        self
    }
}

#[derive(Default)]
pub struct RunNodeJsOptions<'a> {
    pub(crate) flags: Vec<&'a str>,
    pub(crate) inherit_env: bool,
    pub(crate) with_command: Option<Box<dyn Fn(&mut tokio::process::Command) + Send + Sync + 'a>>,
}

impl<'a> RunNodeJsOptions<'a> {
    pub fn node_js_flag(&mut self, flag: &'a str) -> &mut Self {
        self.flags.push(flag);
        self
    }

    pub fn inherit_env(&mut self, inherit_env: bool) -> &mut Self {
        self.inherit_env = inherit_env;
        self
    }

    pub fn with_command(
        &mut self,
        cb: impl Fn(&mut tokio::process::Command) + Send + Sync + 'a,
    ) -> &mut Self {
        self.with_command = Some(Box::new(cb));
        self
    }
}

#[derive(Default)]
pub struct RunPythonOptions<'a> {
    pub(crate) flags: Vec<&'a str>,
    pub(crate) inherit_env: bool,
    pub(crate) with_command: Option<Box<dyn Fn(&mut tokio::process::Command) + Send + Sync + 'a>>,
}

impl<'a> RunPythonOptions<'a> {
    pub fn python_flag(&mut self, flag: &'a str) -> &mut Self {
        self.flags.push(flag);
        self
    }

    pub fn inherit_env(&mut self, inherit_env: bool) -> &mut Self {
        self.inherit_env = inherit_env;
        self
    }

    pub fn with_command(
        &mut self,
        cb: impl Fn(&mut tokio::process::Command) + Send + Sync + 'a,
    ) -> &mut Self {
        self.with_command = Some(Box::new(cb));
        self
    }
}

#[derive(Debug)]
pub enum RunError {
    Std(std::io::Error),
    Exception { details: Option<String> },
}

pub trait DataLoader: Send + Sync {
    fn id(&self) -> u32;
    fn read(
        &self,
        start_index: u32,
        amount: u32,
    ) -> Pin<Box<dyn Future<Output = Box<dyn blob_stream::Blobs + Send + '_>> + Send + '_>>;
    fn shuffle(&self, others: &[u32]) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

pub trait WeightsProvider: Send + Sync {
    fn provide<'a>(
        &'a self,
        names: Vec<String>,
        blobs: Box<dyn blob_stream::Blobs + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

pub trait WeightsLoader: Send + Sync {
    fn read(
        &self,
    ) -> Pin<Box<dyn Future<Output = Pin<Box<dyn AsyncRead + Send + Sync + '_>>> + Send + '_>>;
}

impl<T: AsRef<[u8]> + Send + Sync> WeightsLoader for T {
    fn read(
        &self,
    ) -> Pin<Box<dyn Future<Output = Pin<Box<dyn AsyncRead + Send + Sync + '_>>> + Send + '_>> {
        Box::pin(async move { Box::pin(self.as_ref()) as Pin<Box<dyn AsyncRead + Send + Sync>> })
    }
}

pub trait TrainTracker: Send + Sync {
    fn progress(&self, progress: f32) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    fn metrics<'a>(
        &'a self,
        names: Vec<String>,
        blobs: Box<dyn blob_stream::Blobs + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

#[derive(Debug)]
pub enum CallFunctionError {
    Exception { details: Option<String> },
    Rpc(super::spawn::rpc::CallMethodOnChildError),
}

/// A parameter to be passed to a model function. The data is lazy-loaded using the data loader.
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
    pub mount_path: String,
    pub weights: HashMap<String, WeightKey>,
}

pub struct OtherModel {
    pub mount_path: String,
}

pub struct InitializeWeightsOptions {
    pub params: HashMap<String, Param>,
    pub weights_provider: Box<dyn WeightsProvider>,
    pub other_models: HashMap<String, OtherModelWithWeights>,
}

pub struct InstantiateModelOptions {
    pub weights: HashMap<String, WeightKey>,
    pub other_models: HashMap<String, OtherModel>,
}

pub struct EvaluateOptions<
    F: for<'b> FnOnce(
            Result<
                (
                    Vec<super::spawn::rpc::types::EvaluateOutput>,
                    Box<dyn blob_stream::Blobs + Send + 'b>,
                ),
                CallFunctionError,
            >,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'b>>
        + Send
        + 'static,
> {
    pub params: HashMap<String, Param>,
    pub expected_output_types: Vec<decthings_api::tensor::DecthingsParameterDefinition>,
    pub result_cb: F,
}

pub struct TrainOptions {
    pub params: HashMap<String, Param>,
    pub tracker: Box<dyn TrainTracker>,
}

pub struct GetWeightsOptions {
    pub weights_provider: Box<dyn WeightsProvider>,
}
