use std::{collections::HashMap, future::Future, pin::Pin};

use tokio::io::AsyncRead;

pub struct RunBinOptions<'a> {
    pub(crate) inherit_env: bool,
    pub(crate) with_command: Option<Box<dyn Fn(&mut tokio::process::Command) + Send + Sync + 'a>>,
}

impl<'a> Default for RunBinOptions<'a> {
    fn default() -> Self {
        Self {
            inherit_env: false,
            with_command: None,
        }
    }
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

pub struct RunNodeJsOptions<'a> {
    pub(crate) flags: Vec<&'a str>,
    pub(crate) inherit_env: bool,
    pub(crate) with_command: Option<Box<dyn Fn(&mut tokio::process::Command) + Send + Sync + 'a>>,
}

impl<'a> Default for RunNodeJsOptions<'a> {
    fn default() -> Self {
        Self {
            flags: vec![],
            inherit_env: false,
            with_command: None,
        }
    }
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

pub struct RunPythonOptions<'a> {
    pub(crate) flags: Vec<&'a str>,
    pub(crate) inherit_env: bool,
    pub(crate) with_command: Option<Box<dyn Fn(&mut tokio::process::Command) + Send + Sync + 'a>>,
}

impl<'a> Default for RunPythonOptions<'a> {
    fn default() -> Self {
        Self {
            flags: vec![],
            inherit_env: false,
            with_command: None,
        }
    }
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

pub trait StateProvider: Send + Sync {
    fn provide<'a>(
        &'a self,
        names: Vec<String>,
        blobs: Box<dyn blob_stream::Blobs + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

pub trait StateLoader: Send + Sync {
    fn read(
        &self,
    ) -> Pin<Box<dyn Future<Output = Pin<Box<dyn AsyncRead + Send + '_>>> + Send + '_>>;
}

impl<T: AsRef<[u8]> + Send + Sync> StateLoader for T {
    fn read(
        &self,
    ) -> Pin<Box<dyn Future<Output = Pin<Box<dyn AsyncRead + Send + '_>>> + Send + '_>> {
        Box::pin(async move { Box::pin(self.as_ref()) as Pin<Box<dyn AsyncRead + Send>> })
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

pub struct StateKey {
    pub byte_size: u64,
    pub state_loader: Box<dyn StateLoader>,
}

pub struct OtherModelWithState {
    pub mount_path: String,
    pub state: HashMap<String, StateKey>,
}

pub struct OtherModel {
    pub mount_path: String,
}

pub struct CreateModelStateOptions {
    pub params: HashMap<String, Param>,
    pub state_provider: Box<dyn StateProvider>,
    pub other_models: HashMap<String, OtherModelWithState>,
}

pub struct InstantiateModelOptions {
    pub state: HashMap<String, StateKey>,
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
    pub result_cb: F,
}

pub struct TrainOptions {
    pub params: HashMap<String, Param>,
    pub tracker: Box<dyn TrainTracker>,
}

pub struct GetModelStateOptions {
    pub state_provider: Box<dyn StateProvider>,
}
