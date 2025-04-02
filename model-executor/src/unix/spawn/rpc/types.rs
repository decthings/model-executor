use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum DataEvent {
    #[serde(rename_all = "camelCase")]
    RequestData {
        dataset: String,
        request_id: u32,
        start_index: u32,
        amount: u32,
    },
    #[serde(rename_all = "camelCase")]
    Shuffle { datasets: Vec<String> },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Param {
    pub name: String,
    pub dataset: String,
    pub amount: u32,
    pub total_byte_size: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OtherModelWithWeights {
    pub id: String,
    pub mount_path: String,
    pub weights: Vec<Param>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeCommand {
    pub path: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeWeightsCommand {
    pub params: Vec<Param>,
    pub other_models: Vec<OtherModelWithWeights>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum InitializeWeightsError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeWeightsResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<InitializeWeightsError>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OtherModel {
    pub id: String,
    pub mount_path: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstantiateModelCommand {
    pub instantiated_model_id: String,
    pub weights: Vec<Param>,
    pub other_models: Vec<OtherModel>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum InstantiateModelError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Disposed,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstantiateModelResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<InstantiateModelError>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DisposeInstantiatedModelCommand {
    pub instantiated_model_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainCommand {
    pub instantiated_model_id: String,
    pub training_session_id: String,
    pub params: Vec<Param>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum TrainError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    InstantiatedModelNotFound,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<TrainError>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelTrainCommand {
    pub training_session_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateCommand {
    pub instantiated_model_id: String,
    pub params: Vec<Param>,
    pub expected_output_types: Vec<decthings_api::tensor::DecthingsParameterDefinition>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum EvaluateError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    InstantiatedModelNotFound,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateOutput {
    pub name: String,
    pub byte_sizes: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EvaluateError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<EvaluateOutput>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetWeightsCommand {
    pub instantiated_model_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum GetWeightsError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    InstantiatedModelNotFound,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetWeightsResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GetWeightsError>,
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "method", content = "params")]
pub(crate) enum CommandMessageWithResponseWithId<'a> {
    CallInitializeWeights {
        id: &'a str,
        #[serde(flatten)]
        cmd: &'a InitializeWeightsCommand,
    },
    CallInstantiateModel {
        id: &'a str,
        #[serde(flatten)]
        cmd: &'a InstantiateModelCommand,
    },
    CallTrain {
        id: &'a str,
        #[serde(flatten)]
        cmd: &'a TrainCommand,
    },
    CallEvaluate {
        id: &'a str,
        #[serde(flatten)]
        cmd: &'a EvaluateCommand,
    },
    CallGetWeights {
        id: &'a str,
        #[serde(flatten)]
        cmd: &'a GetWeightsCommand,
    },
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CommandMessageWithResponse<'a> {
    CallInitializeWeights(&'a InitializeWeightsCommand),
    CallInstantiateModel(&'a InstantiateModelCommand),
    CallTrain(&'a TrainCommand),
    CallEvaluate(&'a EvaluateCommand),
    CallGetWeights(&'a GetWeightsCommand),
}

impl CommandMessageWithResponse<'_> {
    pub fn with_id<'a>(&'a self, id: &'a str) -> CommandMessageWithResponseWithId<'a> {
        match self {
            CommandMessageWithResponse::CallInitializeWeights(cmd) => {
                CommandMessageWithResponseWithId::CallInitializeWeights { id, cmd }
            }
            CommandMessageWithResponse::CallInstantiateModel(cmd) => {
                CommandMessageWithResponseWithId::CallInstantiateModel { id, cmd }
            }
            CommandMessageWithResponse::CallTrain(cmd) => {
                CommandMessageWithResponseWithId::CallTrain { id, cmd }
            }
            CommandMessageWithResponse::CallEvaluate(cmd) => {
                CommandMessageWithResponseWithId::CallEvaluate { id, cmd }
            }
            CommandMessageWithResponse::CallGetWeights(cmd) => {
                CommandMessageWithResponseWithId::CallGetWeights { id, cmd }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "method", content = "params")]
pub(crate) enum CommandMessageWithoutResponse<'a> {
    Initialize(&'a InitializeCommand),
    CallDisposeInstantiatedModel(&'a DisposeInstantiatedModelCommand),
    CallCancelTrain(&'a CancelTrainCommand),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultMessage {
    pub id: String,
    pub result: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum ModelSessionInitializedError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "event", content = "params")]
pub enum EventMessage {
    #[serde(rename_all = "camelCase")]
    ModelSessionInitialized {
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<ModelSessionInitializedError>,
    },
    #[serde(rename_all = "camelCase")]
    TrainingProgress {
        training_session_id: String,
        progress: f32,
    },
    #[serde(rename_all = "camelCase")]
    TrainingMetrics {
        training_session_id: String,
        names: Vec<String>,
    },
    #[serde(rename_all = "camelCase")]
    ProvideWeightsData {
        command_id: String,
        names: Vec<String>,
    },
}
