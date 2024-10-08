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
pub struct OtherModelWithState {
    pub id: String,
    pub mount_path: String,
    pub state: Vec<Param>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeCommand {
    pub path: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateModelStateCommand {
    pub params: Vec<Param>,
    pub other_models: Vec<OtherModelWithState>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum CreateModelStateError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateModelStateResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<CreateModelStateError>,
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
    pub state: Vec<Param>,
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
pub struct GetModelStateCommand {
    pub instantiated_model_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum GetModelStateError {
    Exception {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    InstantiatedModelNotFound,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetModelStateResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GetModelStateError>,
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "method", content = "params")]
pub(crate) enum CommandMessageWithResponseWithId<'a> {
    CallCreateModelState {
        id: &'a str,
        #[serde(flatten)]
        cmd: &'a CreateModelStateCommand,
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
    CallGetModelState {
        id: &'a str,
        #[serde(flatten)]
        cmd: &'a GetModelStateCommand,
    },
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CommandMessageWithResponse<'a> {
    CallCreateModelState(&'a CreateModelStateCommand),
    CallInstantiateModel(&'a InstantiateModelCommand),
    CallTrain(&'a TrainCommand),
    CallEvaluate(&'a EvaluateCommand),
    CallGetModelState(&'a GetModelStateCommand),
}

impl CommandMessageWithResponse<'_> {
    pub fn with_id<'a>(&'a self, id: &'a str) -> CommandMessageWithResponseWithId<'a> {
        match self {
            CommandMessageWithResponse::CallCreateModelState(cmd) => {
                CommandMessageWithResponseWithId::CallCreateModelState { id, cmd }
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
            CommandMessageWithResponse::CallGetModelState(cmd) => {
                CommandMessageWithResponseWithId::CallGetModelState { id, cmd }
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
    ProvideStateData {
        command_id: String,
        names: Vec<String>,
    },
}
