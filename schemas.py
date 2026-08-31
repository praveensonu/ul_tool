from enum import Enum
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LoadMethod(str, Enum):
    full = "full"
    lora = "lora"
    adaptor = "adaptor"


class LoadModelRequest(BaseModel):
    base_model_path: str
    adaptor_path: Optional[str] = None
    hf_key: Optional[str] = None
    method: LoadMethod


class LoadModelResponse(BaseModel):
    status: str
    method: str
    base_model_path: str
    adaptor_path: Optional[str] = None
    merged: bool
    message: str


class DatasetUploadResponse(BaseModel):
    status: str
    forget_set_path: str
    retain_set_path: Optional[str]
    forget_rows: int
    retain_rows: Optional[int]
    has_retain_set: bool
    prompt_template: str
    forget_preview: List[Dict[str, Any]]
    retain_preview: Optional[List[Dict[str, Any]]]
    message: str


class LoraTargetModule(str, Enum):
    q_proj = "q_proj"
    v_proj = "v_proj"
    k_proj = "k_proj"
    o_proj = "o_proj"


class GeneralHyperParams(BaseModel):
    max_steps: Optional[int] = Field(None, ge=1)
    epochs: Optional[int] = Field(None, ge=1)
    learning_rate: Union[float, str]
    context_length: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_steps_or_epochs(self):
        if self.max_steps is None and self.epochs is None:
            raise ValueError("Either max_steps or epochs must be provided.")
        if self.max_steps is not None and self.epochs is not None:
            raise ValueError("Only one of max_steps or epochs can be provided.")
        return self

    @field_validator("context_length")
    @classmethod
    def validate_power_of_two(cls, value):
        if value & (value - 1) != 0:
            raise ValueError("context_length must be a power of 2.")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value):
        value = float(value)
        if value <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        return value


class LoraSettings(BaseModel):
    target_modules: List[LoraTargetModule]


class OptimizationHyperParams(BaseModel):
    batch_size: int = Field(..., ge=1)
    grad_accum: int = Field(..., ge=1)
    weight_decay: float = Field(..., ge=0)


class ScheduleHyperParams(BaseModel):
    save_steps: int = Field(..., ge=1)


class MemoryHyperParams(BaseModel):
    assistant_completions_only: bool = True

    @field_validator("assistant_completions_only")
    @classmethod
    def must_be_true(cls, value):
        if value is not True:
            raise ValueError("assistant_completions_only must always be true.")
        return value


class HyperParamsConfig(BaseModel):
    general: GeneralHyperParams
    optimization: OptimizationHyperParams
    schedule: ScheduleHyperParams
    memory: MemoryHyperParams
    lora_settings: Optional[LoraSettings] = None


class FinalTrainingConfigRequest(BaseModel):
    model_name: str
    adaptor_path: Optional[str] = None
    hf_key: Optional[str] = None
    method: LoadMethod
    gpu_id: int

    forget_set_path: str
    retain_set_path: Optional[str] = None

    hyperparams: HyperParamsConfig

    @model_validator(mode="after")
    def validate_method_specific_config(self):
        if self.method == LoadMethod.lora:
            if self.hyperparams.lora_settings is None:
                raise ValueError("lora_settings is required when method='lora'.")

        if self.method != LoadMethod.lora:
            if self.hyperparams.lora_settings is not None:
                raise ValueError("lora_settings is only allowed when method='lora'.")

        if self.method == LoadMethod.adaptor:
            if self.adaptor_path is None:
                raise ValueError("adaptor_path is required when method='adaptor'.")

        return self


class FinalTrainingConfigResponse(BaseModel):
    status: str
    orchestrator_config: Dict[str, Any]
    message: str


class TrainRunSuccessResponse(BaseModel):
    status: Literal["success"]
    orchestrator_config: Dict[str, Any]
    result: Dict[str, Any]


class TrainRunStoppedResponse(BaseModel):
    status: Literal["stopped"]
    orchestrator_config: Dict[str, Any]
    result: None = None
    message: str


TrainRunResponse = Union[TrainRunSuccessResponse, TrainRunStoppedResponse]


class TrainStopResponse(BaseModel):
    status: Literal["stopped", "idle"]
    message: str


class EvaluationTrainingResult(BaseModel):
    """The part of a completed training response needed for evaluation."""

    model_config = ConfigDict(extra="allow")

    output_dir: str = Field(..., min_length=1)


class EvaluationRequest(BaseModel):
    orchestrator_config: Dict[str, Any]
    training_result: EvaluationTrainingResult
    embedding_model_name: str = Field(..., min_length=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    max_new_tokens: int = Field(default=256, ge=1)
    evaluation_output_dir: Optional[str] = None

    @model_validator(mode="after")
    def validate_evaluation_inputs(self):
        model_config = self.orchestrator_config.get("model")
        dataset_config = self.orchestrator_config.get("dataset")
        gpu_config = self.orchestrator_config.get("gpu")

        if not isinstance(model_config, dict) or not model_config.get("model_name"):
            raise ValueError("orchestrator_config.model.model_name is required.")
        if not isinstance(dataset_config, dict):
            raise ValueError("orchestrator_config.dataset is required.")
        if not dataset_config.get("forget_set_path"):
            raise ValueError(
                "orchestrator_config.dataset.forget_set_path is required."
            )
        if not dataset_config.get("retain_set_path"):
            raise ValueError(
                "orchestrator_config.dataset.retain_set_path is required to "
                "compute model utility."
            )
        if not isinstance(gpu_config, dict) or not isinstance(
            gpu_config.get("gpu_id"), int
        ):
            raise ValueError("orchestrator_config.gpu.gpu_id is required.")
        if gpu_config["gpu_id"] < 0:
            raise ValueError("orchestrator_config.gpu.gpu_id must be non-negative.")

        return self


class ForgetQualityScores(BaseModel):
    evaluated_rows: int
    score: float
    average_perplexity: float
    mean_conditional_probability: float
    mean_rouge_l: float
    component_scores: List[float]


class ModelUtilityScores(BaseModel):
    evaluated_rows: int
    score: float
    average_perplexity: float
    mean_conditional_probability: float
    mean_rouge_l: float
    mean_cosine_similarity: float
    component_scores: List[float]


class ModelEvaluationScores(BaseModel):
    forget_quality: ForgetQualityScores
    model_utility: ModelUtilityScores


class EvaluationOutputFiles(BaseModel):
    pre_forget_scores_path: str
    pre_retain_scores_path: str
    post_forget_scores_path: str
    post_retain_scores_path: str


class EvaluationResponse(BaseModel):
    status: Literal["success"]
    model_path: str
    embedding_model_name: str
    forget_set_path: str
    retain_set_path: str
    pre_unlearning: ModelEvaluationScores
    post_unlearning: ModelEvaluationScores
    output_files: EvaluationOutputFiles
    message: str


class EvaluationStartResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"]
    message: str


class EvaluationProgressEvent(BaseModel):
    stage: str
    message: str
    timestamp: str


class EvaluationStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    current_stage: str
    message: str
    progress: List[EvaluationProgressEvent]
    result: Optional[EvaluationResponse] = None
    error: Optional[str] = None
