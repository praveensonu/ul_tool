from enum import Enum
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


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
