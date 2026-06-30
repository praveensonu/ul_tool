import os
import sys
from types import SimpleNamespace
import inspect
from typing import Any, Dict, Optional


def dict_to_namespace(d: Dict[str, Any]):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def build_training_args(api_config: Dict[str, Any]):
    from transformers import TrainingArguments

    hp = api_config["hyperparams"]
    output_dir = api_config.get("output_dir", "outputs/run")

    kwargs = {
        "output_dir": output_dir,
        "learning_rate": float(hp["general"]["learning_rate"]),
        "per_device_train_batch_size": hp["optimization"]["batch_size"],
        "save_strategy": "steps",
        "save_steps": hp["schedule"]["save_steps"],
        "weight_decay": hp["optimization"]["weight_decay"],
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 10,
        "label_names": ["labels"],
        "bf16": True,
        "gradient_accumulation_steps": hp["optimization"]["grad_accum"],
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
    }

    if hp["general"].get("max_steps") is not None:
        kwargs["max_steps"] = hp["general"]["max_steps"]

    if hp["general"].get("epochs") is not None:
        kwargs["num_train_epochs"] = hp["general"]["epochs"]

    # transformers changed eval_strategy/evaluation_strategy across versions
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"

    return TrainingArguments(**kwargs)


def build_trainer_kwargs(
    trainer_cls,
    model,
    tokenizer,
    training_args,
    train_dataset,
    data_collator,
):
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }

    # transformers newer versions use processing_class instead of tokenizer
    sig = inspect.signature(trainer_cls.__init__)
    trainer_sig = inspect.signature(__import__("transformers").Trainer.__init__)

    if "processing_class" in trainer_sig.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return kwargs

def get_trainable_parameter_counts(model):
    trainable_params = 0
    total_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return trainable_params, total_params

def run_orchestrator(api_config: Dict[str, Any]) -> Dict[str, Any]:
    gpu_id = api_config["gpu"]["gpu_id"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    snpo_path = os.path.abspath("unlearning/snpo")
    if snpo_path not in sys.path:
        sys.path.insert(0, snpo_path)

    from accelerate import Accelerator

    from dataset.dataset_loader import read_file
    from model.model_loader import load_model_by_method

    from unlearning.data_helpers.data_module import (
        ForgetOnlyDataset,
        ForgetRetainDataset,
    )
    from unlearning.data_helpers.collators import (
        ForgetCollator,
        RetainCollator,
    )
    from unlearning.snpo.trainer import (
        SimNPOForgetOnlyTrainer,
        SimNPOForgetRetainTrainer,
    )

    accelerator = Accelerator()

    model_cfg = api_config["model"]
    data_cfg = api_config["dataset"]
    hp = api_config["hyperparams"]

    model, tokenizer, merged = load_model_by_method(
        method=model_cfg["method"],
        base_model_path=model_cfg["model_name"],
        adaptor_path=model_cfg.get("adaptor_path"),
        gpu_id=0,  # because CUDA_VISIBLE_DEVICES remaps selected GPU to cuda:0
        hf_key=api_config["model"].get("hf_key"),
    )

    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    trainable_params, total_params = get_trainable_parameter_counts(model)
    if trainable_params == 0:
        raise ValueError(
            "Loaded model has no trainable parameters. "
            "For method='adaptor', keep the PEFT adapter attached instead of "
            "calling merge_and_unload() before training."
        )

    forget_df = read_file(data_cfg["forget_set_path"])

    retain_path = data_cfg.get("retain_set_path")
    retain_df = read_file(retain_path) if retain_path else None

    context_length = hp["general"]["context_length"]

    training_args = build_training_args(api_config)

    if retain_df is None:
        train_dataset = ForgetOnlyDataset(
            forget_data=forget_df,
            tokenizer=tokenizer,
            max_length=context_length,
            question_key="question",
            answer_key="answer",
        )

        trainer_kwargs = build_trainer_kwargs(
            trainer_cls=SimNPOForgetOnlyTrainer,
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=ForgetCollator,
        )

        trainer = SimNPOForgetOnlyTrainer(
            delta=0.0,
            beta=3.5,
            **trainer_kwargs,
        )

        run_type = "forget_only"

    else:
        train_dataset = ForgetRetainDataset(
            forget_data=forget_df,
            retain_data=retain_df,
            tokenizer=tokenizer,
            max_length=context_length,
            question_key="question",
            answer_key="answer",
        )

        trainer_kwargs = build_trainer_kwargs(
            trainer_cls=SimNPOForgetRetainTrainer,
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=RetainCollator,
        )

        trainer = SimNPOForgetRetainTrainer(
            delta=0.0,
            beta=3.5,
            **trainer_kwargs,
        )

        run_type = "forget_retain"

    result = trainer.train()

    accelerator.wait_for_everyone()

    output_dir = os.path.join(
        api_config.get("output_dir", "outputs/run"),
        run_type,
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    import torch
    import gc
    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "status": "success",
        "run_type": run_type,
        "merged_adapter": merged,
        "output_dir": output_dir,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "metrics": getattr(result, "metrics", {}),
    }
