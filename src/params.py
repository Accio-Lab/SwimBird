from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    max_latent_tokens: int = field(default=None)


@dataclass
class TrainingArguments(HFTrainingArguments):
    model_init_kwargs: Optional[dict] = field(default_factory=dict)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, 
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    # use_liger: bool = True
    run_name: Optional[str] = field(default="vscode debugger", metadata={"help": "Name of the run for logging purposes."})

    latent_loss: str = field(default="mse")
    latent_lambda: float = field(default=0.1)

    checkpoint_name: Optional[str] = None


@dataclass
class DataArguments:
    data_path: List[str] = field(
        default=None, metadata={"help": "Path to the training data.", "nargs": "+"}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    nframes: Optional[int] = field(default=None, metadata={"help": "Number of frames for video data."})
    random_seed: Optional[int] = field(default=None)
    max_latent_token: int = field(default=32)