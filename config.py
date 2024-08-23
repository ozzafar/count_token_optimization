from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    # Exp setup
    train: bool
    evaluate: bool
    experiment: bool = False
    evaluate_experiment: bool = False
    experiment_name: str = False
    evaluate_tokens: bool = False
    evaluate_token_reuse: bool = False
    create_images_grid: bool = False
    create_human_study: bool = False

    amount: float = 7
    clazz: str = "oranges"
    _lambda: float = 10
    scale: float = 70
    token_clazz: str = None  # for token reuse evaluation

    is_dynamic_scale_factor: bool = False
    yolo_threshold: float = 0.5
    is_v2: bool = False
    is_controlnet: bool = False

    # Id of the experiment
    exp_id: str = "demo"

    # the counting model (Options: clip-count, clip)
    counting_model_name: str = "clip-count"

    diffusion_steps: int = 1

    # Affect training time
    early_stopping: int = 15
    num_train_epochs: int = 50

    # affect variability of the training images
    # i.e., also sets batch size with accumulation
    epoch_size: int = 1
    number_of_prompts: int = 1  # how many different prompts to use
    batch_size: int = 1  # set to one due to gpu constraints
    gradient_accumulation_steps: int = 1  # same as the epoch size

    # Skip if there exists a token checkpoint
    skip_exists: bool = False

    # Train and Optimization
    lr: float = 0.01
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    weight_decay: float = 1e-2
    eps: float = 1e-08
    max_grad_norm: str = "1"
    seed: int = 35

    # Generative model
    guidance_scale: int = 7
    height: int = 512
    width: int = 512

    # Discrimnative tokens
    placeholder_token: str = "some"
    initializer_token: str = "some"

    # Path to save all outputs to
    output_path: Path = Path("results")
    save_as_full_pipeline = True

    # Cuda related
    device: str = "cuda"
    mixed_precision = "no"
    gradient_checkpointing = True

    # evaluate
    test_size: int = 10


def __post_init__(self):
    self.output_path.mkdir(exist_ok=True, parents=True)
