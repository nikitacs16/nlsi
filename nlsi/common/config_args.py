from dataclasses import dataclass, field



@dataclass(frozen=True)
class ModelArguments:
    """
    Configurations for the generation part
    """

    model_type: str = field(
        default="gpt3",
        metadata={
            "help": "model to use for generation. Only gpt3 is supported as of now. Also use interchangebly with huggingface model"
        },
    )
    model_st_type: str = field(
        default="all-MiniLM-L6-v2",
        metadata={"help": "model to use for sentence transformer"},
    )
    temperature: float = field(
        default=0.0, metadata={"help": "value of temperature used in sampling"}
    )
    #top_p: float = field(default=0.9, metadata={"help": "value of p in top-p sampling"})
    debug_mode: bool = field(default=False, metadata={"help": "prints extra details"})
    max_examples_in_prompt: int = field(
        default=10, metadata={"help": "max_examples_in_prompt"}
    )
    max_length: int = field(
        default=4000, metadata={"help": "max length of generated text"}
    )
    prompt_example_similarity_method: str = field(
        default="bm25", metadata={"help": "prompt example selection method"}
    )
    prompt_instruction: str = field(default="", metadata={"help": "prompt instruction"})
    gpt_engine: str = field(
        default="gpt-4",
        metadata={
            "help": "gpt engine to use for generation. Only gpt3 is supported as of now. "
        },
    )
    is_most_to_least_similar: bool = field(
        default=True,
        metadata={
            "help": "Arrange the examples in the prompt from most to least similar"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size for generation"})
    use_huggingface: bool = field(
        default=False, metadata={"help": "use huggingface model for generation"}
    )
    use_llama: bool = field(
        default=False, metadata={"help": "use llama head for generation"}
    )
    torch_dtype: str = field(
        default="float32", metadata={"help": "torch dtype to use for generation"}
    )
    device: str = field(
        default="cuda", metadata={"help": "device to use for generation"}
    
    )


@dataclass(frozen=True)
class DataArguments:
    """
    Data configuration. Definint training data, prefix data, test data to use
    """

    split_to_use_for_eval: str = field(
        default="test", metadata={"help": "split to use for evaluation. val, test"}
    )
    data_version: str = field(
        default="v1", metadata={"help": "version of data to use. v1, v2, .."}
    )

    custom_file: str = field(
        default=None,
        metadata={
            "help": "file containing the results of rewriter or selector to be used as input for the sv parser"
        },
    )


@dataclass(frozen=True)
class ExperimentArguments:
    """
    Experiment configurations
    """

    exp_name: str = field(
        default="default", metadata={"help": "For logging and dumping file outputs etc"}
    )
    num_examples_to_use: int = field(
        default=-1,
        metadata={"help": "max number of data points from validation_data to test"},
    )
    num_examples_to_use_train: int = field(
        default=-1,
        metadata={"help": "max number of data points from train_data to load"},
    )
    exp_type: str = field(default="selector", metadata={"help": "selector or rewriter"})
    continue_from_last_point: bool = field(
        default=False, metadata={"help": "continue from previously left point"}
    )
