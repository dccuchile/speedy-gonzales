import logging

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from models.ffnn import FFNNConfig, FFNNForSequenceClassification, FFNNModel
from models.lstm import BiLSTMConfig, BiLSTMForSequenceClassification, BiLSTMModel

logger = logging.getLogger(__name__)


# Register custom models to use them properly with AutoModel, AutoConfig, etc
AutoConfig.register("bilstm", BiLSTMConfig)
AutoConfig.register("ffnn", FFNNConfig)

AutoModel.register(BiLSTMConfig, BiLSTMModel)
AutoModel.register(FFNNConfig, FFNNModel)

AutoModelForSequenceClassification.register(
    BiLSTMConfig, BiLSTMForSequenceClassification
)
AutoModelForSequenceClassification.register(FFNNConfig, FFNNForSequenceClassification)


def get_model(task: str, *args, **kwargs) -> PreTrainedModel:
    if task == "sequence_classification":
        return AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
    elif task == "token_classification":
        return AutoModelForTokenClassification.from_pretrained(*args, **kwargs)
    else:
        raise ValueError(
            "task should be one the implemented ones: sequence_classification,"
            " token_classification"
        )


def get_model_from_config(task: str, **kwargs) -> PreTrainedModel:
    if task == "sequence_classification":
        return AutoModelForSequenceClassification.from_config(**kwargs)
    elif task == "token_classification":
        return AutoModelForTokenClassification.from_pretrained(**kwargs)
    else:
        raise ValueError(
            "task should be one the implemented ones: sequence_classification,"
            " token_classification"
        )


def get_config(*args, **kwargs) -> PretrainedConfig:
    return AutoConfig.from_pretrained(*args, **kwargs)


def get_tokenizer(*args, **kwargs) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(*args, **kwargs)


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    task: str = "sequence_classification",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = get_tokenizer(model_name, use_auth_token=True)
    model = get_model(
        task,
        model_name,
        use_auth_token=True,
        num_labels=num_labels,
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #    model_name, use_auth_token=True, num_labels=num_labels
    # )
    return model, tokenizer
