import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForQuestionAnswering

from trainers import QATrainer, QADistillationTrainer

logger = logging.getLogger()


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input
    our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    loading_script_path: str = field(
        metadata={"help": "The path to the loading script"}
    )
    train_file: str = field(
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: str = field(
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: str = field(
        metadata={"help": "A csv or a json file containing the test data."},
    )
    text_column_name: str = field(
        default="tokens", metadata={"help": "The name of the text column"}
    )
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the label column"}
    )
    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by"
                " that word or just on the one (in which case the other tokens will"
                " have a padding index)."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        },
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        },
    )
    student_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        },
    )


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.
    """

    output_dir: str = field(
        metadata={
            "help": (
                "The output directory where the model predictions and checkpoints will"
                " be written."
            )
        },
    )
    train_with_kd: bool = field(
        default=False,
        metadata={
            "help": "Wheter to use knowledge distillation or just simple fine-tune."
        },
    )
    device: str = field(
        default="cpu",
        metadata={"help": "The device to run the training/evaluation on."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    kd_alpha: float = field(
        default=0.5, metadata={"help": "Alpha for knowledge distilation loss."}
    )
    kd_temperature: float = field(
        default=1, metadata={"help": "Temperature to use on softmax in kd"}
    )
    kd_loss_type: str = field(
        default="kldiv",
        metadata={
            "help": (
                "The kind of loss to use as KD loss, it should be one of 'ce', 'kldiv',"
                " 'mse'"
            )
        },
    )

    def __post_init__(self):
        if self.kd_alpha > 1 or self.kd_alpha < 0:
            raise ValueError("kd_alpha should be between 0 and 1")
        if self.kd_temperature == 0:
            raise ValueError("kd_temperature can not be zero")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.train_with_kd:
        if (
            model_args.student_model_name_or_path is None
            or model_args.teacher_model_name_or_path is None
        ):
            raise ValueError("--train_with_kd requires student and teacher models")
    else:
        if model_args.model_name_or_path is None:
            raise ValueError(
                "Simple fine-tuning (--train_with_kd False) requires model_name_or_path"
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.output_dir, "train.log")),
        ],
    )
    logger.setLevel(logging.INFO)

    dataset = load_dataset(
        data_args.loading_script_path,
        data_files={
            "train": os.path.abspath(data_args.train_file),
            "validation": os.path.abspath(data_args.validation_file),
            "test": os.path.abspath(data_args.test_file),
        },
    )

    column_names = dataset["train"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    column_names = dataset["train"].column_names

    trainer = None

    if training_args.train_with_kd:
        student_tokenizer = AutoTokenizer.from_pretrained(
            model_args.student_model_name_or_path
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            model_args.teacher_model_name_or_path
        )
        student_model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.student_model_name_or_path
        )
        teacher_model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.teacher_model_name_or_path
        )

        trainer = QADistillationTrainer(
            student_model,
            teacher_model,
            student_tokenizer,
            teacher_tokenizer,
            question_column_name,
            context_column_name,
            answer_column_name,
            dataset,
            device=training_args.device,
            batch_size=training_args.per_device_train_batch_size,
            epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            output_dir=training_args.output_dir,
            kd_alpha=training_args.kd_alpha,
            kd_temperature=training_args.kd_temperature,
            kd_loss_type=training_args.kd_loss_type,
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path
        )

        trainer = QATrainer(
            question_column_name,
            context_column_name,
            answer_column_name,
            model,
            tokenizer,
            dataset,
            device=training_args.device,
            batch_size=training_args.per_device_train_batch_size,
            epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            output_dir=training_args.output_dir,
        )

    trainer.train()


if __name__ == "__main__":
    main()
