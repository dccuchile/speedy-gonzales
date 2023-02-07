import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from trainers import (
    DistillationTrainer,
    LearnedMultipleTeacherDistillationTrainer,
    # MultipleTeacherDistillationTrainer,
    Trainer,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from datasets import load_dataset

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

    train_file: str = field(
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: str = field(
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: str = field(
        metadata={"help": "A csv or a json file containing the test data."},
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
    multiple_teacher_model_name_or_path: Optional[list[str]] = field(
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
    train_with_multiple_teacher_kd: bool = field(
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
    elif training_args.train_with_multiple_teacher_kd:
        if (
            model_args.student_model_name_or_path is None
            or model_args.multiple_teacher_model_name_or_path is None
        ):
            raise ValueError(
                "--train_with_multiple_teacher requires student and teacher[s] models"
            )
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
        "json",
        data_files={
            "train": data_args.train_file,
            "validation": data_args.validation_file,
            "test": data_args.test_file,
        },
    )

    label_list = dataset["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in id2label.items()}

    dataset = dataset.map(
        lambda example: {"labels": label2id[example["label"]]}, remove_columns=["label"]
    )

    other_columns = [
        column
        for column in dataset["train"].column_names
        if column not in ["labels", "sentence1", "sentence2"]
    ]
    dataset = dataset.remove_columns(other_columns)

    trainer = None

    if training_args.train_with_kd:
        student_tokenizer = AutoTokenizer.from_pretrained(
            model_args.student_model_name_or_path
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            model_args.teacher_model_name_or_path
        )
        student_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.student_model_name_or_path, num_labels=num_labels
        )
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model_name_or_path, num_labels=num_labels
        )

        trainer = DistillationTrainer(
            student_model,
            teacher_model,
            student_tokenizer,
            teacher_tokenizer,
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
    elif training_args.train_with_multiple_teacher_kd:
        teacher_models, teacher_tokenizers = [], []
        for teacher in model_args.multiple_teacher_model_name_or_path:
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher)
            teacher_model = AutoModelForSequenceClassification.from_pretrained(
                teacher, num_labels=num_labels
            )

            teacher_models.append(teacher_model)
            teacher_tokenizers.append(teacher_tokenizer)

        student_tokenizer = AutoTokenizer.from_pretrained(
            model_args.student_model_name_or_path
        )
        student_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.student_model_name_or_path, num_labels=num_labels
        )

        trainer = LearnedMultipleTeacherDistillationTrainer(
            student_model,
            teacher_models,
            student_tokenizer,
            teacher_tokenizers,
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
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_labels=num_labels
        )

        trainer = Trainer(
            model,
            tokenizer,
            dataset,
            training_args.device,
            training_args.per_device_train_batch_size,
            training_args.num_train_epochs,
            training_args.learning_rate,
            training_args.output_dir,
        )

    trainer.train()


if __name__ == "__main__":
    main()
