import itertools
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from trainers import (
    DistillationTrainer,
    QADistillationTrainer,
    TokenClassificationDistillationTrainer,
)
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from utils import get_model_and_tokenizer

from datasets import DatasetDict, load_dataset

logger = logging.getLogger()


class Experiment(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> dict[str, Any]:
        pass

    @abstractmethod
    def search_space(self) -> dict[str, Any]:
        pass


class GridStudy:
    def __init__(self, storage_file: str) -> None:
        self.storage_file = storage_file
        self.experiments_data: list[dict[str, Any]] = []

    def get_all_combinations(self, experiment: Experiment) -> list[dict[str, Any]]:
        search_space = experiment.search_space()
        combinations = itertools.product(*search_space.values())
        kwargs_combinations = []

        for combination in combinations:
            kwargs = {
                key: value for key, value in zip(search_space.keys(), combination)
            }
            kwargs_combinations.append(kwargs)

        return kwargs_combinations

    def run(self, experiment: Experiment) -> None:
        kwargs_combinations = self.get_all_combinations(experiment)

        for kwargs in kwargs_combinations:
            start_time = time.time()
            experiment_data = experiment(**kwargs)
            end_time = time.time()
            experiment_data.update({"experiment_elapsed_time": end_time - start_time})
            self.experiments_data.append(experiment_data)
            self.save_study()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.experiments_data)

    def save_study(self) -> None:
        study_df = self.to_dataframe()
        study_df.to_csv(self.storage_file, index=False)

    def load_study(self) -> None:
        study_df = pd.read_csv(self.storage_file)
        self.experiments_data = study_df.to_dict("records")


class TeacherSearchKDExperiment(Experiment):
    """
    This objective was created to experiment using KD with different teachers
    (with a fixed student) in a grid search way.
    """

    def __init__(
        self,
        student_model_name_or_path: str,
        dataset_name: str,
        device: str,
        datasets_dir: str,
        kd_alpha: float = 0,
        kd_temperature: float = 1,
    ) -> None:
        self.student_model_name_or_path = student_model_name_or_path
        self.dataset_name = dataset_name
        self.device = device
        self.datasets_dir = datasets_dir
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

        self.num_labels, self.dataset = self.load_dataset()

    def load_dataset(self) -> tuple[int, DatasetDict]:
        dataset: DatasetDict = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.datasets_dir, "PAWS-X/es/pawsx-train.json"),
                "validation": os.path.join(
                    self.datasets_dir, "PAWS-X/es/pawsx-dev.json"
                ),
                "test": os.path.join(self.datasets_dir, "PAWS-X/es/pawsx-test.json"),
            },
        )

        label_list = dataset["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in id2label.items()}

        dataset = dataset.map(
            lambda example: {"labels": label2id[example["label"]]},
            remove_columns=["label"],
        )

        other_columns = [
            column
            for column in dataset["train"].column_names
            if column not in ["labels", "sentence1", "sentence2"]
        ]
        dataset = dataset.remove_columns(other_columns)

        return num_labels, dataset

    def __call__(
        self, teacher_model_name_or_path: str, kd_loss_type: str, learning_rate: float
    ) -> dict[str, Any]:

        teacher_model, teacher_tokenizer = get_model_and_tokenizer(
            teacher_model_name_or_path, self.num_labels
        )
        student_model, student_tokenizer = get_model_and_tokenizer(
            self.student_model_name_or_path, self.num_labels
        )

        output_dir = (
            f"/data/jcanete/kd-experiments/{self.dataset_name}"
            f"/student-{self.student_model_name_or_path.replace('/', '-')}"
            f"/{teacher_model_name_or_path.replace('/', '-')}"
        )

        trainer = DistillationTrainer(
            student_model,
            teacher_model,
            student_tokenizer,
            teacher_tokenizer,
            self.dataset,
            device=self.device,
            batch_size=64,
            epochs=50,
            learning_rate=learning_rate,
            kd_alpha=self.kd_alpha,
            kd_temperature=self.kd_temperature,
            kd_loss_type=kd_loss_type,
            load_best_model_at_end=True,
            output_dir=output_dir,
            early_stopping_patience=10,
        )
        completed_epochs = trainer.train()

        task_metrics = trainer.evaluate()

        # TODO: maybe is interesting save the time elapsed on the experiment
        to_save = {
            "student_model_name_or_path": self.student_model_name_or_path,
            "teacher_model_name_or_path": teacher_model_name_or_path,
            "learning_rate": learning_rate,
            "kd_alpha": self.kd_alpha,
            "kd_temperature": self.kd_temperature,
            "kd_loss_type": kd_loss_type,
            "completed_epochs": completed_epochs,
            "validation_loss": task_metrics["validation_loss"],
            "validation_accuracy": task_metrics["accuracy"],
        }

        return to_save

    def search_space(self) -> dict:
        models = []
        learning_rates = [1e-4, 5e-5]
        losses = ["mse", "ce", "kldiv"]
        if "pawsx" in self.dataset_name:
            models = [
                "CenIA/albert-base-spanish-finetuned-pawsx",
                "CenIA/bert-base-spanish-wwm-cased-finetuned-pawsx",
                "CenIA/roberta-large-bne-finetuned-pawsx",
                "CenIA/albert-xxlarge-spanish-finetuned-pawsx",
            ]
        return {
            "teacher_model_name_or_path": models,
            "kd_loss_type": losses,
            "learning_rate": learning_rates,
        }


class TasksKDExperiment(Experiment):
    def __init__(
        self,
        student_model_name_or_path: str,
        device: str,
        datasets_dir: str,
        base_output_dir: str,
        kd_loss_type: str = "kldiv",
        kd_alpha: float = 0,
        kd_temperature: float = 1,
    ) -> None:
        self.student_model_name_or_path = student_model_name_or_path
        self.device = device
        self.datasets_dir = datasets_dir
        self.base_output_dir = base_output_dir
        self.kd_loss_type = kd_loss_type
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature


class SequenceClassificationKDExperiment(TasksKDExperiment):
    def __call__(
        self, dataset_config: dict[str, str], learning_rate: float, batch_size: int
    ) -> dict[str, Any]:

        output_dir = os.path.join(
            self.base_output_dir,
            "seq-class-kd",
            dataset_config["dataset_id"],
            self.student_model_name_or_path.replace("/", "-"),
            f"bs_{batch_size}_lr_{learning_rate}",
        )

        dataset: DatasetDict = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.datasets_dir, dataset_config["train_file"]),
                "validation": os.path.join(
                    self.datasets_dir, dataset_config["validation_file"]
                ),
                "test": os.path.join(self.datasets_dir, dataset_config["test_file"]),
            },
        )

        label_list = dataset["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in id2label.items()}

        dataset = dataset.map(
            lambda example: {"labels": label2id[example["label"]]},
            remove_columns=["label"],
        )

        other_columns = [
            column
            for column in dataset["train"].column_names
            if column not in ["labels", "sentence1", "sentence2"]
        ]
        dataset = dataset.remove_columns(other_columns)

        student_tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_name_or_path
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            dataset_config["teacher_model"]
        )
        student_model = AutoModelForSequenceClassification.from_pretrained(
            self.student_model_name_or_path, num_labels=num_labels
        )
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            dataset_config["teacher_model"], num_labels=num_labels
        )

        trainer = DistillationTrainer(
            student_model,
            teacher_model,
            student_tokenizer,
            teacher_tokenizer,
            dataset,
            device=self.device,
            batch_size=batch_size,
            epochs=50,
            learning_rate=learning_rate,
            output_dir=output_dir,
            kd_alpha=self.kd_alpha,
            kd_temperature=self.kd_temperature,
            kd_loss_type=self.kd_loss_type,
            early_stopping_patience=10,
            load_best_model_at_end=True,
        )

        to_save = {
            "task_type": "sequence-classification",
            "dataset_id": dataset_config["dataset_id"],
            "student_model_name_or_path": self.student_model_name_or_path,
            "teacher_model_name_or_path": dataset_config["teacher_model"],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "kd_alpha": self.kd_alpha,
            "kd_temperature": self.kd_temperature,
            "kd_loss_type": self.kd_loss_type,
            "completed_epochs": -1,
            "validation_loss": -1,
            "validation_accuracy": -1,
        }

        try:
            completed_epochs = trainer.train()
        except RuntimeError:  # out of memory
            logger.error(f"Runtime error (probably out of memory) on config: {to_save}")
            return to_save

        task_metrics = trainer.evaluate()

        to_save["completed_epochs"] = completed_epochs
        to_save["validation_loss"] = task_metrics["validation_loss"]
        to_save["validation_accuracy"] = task_metrics["accuracy"]

        return to_save

    def search_space(self) -> dict[str, Any]:
        dataset_configs = [
            {
                "dataset_id": "mldoc",
                "train_file": "MLDoc/mldoc-train.json",
                "validation_file": "MLDoc/mldoc-dev.json",
                "test_file": "MLDoc/mldoc-test.json",
                "teacher_model": "CenIA/roberta-large-bne-finetuned-mldoc",
            },
            {
                "dataset_id": "pawsx",
                "train_file": "PAWS-X/es/pawsx-train.json",
                "validation_file": "PAWS-X/es/pawsx-dev.json",
                "test_file": "PAWS-X/es/pawsx-test.json",
                "teacher_model": "CenIA/albert-xxlarge-spanish-finetuned-pawsx",
            },
            {
                "dataset_id": "xnli",
                "train_file": "XNLI/xnli-train.json",
                "validation_file": "XNLI/xnli-dev.json",
                "test_file": "XNLI/xnli-test.json",
                "teacher_model": "CenIA/albert-xxlarge-spanish-finetuned-xnli",
            },
        ]
        learning_rates = [5e-5, 1e-4]
        batch_sizes = [64, 32, 16]
        return {
            "dataset_config": dataset_configs,
            "learning_rate": learning_rates,
            "batch_size": batch_sizes,
        }


class TokenClassificationKDExperiment(TasksKDExperiment):
    def __call__(
        self, dataset_config: dict[str, Any], learning_rate: float, batch_size: int
    ) -> dict[str, Any]:

        output_dir = os.path.join(
            self.base_output_dir,
            "token-class-kd",
            dataset_config["dataset_id"],
            self.student_model_name_or_path.replace("/", "-"),
            f"bs_{batch_size}_lr_{learning_rate}",
        )

        text_column = dataset_config["text_column_name"]
        label_column = dataset_config["label_column_name"]
        label_all_tokens = dataset_config["label_all_tokens"]

        dataset: DatasetDict = load_dataset(
            dataset_config["dataset_name"], dataset_config["dataset_config_name"]
        )

        label_list = dataset["train"].features[label_column].feature.names
        num_labels = len(label_list)

        # id2label = {i: label for i, label in enumerate(label_list)}
        # label2id = {label: i for i, label in id2label.items()}

        other_columns = [
            column
            for column in dataset["train"].column_names
            if column not in [text_column, label_column]
        ]
        dataset = dataset.remove_columns(other_columns)

        student_tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_name_or_path
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            dataset_config["teacher_model"]
        )
        student_model = AutoModelForTokenClassification.from_pretrained(
            self.student_model_name_or_path, num_labels=num_labels
        )
        teacher_model = AutoModelForTokenClassification.from_pretrained(
            dataset_config["teacher_model"], num_labels=num_labels
        )

        trainer = TokenClassificationDistillationTrainer(
            student_model,
            teacher_model,
            student_tokenizer,
            teacher_tokenizer,
            text_column,
            label_column,
            label_list,
            dataset,
            label_all_tokens=label_all_tokens,
            device=self.device,
            batch_size=batch_size,
            epochs=50,
            learning_rate=learning_rate,
            output_dir=output_dir,
            kd_alpha=self.kd_alpha,
            kd_temperature=self.kd_temperature,
            kd_loss_type=self.kd_loss_type,
            early_stopping_patience=10,
            load_best_model_at_end=True,
        )

        to_save = {
            "task_type": "token-classification",
            "dataset_id": dataset_config["dataset_id"],
            "student_model_name_or_path": self.student_model_name_or_path,
            "teacher_model_name_or_path": dataset_config["teacher_model"],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "kd_alpha": self.kd_alpha,
            "kd_temperature": self.kd_temperature,
            "kd_loss_type": self.kd_loss_type,
            "completed_epochs": -1,
            "validation_loss": -1,
            "validation_precision": -1,
            "validation_recall": -1,
            "validation_f1": -1,
            "validation_accuracy": -1,
        }

        try:
            completed_epochs = trainer.train()
        except RuntimeError:  # out of memory
            logger.error(f"Runtime error (probably out of memory) on config: {to_save}")
            return to_save

        task_metrics = trainer.evaluate()

        to_save["completed_epochs"] = completed_epochs
        to_save["validation_loss"] = task_metrics["validation_loss"]
        to_save["validation_precision"] = task_metrics["overall_precision"]
        to_save["validation_recall"] = task_metrics["overall_recall"]
        to_save["validation_f1"] = task_metrics["overall_f1"]
        to_save["validation_accuracy"] = task_metrics["overall_accuracy"]

        return to_save

    def search_space(self) -> dict[str, Any]:
        dataset_configs = [
            {
                "dataset_id": "pos",
                "dataset_name": "universal_dependencies",
                "dataset_config_name": "es_ancora",
                "text_column_name": "tokens",
                "label_column_name": "upos",
                "label_all_tokens": False,
                "teacher_model": "CenIA/roberta-base-bne-finetuned-pos",
            },
            {
                "dataset_id": "ner",
                "dataset_name": "conll2002",
                "dataset_config_name": "es",
                "text_column_name": "tokens",
                "label_column_name": "ner_tags",
                "label_all_tokens": False,
                "teacher_model": "CenIA/roberta-base-bne-finetuned-ner",
            },
        ]
        learning_rates = [5e-5, 1e-4]
        batch_sizes = [64, 32, 16]
        return {
            "dataset_config": dataset_configs,
            "learning_rate": learning_rates,
            "batch_size": batch_sizes,
        }


class QAKDExperiment(TasksKDExperiment):
    def __call__(
        self, dataset_config: dict[str, Any], learning_rate: float, batch_size: int
    ) -> dict[str, Any]:

        output_dir = os.path.join(
            self.base_output_dir,
            "qa-kd",
            dataset_config["dataset_id"],
            self.student_model_name_or_path.replace("/", "-"),
            f"bs_{batch_size}_lr_{learning_rate}",
        )

        dataset: DatasetDict = load_dataset(
            os.path.join(self.datasets_dir, dataset_config["loading_script_path"]),
            data_files={
                "train": os.path.join(self.datasets_dir, dataset_config["train_file"]),
                "validation": os.path.join(
                    self.datasets_dir, dataset_config["validation_file"]
                ),
                "test": os.path.join(self.datasets_dir, dataset_config["test_file"]),
            },
        )

        column_names = dataset["train"].column_names
        question_column_name = (
            "question" if "question" in column_names else column_names[0]
        )
        context_column_name = (
            "context" if "context" in column_names else column_names[1]
        )
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        column_names = dataset["train"].column_names

        student_tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_name_or_path
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            dataset_config["teacher_model"]
        )
        student_model = AutoModelForQuestionAnswering.from_pretrained(
            self.student_model_name_or_path
        )
        teacher_model = AutoModelForQuestionAnswering.from_pretrained(
            dataset_config["teacher_model"]
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
            max_seq_length=512,
            device=self.device,
            batch_size=batch_size,
            epochs=50,
            learning_rate=learning_rate,
            output_dir=output_dir,
            kd_alpha=self.kd_alpha,
            kd_temperature=self.kd_temperature,
            kd_loss_type=self.kd_loss_type,
            early_stopping_patience=10,
            load_best_model_at_end=True,
        )

        # TODO: maybe is interesting save the time elapsed on the experiment
        to_save = {
            "task_type": "question-answering",
            "dataset_id": dataset_config["dataset_id"],
            "student_model_name_or_path": self.student_model_name_or_path,
            "teacher_model_name_or_path": dataset_config["teacher_model"],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "kd_alpha": self.kd_alpha,
            "kd_temperature": self.kd_temperature,
            "kd_loss_type": self.kd_loss_type,
            "completed_epochs": -1,
            "validation_loss": -1,
            "validation_f1": -1,
            "validation_exact_match": -1,
        }

        try:
            completed_epochs = trainer.train()
        except RuntimeError:  # out of memory
            logger.error(f"Runtime error (probably out of memory) on config: {to_save}")
            return to_save

        task_metrics = trainer.evaluate()

        to_save["completed_epochs"] = completed_epochs
        to_save["validation_loss"] = task_metrics["validation_loss"]
        to_save["validation_f1"] = task_metrics["f1"]
        to_save["validation_exact_match"] = task_metrics["exact_match"]

        return to_save

    def search_space(self) -> dict[str, Any]:
        dataset_configs = [
            {
                "dataset_id": "mlqa",
                "loading_script_path": "QA/qa_datasets.py",
                "train_file": "QA/MLQA/mlqa-train.json",
                "validation_file": "QA/MLQA/mlqa-dev.json",
                "test_file": "QA/MLQA/mlqa-test.json",
                "teacher_model": "CenIA/albert-xxlarge-spanish-finetuned-qa-mlqa",
            },
            {
                "dataset_id": "sqac",
                "loading_script_path": "QA/qa_datasets.py",
                "train_file": "QA/SQAC/sqac-train.json",
                "validation_file": "QA/SQAC/sqac-dev.json",
                "test_file": "QA/SQAC/sqac-test.json",
                "teacher_model": "CenIA/albert-xxlarge-spanish-finetuned-qa-sqac",
            },
            {
                "dataset_id": "tar",
                "loading_script_path": "QA/qa_datasets.py",
                "train_file": "QA/TAR-XQuAD/tar-train.json",
                "validation_file": "QA/TAR-XQuAD/tar-dev.json",
                "test_file": "QA/TAR-XQuAD/xquad-test.json",
                "teacher_model": "CenIA/albert-xxlarge-spanish-finetuned-qa-tar",
            },
        ]
        learning_rates = [5e-5, 1e-4]
        batch_sizes = [64, 32, 16]
        return {
            "dataset_config": dataset_configs,
            "learning_rate": learning_rates,
            "batch_size": batch_sizes,
        }


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input
    our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    datasets_dir: str = field(
        metadata={"help": "The path of the directory that contains all datasets."},
    )
    output_dir: str = field(
        metadata={"help": "The base directory to store all models from experiments."}
    )


@dataclass
class ExperimentArguments:
    storage_file: str = field(
        metadata={"help": "The path to save the results of the experiments"}
    )
    experiment_type: str = field(
        metadata={
            "help": (
                "What kind of experiment to run, supported types are:"
                " 'teacher-search-kd', 'seq-class-kd', 'token-class-kd', 'qa-kd'"
            )
        }
    )
    device: str = field(
        default="cpu",
        metadata={"help": "The device to run the training/evaluation on."},
    )


@dataclass
class TeacherSearchKDExperimentArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use"}
    )


@dataclass
class TasksKDExperimentArguments:
    student_model_name_or_path: str = field(
        metadata={"help": "The name or path of the model to use as student"},
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
    kd_alpha: float = field(
        default=0.0, metadata={"help": "The alpha parameter of the KD loss"}
    )
    kd_temperature: float = field(
        default=1.0, metadata={"help": "The temperature parameter of the KD loss"}
    )


def main():
    parser = HfArgumentParser(
        (
            DataTrainingArguments,
            ExperimentArguments,
            TeacherSearchKDExperimentArguments,
            TasksKDExperimentArguments,
        )
    )
    (
        data_args,
        exp_args,
        ts_args,
        kd_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join("./", "train.log")),
        ],
    )
    logger.setLevel(logging.INFO)

    experiment = None

    if exp_args.experiment_type == "teacher-search-kd":
        experiment = TeacherSearchKDExperiment(
            kd_args.student_model_name_or_path,
            dataset_name=ts_args.dataset_name,
            device=exp_args.device,
            datasets_dir=data_args.datasets_dir,
            kd_alpha=kd_args.kd_alpha,
            kd_temperature=kd_args.kd_temperature,
        )

    elif exp_args.experiment_type == "seq-class-kd":
        experiment = SequenceClassificationKDExperiment(
            kd_args.student_model_name_or_path,
            exp_args.device,
            data_args.datasets_dir,
            data_args.output_dir,
            kd_loss_type=kd_args.kd_loss_type,
            kd_alpha=kd_args.kd_alpha,
            kd_temperature=kd_args.kd_temperature,
        )
    elif exp_args.experiment_type == "token-class-kd":
        experiment = TokenClassificationKDExperiment(
            kd_args.student_model_name_or_path,
            exp_args.device,
            data_args.datasets_dir,
            data_args.output_dir,
            kd_loss_type=kd_args.kd_loss_type,
            kd_alpha=kd_args.kd_alpha,
            kd_temperature=kd_args.kd_temperature,
        )
    elif exp_args.experiment_type == "qa-kd":
        experiment = QAKDExperiment(
            kd_args.student_model_name_or_path,
            exp_args.device,
            data_args.datasets_dir,
            data_args.output_dir,
            kd_loss_type=kd_args.kd_loss_type,
            kd_alpha=kd_args.kd_alpha,
            kd_temperature=kd_args.kd_temperature,
        )

    study = GridStudy(storage_file=exp_args.storage_file)
    study.run(experiment=experiment)


if __name__ == "__main__":
    main()
