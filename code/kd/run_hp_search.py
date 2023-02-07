import copy
from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Optional
import torch

from benchmark import PerformanceBenchmark
import optuna
from models.lstm import BiLSTMConfig, BiLSTMForSequenceClassification
from models.ffnn import FFNNConfig, FFNNForSequenceClassification
from trainers import DistillationTrainer, Trainer
from utils import get_model_and_tokenizer
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    AlbertConfig,
    AlbertForSequenceClassification,
    AutoModel,
)
from datasets import load_dataset, Dataset


logger = logging.getLogger()


class FTObjective(object):
    """
    This objective was created to experiment with different architectures as student
    model. It allows to choose between bilstm, ffnn, and transformer (albert) and finds
    the best suited configuration for that architecture.
    """

    def __init__(
        self,
        model_type: str,
        num_labels: int,
        dataset: Dataset,
        device: str,
        allow_pretrained_embeddings: bool,
        allow_albert_pretrained: bool,
    ) -> None:
        self.model_type = model_type
        self.num_labels = num_labels
        self.dataset = dataset
        self.device = device
        self.allow_pretrained_embeddings = allow_pretrained_embeddings
        self.allow_albert_pretrained = allow_albert_pretrained

        if self.model_type not in ["albert", "ffnn", "bilstm"]:
            raise ValueError("model type should be ffnn, bilstm or albert")

        if self.allow_pretrained_embeddings and self.allow_albert_pretrained:
            raise ValueError(
                "Allow pretrained embeddings and allow pretrained albert model are not"
                " compatible since the embeddings are the same"
            )

        if self.allow_pretrained_embeddings or allow_albert_pretrained:
            self.albert_pretrained_model = AutoModel.from_pretrained(
                "CenIA/albert-tiny-spanish"
            )

    # NOTE: this method will only work because the three type of models share the same
    # type of embeddings with matching vocabulary and sizes
    def copy_pretrained_embeddings(self, to_model: PreTrainedModel) -> None:
        state_dict = copy.deepcopy(self.albert_pretrained_model.embeddings.state_dict())
        to_model.base_model.embeddings.load_state_dict(state_dict)

    # NOTE: the following method will only work with albert with matching sizes
    def copy_pretrained_albert(self, to_model: PreTrainedModel) -> None:
        state_dict = copy.deepcopy(self.albert_pretrained_model.state_dict())
        to_model.base_model.load_state_dict(state_dict)

    def _freeze_params(self, module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def freeze_embeddings(self, model: PreTrainedModel) -> None:
        self._freeze_params(model.base_model.embeddings)

    def define_ffnn_model(
        self,
        trial: optuna.Trial,
        use_pretrained_embeddings: bool = None,
        freeze_pretrained_embeddings: bool = None,
    ) -> PreTrainedModel:
        hidden_dim = trial.suggest_int("hidden_dim", 128, 1280, step=128)
        num_layers = trial.suggest_int("num_layers", 1, 16)
        config = FFNNConfig(
            vocab_size=31000,
            embedding_dim=128,
            hidden_dim=hidden_dim,
            num_labels=self.num_labels,
            num_layers=num_layers,
        )
        model = FFNNForSequenceClassification(config)

        if use_pretrained_embeddings:
            self.copy_pretrained_embeddings(model)

            if freeze_pretrained_embeddings:
                self.freeze_embeddings(model)

        return model

    def define_bilstm_model(
        self,
        trial: optuna.Trial,
        use_pretrained_embeddings: bool = None,
        freeze_pretrained_embeddings: bool = None,
    ) -> PreTrainedModel:
        hidden_dim = trial.suggest_int("hidden_dim", 128, 768, step=128)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        num_ffnn_layers = trial.suggest_int("num_ffnn_layers", 0, 3)
        config = BiLSTMConfig(
            vocab_size=31000,
            embedding_dim=128,
            hidden_dim=hidden_dim,
            num_labels=self.num_labels,
            num_layers=num_layers,
            num_ffnn_layers=num_ffnn_layers,
        )
        model = BiLSTMForSequenceClassification(config)

        if use_pretrained_embeddings:
            self.copy_pretrained_embeddings(model)

            if freeze_pretrained_embeddings:
                self.freeze_embeddings(model)

        return model

    def define_albert_model(
        self,
        trial: optuna.Trial,
        use_pretrained_embeddings: bool = None,
        freeze_pretrained_embeddings: bool = None,
        use_albert_pretrained: bool = None,
    ) -> PreTrainedModel:
        num_layers = trial.suggest_int("num_layers", 1, 12)
        config = AlbertConfig(
            vocab_size=31000,
            embedding_size=128,
            hidden_size=312,
            num_labels=self.num_labels,
            num_hidden_layers=num_layers,
            num_attention_heads=12,
            intermediate_size=1248,
        )
        model = AlbertForSequenceClassification(config)

        if use_pretrained_embeddings:
            self.copy_pretrained_embeddings(model)

            if freeze_pretrained_embeddings:
                self.freeze_embeddings(model)

        if use_albert_pretrained:
            self.copy_pretrained_albert(model)

        return model

    def define_model(self, trial: optuna.Trial) -> PreTrainedModel:
        use_pretrained_embeddings = None
        freeze_pretrained_embeddings = None
        use_albert_pretrained = None

        if self.allow_pretrained_embeddings:
            use_pretrained_embeddings = trial.suggest_categorical(
                "use_pretrained_embeddings", choices=[False, True]
            )
            freeze_pretrained_embeddings = trial.suggest_categorical(
                "freeze_pretrained_embeddings", choices=[False, True]
            )
        if self.allow_albert_pretrained:
            use_albert_pretrained = trial.suggest_categorical(
                "use_albert_pretrained", choices=[False, True]
            )

        if self.model_type == "ffnn":
            return self.define_ffnn_model(
                trial,
                use_pretrained_embeddings=use_pretrained_embeddings,
                freeze_pretrained_embeddings=freeze_pretrained_embeddings,
            )
        elif self.model_type == "bilstm":
            return self.define_bilstm_model(
                trial,
                use_pretrained_embeddings=use_pretrained_embeddings,
                freeze_pretrained_embeddings=freeze_pretrained_embeddings,
            )
        elif self.model_type == "albert":
            return self.define_albert_model(
                trial,
                use_pretrained_embeddings=use_pretrained_embeddings,
                freeze_pretrained_embeddings=freeze_pretrained_embeddings,
                use_albert_pretrained=use_albert_pretrained,
            )

    def define_trainer(self, trial: optuna.Trial, model: PreTrainedModel) -> Trainer:
        tokenizer = AutoTokenizer.from_pretrained("CenIA/albert-tiny-spanish")
        epochs = trial.suggest_int("epochs", 1, 30)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
        return Trainer(
            model,
            tokenizer,
            self.dataset,
            device=self.device,
            batch_size=64,
            epochs=epochs,
            learning_rate=learning_rate,
            trial=trial,
            load_best_model_at_end=False,
        )

    def __call__(self, trial: optuna.Trial) -> float:
        model = self.define_model(trial)

        trainer = self.define_trainer(trial, model)
        trainer.train()
        task_metrics = trainer.evaluate()

        performance_benchmark = PerformanceBenchmark(
            model_config=model.config, task="sequence_classification"
        )
        performance_metrics = performance_benchmark.run_benchmark(
            batch_sizes=[1], sequence_lengths=[512], device=self.device
        )

        trial.set_user_attr("accuracy", task_metrics["accuracy"])
        trial.set_user_attr("model_size_mb", performance_metrics["size_mb"])
        trial.set_user_attr("model_parameters", performance_metrics["model_parameters"])
        trial.set_user_attr("macs", performance_metrics["times"][0]["macs"])
        trial.set_user_attr(
            "time_avg_ms", performance_metrics["times"][0]["time_avg_ms"]
        )
        trial.set_user_attr(
            "time_std_ms", performance_metrics["times"][0]["time_std_ms"]
        )

        return task_metrics["validation_loss"]


class KDObjective(object):
    """
    This objective was created to find (with fixed teacher and student models) which are
    the best hyperparameters, including: learning rate, KD alpha, KD temperature and
    loss to use in KD (mse, ce or kldiv).
    """

    def __init__(
        self,
        student_model_name_or_path: str,
        teacher_model_name_or_path: str,
        num_labels: int,
        dataset: Dataset,
        device: str,
        batch_size: int = 64,
    ) -> None:
        self.student_model_name_or_path = student_model_name_or_path
        self.teacher_model_name_or_path = teacher_model_name_or_path
        self.num_labels = num_labels
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

    def __call__(self, trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4)
        kd_alpha = trial.suggest_float("kd_alpha", 0, 1)
        kd_temperature = trial.suggest_categorical("kd_temperature", choices=[1, 2, 5])
        kd_loss_type = trial.suggest_categorical(
            "kd_loss_type", choices=["kldiv", "ce", "mse"]
        )

        teacher_model, teacher_tokenizer = get_model_and_tokenizer(
            self.teacher_model_name_or_path, self.num_labels
        )
        student_model, student_tokenizer = get_model_and_tokenizer(
            self.student_model_name_or_path, self.num_labels
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
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
            kd_loss_type=kd_loss_type,
            load_best_model_at_end=True,
            trial=trial,
            output_dir="./output_kd_experiments",
        )
        completed_epochs = trainer.train()

        task_metrics = trainer.evaluate()

        trial.set_user_attr("completed_epochs", completed_epochs)
        trial.set_user_attr("accuracy", task_metrics["accuracy"])

        return task_metrics["validation_loss"]


class TeacherSearchKDObjective(object):
    """
    This objective was created to experiment using KD with different teachers
    (with a fixed student) in a grid search way.
    """

    def __init__(
        self,
        student_model_name_or_path: str,
        num_labels: int,
        dataset: Dataset,
        dataset_name: str,
        device: str,
    ) -> None:
        self.student_model_name_or_path = student_model_name_or_path
        self.num_labels = num_labels
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.device = device

    def __call__(self, trial: optuna.Trial) -> float:
        teacher_model_name_or_path = trial.suggest_categorical(
            "teacher_model_name_or_path", choices=[None]
        )

        teacher_model, teacher_tokenizer = get_model_and_tokenizer(
            teacher_model_name_or_path, self.num_labels
        )
        student_model, student_tokenizer = get_model_and_tokenizer(
            self.student_model_name_or_path, self.num_labels
        )

        output_dir = (
            f"/data/jcanete/kd-experiments/{self.dataset_name}"
            f"/student-{self.student_model_name_or_path}/{teacher_model_name_or_path}"
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
            learning_rate=1e-4,
            kd_alpha=0,
            kd_temperature=1,
            kd_loss_type="mse",
            load_best_model_at_end=True,
            output_dir=output_dir,
        )
        completed_epochs = trainer.train()

        task_metrics = trainer.evaluate()

        trial.set_user_attr("completed_epochs", completed_epochs)
        trial.set_user_attr("accuracy", task_metrics["accuracy"])

        return task_metrics["validation_loss"]


def get_teacher_models_to_try(dataset_name):
    models = []
    if dataset_name == "pawsx":
        models = [
            "CenIA/albert-xxlarge-spanish-finetuned-pawsx",
            "CenIA/bert-base-spanish-wwm-cased-finetuned-pawsx",
            "CenIA/roberta-large-bne-finetuned-pawsx",
            "CenIA/distillbert-base-spanish-uncased-finetuned-pawsx",
        ]
    return {"teacher_model_name_or_path": models}


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
class HyperparameterSearchArguments:
    """
    HyperparameterSearchArguments is the subset of the arguments we use which relates
    to the optuna optimisation itself.
    """

    study_name: str = field(
        metadata={
            "help": (
                "The name of the study."
                "The DB containing the data of all trial will also use this name."
            )
        }
    )
    experiment_type: str = field(
        metadata={"help": "Which type of experiment to run, it can be 'ft' or 'kd'"}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "The device to run the training/evaluation on."},
    )
    num_trials: int = field(
        default=100, metadata={"help": "How many different trials to run."}
    )
    sampler_type: str = field(
        default="tpe",
        metadata={"help": "Which sampler to use, options are: 'tpe' or 'random'"},
    )
    pruner_type: str = field(
        default="hyperband",
        metadata={"help": "Which pruner to use, options are: 'hyperband' or 'median'"},
    )


@dataclass
class FTObjectiveArguments:
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "The kind of model to use on the search, it can be ffnn or bilstm."
        },
    )
    allow_pretrained_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "Wheter to allow the use of pretrained embeddings on the search or not"
            )
        },
    )
    allow_albert_pretrained: bool = field(
        default=False,
        metadata={
            "help": (
                "Wheter to allow the use of pretrained weights of albert-tiny in the"
                " case of search of albert architecture."
            )
        },
    )


@dataclass
class KDObjectiveArguments:
    student_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the model to use as student"},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the model to use as teacher"},
    )


@dataclass
class TeacherSearchKDObjectiveArguments:
    ts_student_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the model to use as student"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use"}
    )


def main():
    parser = HfArgumentParser(
        (
            DataTrainingArguments,
            HyperparameterSearchArguments,
            FTObjectiveArguments,
            KDObjectiveArguments,
            TeacherSearchKDObjectiveArguments,
        )
    )
    (
        data_args,
        hps_args,
        ft_args,
        kd_args,
        ts_args,
    ) = parser.parse_args_into_dataclasses()

    if hps_args.experiment_type == "ft":
        if ft_args.model_type is None:
            raise ValueError("model_type is required when running ft experiments")
    elif hps_args.experiment_type == "kd":
        if (
            kd_args.student_model_name_or_path is None
            or kd_args.teacher_model_name_or_path is None
        ):
            raise ValueError(
                "student_model_name_or_path and teacher_model_name_or_path are required"
                " when running kd experiments"
            )
    elif hps_args.experiment_type == "ts":
        if (
            ts_args.ts_student_model_name_or_path is None
            or ts_args.dataset_name is None
        ):
            raise ValueError(
                "student_model_name_or_path and dataset_name are required on teacher"
                " search"
            )

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

    dataset = load_dataset(
        "json",
        data_files={
            "train": data_args.train_file,
            "dev": data_args.validation_file,
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

    if hps_args.experiment_type == "ft":
        objective = FTObjective(
            model_type=ft_args.model_type,
            num_labels=num_labels,
            dataset=dataset,
            device=hps_args.device,
            allow_pretrained_embeddings=ft_args.allow_pretrained_embeddings,
            allow_albert_pretrained=ft_args.allow_albert_pretrained,
        )
    elif hps_args.experiment_type == "kd":
        objective = KDObjective(
            kd_args.student_model_name_or_path,
            kd_args.teacher_model_name_or_path,
            num_labels=num_labels,
            dataset=dataset,
            device=hps_args.device,
        )
    else:
        objective = TeacherSearchKDObjective(
            ts_args.ts_student_model_name_or_path,
            num_labels=num_labels,
            dataset=dataset,
            dataset_name=ts_args.dataset_name,
            device=hps_args.device,
        )

    storage_name = "sqlite:///{}.db".format(hps_args.study_name)

    if hps_args.sampler_type == "tpe":
        sampler = optuna.samplers.TPESampler()
    else:
        sampler = optuna.samplers.RandomSampler()

    if hps_args.pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.MedianPruner()

    # On teacher search we will just try all models
    if hps_args.experiment_type == "ts":
        search_space = get_teacher_models_to_try(ts_args.dataset_name)
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(
            sampler=sampler,
            study_name=hps_args.study_name,
            storage=storage_name,
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(objective)
    else:
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            study_name=hps_args.study_name,
            storage=storage_name,
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(objective, hps_args.num_trials)


if __name__ == "__main__":
    main()
