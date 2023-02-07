import json
import logging
import os
from typing import Any

import evaluate
import numpy as np
import optuna
import torch
from joblib import Memory
from loss import KnowledgeDistilationLoss
from torch import optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    QuestionAnsweringPipeline,
    default_data_collator,
    get_scheduler,
)
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from datasets import DatasetDict

logger = logging.getLogger(__name__)


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def freeze_parameters(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: DatasetDict,
        device: str = "cpu",
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 3e-5,
        output_dir: str = "./output",
        load_best_model_at_end: bool = True,
        early_stopping_patience: int = 5,
        trial: optuna.trial.Trial = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = torch.device(device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir

        self.dataloaders = self.create_dataloaders(self.tokenizer)

        self.num_training_steps = self.epochs * len(self.dataloaders["train"])

        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()

        self.progress_bar = tqdm(range(self.num_training_steps))
        self.metric = self.load_metric()

        self.train_metrics_history = []  # type: list[dict[str, Any]]
        self.validation_metrics_history = []  # type: list[dict[str, Any]]

        self.metric_to_optimize = "validation_loss"
        self.load_best_model_at_end = load_best_model_at_end
        self.early_stopping_patience = early_stopping_patience

        self.trial = trial

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params}")

    def load_metric(self) -> evaluate.Metric:
        return evaluate.load("accuracy")

    def move_to_device(self) -> None:
        self.model.to(self.device)
        self.loss.to(self.device)

    def train(self) -> int:
        """
        Loop of training. Returns the ammount of completed epochs
        (since it can stop earlier by early stopping)
        """
        self.move_to_device()
        best_metric = 1e10
        last_improve = 0

        for epoch in range(1, self.epochs + 1):
            logger.info(f"Epoch {epoch}")
            train_metrics = self.train_epoch()
            validation_metrics = self.evaluate()
            train_metrics.update({"epoch": epoch})
            validation_metrics.update({"epoch": epoch})
            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Validation metrics: {validation_metrics}")
            self.train_metrics_history.append(train_metrics)
            self.validation_metrics_history.append(validation_metrics)

            if validation_metrics[self.metric_to_optimize] < best_metric:
                best_metric = validation_metrics[self.metric_to_optimize]
                last_improve = 0
                self.save_model()
            else:
                last_improve += 1

            if last_improve > self.early_stopping_patience:
                logger.info(f"Early stopped at epoch: {epoch}")
                break

            if self.trial:
                self.trial.report(validation_metrics[self.metric_to_optimize], epoch)

                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        self.save_metrics()

        if self.load_best_model_at_end:
            self.model = self.model.from_pretrained(self.output_dir)

        return epoch

    def evaluate(self) -> dict[str, float]:
        validation_loss = 0.0
        num_batches = len(self.dataloaders["validation"])

        self.model.eval()
        for batch in self.dataloaders["validation"]:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            validation_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            self.metric.add_batch(predictions=predictions, references=batch["labels"])

        validation_loss /= num_batches
        all_metrics = self.metric.compute()
        all_metrics.update({"validation_loss": validation_loss})

        return all_metrics

    def save_model(self) -> None:
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def save_metrics(self) -> None:
        """
        Saves metrics as json file, it uses the np_encoder function to encode properly
        objects from numpy
        """
        with open(os.path.join(self.output_dir, "train_metrics.json"), "w") as f:
            json.dump(self.train_metrics_history, f, indent=4, default=np_encoder)
        with open(os.path.join(self.output_dir, "validation_metrics.json"), "w") as f:
            json.dump(self.validation_metrics_history, f, indent=4, default=np_encoder)

    def train_step(self) -> None:
        pass

    def train_epoch(self) -> dict[str, float]:
        ce_train_loss = 0.0
        num_batches = len(self.dataloaders["train"])
        current_batch = 0

        self.model.train()

        for batch in self.dataloaders["train"]:
            self.optimizer.zero_grad()
            current_batch += 1
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            ce_train_loss_value = outputs.loss.item()
            ce_train_loss += ce_train_loss_value

            outputs.loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.progress_bar.update(1)
            self.progress_bar.set_description(
                "Loss: {:.3f}".format(ce_train_loss_value)
            )
        train_metrics = {
            "ce_train_loss": ce_train_loss / num_batches,
        }
        return train_metrics

    def create_optimizer(self) -> optim.Optimizer:
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        return get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

    def create_dataloaders(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> dict[str, DataLoader]:
        if "sentence2" not in self.dataset["train"].column_names:
            tokenized = self.dataset.map(
                lambda examples: tokenizer(examples["sentence1"], truncation=True),
                batched=True,
                remove_columns=["sentence1"],
            )
        else:
            tokenized = self.dataset.map(
                lambda examples: tokenizer(
                    examples["sentence1"],
                    examples["sentence2"],
                    truncation=True,
                ),
                batched=True,
                remove_columns=["sentence1", "sentence2"],
            )

        tokenized.set_format("torch")

        data_collator = DataCollatorWithPadding(tokenizer)

        return {
            key: DataLoader(
                tokenized[key], batch_size=self.batch_size, collate_fn=data_collator
            )
            for key in tokenized.keys()
        }


class TokenClassificationTrainer(Trainer):
    def __init__(
        self,
        text_column_name: str,
        label_column_name: str,
        label_list: list[str],
        *args,
        b_to_i_label: dict = {},
        label_all_tokens: bool = False,
        **kwargs,
    ) -> None:
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.label_list = label_list

        self.b_to_i_label = b_to_i_label
        self.label_all_tokens = label_all_tokens

        super().__init__(*args, **kwargs)

    def create_dataloaders(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> dict[str, DataLoader]:
        tokenized = self.dataset.map(
            lambda examples: self.tokenize_and_align_labels(
                tokenizer=tokenizer,
                examples=examples,
                text_column_name=self.text_column_name,
                label_column_name=self.label_column_name,
                b_to_i_label=self.b_to_i_label,
                label_all_tokens=self.label_all_tokens,
            ),
            batched=True,
            remove_columns=[self.text_column_name, self.label_column_name],
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        return {
            key: DataLoader(
                tokenized[key], batch_size=self.batch_size, collate_fn=data_collator
            )
            for key in tokenized.keys()
        }

    def load_metric(self) -> evaluate.Metric:
        return evaluate.load("seqeval")

    def tokenize_and_align_labels(
        self,
        tokenizer,
        examples,
        text_column_name,
        label_column_name,
        b_to_i_label,
        label_all_tokens,
    ):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words
            # (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the
                # current label or -100, depending on the label_all_tokens flag.
                else:
                    if label_all_tokens:
                        label_ids.append(b_to_i_label[label[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def evaluate(self) -> dict[str, float]:
        validation_loss = 0.0
        num_batches = len(self.dataloaders["validation"])

        self.model.eval()
        for batch in self.dataloaders["validation"]:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            validation_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Remove ignored index (special tokens)
            true_predictions = []
            true_labels = []

            for prediction, label in zip(
                predictions.cpu().numpy(), batch["labels"].cpu().numpy()
            ):
                true_prediction = []
                true_label = []
                for token_prediction, token_label in zip(prediction, label):
                    if token_label != -100:
                        true_prediction.append(self.label_list[token_prediction])
                        true_label.append(self.label_list[token_label])
                true_predictions.append(true_prediction)
                true_labels.append(true_label)

            self.metric.add_batch(predictions=true_predictions, references=true_labels)

        validation_loss /= num_batches
        all_metrics = self.metric.compute()
        all_metrics.update({"validation_loss": validation_loss})

        return all_metrics


class QATrainer(Trainer):
    def __init__(
        self,
        question_column_name: str,
        context_column_name: str,
        answer_column_name: str,
        *args,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        pad_to_max_length: bool = False,
        **kwargs,
    ) -> None:
        self.question_column_name = question_column_name
        self.context_column_name = context_column_name
        self.answer_column_name = answer_column_name
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.pad_to_max_length = pad_to_max_length
        super().__init__(*args, **kwargs)

        self.inference_pipeline = QuestionAnsweringPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
        )

    def load_metric(self) -> evaluate.Metric:
        return evaluate.load("squad")

    def create_dataloaders(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> dict[str, DataLoader]:
        column_names = self.dataset["train"].column_names

        preprocessed_dataset = self.dataset.map(
            lambda examples: self.prepare_train_features(examples, tokenizer),
            batched=True,
            remove_columns=column_names,
        )

        data_collator = (
            default_data_collator
            if self.pad_to_max_length
            else DataCollatorWithPadding(tokenizer)
        )

        return {
            key: DataLoader(
                preprocessed_dataset[key],
                batch_size=self.batch_size,
                collate_fn=data_collator,
            )
            for key in preprocessed_dataset.keys()
        }

    def evaluate(self) -> dict[str, float]:
        validation_loss = 0.0
        num_batches = len(self.dataloaders["validation"])

        self.model.eval()
        # first we will only calculate the validation loss
        for batch in self.dataloaders["validation"]:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            validation_loss += outputs.loss.item()

        validation_loss /= num_batches

        # next we will calculate f1 score / exact match using a inference pipeline
        validation_dataset_with_answers = self.dataset["validation"].map(
            lambda examples: self.inference_pipeline(
                question=examples["question"],
                context=examples["context"],
                max_seq_len=self.max_seq_length,
                doc_stride=self.doc_stride,
                align_to_words=True,
            ),
        )

        predictions = [
            {"prediction_text": example["answer"], "id": example["id"]}
            for example in validation_dataset_with_answers
        ]
        references = [
            {"answers": example["answers"], "id": example["id"]}
            for example in validation_dataset_with_answers
        ]

        all_metrics = self.metric.compute(
            predictions=predictions, references=references
        )
        all_metrics.update({"validation_loss": validation_loss})

        return all_metrics

    # Training preprocessing
    def prepare_train_features(self, examples, tokenizer):
        pad_on_right = tokenizer.padding_side == "right"
        # Some of the questions have lots of whitespace on the left, which is not useful
        # and will make the truncation of the context fail (the tokenized question will
        # take a lots of space). So we remove that left whitespace
        examples[self.question_column_name] = [
            q.lstrip() for q in examples[self.question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the
        # overflows using a stride. This results in one example possible giving several
        # features when a context is long, each of those features having a context that
        # overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[
                self.question_column_name if pad_on_right else self.context_column_name
            ],
            examples[
                self.context_column_name if pad_on_right else self.question_column_name
            ],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we
        # need a map from a feature to its corresponding example. This key gives us just
        # that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the
        # original context. This will help us compute the start_positions and
        # end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the
            # context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example
            # containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is
                # labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the
                    # two ends of the answer. Note: we could go after the last offset if
                    #  the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples


class DistillationTrainer(Trainer):
    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
        *args,
        kd_loss_type: str = "kldiv",
        kd_alpha: float = 0.5,
        kd_temperature: float = 1.0,
        do_cache_teacher_outputs: bool = True,
        teacher_cache_dir: str = "./teacher_cache",
        **kwargs,
    ) -> None:
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer

        self.kd_loss_type = kd_loss_type
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

        self.do_cache_teacher_outputs = do_cache_teacher_outputs
        self.teacher_cache_dir = teacher_cache_dir

        super().__init__(student_model, student_tokenizer, *args, **kwargs)

        self.loss = KnowledgeDistilationLoss(
            self.kd_loss_type, self.kd_alpha, self.kd_temperature
        )
        self.teacher_dataloaders = self.create_dataloaders(self.teacher_tokenizer)

        freeze_parameters(self.teacher_model)

        self.memory = Memory(self.teacher_cache_dir, verbose=0)
        self.cached_teacher_forward = self.memory.cache(
            self.seq_kd_cached_teacher_forward, ignore=["self"]
        )

        if self.do_cache_teacher_outputs:
            self.cache_all_teacher_outputs()

    def seq_kd_cached_teacher_forward(self, numpy_batch, model_name_or_path):
        """
        This method is the key to cache the forward of teacher models and thus achieve
        much more training efficiency.
        The model_name_or_path parameter is not used but it is important to make the
        cache not depending only on the batch but also on the model. I think the correct
        way would be passing the model, but that makes the serialization much heavier.
        """
        teacher_batch = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in numpy_batch.items()
        }
        return self.teacher_model(**teacher_batch).logits.cpu().numpy()

    def format_and_do_cached_teacher_forward(
        self, pytorch_batch
    ) -> SequenceClassifierOutput:
        numpy_batch = {key: value.cpu().numpy() for key, value in pytorch_batch.items()}
        teacher_outputs = self.cached_teacher_forward(
            numpy_batch, self.teacher_model.name_or_path
        )
        pytorch_teacher_logits = torch.from_numpy(teacher_outputs).to(self.device)
        return SequenceClassifierOutput(logits=pytorch_teacher_logits)

    def cache_all_teacher_outputs(self) -> None:
        self.move_to_device()
        for batch in tqdm(
            self.teacher_dataloaders["train"], desc="Caching teacher outputs"
        ):
            self.format_and_do_cached_teacher_forward(batch)

    def move_to_device(self) -> None:
        super().move_to_device()
        self.teacher_model.to(self.device)

    def train_step(self) -> None:
        pass

    def train_epoch(self) -> dict[str, float]:
        kd_train_loss = 0.0
        ce_train_loss = 0.0
        num_batches = len(self.dataloaders["train"])
        current_batch = 0

        self.model.train()
        self.teacher_model.eval()
        dataloaders = zip(self.dataloaders["train"], self.teacher_dataloaders["train"])
        for student_batch, teacher_batch in dataloaders:
            self.optimizer.zero_grad()
            current_batch += 1
            teacher_batch = {
                k: v.to(self.teacher_model.device) for k, v in teacher_batch.items()
            }
            student_batch = {
                k: v.to(self.model.device) for k, v in student_batch.items()
            }
            student_outputs = self.model(**student_batch)
            with torch.no_grad():
                if self.do_cache_teacher_outputs:
                    teacher_outputs = self.format_and_do_cached_teacher_forward(
                        teacher_batch
                    )
                else:
                    teacher_outputs = self.teacher_model(**teacher_batch)

            kd_loss = self.loss(
                student_outputs.logits,
                student_batch["labels"],
                teacher_outputs.logits,
            )
            kd_loss_value = kd_loss.item()

            kd_train_loss += kd_loss_value
            ce_train_loss += student_outputs.loss.item()

            kd_loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.progress_bar.update(1)
            self.progress_bar.set_description("Loss: {:.3f}".format(kd_loss_value))
        train_metrics = {
            "kd_train_loss": kd_train_loss / num_batches,
            "ce_train_loss": ce_train_loss / num_batches,
        }
        return train_metrics


# NOTE: This class is almost the same as TokenClassifierTrainer + DistillationTrainer
# all methods (except from train_epoch) are copies from those classes code, so probably
# there is a better design for this (maybe using multiple inheritance? but im not sure)
class TokenClassificationDistillationTrainer(Trainer):
    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
        text_column_name: str,
        label_column_name: str,
        label_list: list[str],
        *args,
        label_all_tokens: bool = False,
        b_to_i_label: dict = {},
        kd_loss_type: str = "kldiv",
        kd_alpha: float = 0.5,
        kd_temperature: float = 1,
        do_cache_teacher_outputs: bool = True,
        teacher_cache_dir: str = "./teacher_cache",
        **kwargs,
    ) -> None:
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.label_list = label_list

        self.b_to_i_label = b_to_i_label
        self.label_all_tokens = label_all_tokens

        self.kd_loss_type = kd_loss_type
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

        self.do_cache_teacher_outputs = do_cache_teacher_outputs
        self.teacher_cache_dir = teacher_cache_dir

        super().__init__(student_model, student_tokenizer, *args, **kwargs)

        self.loss = KnowledgeDistilationLoss(
            self.kd_loss_type, self.kd_alpha, self.kd_temperature
        )
        self.teacher_dataloaders = self.create_dataloaders(self.teacher_tokenizer)

        freeze_parameters(self.teacher_model)

        self.memory = Memory(self.teacher_cache_dir, verbose=0)
        self.cached_teacher_forward = self.memory.cache(
            self.token_kd_cached_teacher_forward, ignore=["self"]
        )

        if self.do_cache_teacher_outputs:
            self.cache_all_teacher_outputs()

    def token_kd_cached_teacher_forward(self, numpy_batch, model_name_or_path):
        """
        This method is the key to cache the forward of teacher models and thus achieve
        much more training efficiency.
        The model_name_or_path parameter is not used but it is important to make the
        cache not depending only on the batch but also on the model. I think the correct
        way would be passing the model, but that makes the serialization much heavier.
        """
        teacher_batch = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in numpy_batch.items()
        }
        return self.teacher_model(**teacher_batch).logits.cpu().numpy()

    def format_and_do_cached_teacher_forward(
        self, pytorch_batch
    ) -> TokenClassifierOutput:
        numpy_batch = {key: value.cpu().numpy() for key, value in pytorch_batch.items()}
        teacher_outputs = self.cached_teacher_forward(
            numpy_batch, self.teacher_model.name_or_path
        )
        pytorch_teacher_logits = torch.from_numpy(teacher_outputs).to(self.device)
        return TokenClassifierOutput(logits=pytorch_teacher_logits)

    def cache_all_teacher_outputs(self) -> None:
        self.move_to_device()
        for batch in tqdm(
            self.teacher_dataloaders["train"], desc="Caching teacher outputs"
        ):
            self.format_and_do_cached_teacher_forward(batch)

    def move_to_device(self) -> None:
        super().move_to_device()
        self.teacher_model.to(self.device)

    def train_epoch(self) -> dict[str, float]:
        kd_train_loss = 0.0
        ce_train_loss = 0.0
        num_batches = len(self.dataloaders["train"])
        current_batch = 0

        self.model.train()
        self.teacher_model.eval()
        dataloaders = zip(self.dataloaders["train"], self.teacher_dataloaders["train"])
        for student_batch, teacher_batch in dataloaders:
            self.optimizer.zero_grad()
            current_batch += 1
            teacher_batch = {
                k: v.to(self.teacher_model.device) for k, v in teacher_batch.items()
            }
            student_batch = {
                k: v.to(self.model.device) for k, v in student_batch.items()
            }
            student_outputs = self.model(**student_batch)
            with torch.no_grad():
                if self.do_cache_teacher_outputs:
                    teacher_outputs = self.format_and_do_cached_teacher_forward(
                        teacher_batch
                    )
                else:
                    teacher_outputs = self.teacher_model(**teacher_batch)

            num_labels = self.model.config.num_labels

            student_mask = student_batch["labels"].ne(-100)
            student_labels = torch.masked_select(student_batch["labels"], student_mask)

            student_mask = student_mask.unsqueeze(-1)
            student_logits = torch.masked_select(student_outputs.logits, student_mask)

            teacher_mask = teacher_batch["labels"].ne(-100).unsqueeze(-1)
            teacher_logits = torch.masked_select(teacher_outputs.logits, teacher_mask)

            kd_loss = self.loss(
                student_logits.view(-1, num_labels),
                student_labels.view(-1),
                teacher_logits.view(-1, num_labels),
            )
            kd_loss_value = kd_loss.item()

            kd_train_loss += kd_loss_value
            ce_train_loss += student_outputs.loss.item()

            kd_loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.progress_bar.update(1)
            self.progress_bar.set_description("Loss: {:.3f}".format(kd_loss_value))
        train_metrics = {
            "kd_train_loss": kd_train_loss / num_batches,
            "ce_train_loss": ce_train_loss / num_batches,
        }
        return train_metrics

    def create_dataloaders(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> dict[str, DataLoader]:
        truncated = self.dataset.map(
            lambda examples: {
                self.text_column_name: [
                    example[:128] for example in examples[self.text_column_name]
                ]
            },
            batched=True,
        )
        tokenized = truncated.map(
            lambda examples: self.tokenize_and_align_labels(
                tokenizer=tokenizer,
                examples=examples,
                text_column_name=self.text_column_name,
                label_column_name=self.label_column_name,
                b_to_i_label=self.b_to_i_label,
                label_all_tokens=self.label_all_tokens,
            ),
            batched=True,
            remove_columns=[self.text_column_name, self.label_column_name],
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        return {
            key: DataLoader(
                tokenized[key],
                batch_size=self.batch_size,
                collate_fn=data_collator,
            )
            for key in tokenized.keys()
        }

    def load_metric(self) -> evaluate.Metric:
        return evaluate.load("seqeval")

    def tokenize_and_align_labels(
        self,
        tokenizer,
        examples,
        text_column_name,
        label_column_name,
        b_to_i_label,
        label_all_tokens,
    ):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=True,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words
            # (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the
                # current label or -100, depending on the label_all_tokens flag.
                else:
                    if label_all_tokens:
                        label_ids.append(b_to_i_label[label[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def evaluate(self) -> dict[str, float]:
        validation_loss = 0.0
        num_batches = len(self.dataloaders["validation"])

        self.model.eval()
        for batch in self.dataloaders["validation"]:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            validation_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Remove ignored index (special tokens)
            true_predictions = []
            true_labels = []

            for prediction, label in zip(
                predictions.cpu().numpy(), batch["labels"].cpu().numpy()
            ):
                true_prediction = []
                true_label = []
                for token_prediction, token_label in zip(prediction, label):
                    if token_label != -100:
                        true_prediction.append(self.label_list[token_prediction])
                        true_label.append(self.label_list[token_label])
                true_predictions.append(true_prediction)
                true_labels.append(true_label)

            self.metric.add_batch(predictions=true_predictions, references=true_labels)

        validation_loss /= num_batches
        all_metrics = self.metric.compute()
        all_metrics.update({"validation_loss": validation_loss})

        return all_metrics

    def data_collator(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding=True,
            # Conversion to tensors will fail if we have labels as they are not
            # of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [-100] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [-100] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


class QADistillationTrainer(Trainer):
    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
        question_column_name: str,
        context_column_name: str,
        answer_column_name: str,
        *args,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        pad_to_max_length: bool = False,
        kd_loss_type: str = "kldiv",
        kd_alpha: float = 0.5,
        kd_temperature: float = 1.0,
        do_cache_teacher_outputs: bool = True,
        teacher_cache_dir: str = "./teacher_cache",
        **kwargs,
    ) -> None:
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer

        self.question_column_name = question_column_name
        self.context_column_name = context_column_name
        self.answer_column_name = answer_column_name

        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.pad_to_max_length = pad_to_max_length

        self.kd_loss_type = kd_loss_type
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

        self.do_cache_teacher_outputs = do_cache_teacher_outputs
        self.teacher_cache_dir = teacher_cache_dir

        super().__init__(student_model, student_tokenizer, *args, **kwargs)

        self.loss = KnowledgeDistilationLoss(
            self.kd_loss_type, self.kd_alpha, self.kd_temperature
        )
        self.teacher_dataloaders = self.create_dataloaders(teacher_tokenizer)
        self.inference_pipeline = QuestionAnsweringPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
        )

        freeze_parameters(self.teacher_model)

        self.memory = Memory(self.teacher_cache_dir, verbose=0)
        self.cached_teacher_forward = self.memory.cache(
            self.qa_kd_cached_teacher_forward, ignore=["self"]
        )

        if self.do_cache_teacher_outputs:
            self.cache_all_teacher_outputs()

    def qa_kd_cached_teacher_forward(self, numpy_batch, model_name_or_path):
        """
        This method is the key to cache the forward of teacher models and thus achieve
        much more training efficiency.
        The model_name_or_path parameter is not used but it is important to make the
        cache not depending only on the batch but also on the model. I think the correct
        way would be passing the model, but that makes the serialization much heavier.
        """
        teacher_batch = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in numpy_batch.items()
        }
        outputs = self.teacher_model(**teacher_batch)
        return outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy()

    def format_and_do_cached_teacher_forward(
        self, pytorch_batch
    ) -> QuestionAnsweringModelOutput:
        numpy_batch = {key: value.cpu().numpy() for key, value in pytorch_batch.items()}
        teacher_start_outputs, teacher_end_outputs = self.cached_teacher_forward(
            numpy_batch, self.teacher_model.name_or_path
        )
        pytorch_teacher_start_logits = torch.from_numpy(teacher_start_outputs).to(
            self.device
        )
        pytorch_teacher_end_logits = torch.from_numpy(teacher_end_outputs).to(
            self.device
        )
        return QuestionAnsweringModelOutput(
            start_logits=pytorch_teacher_start_logits,
            end_logits=pytorch_teacher_end_logits,
        )

    def cache_all_teacher_outputs(self) -> None:
        self.move_to_device()
        for batch in tqdm(
            self.teacher_dataloaders["train"], desc="Caching teacher outputs"
        ):
            self.format_and_do_cached_teacher_forward(batch)

    def move_to_device(self) -> None:
        super().move_to_device()
        self.teacher_model.to(self.device)

    def load_metric(self) -> evaluate.Metric:
        return evaluate.load("squad")

    def create_dataloaders(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> dict[str, DataLoader]:
        column_names = self.dataset["train"].column_names

        preprocessed_dataset = self.dataset.map(
            lambda examples: self.prepare_train_features(examples, tokenizer),
            batched=True,
            remove_columns=column_names,
        )

        data_collator = (
            default_data_collator
            if self.pad_to_max_length
            else DataCollatorWithPadding(tokenizer)
        )

        return {
            key: DataLoader(
                preprocessed_dataset[key],
                batch_size=self.batch_size,
                collate_fn=data_collator,
            )
            for key in preprocessed_dataset.keys()
        }

    def train_epoch(self) -> dict[str, float]:
        kd_train_loss = 0.0
        ce_train_loss = 0.0
        num_batches = len(self.dataloaders["train"])
        current_batch = 0

        self.model.train()
        self.teacher_model.eval()
        dataloaders = zip(self.dataloaders["train"], self.teacher_dataloaders["train"])
        for student_batch, teacher_batch in dataloaders:
            self.optimizer.zero_grad()
            current_batch += 1
            teacher_batch = {
                k: v.to(self.teacher_model.device) for k, v in teacher_batch.items()
            }
            student_batch = {
                k: v.to(self.model.device) for k, v in student_batch.items()
            }
            student_outputs = self.model(**student_batch)
            with torch.no_grad():
                if self.do_cache_teacher_outputs:
                    teacher_outputs = self.format_and_do_cached_teacher_forward(
                        teacher_batch
                    )
                else:
                    teacher_outputs = self.teacher_model(**teacher_batch)

            kd_loss_starts = self.loss(
                student_outputs.start_logits,
                student_batch["start_positions"],
                teacher_outputs.start_logits,
            )
            kd_loss_ends = self.loss(
                student_outputs.end_logits,
                student_batch["end_positions"],
                teacher_outputs.end_logits,
            )

            kd_loss = (kd_loss_starts + kd_loss_ends) / 2

            kd_loss_value = kd_loss.item()

            kd_train_loss += kd_loss_value
            ce_train_loss += student_outputs.loss.item()

            kd_loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.progress_bar.update(1)
            self.progress_bar.set_description("Loss: {:.3f}".format(kd_loss_value))
        train_metrics = {
            "kd_train_loss": kd_train_loss / num_batches,
            "ce_train_loss": ce_train_loss / num_batches,
        }
        return train_metrics

    def evaluate(self) -> dict[str, float]:
        validation_loss = 0.0
        num_batches = len(self.dataloaders["validation"])

        self.model.eval()
        # first we will only calculate the validation loss
        for batch in self.dataloaders["validation"]:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            validation_loss += outputs.loss.item()

        validation_loss /= num_batches

        # next we will calculate f1 score / exact match using a inference pipeline
        validation_dataset_with_answers = self.dataset["validation"].map(
            lambda examples: self.inference_pipeline(
                question=examples["question"],
                context=examples["context"],
                max_seq_len=self.max_seq_length,
                doc_stride=self.doc_stride,
                align_to_words=True,
            ),
        )

        predictions = [
            {"prediction_text": example["answer"], "id": example["id"]}
            for example in validation_dataset_with_answers
        ]
        references = [
            {"answers": example["answers"], "id": example["id"]}
            for example in validation_dataset_with_answers
        ]

        all_metrics = self.metric.compute(
            predictions=predictions, references=references
        )
        all_metrics.update({"validation_loss": validation_loss})

        return all_metrics

        # Training preprocessing

    def prepare_train_features(self, examples, tokenizer):
        pad_on_right = tokenizer.padding_side == "right"
        # Some of the questions have lots of whitespace on the left, which is not useful
        # and will make the truncation of the context fail (the tokenized question will
        # take a lots of space). So we remove that left whitespace
        examples[self.question_column_name] = [
            q.lstrip() for q in examples[self.question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the
        # overflows using a stride. This results in one example possible giving several
        # features when a context is long, each of those features having a context that
        # overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[
                self.question_column_name if pad_on_right else self.context_column_name
            ],
            examples[
                self.context_column_name if pad_on_right else self.question_column_name
            ],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we
        # need a map from a feature to its corresponding example. This key gives us just
        # that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the
        # original context. This will help us compute the start_positions and
        # end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the
            # context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example
            # containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is
                # labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the
                    # two ends of the answer. Note: we could go after the last offset if
                    #  the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples


class MultipleTeacherDistillationTrainer(Trainer):
    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_models: list[PreTrainedModel],
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: list[PreTrainedTokenizerBase],
        *args,
        kd_loss_type: str = "kldiv",
        kd_alpha: float = 0.5,
        kd_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(student_model, student_tokenizer, *args, **kwargs)
        self.teacher_models = teacher_models
        self.teacher_tokenizers = teacher_tokenizers
        self.loss = KnowledgeDistilationLoss(kd_loss_type, kd_alpha, kd_temperature)
        self.teacher_dataloaders = [
            self.create_dataloaders(tokenizer) for tokenizer in self.teacher_tokenizers
        ]

    def move_to_device(self) -> None:
        super().move_to_device()
        for teacher in self.teacher_models:
            teacher.to(self.device)

    def train_step(self) -> None:
        pass

    def train_epoch(self) -> dict[str, float]:
        kd_train_loss = 0.0
        ce_train_loss = 0.0
        num_batches = len(self.dataloaders["train"])
        current_batch = 0

        self.model.train()
        for teacher in self.teacher_models:
            teacher.eval()

        teacher_dataloaders = (
            teacher_dataloader["train"]
            for teacher_dataloader in self.teacher_dataloaders
        )
        dataloaders = zip(self.dataloaders["train"], *teacher_dataloaders)

        for student_batch, *teachers_batch in dataloaders:
            self.optimizer.zero_grad()
            current_batch += 1
            teachers_batch = [
                {k: v.to(self.model.device) for k, v in teacher_batch.items()}
                for teacher_batch in teachers_batch
            ]
            student_batch = {
                k: v.to(self.model.device) for k, v in student_batch.items()
            }
            student_outputs = self.model(**student_batch)
            with torch.no_grad():
                teachers_outputs = []
                for teacher, teacher_batch in zip(self.teacher_models, teachers_batch):
                    teacher_output = teacher(**teacher_batch)
                    teachers_outputs.append(teacher_output)

            multi_teacher_outputs = self.combine_teacher_outputs(teachers_outputs)

            kd_loss = self.loss(
                student_outputs.logits,
                student_batch["labels"],
                multi_teacher_outputs.logits,
            )
            kd_loss_value = kd_loss.item()

            kd_train_loss += kd_loss_value
            ce_train_loss += student_outputs.loss.item()

            kd_loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.progress_bar.update(1)
            self.progress_bar.set_description("Loss: {:.3f}".format(kd_loss_value))
        train_metrics = {
            "kd_train_loss": kd_train_loss / num_batches,
            "ce_train_loss": ce_train_loss / num_batches,
        }
        return train_metrics

    def combine_teacher_outputs(self, teachers_outputs):
        # just mean for now
        teacher_logits = torch.stack(
            [teacher_output.logits for teacher_output in teachers_outputs]
        )
        output = SequenceClassifierOutput(logits=teacher_logits.mean(0))
        return output


class LearnedMultipleTeacherDistillationTrainer(Trainer):
    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_models: list[PreTrainedModel],
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: list[PreTrainedTokenizerBase],
        *args,
        kd_loss_type: str = "kldiv",
        kd_alpha: float = 0.5,
        kd_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(student_model, student_tokenizer, *args, **kwargs)
        self.teacher_models = teacher_models
        self.teacher_tokenizers = teacher_tokenizers
        self.loss = KnowledgeDistilationLoss(kd_loss_type, kd_alpha, kd_temperature)
        self.teacher_dataloaders = [
            self.create_dataloaders(tokenizer) for tokenizer in self.teacher_tokenizers
        ]
        self.learned_combination = TransformerCombination(len(teacher_models))
        self.learned_combination.to(self.device)
        self.learned_combination_loss = torch.nn.CrossEntropyLoss()

        self.optimizer = self.replace_optimizer()
        self.scheduler = self.create_scheduler()

    def move_to_device(self) -> None:
        super().move_to_device()
        for teacher in self.teacher_models:
            teacher.to(self.device)
            for param in teacher.parameters():
                param.requires_grad = False

    def train_step(self) -> None:
        pass

    def train_epoch(self) -> dict[str, float]:
        kd_train_loss = 0.0
        ce_train_loss = 0.0
        teachers_train_loss = 0.0
        num_batches = len(self.dataloaders["train"])
        current_batch = 0

        self.model.train()
        self.learned_combination.train()
        for teacher in self.teacher_models:
            teacher.eval()

        teacher_dataloaders = (
            teacher_dataloader["train"]
            for teacher_dataloader in self.teacher_dataloaders
        )
        dataloaders = zip(self.dataloaders["train"], *teacher_dataloaders)

        for student_batch, *teachers_batch in dataloaders:
            self.optimizer.zero_grad()
            current_batch += 1
            teachers_batch = [
                {k: v.to(self.model.device) for k, v in teacher_batch.items()}
                for teacher_batch in teachers_batch
            ]
            student_batch = {
                k: v.to(self.model.device) for k, v in student_batch.items()
            }
            student_outputs = self.model(**student_batch)

            teachers_outputs = []
            for teacher, teacher_batch in zip(self.teacher_models, teachers_batch):
                teacher_output = teacher(**teacher_batch)
                teachers_outputs.append(teacher_output)

            multi_teacher_outputs = self.combine_teacher_outputs(teachers_outputs)
            teachers_outputs = self.learned_combination(multi_teacher_outputs)

            teachers_loss = self.learned_combination_loss(
                teachers_outputs.logits, student_batch["labels"]
            )
            teachers_loss_value = teachers_loss.item()

            kd_loss = self.loss(
                student_outputs.logits,
                student_batch["labels"],
                teachers_outputs.logits,
            )
            kd_loss_value = kd_loss.item()

            kd_train_loss += kd_loss_value
            ce_train_loss += student_outputs.loss.item()
            teachers_train_loss += teachers_loss_value

            combined_loss = teachers_loss + kd_loss

            combined_loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.progress_bar.update(1)
            self.progress_bar.set_description("Loss: {:.3f}".format(kd_loss_value))
        train_metrics = {
            "teachers_train_loss": teachers_train_loss / num_batches,
            "kd_train_loss": kd_train_loss / num_batches,
            "ce_train_loss": ce_train_loss / num_batches,
        }
        return train_metrics

    def combine_teacher_outputs(self, teachers_outputs):
        teacher_logits = torch.stack(
            [teacher_output.logits for teacher_output in teachers_outputs]
        )
        return teacher_logits

    def replace_optimizer(self) -> optim.Optimizer:
        return AdamW(
            [*self.model.parameters(), *self.learned_combination.parameters()],
            lr=self.learning_rate,
        )


class WeightedCombination(torch.nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.learned_weights = torch.nn.Linear(num_experts, 1)

    def forward(self, teachers_outputs):
        teachers_outputs = teachers_outputs.permute(1, 2, 0)
        output = self.learned_weights(teachers_outputs)
        output = output.squeeze()
        output = SequenceClassifierOutput(logits=output)
        return output


class TransformerCombination(torch.nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=num_experts, nhead=1
        )
        self.learned_weights = torch.nn.Linear(num_experts, 1)

    def forward(self, teachers_outputs):
        teachers_outputs = teachers_outputs.permute(1, 2, 0)
        output = self.transformer(teachers_outputs)
        output = self.learned_weights(output)
        output = output.squeeze()
        output = SequenceClassifierOutput(logits=output)
        return output
