import logging
from typing import Optional
from packaging import version

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from .configuration_lstm import BiLSTMConfig

logger = logging.getLogger(__name__)


class BiLSTMPreTrainedModel(PreTrainedModel):
    config_class = BiLSTMConfig
    base_model_prefix = "bilstm"

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Copied from transformers.models.bert.modeling_albert.AlbertEmbeddings
class BiLSTMEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: BiLSTMConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_dim
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_dim
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name
        #  and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported
        # when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is
        # all zeros, which usually occurs when its auto-generated, registered buffer
        # helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BiLSTMModel(BiLSTMPreTrainedModel):
    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = BiLSTMEmbeddings(config)
        # self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional_lstm,
            dropout=config.dropout_prob,
        )

        self.hidden_ffnn = nn.Sequential(
            *(
                x
                for _ in range(config.num_ffnn_layers)
                for x in (
                    nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
                    nn.Dropout(config.dropout_prob),
                    nn.GELU(),
                )
            ),
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        x = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        # x = self.layer_norm(x)

        # the output x shape is (batch_size, sequence_len, hidden_size * 2)
        # the output hn shape is (2 * num_layers, sequence_len, hidden_size)
        x, (hn, cn) = self.lstm(x)
        x = self.hidden_ffnn(x)

        return x


class BiLSTMForSequenceClassification(BiLSTMPreTrainedModel):
    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bilstm = BiLSTMModel(config)
        self.classifier = nn.Linear(config.hidden_dim * 2, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: torch.LongTensor | None = None,
        *args,
        **kwargs,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:

        x = self.bilstm(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        x = torch.mean(x, dim=1)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(logits=logits, loss=loss)
