import logging

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class BiLSTMConfig(PretrainedConfig):
    model_type = "bilstm"

    def __init__(
        self,
        vocab_size: int = 10,
        type_vocab_size: int = 2,
        embedding_dim: int = 10,
        hidden_dim: int = 10,
        num_layers: int = 1,
        num_ffnn_layers: int = 0,
        initializer_range: float = 0.1,
        dropout_prob: float = 0.5,
        pad_token_id: int = 0,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 512,
        bidirectional_lstm: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_ffnn_layers = num_ffnn_layers
        self.initializer_range = initializer_range
        self.dropout_prob = dropout_prob
        self.pad_token_id = pad_token_id
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.bidirectional_lstm = bidirectional_lstm
        super().__init__(**kwargs)
