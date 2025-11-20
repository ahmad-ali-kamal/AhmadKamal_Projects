from transformers import AutoTokenizer

MAX_LENGTH = 128
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config = {
    "vocabulary_size": tokenizer.vocab_size,
    "num_classes": 2,
    "d_embed": 128,
    "context_size": MAX_LENGTH,
    "layers_num": 4,
    "heads_num": 4,
    "head_size": 32,
    "dropout_rate": 0.1,
    "use_bias": True
}
