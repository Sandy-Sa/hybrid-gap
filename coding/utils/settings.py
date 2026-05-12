from pathlib import Path

# Column names
ID_COLUMN = "event_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"

MODELS = {
    "llama3": {
        "target_token": "pad",
        "pad_id": 128009,
        "pad": "<|eot_id|>",
        "sos": "<|begin_of_text|>",
        "eos": "<|end_of_text|>",
    },
    "llama3.2": {
        "target_token": "pad",
        "pad_id": 128004,
        "pad": "<|finetune_right_pad_id|>",
        "sos": "<|sos|>",
        "eos": "<|eos|>",
    },
    "qwen3": {"target_token": "pad", "pad_id": 151643, "pad": "<|endoftext|>"},
    "qwen3_embedding": {
        "target_token": "pad",
        "pad_id": 151643,
        "pad": "<|endoftext|>",
    },
    "mediphi": {"target_token": "pad", "pad_id": 32000, "pad": "<|endoftext|>"},
    "bioclinical": {
        "target_token": "cls",
        "pad_id": 1,
        "pad": "<|padding|>",
        "cls_id": 50281,
        "cls": "[CLS]",
        "mask": "[MASK]",
    },
    "medgemma27": {
        "target_token": "pad",
        "pad_id": 0,
        "pad": "<pad>",
    },
}
