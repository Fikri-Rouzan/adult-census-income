"""Modul untuk preprocessing data dalam komponen TFX Transform."""

from typing import Dict
import tensorflow as tf
import tensorflow_transform as tft

# Definisi nama fitur numerik dan kategorikal
NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

LABEL_KEY = "income"


def transformed_name(key: str) -> str:
    """Helper untuk menambahkan suffix pada nama fitur yang telah ditransformasikan."""
    return f"{key}_xf"


def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Callback untuk komponen Transform TFX.

    Args:
        inputs: Dictionary dari fitur input.

    Returns:
        Dictionary dari fitur yang telah ditransformasikan.
    """
    outputs: Dict[str, tf.Tensor] = {}

    # Transformasi fitur numerik (Standarisasi Z-score)
    for feature in NUMERICAL_FEATURES:
        numeric_val = tf.cast(inputs[feature], tf.float32)
        outputs[transformed_name(feature)] = tft.scale_to_z_score(numeric_val)

    # Transformasi fitur kategorikal (Vocabulary index)
    for feature in CATEGORICAL_FEATURES:
        cat_val = inputs[feature]
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(cat_val)

    # Transformasi label
    label = inputs[LABEL_KEY]
    is_above_50k = tf.logical_or(tf.equal(label, ">50K"), tf.equal(label, ">50K."))
    outputs[transformed_name(LABEL_KEY)] = tf.cast(is_above_50k, tf.int64)

    return outputs
