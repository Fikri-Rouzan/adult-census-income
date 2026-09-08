"""Modul untuk tuning hyperparameter menggunakan KerasTuner dalam TFX."""

# pylint: disable=duplicate-code

from typing import NamedTuple, Dict, Any, List
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.transform import (  # pylint: disable=import-error
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    transformed_name,
)

TunerFnResult = NamedTuple(
    "TunerFnResult",
    [("tuner", kt.Tuner), ("fit_kwargs", Dict[str, Any])],
)


def build_model(
    hp: kt.HyperParameters, tf_transform_output: tft.TFTransformOutput
) -> tf.keras.Model:
    """Build model Keras dengan hyperparameter dinamis."""
    del tf_transform_output

    input_features: List[tf.keras.Input] = []
    encoded_features: List[tf.Tensor] = []

    for feature in NUMERICAL_FEATURES:
        inp = tf.keras.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.float32
        )
        input_features.append(inp)
        encoded_features.append(inp)

    for feature in CATEGORICAL_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.int64)
        input_features.append(inp)
        encoded_features.append(tf.cast(inp, tf.float32))

    x = tf.keras.layers.concatenate(encoded_features)

    for i in range(hp.Int("num_layers", min_value=1, max_value=3, step=1)):
        x = tf.keras.layers.Dense(
            hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(
            hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.4, step=0.1)
        )(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def input_fn(
    file_pattern: List[str],
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Generate fitur dan label untuk tuning/training."""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(
            filenames, compression_type="GZIP"
        ),
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Callback untuk komponen Tuner TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, tf_transform_output),
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=5,
        directory=fn_args.working_dir,
        project_name="census_income_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
