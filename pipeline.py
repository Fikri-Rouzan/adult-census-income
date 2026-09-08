"""Pipeline orchestrator menggunakan Apache Beam."""

import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.orchestration import pipeline, metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# Menentukan root direktori proyek secara absolut
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PIPELINE_NAME = "fikri_rouzan-pipeline"
PIPELINE_ROOT = os.path.join(PROJECT_ROOT, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODULES_DIR = os.path.join(PROJECT_ROOT, "modules")
SERVING_MODEL_DIR = os.path.join(PROJECT_ROOT, "serving_model", "adult-income-model")


def init_pipeline(pipeline_name: str, pipeline_root: str) -> pipeline.Pipeline:
    """Inisialisasi komponen pipeline TFX dengan orchestrator Apache Beam."""

    # ExampleGen (Ingestion & Splitting 80:20)
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output_config)

    # StatisticsGen
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # SchemaGen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    # ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # Transform
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.join(MODULES_DIR, "transform.py"),
    )

    # Tuner
    tuner = Tuner(
        module_file=os.path.join(MODULES_DIR, "tuner.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=20),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=10),
    )

    # Trainer
    trainer = Trainer(
        module_file=os.path.join(MODULES_DIR, "trainer.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=20),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=10),
    )

    # Resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    # Evaluator
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="income")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.6}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # Pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            tuner,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )


if __name__ == "__main__":
    BeamDagRunner().run(init_pipeline(PIPELINE_NAME, PIPELINE_ROOT))
