import os

import tensorflow_model_analysis as tfma
from tfx.components import (CsvExampleGen, Evaluator, ExampleValidator, Pusher,
                            SchemaGen, StatisticsGen, Trainer, Transform,
                            Tuner)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import \
    LatestBlessedModelStrategy
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def init_component(
    data_dir,
    transform_module,
    trainer_module,
    tuner_module,
    train_steps,
    eval_steps,
    serving_model_dir,
):
    """Initiate tfx pipeline components

    Args:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform module file
        trainer_module (str): a path to the trainer module file
        tuner_module (str): a path to the tuner module file
        train_steps (int): num of training steps
        eval_steps (int): num of eval steps
        serving_model_dir (str): a path to the serving model directory

    Returns:
        tuple: TFX pipeline components
    """
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_bucket=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_bucket=2),
        ]),
    )

    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=output,
    )

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(transform_module),
    )

    tuner = Tuner(
        module_file=os.path.abspath(tuner_module),
        examples=transform.outputs["transformed_example"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(
            name="train",
            num_steps=train_steps,
        ),
        eval_steps=trainer_pb2.EvalArgs(
            name="eval",
            num_steps=eval_steps,
        ),
    )

    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs["transformed_example"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"],
            num_steps=train_steps,
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=["eval"],
            num_steps=eval_steps
        ),
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolve")

    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(features_key=[
            "gender",
            "smoking_status",
        ]),
    ]

    metric_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name="AUC"),
            tfma.MetricConfig(class_name="Precision"),
            tfma.MetricConfig(class_name="Recall"),
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(class_name="TruePositives"),
            tfma.MetricConfig(class_name="FalsePositives"),
            tfma.MetricConfig(class_name="TrueNegatives"),
            tfma.MetricConfig(class_name="FalseNegatives"),
            tfma.MetricConfig(class_name="BinaryAccuracy",
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={"value": .6},
                                  ),
                                  change_threshold=tfma.GenericChangeThreshold(
                                      direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute={"value": 1e-4},
                                  ),
                              ),
                              ),
        ]),
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="stroke")],
        slicing_specs=slicing_specs,
        metric_specs=metric_specs,
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir,
            ),
        ),
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )

    return components
