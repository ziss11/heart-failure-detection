import os

import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from keras.utils.vis_utils import plot_model
from transform import (CATEGORICAL_FEATURES, LABEL_KEY, NUMERICAL_FEATURES,
                       transformed_name)
from tuner import input_fn


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Return a function that parses a serialized tf.Example"""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"),
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Return the output to be used in the serving signature."""

        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec,
        )

        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)

        return {"outputs": outputs}

    return serve_tf_examples_fn


def get_model(hp):
    """This model defines a keras Model

    Args:
        hp (kt.HyperParameters): object that contains best hyperparameters
        from tuner

    Returns:
        tf.keras.Model: model as a Keras object
    """

    input_features = list()

    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            layers.Input(shape=(dim+1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            layers.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = layers.concatenate(input_features)
    x = layers.Dense(hp["dense_unit"], activation=tf.nn.relu)(concatenate)

    for _ in range(hp["num_hidden_layers"]):
        x = layers.Dense(hp["dense_unit"], activation=tf.nn.relu)(x)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["binary_accuracy"],
    )

    model.summary()

    return model


def run_fn(fn_args):
    """Train the model based on given args

    Args:
        fn_args (FnArgs): Holds args used to train the model as name/value pairs.
    """

    hp = fn_args.hyperparameters["values"]
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_set = input_fn(fn_args.train_files, tf_transform_output)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output)

    model = get_model(hp)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq="batch"
    )

    early_stop_callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        patience=10,
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    callbacks = [
        tensorboard_callback,
        early_stop_callbacks,
        model_checkpoint_callback
    ]

    model.fit(
        train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_set,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        epochs=hp["tuner/initial_epoch"],
        verbose=1,
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output,
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures,
    )

    plot_model(
        model,
        to_file="images/model_plot.png",
        show_shapes=True,
        show_layer_names=True,
    )
