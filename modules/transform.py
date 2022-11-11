import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURE = {
    "ever_married": 2,
    "work_type": 5,
    "Residence_type": 2,
    "smoking_status": 4,
}

NUMERICAL_FEATURE = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

LABEL_KEY = "stroke"


def transformed_name(key):
    """Transform feature key

    Args:
        key (str): the key to be transformed

    Returns:
        str: transformed key
    """

    return f"{key}_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Convert a label (0 or 1) into a one-hot vector

    Args:
        label_tensor (int): label tensor (0 or 1)
        num_labels (int, optional): num of label. Defaults to 2.

    Returns:
        tf.Tensor: label tensor
    """

    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features

    Args:
        inputs (dict): map from feature keys to raw features

    Returns:
        dict: map from features keys to transformed features
    """

    outputs = dict()

    for key in CATEGORICAL_FEATURE:
        dim = CATEGORICAL_FEATURE[key]

        int_value = tft.compute_and_apply_vocabulary(inputs[key], top_k=dim+1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labesl=dim+1)

    for feature in NUMERICAL_FEATURE:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
