"""CNN architecture for 3-class dementia classification on 208x176 MRI slices."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(208, 176, 1), num_classes=3, l2=1e-5):
    # No horizontal flip: hemisphere asymmetry is itself a signal for
    # atrophy, so mirroring the scan would throw that away.
    inputs = keras.Input(shape=input_shape)

    x = layers.RandomRotation(0.02)(inputs)
    x = layers.RandomTranslation(0.02, 0.02)(x)

    # No BatchNorm: on this dataset (only ~50 steps/epoch) its running
    # statistics consistently lagged the weights enough that eval-mode
    # predictions collapsed to a single class every run, even though
    # train-mode metrics looked fine. L2 + dropout is the more standard,
    # more stable choice for a dataset this small anyway.
    reg = keras.regularizers.l2(l2)
    for filters in (24, 48, 96):
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu", kernel_regularizer=reg
        )(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="dementia_cnn")
