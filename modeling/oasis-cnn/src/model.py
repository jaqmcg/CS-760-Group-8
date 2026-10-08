import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""CNN architecture for 3-class dementia classification on 208x176 MRI slices."""

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


class PatchExtractor(layers.Layer):
    """Safely extracts patches inside a Keras 3 Functional pipeline."""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        # Native TensorFlow ops must live safely inside a Layer's call method
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return patches


class PatchEncoder(layers.Layer):
    """Cuts the MRI into patches and injects position embeddings."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

"""ViT architecture for 3-class dementia classification on 208x176 MRI slices."""

def build_vit_model(input_shape=(208, 176, 1), num_classes=3, l2_reg=1e-5):
    # Setup patches (Must divide input cleanly: 208/16 = 13, 176/16 = 11)
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    projection_dim = 64
    num_heads = 4
    transformer_layers = 3

    reg = keras.regularizers.l2(l2_reg)
    inputs = keras.Input(shape=input_shape)

    # 1. Custom Small-Scale Augmentation (NO FLIPS)
    x = layers.RandomRotation(0.02)(inputs)
    x = layers.RandomTranslation(0.02, 0.02)(x)

    # 2. Extract Patches using our new Keras-compliant layer
    patches = PatchExtractor(patch_size=patch_size)(x)
    
    # Reshape to: (batch, num_patches, patch_area * channels)
    patches = layers.Reshape((num_patches, patch_size * patch_size * input_shape[-1]))(
        patches
    )
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # 3. Transformer Encoder Blocks (LayerNorm replaces BatchNorm safely)
    for _ in range(transformer_layers):
        # Layer Norm 1 + Multi-Head Attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.2
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Norm 2 + MLP Network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(
            projection_dim * 2, activation="relu", kernel_regularizer=reg
        )(x3)
        x3 = layers.Dropout(0.2)(x3)
        x3 = layers.Dense(projection_dim, kernel_regularizer=reg)(x3)
        x3 = layers.Dropout(0.2)(x3)
        encoded_patches = layers.Add()([x3, x2])

    # 4. Output Decision Head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    features = layers.Dense(32, activation="relu", kernel_regularizer=reg)(
        representation
    )
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    return keras.Model(inputs, outputs, name="dementia_vit")

"""CNN-SVM architecture for 3-class dementia classification on 208x176 MRI slices."""

def build_cnn_svm_model(input_shape=(208, 176, 1), num_classes=3, l2=1e-5):
    inputs = keras.Input(shape=input_shape)

    # Custom Augmentation (NO FLIPS)
    x = layers.RandomRotation(0.02)(inputs)
    x = layers.RandomTranslation(0.02, 0.02)(x)

    reg = keras.regularizers.l2(l2)

    # Core CNN Feature Extractor (No BatchNorm)
    for filters in (24, 48, 96):
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu", kernel_regularizer=reg
        )(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.3)(x)

    # SVM OUTPUT HEAD:
    # Linear activation paired with L2 penalty behaves as a Multi-class SVM
    # Note: Use 'squared_hinge' as the loss function during model.compile()!
    outputs = layers.Dense(
        num_classes,
        activation="linear",
        kernel_regularizer=keras.regularizers.l2(1e-3),
    )(x)

    return keras.Model(inputs, outputs, name="dementia_cnn_svm")

"""CNN-Transformer architecture for 3-class dementia classification on 208x176 MRI slices."""

def build_hybrid_cnn_transformer(
    input_shape=(208, 176, 1), num_classes=3, l2=1e-5
):
    inputs = keras.Input(shape=input_shape)

    # Global Augmentation Setup (NO FLIPS)
    aug = layers.RandomRotation(0.02)(inputs)
    aug = layers.RandomTranslation(0.02, 0.02)(aug)

    reg = keras.regularizers.l2(l2)

    # --- BRANCH 1: CNN (Local Textures) ---
    cnn_x = layers.Conv2D(
        32, 3, padding="same", activation="relu", kernel_regularizer=reg
    )(aug)
    cnn_x = layers.MaxPooling2D()(cnn_x)
    cnn_x = layers.Dropout(0.2)(cnn_x)
    cnn_features = layers.GlobalAveragePooling2D()(cnn_x)

    # --- BRANCH 2: Transformer (Global Dependencies) ---
    # Downscale image to a lower resolution patch grid
    patch_proj = layers.Conv2D(
        32,
        kernel_size=16,
        strides=16,
        padding="valid",
        kernel_regularizer=reg,
    )(aug)
    # Flatten grid to sequence
    seq_len = (input_shape[0] // 16) * (input_shape[1] // 16)
    trans_x = layers.Reshape((seq_len, 32))(patch_proj)

    # Lightweight Self-Attention Layer
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.2)(
        trans_x, trans_x
    )
    transformer_features = layers.GlobalAveragePooling1D()(attn_out)

    # --- FEATURE FUSION & DECISION HEAD ---
    # Merge both structural views together
    fused = layers.Concatenate()([cnn_features, transformer_features])

    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(fused)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="dementia_hybrid_net")