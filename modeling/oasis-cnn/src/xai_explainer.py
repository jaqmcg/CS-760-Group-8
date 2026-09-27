import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from lime import lime_image
import shap

# Assuming your image size parameter is fixed
IMG_SIZE = (208, 176)

# =====================================================================
# 1. GRAD-CAM (Gradient-weighted Class Activation Mapping)
# =====================================================================
def generate_gradcam(model, img_array, last_conv_layer_name="conv2d_2", pred_index=None):
    """Generates a Grad-CAM heatmap overlay for a given convolutional layer."""
    # 1. Create a model that outputs the activations of the last conv layer and the predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the gradient of the top predicted class with respect to the conv layer activations
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 3. Extract the gradient map
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Compute the feature map weights via global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map by its gradient weight
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap using ReLU (keeping only positive influences)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    """Overlays the heatmap onto the original greyscale MRI scan."""
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use Jet colormap to colorize the heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    # Convert grayscale input back to RGB for merging colors
    if img.shape[-1] == 1:
        img_rgb = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = np.uint8(img * 255)

    # Blend the images
    superimposed_img = jet * alpha + img_rgb
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img


# =====================================================================
# 2. LIME (Local Interpretable Model-agnostic Explanations)
# =====================================================================
def explain_with_lime(model, img_single_channel):
    """LIME requires a 3-channel RGB image format natively. 
    We mirror the grayscale channel to pass through, then slice it back for the model.
    """
    explainer = lime_image.LimeImageExplainer()
    
    # Replicate greyscale into 3 channels for LIME segmentation math
    img_rgb = np.repeat(img_single_channel, 3, axis=-1)

    # Custom wrapper function so LIME can score single-channel inputs
    def predict_fn(images):
        # Convert incoming RGB batch back to single channel grayscale
        gray_images = images[..., 0:1]
        return model.predict(gray_images, verbose=0)

    # Generate explanation
    explanation = explainer.explain_instance(
        img_rgb.astype('float64'), 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=200
    )
    
    # Extract bounding boundaries of key positive regions
    top_label = explanation.top_labels[0]
    mask_image, mask = explanation.get_image_and_mask(
        top_label, 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    return mask_image, mask


# =====================================================================
# 3. SHAP (SHapley Additive exPlanations)
# =====================================================================
def explain_with_shap(model, background_images, test_image):
    """Calculates feature importance values across a subset of background contexts."""
    # Explainer uses a small batch of training images to construct baseline expectations
    explainer = shap.GradientExplainer(model, background_images)
    
    # Compute Shapley values for the specific image
    shap_values = explainer.shap_values(test_image)
    return shap_values


# =====================================================================
# 4. PLOTTING AND EVALUATION EXECUTION
# =====================================================================
def main_explain(model, sample_mri, background_batch=None):
    """
    Args:
        model: Trained Keras/TF model instance
        sample_mri: Shape (208, 176, 1), scaled 0.0 to 1.0
        background_batch: Batch of training scans (e.g. 50 images) required for SHAP baseline
    """
    # Create batch dimension required for model execution pipeline
    img_batch = np.expand_dims(sample_mri, axis=0)

    # --- Run Grad-CAM ---
    # Change 'conv2d_2' if you are targeting the ViT/Hybrid variant
    heatmap, pred_idx = generate_gradcam(model, img_batch, last_conv_layer_name="conv2d_2")
    gradcam_result = overlay_gradcam(sample_mri, heatmap)

    # --- Run LIME ---
    lime_img, lime_mask = explain_with_lime(model, sample_mri)

    # --- Run SHAP ---
    if background_batch is not None:
        shap_values = explain_with_shap(model, background_batch, img_batch)
    else:
        shap_values = None

    # --- Visualize All Results ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    # Original scan
    axes[0].imshow(sample_mri.squeeze(), cmap='gray')
    axes[0].set_title(f"Original Scan\n(Pred Class: {pred_idx})")
    axes[0].axis('off')

    # Grad-CAM Display
    axes[1].imshow(gradcam_result)
    axes[1].set_title("Grad-CAM\n(Global Regions)")
    axes[1].axis('off')

    # LIME Display
    axes[2].imshow(lime_img[..., 0], cmap='gray')
    # Overlay outline boundary
    axes[2].contour(lime_mask, colors='r', levels=[0.5], linewidths=1.5)
    axes[2].set_title("LIME\n(Local Boundaries)")
    axes[2].axis('off')

    # SHAP Display
    if shap_values is not None:
        # shap_values[pred_idx] maps importance map specific to targeted predicted category
        shap_slice = shap_values[pred_idx][0].squeeze()
        pos_shap = np.maximum(shap_slice, 0) # Focus on features increasing probability
        axes[3].imshow(pos_shap, cmap='bwr')
        axes[3].set_title("SHAP\n(Pixel Attribution)")
    else:
        axes[3].text(0.5, 0.5, 'SHAP Skipped\n(No Background Batch Provided)', ha='center', va='center')
        axes[3].set_title("SHAP")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/mri_xai_explanations.png", bbox_inches='tight')
    plt.show()
