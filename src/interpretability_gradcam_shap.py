"""Analiza las predicciones mediante Grad-CAM y variables demográficas."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

tf.config.set_visible_devices([], 'GPU')

CLASSES = ['Non-Demented', 'Very Mild', 'Demented']
DATA_DIR = Path('data/processed')
MODEL_PATH = Path('models/best_multimodal_model.keras')
IMG_SIZE = 128
VIZ_DIR = Path('results_interpretability')
VIZ_DIR.mkdir(exist_ok=True)


def load_test_images_with_labels(num_samples=10):
    """Carga algunas imágenes de test junto con sus etiquetas."""
    test_csv = DATA_DIR / 'test_demographics.csv'
    df = pd.read_csv(test_csv)
    train_df = pd.read_csv(DATA_DIR / 'train_demographics.csv')
    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    scaler = StandardScaler()
    scaler.fit(train_df[demo_features].fillna(0).astype(float))
    
    df = df.drop_duplicates(subset=['subject_id'], keep='first').reset_index(drop=True)
    
    if len(df) < num_samples:
        num_samples = len(df)
    
    sample_indices = np.random.choice(len(df), num_samples, replace=False)
    df_sample = df.iloc[sample_indices].reset_index(drop=True)
    
    images = []
    demos = []
    labels = []
    subjects = []
    
    for idx, row in df_sample.iterrows():
        try:
            img_path = DATA_DIR / 'test' / row['label'] / row['image_filename']
            img = keras.preprocessing.image.load_img(
                img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale'
            )
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
            
            demo_vals = row[demo_features].values.astype(float)
            demo_vals = np.nan_to_num(demo_vals, nan=0.0)
            demo_vals = scaler.transform([demo_vals])[0]
            demos.append(demo_vals)
            
            labels.append(row['label'])
            subjects.append(row['subject_id'])
        except Exception as e:
            logger.warning(f"Error cargando {row['subject_id']}: {e}")
    
    return np.array(images), np.array(demos), labels, subjects


def create_grad_cam_heatmap(model, img_array, demo_array, class_idx):
    """Genera un mapa de calor Grad-CAM."""
    base_model = next(
        (layer for layer in model.layers if isinstance(layer, keras.Model)),
        None
    )
    if base_model is None:
        raise ValueError("No se encontró la rama ResNet50 dentro del modelo")

    conv_layers = [
        layer for layer in base_model.layers
        if isinstance(layer, keras.layers.Conv2D)
    ]
    if not conv_layers:
        raise ValueError("No se encontró una capa convolucional en ResNet50")

    last_conv_layer = conv_layers[-1]
    activation_model = keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    def get_layer(name):
        return model.get_layer(name)
    
    with tf.GradientTape() as tape:
        image_rgb = tf.concat([img_array, img_array, img_array], axis=-1)
        last_conv_output, base_output = activation_model(image_rgb)

        image_features = get_layer('global_average_pooling2d')(base_output)
        image_features = get_layer('img_dense_1')(image_features)
        image_features = get_layer('dropout')(image_features, training=False)
        image_features = get_layer('img_dense_2')(image_features)

        demo_features = get_layer('demo_dense_1')(demo_array)
        demo_features = get_layer('dropout_1')(demo_features, training=False)
        demo_features = get_layer('demo_dense_2')(demo_features)

        merged = get_layer('concatenate_1')([image_features, demo_features])
        merged = get_layer('merged_dense_1')(merged)
        merged = get_layer('dropout_2')(merged, training=False)
        merged = get_layer('merged_dense_2')(merged)
        preds = get_layer('classification')(merged)
        class_channel = preds[:, class_idx]
    
    grads = tape.gradient(class_channel, last_conv_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()


def overlay_gradcam_on_image(img_array, heatmap, subject_id, label, pred_class, confidence):
    """Crea una visualización de la imagen y su mapa Grad-CAM."""
    img = (img_array[:, :, 0] * 255).astype(np.uint8)
    
    heatmap_resized = np.kron(heatmap, np.ones((IMG_SIZE // heatmap.shape[0], IMG_SIZE // heatmap.shape[1])))
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original MRI', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.6)
    axes[1].set_title('Grad-CAM Attention', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction info
    axes[2].axis('off')
    info_text = f"Subject: {subject_id}\n\n"
    info_text += f"True Label: {label}\n"
    info_text += f"Prediction: {pred_class}\n"
    info_text += f"Confidence: {confidence:.2%}\n\n"
    info_text += "Activation regions (red):\n"
    info_text += "Show brain areas used\n"
    info_text += "for classification"
    axes[2].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    save_path = VIZ_DIR / f'gradcam_{subject_id}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_model_architecture_diagram():
    """Crea una representación visual de la arquitectura."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Title
    title = "Multimodal Alzheimer's Detection Model Architecture"
    ax.text(0.5, 0.95, title, fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Image branch
    ax.text(0.2, 0.85, "Image Branch", fontsize=13, fontweight='bold')
    ax.text(0.2, 0.80, "MRI Input (128x128x1)", fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    ax.text(0.2, 0.75, "Grayscale → RGB", fontsize=9, style='italic')
    ax.text(0.2, 0.70, "ResNet50 (frozen)", fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    ax.text(0.2, 0.65, "GlobalAveragePooling", fontsize=9)
    ax.text(0.2, 0.60, "Dense(256) + Dropout", fontsize=9)
    ax.text(0.2, 0.55, "Dense(128)", fontsize=9)
    ax.text(0.2, 0.50, "128D Features", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Demographics branch
    ax.text(0.8, 0.85, "Demographics Branch", fontsize=13, fontweight='bold')
    ax.text(0.8, 0.80, "Clinical Input (6 features)", fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax.text(0.8, 0.75, "Gender, Age, MMSE", fontsize=9, style='italic')
    ax.text(0.8, 0.70, "nWBV, eTIV, ASF", fontsize=9, style='italic')
    ax.text(0.8, 0.65, "StandardScaler", fontsize=9)
    ax.text(0.8, 0.60, "Dense(32) + Dropout", fontsize=9)
    ax.text(0.8, 0.55, "Dense(16)", fontsize=9)
    ax.text(0.8, 0.50, "16D Features", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Fusion
    ax.arrow(0.3, 0.48, 0.15, -0.08, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
    ax.arrow(0.7, 0.48, -0.15, -0.08, head_width=0.02, head_length=0.02, fc='red', ec='red')
    
    ax.text(0.5, 0.35, "Concatenate [128D, 16D] = 144D", fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(0.5, 0.28, "Dense(128) + Dropout(0.5)", fontsize=9, ha='center')
    ax.text(0.5, 0.23, "Dense(64)", fontsize=9, ha='center')
    ax.text(0.5, 0.18, "Dense(3, softmax)", fontsize=9, ha='center')
    
    ax.text(0.5, 0.10, "Output: [Non-Demented | Very Mild | Demented]", fontsize=11, 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    save_path = VIZ_DIR / 'model_architecture.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def main():
    logger.info("Iniciando análisis de interpretabilidad")
    
    logger.info("Cargando modelo multimodal")
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo: {e}")
        logger.info("Modelo no encontrado; se generarán visualizaciones limitadas")
        model = None
    
    logger.info("Generando diagrama de arquitectura")
    arch_path = plot_model_architecture_diagram()
    logger.info(f"Guardado en: {arch_path}")
    
    if model is not None:
        logger.info("Cargando muestras de test para Grad-CAM")
        images, demos, labels, subjects = load_test_images_with_labels(num_samples=6)
        
        logger.info(f"Generando visualizaciones Grad-CAM para {len(images)} muestras")
        
        for i in range(len(images)):
            img = images[i:i+1]
            demo = demos[i:i+1]
            label = labels[i]
            subject = subjects[i]
            
            pred_probs = model.predict([img, demo], verbose=0)[0]
            pred_class = CLASSES[np.argmax(pred_probs)]
            confidence = np.max(pred_probs)
            
            pred_idx = np.argmax(pred_probs)
            heatmap = create_grad_cam_heatmap(model, img, demo, pred_idx)
            
            viz_path = overlay_gradcam_on_image(img[0], heatmap, subject, label, pred_class, confidence)
            logger.info(f"  {subject}: real={label}, predicción={pred_class} ({confidence:.1%}) - guardado en {viz_path}")
    
    logger.info(f"Visualizaciones guardadas en: {VIZ_DIR}")
    logger.info("Análisis de interpretabilidad completado")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
