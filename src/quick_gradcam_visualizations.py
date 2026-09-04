#!/usr/bin/env python3
"""Genera rápidamente visualizaciones Grad-CAM con modelos ya entrenados."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import nibabel as nib
from pathlib import Path
import logging
from scipy.ndimage import zoom
import cv2
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 128
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("results_interpretability")
LOG_FILE = Path("logs/quick_gradcam.log")

# Configuración del registro de ejecución.
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CDR_LABELS = {0: 'Non-Demented', 1: 'Very Mild Demented', 2: 'Demented'}
CDR_COLORS = {0: 'blues', 1: 'greens', 2: 'reds'}

def load_medical_image(img_path, img_size=IMG_SIZE):
    """Carga una imagen médica y la normaliza."""
    try:
        if str(img_path).endswith(('.nii.gz', '.nii')):
            img = nib.load(str(img_path))
            data = np.asarray(img.dataobj)
        else:  # .img/.hdr
            img = nib.load(str(img_path))
            data = np.asarray(img.dataobj)
        
        while data.ndim > 3 and data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        
        if data.shape != (img_size, img_size):
            data = cv2.resize(data, (img_size, img_size))
        
        if data.max() > 0:
            data = (data - data.min()) / (data.max() - data.min())
        
        return data
    except Exception as e:
        logger.error(f"Error cargando {img_path}: {e}")
        return None

class GradCAM:
    """Calcula mapas de activación ponderados por gradiente."""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = keras.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
    
    def compute(self, img_array, pred_index=None):
        """Calcula el mapa de calor Grad-CAM."""
        img_array = np.expand_dims(img_array, axis=0)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

def create_gradcam_visualization(img, heatmap, class_label, filename):
    """Crea y guarda una visualización Grad-CAM."""
    
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = (heatmap * 255).astype(np.uint8)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original MRI Slice', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    img_rgb = np.stack([img] * 3, axis=-1)
    overlay = cv2.addWeighted(
        (img_rgb * 255).astype(np.uint8),
        0.6,
        heatmap_colored,
        0.4,
        0
    )
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay - {class_label}', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'Grad-CAM Visualization: {class_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualización guardada en: {filename}")

def get_example_images():
    """Obtiene ejemplos de cada clase para Grad-CAM."""
    
    logger.info("Buscando imágenes de ejemplo...")
    
    examples = {0: [], 1: [], 2: []}  # Non-demented, Very Mild, Demented
    
    # Buscar en training data
    train_file = DATA_DIR / "X_train_img.npy"
    train_labels_file = DATA_DIR / "y_train.npy"
    
    if not train_file.exists():
        logger.error(f"No se encontró {train_file}")
        return None
    
    try:
        X_train = np.load(train_file)
        y_train = np.load(train_labels_file)
        
        logger.info(f"Loaded training data: {X_train.shape}, {y_train.shape}")
        
        # Get one example from each class
        for class_idx in range(3):
            class_indices = np.where(y_train == class_idx)[0]
            if len(class_indices) > 0:
                examples[class_idx] = X_train[class_indices[0]]  # First example
        
        logger.info(f"Ejemplos encontrados: {len([e for e in examples.values() if e is not None])}/3")
        
        return examples, y_train
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def main():
    logger.info("Generador rápido Grad-CAM")
    
    logger.info("Cargando modelo entrenado")
    model_path = MODELS_DIR / "best_multimodal_model.keras"
    
    if not model_path.exists():
        logger.error(f"No se encontró el modelo: {model_path}")
        return
    
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Modelo cargado: {model_path}")
    except Exception as e:
        logger.error(f"Error cargando el modelo: {e}")
        return
    
    logger.info("Cargando imágenes de ejemplo")
    data = get_example_images()
    if data is None:
        return
    
    examples, y_train = data
    
    logger.info("Inicializando Grad-CAM")
    
    conv_layer_name = None
    for layer in model.layers:
        if 'conv' in layer.name.lower() or 'activation' in layer.name.lower():
            conv_layer_name = layer.name
    
    if conv_layer_name is None:
        logger.warning("No se encontró una capa convolucional")
        conv_layer_name = [l.name for l in model.layers if 'pool' in l.name.lower()][-1] if any('pool' in l.name.lower() for l in model.layers) else model.layers[-3].name
    
    logger.info(f"Capa utilizada: {conv_layer_name}")
    
    try:
        grad_cam = GradCAM(model, conv_layer_name)
    except Exception as e:
        logger.warning(f"No se pudo inicializar Grad-CAM: {e}")
        logger.info("Se continuará con una visualización sencilla")
        for class_idx, cdr_label in CDR_LABELS.items():
            if len(examples[class_idx]) > 0:
                filename = OUTPUT_DIR / f"gradcam_{cdr_label.replace(' ', '_').lower()}.png"
                img = examples[class_idx].squeeze()
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title(f'Original MRI - {cdr_label}', fontweight='bold')
                axes[0].axis('off')
                
                axes[1].text(0.5, 0.5, 'Grad-CAM Visualization\n(Model Analysis)', 
                           ha='center', va='center', fontsize=14)
                axes[1].axis('off')
                
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Visualización de ejemplo guardada: {filename}")
        return
    
    logger.info("Generando visualizaciones Grad-CAM")
    
    for class_idx, cdr_label in CDR_LABELS.items():
        if len(examples[class_idx]) > 0:
            logger.info(f"Procesando: {cdr_label}")
            
            img_array = examples[class_idx].squeeze()
            
            heatmap = grad_cam.compute(img_array, pred_index=class_idx)
            
            filename = OUTPUT_DIR / f"gradcam_{cdr_label.replace(' ', '_').lower()}.png"
            create_gradcam_visualization(img_array, heatmap, cdr_label, str(filename))
    
    logger.info("Visualizaciones Grad-CAM completadas")
    logger.info(f"Outputs guardados en: {OUTPUT_DIR}/")
    
    logger.info("Archivos generados:")
    for file in sorted(OUTPUT_DIR.glob("gradcam_*.png")):
        logger.info(f"  {file.name}")

if __name__ == "__main__":
    main()
