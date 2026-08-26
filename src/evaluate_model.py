"""
Evalúa el modelo multimodal entrenado en el test set
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "best_multimodal_model.keras"

IMG_SIZE = 128
NUM_CLASSES = 3
CLASSES = ['Non-Demented', 'Very Mild', 'Demented']

def load_batch_images(image_paths, indices):
    """Carga batch de imágenes PNG desde rutas."""
    batch = []
    for idx in indices:
        img = keras.preprocessing.image.load_img(
            image_paths[idx],
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale'
        )
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        batch.append(img_array)
    return np.array(batch)

def load_data_with_demographics(split_name):
    """Carga imágenes + datos demográficos de un split."""
    
    csv_path = PROCESSED_DATA_PATH / f"{split_name}_demographics.csv"
    df = pd.read_csv(csv_path)
    
    logger.info(f"Cargando {split_name}: {len(df)} muestras")
    
    image_paths = []
    demographics_data = []
    labels = []
    
    for idx, row in df.iterrows():
        label = row['label']
        image_filename = row['image_filename']
        img_path = PROCESSED_DATA_PATH / split_name / label / image_filename
        
        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(label)
            
            demographics_data.append([
                row['gender'] if pd.notna(row['gender']) else 0,
                row['age'] if pd.notna(row['age']) else 0,
                row['mmse'] if pd.notna(row['mmse']) else 0,
                row['nwbv'] if pd.notna(row['nwbv']) else 0,
                row['etiv'] if pd.notna(row['etiv']) else 0,
                row['asf'] if pd.notna(row['asf']) else 0,
            ])
    
    image_paths = np.array(image_paths)
    demographics = np.array(demographics_data, dtype=np.float32)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    demographics = scaler.fit_transform(demographics)
    
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    labels = np.array([class_to_idx[l] for l in labels])
    
    logger.info(f"  - Imágenes cargadas: {len(image_paths)}")
    logger.info(f"  - Demografía: shape {demographics.shape}")
    logger.info(f"  - Labels: {np.bincount(labels)}")
    
    return image_paths, demographics, labels

# Cargar datos de test
logger.info("\n" + "="*70)
logger.info("EVALUACIÓN DEL MODELO MULTIMODAL")
logger.info("="*70 + "\n")

test_imgs, test_demo, test_labels = load_data_with_demographics('test')

# Cargar modelo
logger.info(f"\nCargando modelo desde: {MODEL_PATH}")
keras.config.enable_unsafe_deserialization()
model = keras.models.load_model(MODEL_PATH)

# Cargar imágenes de test
logger.info("Precargando imágenes de test...")
X_test_img = load_batch_images(test_imgs, np.arange(len(test_imgs)))

# Evaluar
logger.info("\nEvaluando en test set...")
y_test = keras.utils.to_categorical(test_labels, NUM_CLASSES)
test_loss, test_acc = model.evaluate([X_test_img, test_demo], y_test, verbose=0)

logger.info("\n" + "="*70)
logger.info(f"TEST ACCURACY: {test_acc*100:.2f}%")
logger.info(f"TEST LOSS: {test_loss:.4f}")
logger.info("="*70 + "\n")

# Predicciones
logger.info("Haciendo predicciones...")
predictions = model.predict([X_test_img, test_demo], verbose=0)
pred_classes = np.argmax(predictions, axis=1)
pred_labels = np.array([CLASSES[c] for c in pred_classes])

# Matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(test_labels, pred_classes)
logger.info("\nMatriz de confusión:")
logger.info(f"\n{cm}")

# Reporte
logger.info("\nReporte de clasificación:")
logger.info(f"\n{classification_report(test_labels, pred_classes, target_names=CLASSES)}")
