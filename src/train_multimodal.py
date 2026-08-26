# Entrenamiento multimodal combinando imagen RM y datos clínicos del paciente.

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

IMG_SIZE = 128
NUM_CLASSES = 3
CLASSES = ['Non-Demented', 'Very Mild', 'Demented']
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001


def load_data_with_demographics(split_name):
    """Carga imágenes y datos demográficos de un split desde el CSV."""
    
    # Cargar CSV con demografía
    csv_path = PROCESSED_DATA_PATH / f"{split_name}_demographics.csv"
    df = pd.read_csv(csv_path)
    
    logger.info(f"Cargando {split_name}: {len(df)} muestras")
    
    # Construir rutas de imágenes
    image_paths = []
    demographics_data = []
    labels = []
    
    for idx, row in df.iterrows():
        subject_id = row['subject_id']
        label = row['label']
        image_filename = row['image_filename']  # NUEVO: usar nombre del archivo del CSV
        
        # Ruta de imagen usando nombre completo del archivo
        img_path = PROCESSED_DATA_PATH / split_name / label / image_filename
        
        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(label)
            
            # Extraer demografía (rellenar NaN con 0)
            demographics_data.append([
                row['gender'] if pd.notna(row['gender']) else 0,
                row['age'] if pd.notna(row['age']) else 0,
                row['mmse'] if pd.notna(row['mmse']) else 0,
                row['nwbv'] if pd.notna(row['nwbv']) else 0,
                row['etiv'] if pd.notna(row['etiv']) else 0,
                row['asf'] if pd.notna(row['asf']) else 0,
            ])
        else:
            logger.warning(f"Imagen no encontrada: {img_path}")
    
    image_paths = np.array(image_paths)
    demographics = np.array(demographics_data, dtype=np.float32)
    
    scaler = StandardScaler()
    demographics = scaler.fit_transform(demographics)
    
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    labels = np.array([class_to_idx[l] for l in labels])
    
    logger.info(f"{split_name}: {len(image_paths)} imágenes")
    
    feature_names = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    
    return image_paths, demographics, labels, feature_names, scaler


def build_multimodal_model():
    """Construye el modelo con dos ramas: imagen (ResNet50) y datos clínicos."""
    
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='image_input')
    
    x = layers.Concatenate(axis=-1)([img_input, img_input, img_input])
    
    # ResNet50 pretrained
    resnet = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    resnet.trainable = False
    
    x = resnet(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    img_features = layers.Dense(256, activation='relu', name='img_dense_1')(x)
    img_features = layers.Dropout(0.3)(img_features)
    img_features = layers.Dense(128, activation='relu', name='img_dense_2')(img_features)
    
    demo_input = layers.Input(shape=(6,), name='demographics_input')
    
    y = layers.Dense(32, activation='relu', name='demo_dense_1')(demo_input)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(16, activation='relu', name='demo_dense_2')(y)
    
    demo_features = y
    
    merged = layers.Concatenate()([img_features, demo_features])
    merged = layers.Dense(128, activation='relu', name='merged_dense_1')(merged)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(64, activation='relu', name='merged_dense_2')(merged)
    
    # Output: 3 clases
    output = layers.Dense(NUM_CLASSES, activation='softmax', name='classification')(merged)
    
    # Crear modelo
    model = keras.Model(
        inputs=[img_input, demo_input],
        outputs=output,
        name='alzheimer_multimodal'
    )
    
    logger.info("Modelo multimodal creado:")
    model.summary()
    
    return model


def load_batch_images(image_paths, indices):
    """Carga un lote de imágenes PNG desde rutas."""
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


class MultimodalGenerator(keras.utils.Sequence):
    """Generador que combina imágenes y datos demográficos en lotes."""
    
    def __init__(self, image_paths, demographics, labels, batch_size=32, augment=False):
        self.image_paths = image_paths
        self.demographics = demographics
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(labels))
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        # Seleccionar índices para este lote
        start = batch_idx * self.batch_size
        end = min((batch_idx + 1) * self.batch_size, len(self.labels))
        batch_indices = self.indices[start:end]
        
        batch_images = load_batch_images(self.image_paths, batch_indices)
        
        # Data augmentation (si aplica)
        if self.augment:
            batch_images = self._augment(batch_images)
        
        batch_demographics = self.demographics[batch_indices]
        batch_labels = self.labels[batch_indices]
        batch_labels = keras.utils.to_categorical(batch_labels, NUM_CLASSES)
        
        return [batch_images, batch_demographics], batch_labels
    
    def _augment(self, images):
        """Data augmentation ligero."""
        augmented = []
        for img in images:
            # Rotación ±15°
            angle = np.random.uniform(-15, 15)
            img = keras.preprocessing.image.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
            
            # Zoom
            zoom = np.random.uniform(0.9, 1.1)
            img = keras.preprocessing.image.random_zoom(img, [zoom, zoom], row_axis=0, col_axis=1, channel_axis=2)
            
            augmented.append(img)
        return np.array(augmented)


def train_multimodal():
    """Pipeline de entrenamiento del modelo multimodal."""
    
    logger.info("Iniciando entrenamiento multimodal")
    
    logger.info("Cargando datos...")
    train_imgs, train_demo, train_labels, feature_names, scaler = load_data_with_demographics('train')
    val_imgs, val_demo, val_labels, _, _ = load_data_with_demographics('validation')
    test_imgs, test_demo, test_labels, _, _ = load_data_with_demographics('test')
    
    train_gen = MultimodalGenerator(train_imgs, train_demo, train_labels, batch_size=BATCH_SIZE, augment=True)
    val_gen = MultimodalGenerator(val_imgs, val_demo, val_labels, batch_size=BATCH_SIZE, augment=False)
    test_gen = MultimodalGenerator(test_imgs, test_demo, test_labels, batch_size=BATCH_SIZE, augment=False)
    
    model = build_multimodal_model()
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_multimodal_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # Entrenar
    logger.info("\nIniciando entrenamiento...")
    
    X_train_img = load_batch_images(train_imgs, np.arange(len(train_imgs)))
    X_val_img = load_batch_images(val_imgs, np.arange(len(val_imgs)))
    X_test_img = load_batch_images(test_imgs, np.arange(len(test_imgs)))
    
    y_train = keras.utils.to_categorical(train_labels, NUM_CLASSES)
    y_val = keras.utils.to_categorical(val_labels, NUM_CLASSES)
    y_test = keras.utils.to_categorical(test_labels, NUM_CLASSES)
    
    # Entrenar con arrays en lugar de generadores
    history = model.fit(
        [X_train_img, train_demo],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_val_img, val_demo], y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate([X_test_img, test_demo], y_test, verbose=0)
    logger.info(f"Test accuracy: {test_acc*100:.2f}%, loss: {test_loss:.4f}")
    
    return model, history


if __name__ == '__main__':
    Path('models').mkdir(exist_ok=True)
    model, history = train_multimodal()
    model.save('models/final_multimodal_model.keras')
    logger.info("Modelo guardado: models/final_multimodal_model.keras")
