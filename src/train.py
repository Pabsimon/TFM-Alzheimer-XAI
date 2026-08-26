# Entrenamiento de un clasificador de Alzheimer con ResNet50 y transfer learning.

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
IMG_CHANNELS = 1
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 3
CLASSES = ['Non-Demented', 'Very Mild', 'Demented']

LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

PATIENCE_EARLY_STOPPING = 15
PATIENCE_LR_REDUCE = 5
MIN_LR = 1e-7


# =============================================================================
# FUNCIÓN: CONSTRUCCIÓN DEL MODELO
# =============================================================================

def build_model(freeze_base=True, freeze_until_layer=100):
    """Construye el modelo ResNet50 con transfer learning para clasificación de Alzheimer."""
    
    logger.info("Construyendo modelo...")
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    if freeze_base:
        # Se descongelan las últimas 20 capas para adaptación al dominio médico.
        base_model.trainable = True
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
        total_count = len(base_model.layers)
        logger.info(f"Fine-tuning: {trainable_count}/{total_count} capas entrenables")
    else:
        base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
        # Expandir de 1 canal a 3 para compatibilidad con ResNet50
        layers.Lambda(lambda x: tf.concat([x, x, x], axis=-1)) if IMG_CHANNELS == 1
            else layers.Lambda(lambda x: x),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(rate=DROPOUT_RATE),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


# =============================================================================
# FUNCIÓN: GENERADORES DE DATOS
# =============================================================================

def get_data_generators():
    """Crea generadores de datos con augmentación para train y sin augmentación para val/test."""
    
    logger.info("Cargando datos...")
    
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    train_data_path = DATA_PATH / "train"
    val_data_path = DATA_PATH / "validation"
    test_data_path = DATA_PATH / "test"
    
    if not train_data_path.exists():
        logger.warning(f"Directorio de entrenamiento no encontrado: {train_data_path}")
        logger.warning("Asegúrese de que data/processed/ contiene subdirectorios train/, validation/, test/")
    
    common_flow_args = {
        'target_size': (IMG_SIZE, IMG_SIZE),
        'color_mode': 'grayscale',
        'class_mode': 'categorical',
        'batch_size': BATCH_SIZE,
        'shuffle': True,
    }
    
    train_generator = train_datagen.flow_from_directory(
        directory=str(train_data_path),
        **common_flow_args
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        directory=str(val_data_path),
        **common_flow_args
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        directory=str(test_data_path),
        **common_flow_args
    )
    
    logger.info(f"train: {train_generator.samples}, val: {validation_generator.samples}, test: {test_generator.samples}")
    
    return {
        'train': train_generator,
        'validation': validation_generator,
        'test': test_generator
    }


# =============================================================================
# FUNCIÓN: CREAR CALLBACKS
# =============================================================================

def create_callbacks():
    """Configura EarlyStopping, ModelCheckpoint y ReduceLROnPlateau."""
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_EARLY_STOPPING,
        verbose=1,
        restore_best_weights=True,
        min_delta=0.001,
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=str(MODELS_PATH / 'best_model.keras'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE_LR_REDUCE,
        verbose=1,
        min_lr=MIN_LR,
    )
    
    return [early_stopping, model_checkpoint, reduce_lr]


# =============================================================================
# FUNCIÓN PRINCIPAL: ENTRENAMIENTO
# =============================================================================

def train_model(model, data_generators, callbacks):
    """Lanza el entrenamiento con balanceo de clases por pesos."""
    
    logger.info("Iniciando entrenamiento")
    
    train_gen = data_generators['train']
    val_gen = data_generators['validation']
    
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE
    
    # Balanceo de clases: OASIS está desbalanceado (muchas Non-Demented)
    class_indices = train_gen.class_indices
    class_counts = np.array([np.sum(train_gen.classes == i) for i in range(NUM_CLASSES)])
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_gen.classes
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
        logger.info(f"{class_name}: {class_counts[idx]} muestras, peso={class_weights[idx]:.3f}")

    
    history = model.fit(
        x=train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )
    
    logger.info("Entrenamiento completado")
    return history


# =============================================================================
# FUNCIÓN: EVALUAR EN TEST
# =============================================================================

def evaluate_on_test(model, data_generators):
    """Evalúa el modelo en el conjunto de test."""
    
    test_gen = data_generators['test']
    test_steps = test_gen.samples // BATCH_SIZE
    
    test_loss, test_accuracy = model.evaluate(x=test_gen, steps=test_steps, verbose=1)
    
    logger.info(f"Test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy


# =============================================================================
# BLOQUE PRINCIPAL: EJECUCIÓN DEL PIPELINE
# =============================================================================

if __name__ == "__main__":
    try:
        logger.info("Iniciando pipeline de entrenamiento")
        
        model = build_model(freeze_base=True)
        data_generators = get_data_generators()
        callbacks = create_callbacks()
        history = train_model(model, data_generators, callbacks)
        test_loss, test_accuracy = evaluate_on_test(model, data_generators)
        
        # Cargar el mejor modelo guardado
        try:
            # Cargar formato .keras (nativo Keras 3+)
            best_model = keras.models.load_model(
                str(MODELS_PATH / 'best_model.keras')
            )
            logger.info("Modelo cargado exitosamente (formato .keras)")
        except FileNotFoundError:
            # Fallback a .h5 si existe
            try:
                best_model = keras.models.load_model(
                    str(MODELS_PATH / 'best_model.h5'),
                    safe_mode=False
                )
                logger.info("Modelo cargado desde .h5 (fallback)")
            except Exception as e:
                logger.warning(f"No se pudo cargar el mejor modelo: {e}")
                best_model = None
        except Exception as e:
            logger.warning(f"No se pudo cargar el mejor modelo: {e}")
            best_model = None
        
        summary_path = LOGS_PATH / "training_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Test loss: {test_loss:.4f}\n")
            f.write(f"Test accuracy: {test_accuracy:.4f}\n")
        
        logger.info("Pipeline completado")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {str(e)}", exc_info=True)
        sys.exit(1)
