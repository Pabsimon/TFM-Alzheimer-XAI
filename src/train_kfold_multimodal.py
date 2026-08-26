"""Entrenamiento multimodal con validación cruzada por sujetos."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import warnings
import shutil

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

CLASSES = ['Non-Demented', 'Very Mild', 'Demented']
DATA_DIR = Path('data/processed')
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
N_SPLITS = 5

# Se fuerza el uso de CPU para mantener resultados comparables.
tf.config.set_visible_devices([], 'GPU')


def load_batch_images(image_paths, img_size=IMG_SIZE):
    """Carga las imágenes PNG en memoria."""
    images = []
    for img_path in image_paths:
        try:
            img = keras.preprocessing.image.load_img(
                img_path, 
                target_size=(img_size, img_size), 
                color_mode='grayscale'
            )
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            logger.warning(f"Error cargando {img_path}: {e}")
            continue
    
    return np.array(images)


def load_all_training_data():
    """Carga los datos de entrenamiento y las variables demográficas."""
    train_csv = DATA_DIR / 'train_demographics.csv'
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    
    df = pd.read_csv(train_csv)
    
    image_paths = []
    for idx, row in df.iterrows():
        img_path = DATA_DIR / 'train' / row['label'] / row['image_filename']
        image_paths.append(str(img_path))
    
    images = load_batch_images(image_paths)
    
    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    X_demo = df[demo_features].values.astype(float)
    
    X_demo = np.nan_to_num(X_demo, nan=0.0)
    
    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)
    
    y = np.array([CLASSES.index(label) for label in df['label']])
    
    logger.info(f"Datos de entrenamiento: {len(images)} imágenes")
    logger.info(f"Distribución de clases: {np.bincount(y)}")
    
    return images, X_demo, y, df


def load_test_data():
    """Carga los datos de test y las variables demográficas."""
    test_csv = DATA_DIR / 'test_demographics.csv'
    
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    df = pd.read_csv(test_csv)
    
    image_paths = []
    for idx, row in df.iterrows():
        img_path = DATA_DIR / 'test' / row['label'] / row['image_filename']
        image_paths.append(str(img_path))
    
    images = load_batch_images(image_paths)
    
    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    X_demo = df[demo_features].values.astype(float)
    
    X_demo = np.nan_to_num(X_demo, nan=0.0)
    
    # Se utiliza la misma transformación para las variables demográficas.
    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)
    
    y = np.array([CLASSES.index(label) for label in df['label']])
    
    logger.info(f"Datos de test: {len(images)} imágenes")
    logger.info(f"Distribución en test: {np.bincount(y)}")
    
    return images, X_demo, y


def build_multimodal_model():
    """Construye el modelo multimodal con ResNet50 y datos demográficos."""
    # Input de imagen (128x128x1)
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='image_input')
    
    # Expandir grayscale a RGB concatenando 3 veces
    img_expanded = layers.Concatenate(axis=-1)([img_input, img_input, img_input])
    
    # ResNet50 pretrained
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congelar base inicialmente
    base_model.trainable = False
    
    # Rama de imagen
    x = base_model(img_expanded, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Input de demográficos
    demo_input = layers.Input(shape=(6,), name='demographics_input')
    
    # Rama de demográficos
    y = layers.Dense(32, activation='relu')(demo_input)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(16, activation='relu')(y)
    
    # Fusionar ramas
    merged = layers.Concatenate()([x, y])
    merged = layers.Dense(128, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(64, activation='relu')(merged)
    
    # Output
    output = layers.Dense(len(CLASSES), activation='softmax')(merged)
    
    model = keras.Model(inputs=[img_input, demo_input], outputs=output)
    
    return model, base_model


def train_model_on_fold(model, base_model, X_train_img, X_train_demo, y_train, fold_idx):
    """Entrena el modelo en un fold."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx + 1}/{N_SPLITS}")
    logger.info(f"{'='*60}")
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=0
    )
    
    # Entrenar
    history = model.fit(
        [X_train_img, X_train_demo],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    logger.info(f"Fold {fold_idx + 1} training completed in {len(history.history['loss'])} epochs")
    
    return model, history


def main():
    """Ejecuta el entrenamiento con validación cruzada."""
    
    logger.info("="*60)
    logger.info("K-FOLD CROSS-VALIDATION TRAINING")
    logger.info(f"K = {N_SPLITS}")
    logger.info("="*60)
    
    X_train_img, X_train_demo, y_train, train_df = load_all_training_data()
    
    X_test_img, X_test_demo, y_test = load_test_data()
    
    # Inicializar K-Fold por sujeto para evitar fuga entre cortes del mismo paciente
    if HAS_STRATIFIED_GROUP_KFOLD:
        kfold = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        logger.info("Using StratifiedGroupKFold (split by subject_id)")
    else:
        logger.warning(
            "StratifiedGroupKFold not available. Falling back to StratifiedKFold over subjects. "
            "Please upgrade scikit-learn for strict group-aware stratification."
        )
        kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    subject_df = train_df[['subject_id', 'label']].drop_duplicates('subject_id').copy()
    subject_df['y'] = subject_df['label'].apply(CLASSES.index)
    subject_indices = np.arange(len(subject_df))
    
    fold_results = []
    fold_accuracies = []
    
    # Iterar sobre K folds con partición por sujeto
    if HAS_STRATIFIED_GROUP_KFOLD:
        split_iter = kfold.split(subject_indices, subject_df['y'].values, groups=subject_df['subject_id'].values)
    else:
        split_iter = kfold.split(subject_indices, subject_df['y'].values)

    for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(split_iter):
        logger.info(f"\nProcesando Fold {fold_idx + 1}/{N_SPLITS}...")

        train_subjects = set(subject_df.iloc[train_subject_idx]['subject_id'])
        val_subjects = set(subject_df.iloc[val_subject_idx]['subject_id'])
        overlap_subjects = train_subjects.intersection(val_subjects)

        if overlap_subjects:
            raise RuntimeError(
                f"Subject leakage detected in fold {fold_idx + 1}: {len(overlap_subjects)} overlapping subjects"
            )

        train_idx = train_df.index[train_df['subject_id'].isin(train_subjects)].to_numpy()
        val_idx = train_df.index[train_df['subject_id'].isin(val_subjects)].to_numpy()

        logger.info(
            f"Fold {fold_idx + 1}: train subjects={len(train_subjects)}, "
            f"val subjects={len(val_subjects)}, train slices={len(train_idx)}, val slices={len(val_idx)}"
        )
        
        # Dividir datos de training en train/validation para este fold
        X_fold_train_img = X_train_img[train_idx]
        X_fold_train_demo = X_train_demo[train_idx]
        y_fold_train = y_train[train_idx]
        
        # Construir nuevo modelo para este fold
        model, base_model = build_multimodal_model()
        
        # Entrenar modelo en este fold
        model, history = train_model_on_fold(
            model, base_model, 
            X_fold_train_img, X_fold_train_demo, y_fold_train,
            fold_idx
        )
        
        # Evaluar en TEST set
        y_pred_probs = model.predict([X_test_img, X_test_demo], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calcular métrica
        test_accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(test_accuracy)
        
        logger.info(f"Fold {fold_idx + 1} TEST ACCURACY: {test_accuracy*100:.2f}%")
        
        # Guardar resultados del fold
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': test_accuracy,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=CLASSES)
        })
    
    # Calcular estadísticas
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS DE K-FOLD CROSS-VALIDATION")
    logger.info("="*60)
    
    for i, acc in enumerate(fold_accuracies):
        logger.info(f"Fold {i+1}: {acc*100:.2f}%")
    
    logger.info(f"\n{'─'*60}")
    logger.info(f"MEDIA:        {mean_accuracy*100:.2f}%")
    logger.info(f"DESV. ESTÁNDAR: {std_accuracy*100:.2f}%")
    logger.info(f"INTERVALO:    {(mean_accuracy - std_accuracy)*100:.2f}% - {(mean_accuracy + std_accuracy)*100:.2f}%")
    logger.info(f"{'─'*60}\n")
    
    # Generar reporte
    report_path = Path('results_kfold.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("K-FOLD CROSS-VALIDATION RESULTS\n")
        f.write(f"K = {N_SPLITS}\n")
        f.write(f"Test Set Size: {len(y_test)} images\n")
        f.write("="*60 + "\n\n")
        
        for i, acc in enumerate(fold_accuracies):
            f.write(f"Fold {i+1}: {acc*100:.2f}%\n")
        
        f.write(f"\n{'-'*60}\n")
        f.write(f"MEDIA:              {mean_accuracy*100:.2f}%\n")
        f.write(f"DESV. ESTANDAR:     {std_accuracy*100:.2f}%\n")
        f.write(f"INTERVALO (±1σ):    {(mean_accuracy - std_accuracy)*100:.2f}% - {(mean_accuracy + std_accuracy)*100:.2f}%\n")
        f.write(f"{'-'*60}\n\n")
        
        for fold_result in fold_results:
            f.write(f"\nFold {fold_result['fold']} - Matriz de Confusión:\n")
            f.write(str(fold_result['confusion_matrix']) + "\n\n")
            f.write(f"Fold {fold_result['fold']} - Classification Report:\n")
            f.write(fold_result['classification_report'] + "\n")
    
    logger.info(f"Reporte guardado en: {report_path}")
    
    return mean_accuracy, std_accuracy, fold_accuracies


if __name__ == '__main__':
    try:
        mean_acc, std_acc, fold_accs = main()
        logger.info("\n✅ K-FOLD TRAINING COMPLETADO")
    except Exception as e:
        logger.error(f"Error during K-Fold training: {e}", exc_info=True)
        raise
