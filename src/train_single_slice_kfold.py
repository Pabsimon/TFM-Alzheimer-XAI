"""Entrenamiento de referencia usando solo el corte central."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
import logging
import warnings

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Se usa CPU para mantener las ejecuciones comparables.
tf.config.set_visible_devices([], 'GPU')

CLASSES = ['Non-Demented', 'Very Mild', 'Demented']
DATA_DIR = Path('data/processed')
IMG_SIZE = 128
N_SPLITS = 5


def load_batch_images(image_paths, img_size=IMG_SIZE):
    """Carga las imágenes PNG en memoria."""
    images = []
    for img_path in image_paths:
        try:
            img = keras.preprocessing.image.load_img(
                img_path, target_size=(img_size, img_size), color_mode='grayscale'
            )
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            logger.warning(f"Error cargando {img_path}: {e}")
    return np.array(images)


def get_central_slice_only(image_filenames):
    """Devuelve solo los nombres correspondientes al corte central."""
    central_only = []
    for fname in image_filenames:
        if '_s1.png' in fname:
            central_only.append(fname)
    return central_only


def load_single_slice_training_data():
    """Carga los datos de entrenamiento usando el corte central."""
    train_csv = DATA_DIR / 'train_demographics.csv'
    df = pd.read_csv(train_csv)
    
    df = df[df['image_filename'].str.contains('_s1.png')].reset_index(drop=True)
    
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
    
    logger.info(f"Entrenamiento single-slice: {len(images)} imágenes")
    logger.info(f"Distribución de clases: {np.bincount(y)}")
    
    return images, X_demo, y, df


def load_single_slice_test_data():
    """Carga los datos de test usando el corte central."""
    test_csv = DATA_DIR / 'test_demographics.csv'
    df = pd.read_csv(test_csv)
    
    df = df[df['image_filename'].str.contains('_s1.png')].reset_index(drop=True)
    
    image_paths = []
    for idx, row in df.iterrows():
        img_path = DATA_DIR / 'test' / row['label'] / row['image_filename']
        image_paths.append(str(img_path))
    
    images = load_batch_images(image_paths)
    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    X_demo = df[demo_features].values.astype(float)
    X_demo = np.nan_to_num(X_demo, nan=0.0)
    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)
    y = np.array([CLASSES.index(label) for label in df['label']])
    
    logger.info(f"Test single-slice: {len(images)} imágenes")
    
    return images, X_demo, y


def build_multimodal_model():
    """Construye el modelo multimodal para la comparación."""
    img_input = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='image_input')
    img_expanded = keras.layers.Concatenate(axis=-1)([img_input, img_input, img_input])
    
    base_model = keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    x = base_model(img_expanded, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    
    demo_input = keras.layers.Input(shape=(6,), name='demographics_input')
    y = keras.layers.Dense(32, activation='relu')(demo_input)
    y = keras.layers.Dropout(0.2)(y)
    y = keras.layers.Dense(16, activation='relu')(y)
    
    merged = keras.layers.Concatenate()([x, y])
    merged = keras.layers.Dense(128, activation='relu')(merged)
    merged = keras.layers.Dropout(0.5)(merged)
    merged = keras.layers.Dense(64, activation='relu')(merged)
    output = keras.layers.Dense(len(CLASSES), activation='softmax')(merged)
    
    model = keras.Model(inputs=[img_input, demo_input], outputs=output)
    return model, base_model


def load_multi_slice_fold_accuracies(report_path=Path('results_kfold.txt')):
    """Lee las precisiones de los folds del resultado multi-slice."""
    if not report_path.exists():
        return None

    pattern = re.compile(r"^Fold\s+\d+\s*:\s*([0-9]+\.?[0-9]*)%")
    accuracies = []

    with open(report_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            match = pattern.match(line)
            if match:
                accuracies.append(float(match.group(1)) / 100.0)

    if len(accuracies) == N_SPLITS:
        return np.array(accuracies)

    logger.warning(
        "No se pudieron leer %d precisiones de folds de %s (encontradas %d).",
        N_SPLITS,
        report_path,
        len(accuracies),
    )
    return None


def main():
    logger.info("Comparación single-slice frente a multi-slice")
    
    logger.info("Cargando datos single-slice")
    X_train_img, X_train_demo, y_train, train_df = load_single_slice_training_data()
    X_test_img, X_test_demo, y_test = load_single_slice_test_data()
    
    if HAS_STRATIFIED_GROUP_KFOLD:
        kfold = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        logger.info("Usando StratifiedGroupKFold por sujeto")
    else:
        logger.warning(
            "StratifiedGroupKFold not available. Falling back to StratifiedKFold over subjects."
        )
        kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    subject_df = train_df[['subject_id', 'label']].drop_duplicates('subject_id').copy()
    subject_df['y'] = subject_df['label'].apply(CLASSES.index)
    subject_indices = np.arange(len(subject_df))
    
    single_slice_accuracies = []
    
    logger.info("Iniciando entrenamiento single-slice")
    
    if HAS_STRATIFIED_GROUP_KFOLD:
        split_iter = kfold.split(subject_indices, subject_df['y'].values, groups=subject_df['subject_id'].values)
    else:
        split_iter = kfold.split(subject_indices, subject_df['y'].values)

    for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(split_iter):
        logger.info(f"Fold {fold_idx + 1}/{N_SPLITS}")

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
        
        X_fold_train_img = X_train_img[train_idx]
        X_fold_train_demo = X_train_demo[train_idx]
        y_fold_train = y_train[train_idx]
        
        model, base_model = build_multimodal_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0
        )
        
        history = model.fit(
            [X_fold_train_img, X_fold_train_demo], y_fold_train,
            batch_size=32, epochs=100, validation_split=0.2,
            callbacks=[early_stop, reduce_lr], verbose=0
        )
        
        y_pred_probs = model.predict([X_test_img, X_test_demo], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        single_slice_accuracies.append(accuracy)
        
        logger.info(f"Accuracy single-slice: {accuracy*100:.2f}%")
    
    mean_single = np.mean(single_slice_accuracies)
    std_single = np.std(single_slice_accuracies)
    
    multi_slice_accuracies = load_multi_slice_fold_accuracies()
    if multi_slice_accuracies is None:
        logger.warning(
            "Multi-slice fold accuracies not available in results_kfold.txt. "
            "Run train_kfold_multimodal.py first to enable statistical comparison."
        )
        mean_multi = np.nan
        std_multi = np.nan
        t_stat = np.nan
        p_value = np.nan
    else:
        mean_multi = np.mean(multi_slice_accuracies)
        std_multi = np.std(multi_slice_accuracies)
        t_stat, p_value = stats.ttest_rel(multi_slice_accuracies, single_slice_accuracies)
    
    logger.info("Resultados de la comparación")
    
    logger.info("\nSingle-Slice K-Fold Accuracies:")
    for i, acc in enumerate(single_slice_accuracies):
        logger.info(f"  Fold {i+1}: {acc*100:.2f}%")
    logger.info(f"  Mean:     {mean_single*100:.2f}%")
    logger.info(f"  Std Dev:  {std_single*100:.2f}%")
    
    if multi_slice_accuracies is not None:
        logger.info("Multi-slice, accuracies de los folds:")
        for i, acc in enumerate(multi_slice_accuracies):
            logger.info(f"  Fold {i+1}: {acc*100:.2f}%")
        logger.info(f"  Mean:     {mean_multi*100:.2f}%")
        logger.info(f"  Std Dev:  {std_multi*100:.2f}%")

        logger.info("Comparación estadística:")
        logger.info(f"  Single-Slice Mean:   {mean_single*100:.2f}%")
        logger.info(f"  Multi-Slice Mean:    {mean_multi*100:.2f}%")
        logger.info(f"  Difference:          +{(mean_multi - mean_single)*100:.2f}pp")
        logger.info(f"  t-statistic:         {t_stat:.4f}")
        logger.info(f"  p-value:             {p_value:.4f}")

        if p_value < 0.05:
            logger.info("  Resultado: diferencia estadísticamente significativa (p < 0.05)")
        else:
            logger.info("  Resultado: diferencia no significativa (p >= 0.05)")
    else:
        logger.info("\nStatistical comparison skipped (missing multi-slice fold report).")
    
    report_path = Path('results_single_vs_multi_comparison.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SINGLE-SLICE vs MULTI-SLICE COMPARISON\n")
        f.write("SINGLE-SLICE (Central slice only - _s1.png)\n")
        f.write(f"Muestras de entrenamiento: {len(X_train_img)} imágenes\n")
        f.write(f"Muestras de test:           {len(X_test_img)} imágenes\n")
        f.write(f"\nFold Accuracies:\n")
        for i, acc in enumerate(single_slice_accuracies):
            f.write(f"  Fold {i+1}: {acc*100:.2f}%\n")
        f.write(f"\nMean Accuracy:   {mean_single*100:.2f}%\n")
        f.write(f"Std Deviation:   {std_single*100:.2f}%\n")
        f.write(f"CI (±1σ):        {(mean_single-std_single)*100:.2f}% - {(mean_single+std_single)*100:.2f}%\n")
        
        if multi_slice_accuracies is not None:
            f.write("MULTI-SLICE (3 slices per subject - _s0, _s1, _s2)\n")
            f.write("Fuente: results_kfold.txt generado por train_kfold_multimodal.py\n")
            f.write(f"\nFold Accuracies:\n")
            for i, acc in enumerate(multi_slice_accuracies):
                f.write(f"  Fold {i+1}: {acc*100:.2f}%\n")
            f.write(f"\nMean Accuracy:   {mean_multi*100:.2f}%\n")
            f.write(f"Std Deviation:   {std_multi*100:.2f}%\n")
            f.write(f"CI (±1σ):        {(mean_multi-std_multi)*100:.2f}% - {(mean_multi+std_multi)*100:.2f}%\n")

            f.write("STATISTICAL COMPARISON\n")
            f.write(f"Single-Slice Mean:   {mean_single*100:.2f}%\n")
            f.write(f"Multi-Slice Mean:    {mean_multi*100:.2f}%\n")
            f.write(f"Absolute Difference: {abs(mean_multi - mean_single)*100:.2f}pp\n")
            f.write(f"Relative Improvement: {((mean_multi - mean_single) / mean_single * 100):.2f}%\n")
            f.write(f"\nPaired t-test:\n")
            f.write(f"  t-statistic: {t_stat:.4f}\n")
            f.write(f"  p-value:     {p_value:.4f}\n")

            if p_value < 0.05:
                f.write("  Significance: YES (p < 0.05) - Multi-slice significantly better\n")
            else:
                f.write("  Significance: NO (p >= 0.05) - Not statistically different\n")

            f.write("CONCLUSION\n")
            f.write(f"Multi-slice approach provides +{(mean_multi - mean_single)*100:.2f}pp improvement\n")
            f.write("over single-slice baseline under the same cross-validation setup.\n")
        else:
            f.write("STATISTICAL COMPARISON\n")
            f.write("Skipped: multi-slice fold results not found in results_kfold.txt\n")
            f.write("Run train_kfold_multimodal.py first, then rerun this script.\n")
    
    logger.info(f"Informe guardado en: {report_path}")
    logger.info("Comparación completada")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
