"""Compara el rendimiento de los modelos single-slice y multi-slice."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
import logging
import warnings
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/single_vs_multi_comparison.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

CLASSES = ['Non-Demented', 'Very Mild', 'Demented']
DATA_DIR = Path('data/processed')
IMG_SIZE = 128
RESULTS_DIR = Path('results_kfold_analysis')
RESULTS_DIR.mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# Se usa CPU para mantener las ejecuciones comparables.
tf.config.set_visible_devices([], 'GPU')

logger.info("Iniciando comparación single-slice frente a multi-slice")


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
            logger.debug(f"Error cargando {img_path}: {e}")
    return np.array(images)


def load_all_training_data():
    """Carga los datos de entrenamiento."""
    logger.info("Cargando datos de entrenamiento")
    train_csv = DATA_DIR / 'train_demographics.csv'
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
    logger.info(f"Datos de entrenamiento: {images.shape}")
    
    return images, X_demo, y, df


def load_test_data():
    """Carga los datos de test."""
    logger.info("Cargando datos de test")
    test_csv = DATA_DIR / 'test_demographics.csv'
    df = pd.read_csv(test_csv)
    
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
    logger.info(f"Datos de test: {images.shape}")
    
    return images, X_demo, y, df


def build_multimodal_model():
    """Construye el modelo multimodal."""
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
    return model


def run_kfold_analysis(
    X_train_img,
    X_train_demo,
    y_train,
    train_df,
    X_test_img,
    X_test_demo,
    y_test,
    model_type='multi',
):
    """Ejecuta la validación cruzada para un tipo de entrada."""
    logger.info(f"Analizando modelo {model_type}-slice")
    
    if HAS_STRATIFIED_GROUP_KFOLD:
        kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        logger.info("Usando StratifiedGroupKFold por sujeto")
    else:
        logger.warning("StratifiedGroupKFold unavailable. Falling back to StratifiedKFold over unique subjects.")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    subject_df = train_df[['subject_id', 'label']].drop_duplicates('subject_id').copy()
    subject_df['y'] = subject_df['label'].apply(CLASSES.index)
    subject_idx = np.arange(len(subject_df))

    if HAS_STRATIFIED_GROUP_KFOLD:
        split_iter = kfold.split(subject_idx, subject_df['y'].values, groups=subject_df['subject_id'].values)
    else:
        split_iter = kfold.split(subject_idx, subject_df['y'].values)
    accuracies = []
    
    fold_num = 0
    for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(split_iter):
        fold_num += 1
        logger.info(f"  Fold {fold_num}/5...")

        train_subjects = set(subject_df.iloc[train_subject_idx]['subject_id'])
        val_subjects = set(subject_df.iloc[val_subject_idx]['subject_id'])
        overlap_subjects = train_subjects.intersection(val_subjects)
        if overlap_subjects:
            raise RuntimeError(f"Subject leakage detected in fold {fold_num}: {len(overlap_subjects)}")

        train_idx = train_df.index[train_df['subject_id'].isin(train_subjects)].to_numpy()
        
        X_fold_train_img = X_train_img[train_idx]
        X_fold_train_demo = X_train_demo[train_idx]
        y_fold_train = y_train[train_idx]
        
        model = build_multimodal_model()
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
        
        model.fit(
            [X_fold_train_img, X_fold_train_demo], y_fold_train,
            batch_size=32, epochs=100, validation_split=0.2,
            callbacks=[early_stop, reduce_lr], verbose=0
        )
        
        y_pred_probs = model.predict([X_test_img, X_test_demo], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        logger.info(f"    Fold {fold_num}: {accuracy*100:.2f}%")
        
        del model
        keras.backend.clear_session()
    
    return np.array(accuracies)


def main():
    logger.info("Cargando datos para la comparación")
    X_train_img, X_train_demo, y_train, train_df = load_all_training_data()
    X_test_img, X_test_demo, y_test, test_df = load_test_data()
    
    logger.info("Ejecutando análisis k-fold")
    
    # El corte central se identifica por el sufijo del nombre de archivo.
    logger.info("Extrayendo cortes centrales")
    train_single_mask = train_df['image_filename'].str.contains('_s1.png', na=False).values
    test_single_mask = test_df['image_filename'].str.contains('_s1.png', na=False).values

    X_train_single = X_train_img[train_single_mask]
    y_train_single = y_train[train_single_mask]
    X_train_demo_single = X_train_demo[train_single_mask]
    train_df_single = train_df.loc[train_single_mask].reset_index(drop=True)

    X_test_single = X_test_img[test_single_mask]
    y_test_single = y_test[test_single_mask]
    X_test_demo_single = X_test_demo[test_single_mask]
    
    logger.info(f"Single-slice dataset: train={X_train_single.shape[0]}, test={X_test_single.shape[0]}")
    logger.info(f"Multi-slice dataset: train={X_train_img.shape[0]}, test={X_test_img.shape[0]}")
    
    logger.info("Análisis k-fold single-slice")
    single_accs = run_kfold_analysis(
        X_train_single, X_train_demo_single, y_train_single, train_df_single,
        X_test_single, X_test_demo_single, y_test_single,
        model_type='single'
    )
    
    logger.info("Análisis k-fold multi-slice")
    multi_accs = run_kfold_analysis(
        X_train_img, X_train_demo, y_train, train_df,
        X_test_img, X_test_demo, y_test,
        model_type='multi'
    )
    
    logger.info("Comparación estadística")
    
    single_mean = np.mean(single_accs)
    single_std = np.std(single_accs)
    multi_mean = np.mean(multi_accs)
    multi_std = np.std(multi_accs)
    
    logger.info(f"\nSingle-Slice Results:")
    logger.info(f"  Mean Accuracy: {single_mean*100:.2f}%")
    logger.info(f"  Std Dev:       {single_std*100:.2f}%")
    logger.info(f"  Accuracies:    {[f'{a*100:.2f}%' for a in single_accs]}")
    
    logger.info(f"\nMulti-Slice Results:")
    logger.info(f"  Mean Accuracy: {multi_mean*100:.2f}%")
    logger.info(f"  Std Dev:       {multi_std*100:.2f}%")
    logger.info(f"  Accuracies:    {[f'{a*100:.2f}%' for a in multi_accs]}")
    
    t_stat, p_value = stats.ttest_rel(multi_accs, single_accs)
    logger.info(f"\nPaired t-test (multi vs single):")
    logger.info(f"  t-statistic:  {t_stat:.4f}")
    logger.info(f"  p-value:      {p_value:.6f}")
    
    if p_value < 0.05:
        logger.info("  Resultado: diferencia estadísticamente significativa (p < 0.05)")
    else:
        logger.info("  Resultado: diferencia no significativa (p >= 0.05)")
    
    improvement = (multi_mean - single_mean) * 100
    logger.info(f"\nAbsolute Improvement: +{improvement:.2f} percentage points")
    logger.info(f"Relative Improvement: +{(improvement / (single_mean*100)):.1f}%")
    
    logger.info("Generando gráfico de comparación")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(single_accs))
    width = 0.35
    
    ax.bar(x - width/2, single_accs*100, width, label='Single-Slice', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, multi_accs*100, width, label='Multi-Slice', color='coral', alpha=0.8)
    
    ax.axhline(y=single_mean*100, color='steelblue', linestyle='--', linewidth=2, 
               label=f'Single-Slice Mean: {single_mean*100:.2f}%', alpha=0.7)
    ax.axhline(y=multi_mean*100, color='coral', linestyle='--', linewidth=2, 
               label=f'Multi-Slice Mean: {multi_mean*100:.2f}%', alpha=0.7)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Single-Slice vs Multi-Slice Comparison\n(Improvement: +{improvement:.2f}pp, p={p_value:.4f})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(single_accs))])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([60, 95])
    
    plot_path = RESULTS_DIR / 'single_vs_multi_comparison.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico guardado en: {plot_path}")
    
    report_path = RESULTS_DIR / 'single_vs_multi_comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SINGLE-SLICE vs MULTI-SLICE K-FOLD COMPARISON\n")
        f.write("INFORMACIÓN DEL DATASET\n")
        f.write(f"Muestras de entrenamiento single-slice: {X_train_single.shape[0]}\n")
        f.write(f"Muestras de test single-slice:           {X_test_single.shape[0]}\n")
        f.write(f"Muestras de entrenamiento multi-slice:   {X_train_img.shape[0]}\n")
        f.write(f"Muestras de test multi-slice:             {X_test_img.shape[0]}\n\n")
        
        f.write("SINGLE-SLICE RESULTS\n")
        f.write(f"Fold Accuracies:\n")
        for i, acc in enumerate(single_accs):
            f.write(f"  Fold {i+1}: {acc*100:.2f}%\n")
        f.write(f"\nMean Accuracy:   {single_mean*100:.2f}%\n")
        f.write(f"Std Deviation:   {single_std*100:.2f}%\n")
        f.write(f"Min:             {np.min(single_accs)*100:.2f}%\n")
        f.write(f"Max:             {np.max(single_accs)*100:.2f}%\n\n")
        
        f.write("MULTI-SLICE RESULTS\n")
        f.write(f"Fold Accuracies:\n")
        for i, acc in enumerate(multi_accs):
            f.write(f"  Fold {i+1}: {acc*100:.2f}%\n")
        f.write(f"\nMean Accuracy:   {multi_mean*100:.2f}%\n")
        f.write(f"Std Deviation:   {multi_std*100:.2f}%\n")
        f.write(f"Min:             {np.min(multi_accs)*100:.2f}%\n")
        f.write(f"Max:             {np.max(multi_accs)*100:.2f}%\n\n")
        
        f.write("STATISTICAL COMPARISON\n")
        f.write(f"Absolute Improvement: +{improvement:.2f} percentage points\n")
        f.write(f"Relative Improvement: +{(improvement / (single_mean*100)):.1f}%\n\n")
        f.write(f"Paired t-test (Multi vs Single):\n")
        f.write(f"  t-statistic: {t_stat:.4f}\n")
        f.write(f"  p-value:     {p_value:.6f}\n")
        f.write(f"  Significance: {'✓ YES (p < 0.05)' if p_value < 0.05 else '✗ NO (p >= 0.05)'}\n\n")
        
        f.write("INTERPRETACIÓN\n")
        f.write(f"The multi-slice approach provides a {improvement:.2f}pp improvement over single-slice.\n")
        if p_value < 0.05:
            f.write(f"This improvement is statistically significant (p={p_value:.6f} < 0.05).\n")
        else:
            f.write(f"This improvement is not statistically significant (p={p_value:.6f} >= 0.05).\n")
        f.write(f"\nConclusion: Multi-slice MRI processing {'significantly' if p_value < 0.05 else 'marginally'} improves\n")
        f.write(f"Alzheimer's detection accuracy compared to single-slice processing.\n")
    
    logger.info(f"Informe guardado en: {report_path}")
    logger.info("Análisis de comparación completado")
    
    return single_accs, multi_accs


if __name__ == '__main__':
    try:
        single_accs, multi_accs = main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
