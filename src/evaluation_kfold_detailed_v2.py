"""Analiza la validación cruzada y guarda métricas y gráficos por fold."""

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
import warnings
import traceback
from datetime import datetime

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/detailed_analysis.log'),
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

# Se fuerza el uso de CPU para mantener resultados comparables.
tf.config.set_visible_devices([], 'GPU')

logger.info("Iniciando análisis k-fold")


def load_batch_images(image_paths, img_size=IMG_SIZE):
    """Carga las imágenes PNG en memoria."""
    images = []
    failed = 0
    logger.info(f"Cargando {len(image_paths)} imágenes")
    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 100 == 0:
            logger.info(f"Cargadas {idx+1}/{len(image_paths)} imágenes")
        try:
            img = keras.preprocessing.image.load_img(
                img_path, target_size=(img_size, img_size), color_mode='grayscale'
            )
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            failed += 1
            logger.debug(f"Error cargando {img_path}: {e}")
    
    if failed > 0:
        logger.warning(f"No se pudieron cargar {failed}/{len(image_paths)} imágenes")
    
    logger.info(f"Imágenes cargadas: {len(images)}")
    return np.array(images)


def load_all_training_data():
    """Carga los datos de entrenamiento."""
    logger.info("\n>>> Loading TRAINING data...")
    train_csv = DATA_DIR / 'train_demographics.csv'
    
    if not train_csv.exists():
        logger.error(f"Training CSV not found: {train_csv}")
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    
    df = pd.read_csv(train_csv)
    logger.info(f"Loaded {len(df)} training samples from CSV")
    
    image_paths = []
    for idx, row in df.iterrows():
        img_path = DATA_DIR / 'train' / row['label'] / row['image_filename']
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
        image_paths.append(str(img_path))
    
    images = load_batch_images(image_paths)
    
    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    X_demo = df[demo_features].values.astype(float)
    X_demo = np.nan_to_num(X_demo, nan=0.0)
    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)
    
    y = np.array([CLASSES.index(label) for label in df['label']])
    logger.info(f"✓ Training data loaded: images={images.shape}, demo={X_demo.shape}, labels={y.shape}")
    logger.info(f"  Class distribution: {np.bincount(y)}")
    
    return images, X_demo, y, df


def load_test_data():
    """Carga los datos de test."""
    logger.info("\n>>> Loading TEST data...")
    test_csv = DATA_DIR / 'test_demographics.csv'
    
    if not test_csv.exists():
        logger.error(f"Test CSV not found: {test_csv}")
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    df = pd.read_csv(test_csv)
    logger.info(f"Loaded {len(df)} test samples from CSV")
    
    image_paths = []
    for idx, row in df.iterrows():
        img_path = DATA_DIR / 'test' / row['label'] / row['image_filename']
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
        image_paths.append(str(img_path))
    
    images = load_batch_images(image_paths)
    
    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    X_demo = df[demo_features].values.astype(float)
    X_demo = np.nan_to_num(X_demo, nan=0.0)
    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)
    
    y = np.array([CLASSES.index(label) for label in df['label']])
    logger.info(f"✓ Test data loaded: images={images.shape}, demo={X_demo.shape}, labels={y.shape}")
    logger.info(f"  Class distribution: {np.bincount(y)}")
    
    return images, X_demo, y


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
    return model, base_model


def create_confusion_matrix_plot(y_true, y_pred, fold_idx, save_path):
    """Crea y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    threshold = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            text_color = 'white' if cm[i, j] > threshold else 'black'
            ax.text(j, i, cm[i, j], ha="center", va="center", color=text_color, fontsize=12)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix - Fold {fold_idx}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def create_roc_auc_plot(y_true_onehot, y_pred_probs, fold_idx, save_path):
    """Crea y guarda las curvas ROC-AUC."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'orange', 'green']
    roc_aucs = {}
    
    for i, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs[class_name] = roc_auc
        ax.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.3f})', 
                color=colors[i], linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC-AUC Curves - Fold {fold_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_aucs


def main():
    start_time = datetime.now()
    
    try:
        logger.info("\n[STEP 1] Loading data...")
        X_train_img, X_train_demo, y_train, train_df = load_all_training_data()
        X_test_img, X_test_demo, y_test = load_test_data()
        
        y_test_onehot = np.eye(len(CLASSES))[y_test]
        
        logger.info(f"\n[STEP 2] Starting K-Fold analysis (5 folds)...")
        
        if HAS_STRATIFIED_GROUP_KFOLD:
            kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            logger.info("Using StratifiedGroupKFold (split by subject_id)")
        else:
            logger.warning(
                "StratifiedGroupKFold not available. Falling back to StratifiedKFold over subjects."
            )
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        subject_df = train_df[['subject_id', 'label']].drop_duplicates('subject_id').copy()
        subject_df['y'] = subject_df['label'].apply(CLASSES.index)
        subject_indices = np.arange(len(subject_df))
        
        fold_results = []
        all_y_pred = []
        all_y_pred_probs = []
        
        fold_num = 0
        if HAS_STRATIFIED_GROUP_KFOLD:
            split_iter = kfold.split(subject_indices, subject_df['y'].values, groups=subject_df['subject_id'].values)
        else:
            split_iter = kfold.split(subject_indices, subject_df['y'].values)

        for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(split_iter):
            fold_num += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"FOLD {fold_num}/5 - Started at {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"{'='*70}")

            train_subjects = set(subject_df.iloc[train_subject_idx]['subject_id'])
            val_subjects = set(subject_df.iloc[val_subject_idx]['subject_id'])
            overlap_subjects = train_subjects.intersection(val_subjects)
            if overlap_subjects:
                raise RuntimeError(
                    f"Subject leakage detected in fold {fold_num}: {len(overlap_subjects)} overlapping subjects"
                )

            train_idx = train_df.index[train_df['subject_id'].isin(train_subjects)].to_numpy()
            val_idx = train_df.index[train_df['subject_id'].isin(val_subjects)].to_numpy()

            logger.info(
                f"Fold {fold_num}: train subjects={len(train_subjects)}, "
                f"val subjects={len(val_subjects)}, train slices={len(train_idx)}, val slices={len(val_idx)}"
            )
            
            X_fold_train_img = X_train_img[train_idx]
            X_fold_train_demo = X_train_demo[train_idx]
            y_fold_train = y_train[train_idx]
            
            logger.info(f"Building model...")
            model, base_model = build_multimodal_model()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Training model...")
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0
            )
            
            history = model.fit(
                [X_fold_train_img, X_fold_train_demo], y_fold_train,
                batch_size=32, epochs=100, validation_split=0.2,
                callbacks=[early_stop, reduce_lr], verbose=1
            )
            
            logger.info("Generando predicciones sobre test")
            y_pred_probs = model.predict([X_test_img, X_test_demo], verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Métricas del fold.
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"  ✓ Accuracy:  {accuracy*100:.2f}%")
            logger.info(f"  ✓ Precision: {precision:.4f}")
            logger.info(f"  ✓ Recall:    {recall:.4f}")
            logger.info(f"  ✓ F1-Score:  {f1:.4f}")
            
            # Gráficos del fold.
            logger.info("Generando gráficos")
            cm_path = RESULTS_DIR / f'confusion_matrix_fold{fold_idx+1}.png'
            cm = create_confusion_matrix_plot(y_test, y_pred, fold_idx+1, cm_path)
            logger.info("Matriz de confusión guardada")
            
            roc_path = RESULTS_DIR / f'roc_auc_fold{fold_idx+1}.png'
            roc_aucs = create_roc_auc_plot(y_test_onehot, y_pred_probs, fold_idx+1, roc_path)
            logger.info("Gráfico ROC-AUC guardado")
            
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'roc_aucs': roc_aucs,
                'y_pred': y_pred,
                'y_pred_probs': y_pred_probs
            })
            
            all_y_pred.append(y_pred)
            all_y_pred_probs.append(y_pred_probs)
            
            del model, base_model
            logger.info(f"Fold {fold_num} completed ✓")
        
        # Métricas del conjunto de modelos.
        logger.info("Analizando el conjunto de modelos")
        
        ensemble_probs = np.mean(all_y_pred_probs, axis=0)
        ensemble_pred = np.argmax(ensemble_probs, axis=0)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        
        logger.info(f"  ✓ Ensemble Accuracy:  {ensemble_accuracy*100:.2f}%")
        logger.info(f"  ✓ Ensemble Precision: {ensemble_precision:.4f}")
        logger.info(f"  ✓ Ensemble Recall:    {ensemble_recall:.4f}")
        logger.info(f"  ✓ Ensemble F1-Score:  {ensemble_f1:.4f}")
        
        # Matriz de confusión del conjunto.
        ensemble_cm_path = RESULTS_DIR / 'confusion_matrix_ensemble.png'
        ensemble_cm = create_confusion_matrix_plot(y_test, ensemble_pred, 'Ensemble', ensemble_cm_path)
        logger.info(f"  → Ensemble confusion matrix saved")
        
        # Informe resumen.
        logger.info("Generando informe final")
        report_path = RESULTS_DIR / 'detailed_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("K-FOLD DETAILED ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("PER-FOLD METRICS\n")
            f.write("-"*70 + "\n")
            for result in fold_results:
                f.write(f"\nFold {result['fold']}:\n")
                f.write(f"  Accuracy:  {result['accuracy']*100:.2f}%\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall:    {result['recall']:.4f}\n")
                f.write(f"  F1-Score:  {result['f1']:.4f}\n")
                
                f.write(f"  Per-Class ROC-AUC:\n")
                for class_name, auc_val in result['roc_aucs'].items():
                    f.write(f"    {class_name}: {auc_val:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("ENSEMBLE METRICS (Average of 5 Models)\n")
            f.write("="*70 + "\n")
            f.write(f"Accuracy:  {ensemble_accuracy*100:.2f}%\n")
            f.write(f"Precision: {ensemble_precision:.4f}\n")
            f.write(f"Recall:    {ensemble_recall:.4f}\n")
            f.write(f"F1-Score:  {ensemble_f1:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*70 + "\n")
            accs = [r['accuracy'] for r in fold_results]
            f.write(f"Mean Accuracy:     {np.mean(accs)*100:.2f}%\n")
            f.write(f"Std Accuracy:      {np.std(accs)*100:.2f}%\n")
            f.write(f"Min Accuracy:      {np.min(accs)*100:.2f}%\n")
            f.write(f"Max Accuracy:      {np.max(accs)*100:.2f}%\n")
            f.write(f"Ensemble Accuracy: {ensemble_accuracy*100:.2f}% (+{(ensemble_accuracy - np.mean(accs))*100:.2f}pp vs mean)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("CLASSIFICATION REPORTS\n")
            f.write("="*70 + "\n")
            
            f.write("\nEnsemble Classification Report:\n")
            f.write(classification_report(y_test, ensemble_pred, target_names=CLASSES))
        
        logger.info(f"✓ Report saved to: {report_path}")
        logger.info(f"✓ All analysis files saved to: {RESULTS_DIR}")
        
        elapsed = datetime.now() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ COMPLETED - Execution time: {elapsed}")
        logger.info(f"{'='*70}\n")
        
        return fold_results, ensemble_accuracy
    
    except Exception as e:
        logger.error(f"ERROR: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    try:
        results, ens_acc = main()
        logger.info("✅ DETAILED K-FOLD ANALYSIS COMPLETED SUCCESSFULLY")
    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        sys.exit(1)
