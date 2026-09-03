"""Compara modelos usando imagen, datos tabulares y ambas modalidades."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import time
import platform
import ctypes
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
import warnings

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ablation_modalities_kfold.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

CLASSES = ['Non-Demented', 'Very Mild', 'Demented']
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data' / 'processed'
RESULTS_DIR = REPO_ROOT / 'results_kfold_analysis'
RESULTS_DIR.mkdir(exist_ok=True)
(REPO_ROOT / 'logs').mkdir(exist_ok=True)

IMG_SIZE = 128
N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
SEED = 42


def collect_system_info():
    """Recoge información básica del entorno de ejecución."""
    cpu_count_logical = os.cpu_count()
    gpu_devices = tf.config.list_physical_devices('GPU')

    info = {
        'platform_system': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor() or 'unknown',
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'cpu_logical_cores': int(cpu_count_logical) if cpu_count_logical is not None else None,
        'gpu_detected_count': len(gpu_devices),
        'gpu_detected_names': [d.name for d in gpu_devices],
        'gpu_enabled_for_run': False,
    }

    try:
        total_memory_gb = None
        if platform.system().lower().startswith('win'):
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_memory_gb = stat.ullTotalPhys / (1024 ** 3)
        if hasattr(os, 'sysconf'):
            page_size = os.sysconf('SC_PAGE_SIZE')
            phys_pages = os.sysconf('SC_PHYS_PAGES')
            total_memory_gb = (page_size * phys_pages) / (1024 ** 3)
        if total_memory_gb is not None:
            info['ram_total_gb_estimated'] = round(float(total_memory_gb), 2)
    except Exception:
        pass

    return info

# Se usa CPU para mantener las ejecuciones comparables.
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass


def load_batch_images(image_paths, img_size=IMG_SIZE):
    images = []
    for img_path in image_paths:
        try:
            img = keras.preprocessing.image.load_img(
                img_path, target_size=(img_size, img_size), color_mode='grayscale'
            )
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
        except Exception as exc:
            logger.warning(f"Error cargando {img_path}: {exc}")
    return np.array(images)


def load_training_data():
    train_csv = DATA_DIR / 'train_demographics.csv'
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)

    image_paths = [str(DATA_DIR / 'train' / row['label'] / row['image_filename']) for _, row in df.iterrows()]
    X_img = load_batch_images(image_paths)

    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    X_demo = df[demo_features].values.astype(float)
    X_demo = np.nan_to_num(X_demo, nan=0.0)

    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)

    y = np.array([CLASSES.index(label) for label in df['label']])

    return X_img, X_demo, y, df


def load_test_data():
    test_csv = DATA_DIR / 'test_demographics.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df = pd.read_csv(test_csv)

    image_paths = [str(DATA_DIR / 'test' / row['label'] / row['image_filename']) for _, row in df.iterrows()]
    X_img = load_batch_images(image_paths)

    demo_features = ['gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
    X_demo = df[demo_features].values.astype(float)
    X_demo = np.nan_to_num(X_demo, nan=0.0)

    scaler = StandardScaler()
    X_demo = scaler.fit_transform(X_demo)

    y = np.array([CLASSES.index(label) for label in df['label']])

    return X_img, X_demo, y


def build_image_only_model():
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
    output = keras.layers.Dense(len(CLASSES), activation='softmax')(x)

    model = keras.Model(inputs=img_input, outputs=output)
    return model


def build_tabular_only_model():
    demo_input = keras.layers.Input(shape=(6,), name='demographics_input')
    y = keras.layers.Dense(64, activation='relu')(demo_input)
    y = keras.layers.Dropout(0.2)(y)
    y = keras.layers.Dense(32, activation='relu')(y)
    y = keras.layers.Dropout(0.2)(y)
    output = keras.layers.Dense(len(CLASSES), activation='softmax')(y)

    model = keras.Model(inputs=demo_input, outputs=output)
    return model


def build_multimodal_model():
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


def fit_and_eval(mode, model, X_fold_img, X_fold_demo, y_fold, X_test_img, X_test_demo, y_test):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0),
    ]

    train_start = time.perf_counter()

    if mode == 'image':
        history = model.fit(
            X_fold_img,
            y_fold,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )
        train_seconds = time.perf_counter() - train_start
        infer_start = time.perf_counter()
        y_pred_probs = model.predict(X_test_img, verbose=0)
        infer_seconds = time.perf_counter() - infer_start
    elif mode == 'tabular':
        history = model.fit(
            X_fold_demo,
            y_fold,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )
        train_seconds = time.perf_counter() - train_start
        infer_start = time.perf_counter()
        y_pred_probs = model.predict(X_test_demo, verbose=0)
        infer_seconds = time.perf_counter() - infer_start
    else:
        history = model.fit(
            [X_fold_img, X_fold_demo],
            y_fold,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )
        train_seconds = time.perf_counter() - train_start
        infer_start = time.perf_counter()
        y_pred_probs = model.predict([X_test_img, X_test_demo], verbose=0)
        infer_seconds = time.perf_counter() - infer_start

    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)

    timing = {
        'train_seconds': float(train_seconds),
        'inference_seconds': float(infer_seconds),
        'epochs_ran': int(len(history.history.get('loss', []))),
        'test_samples': int(len(y_test)),
    }

    return acc, f1w, report, timing


def save_ablation_plots(summary, output_dir):
    """Guarda gráficos sencillos con los resultados de la ablación."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"No se pudieron generar los gráficos: {exc}")
        return

    mode_order = ['image', 'tabular', 'multimodal']
    labels = ['Solo imagen', 'Solo tabular', 'Multimodal']

    acc_means = [summary['results'][m]['mean_accuracy'] * 100.0 for m in mode_order]
    acc_stds = [summary['results'][m]['std_accuracy'] * 100.0 for m in mode_order]
    train_means = [summary['results'][m]['mean_train_seconds'] / 60.0 for m in mode_order]
    epoch_means = [summary['results'][m]['mean_epochs_ran'] for m in mode_order]
    infer_ms = [summary['results'][m]['mean_inference_ms_per_sample'] for m in mode_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(labels, acc_means, yerr=acc_stds, capsize=5, color=['#4C78A8', '#54A24B', '#E45756'])
    ax1.set_ylabel('Accuracy media (%)')
    ax1.set_title('Ablacion: rendimiento por modalidad')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.25)

    ax2.bar(labels, train_means, color=['#4C78A8', '#54A24B', '#E45756'])
    ax2.set_ylabel('Tiempo medio de entrenamiento (min)')
    ax2.set_title('Costo temporal por modalidad')
    ax2.grid(axis='y', alpha=0.25)
    for idx, value in enumerate(epoch_means):
        ax2.text(idx, train_means[idx], f"{value:.1f} ep", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / 'ablation_accuracy_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, infer_ms, color=['#4C78A8', '#54A24B', '#E45756'])
    ax.set_ylabel('Latencia media de inferencia por muestra (ms)')
    ax.set_title('Costo de inferencia por modalidad')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / 'ablation_inference_latency.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info("Gráficos de ablación guardados")


def run_ablation():
    logger.info("Iniciando estudio de ablación")

    X_train_img, X_train_demo, y_train, train_df = load_training_data()
    X_test_img, X_test_demo, y_test = load_test_data()

    logger.info(
        f"Datos cargados: train={len(X_train_img)}, test={len(X_test_img)}, "
        f"clases={np.bincount(y_train).tolist()}"
    )

    if HAS_STRATIFIED_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        logger.info("Usando StratifiedGroupKFold por sujeto")
    else:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        logger.warning("StratifiedGroupKFold no disponible; se usa StratifiedKFold por sujeto.")

    subject_df = train_df[['subject_id', 'label']].drop_duplicates('subject_id').copy()
    subject_df['y'] = subject_df['label'].apply(CLASSES.index)
    subject_idx = np.arange(len(subject_df))

    if HAS_STRATIFIED_GROUP_KFOLD:
        split_iter = splitter.split(subject_idx, subject_df['y'].values, groups=subject_df['subject_id'].values)
    else:
        split_iter = splitter.split(subject_idx, subject_df['y'].values)

    mode_results = {
        'image': {'acc': [], 'f1w': [], 'reports': [], 'train_seconds': [], 'inference_seconds': [], 'epochs_ran': []},
        'tabular': {'acc': [], 'f1w': [], 'reports': [], 'train_seconds': [], 'inference_seconds': [], 'epochs_ran': []},
        'multimodal': {'acc': [], 'f1w': [], 'reports': [], 'train_seconds': [], 'inference_seconds': [], 'epochs_ran': []},
    }

    for fold_idx, (train_subject_idx, val_subject_idx) in enumerate(split_iter, start=1):
        train_subjects = set(subject_df.iloc[train_subject_idx]['subject_id'])
        val_subjects = set(subject_df.iloc[val_subject_idx]['subject_id'])

        overlap = train_subjects.intersection(val_subjects)
        if overlap:
            raise RuntimeError(f"Subject leakage detected in fold {fold_idx}: {len(overlap)}")

        fold_train_idx = train_df.index[train_df['subject_id'].isin(train_subjects)].to_numpy()

        X_fold_img = X_train_img[fold_train_idx]
        X_fold_demo = X_train_demo[fold_train_idx]
        y_fold = y_train[fold_train_idx]

        logger.info(
            f"Fold {fold_idx}/{N_SPLITS}: train subjects={len(train_subjects)}, "
            f"val subjects={len(val_subjects)}, train slices={len(fold_train_idx)}"
        )

        img_model = build_image_only_model()
        acc, f1w, report, timing = fit_and_eval('image', img_model, X_fold_img, X_fold_demo, y_fold, X_test_img, X_test_demo, y_test)
        mode_results['image']['acc'].append(acc)
        mode_results['image']['f1w'].append(f1w)
        mode_results['image']['reports'].append(report)
        mode_results['image']['train_seconds'].append(timing['train_seconds'])
        mode_results['image']['inference_seconds'].append(timing['inference_seconds'])
        mode_results['image']['epochs_ran'].append(timing['epochs_ran'])
        logger.info(
            f"  Image-only:    acc={acc*100:.2f}% | f1w={f1w:.4f} | "
            f"train={timing['train_seconds']/60.0:.2f} min | "
            f"infer={timing['inference_seconds']*1000.0/max(timing['test_samples'], 1):.2f} ms/muestra"
        )
        del img_model
        keras.backend.clear_session()

        tab_model = build_tabular_only_model()
        acc, f1w, report, timing = fit_and_eval('tabular', tab_model, X_fold_img, X_fold_demo, y_fold, X_test_img, X_test_demo, y_test)
        mode_results['tabular']['acc'].append(acc)
        mode_results['tabular']['f1w'].append(f1w)
        mode_results['tabular']['reports'].append(report)
        mode_results['tabular']['train_seconds'].append(timing['train_seconds'])
        mode_results['tabular']['inference_seconds'].append(timing['inference_seconds'])
        mode_results['tabular']['epochs_ran'].append(timing['epochs_ran'])
        logger.info(
            f"  Tabular-only:  acc={acc*100:.2f}% | f1w={f1w:.4f} | "
            f"train={timing['train_seconds']/60.0:.2f} min | "
            f"infer={timing['inference_seconds']*1000.0/max(timing['test_samples'], 1):.2f} ms/muestra"
        )
        del tab_model
        keras.backend.clear_session()

        mm_model = build_multimodal_model()
        acc, f1w, report, timing = fit_and_eval('multimodal', mm_model, X_fold_img, X_fold_demo, y_fold, X_test_img, X_test_demo, y_test)
        mode_results['multimodal']['acc'].append(acc)
        mode_results['multimodal']['f1w'].append(f1w)
        mode_results['multimodal']['reports'].append(report)
        mode_results['multimodal']['train_seconds'].append(timing['train_seconds'])
        mode_results['multimodal']['inference_seconds'].append(timing['inference_seconds'])
        mode_results['multimodal']['epochs_ran'].append(timing['epochs_ran'])
        logger.info(
            f"  Multimodal:    acc={acc*100:.2f}% | f1w={f1w:.4f} | "
            f"train={timing['train_seconds']/60.0:.2f} min | "
            f"infer={timing['inference_seconds']*1000.0/max(timing['test_samples'], 1):.2f} ms/muestra"
        )
        del mm_model
        keras.backend.clear_session()

    summary = {
        'config': {
            'n_splits': N_SPLITS,
            'seed': SEED,
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'test_samples': int(len(y_test)),
            'split_strategy': 'StratifiedGroupKFold by subject_id' if HAS_STRATIFIED_GROUP_KFOLD else 'StratifiedKFold over unique subjects',
        },
        'system_info': collect_system_info(),
        'results': {}
    }

    fold_rows = []

    for mode, values in mode_results.items():
        accs = np.array(values['acc'])
        f1ws = np.array(values['f1w'])
        train_secs = np.array(values['train_seconds'])
        infer_secs = np.array(values['inference_seconds'])
        epochs_arr = np.array(values['epochs_ran'])

        for fold_id, (acc, f1w, tsec, isec, eps) in enumerate(
            zip(values['acc'], values['f1w'], values['train_seconds'], values['inference_seconds'], values['epochs_ran']),
            start=1,
        ):
            fold_rows.append({
                'mode': mode,
                'fold': fold_id,
                'accuracy': float(acc),
                'f1_weighted': float(f1w),
                'train_seconds': float(tsec),
                'train_minutes': float(tsec / 60.0),
                'inference_seconds': float(isec),
                'inference_ms_per_sample': float((isec * 1000.0) / max(len(y_test), 1)),
                'epochs_ran': int(eps),
            })

        mean_infer_ms_sample = float((np.mean(infer_secs) * 1000.0) / max(len(y_test), 1))
        summary['results'][mode] = {
            'fold_accuracy': values['acc'],
            'fold_f1_weighted': values['f1w'],
            'fold_train_seconds': values['train_seconds'],
            'fold_inference_seconds': values['inference_seconds'],
            'fold_epochs_ran': values['epochs_ran'],
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_f1_weighted': float(np.mean(f1ws)),
            'std_f1_weighted': float(np.std(f1ws)),
            'mean_train_seconds': float(np.mean(train_secs)),
            'std_train_seconds': float(np.std(train_secs)),
            'mean_inference_seconds': float(np.mean(infer_secs)),
            'std_inference_seconds': float(np.std(infer_secs)),
            'mean_inference_ms_per_sample': mean_infer_ms_sample,
            'mean_epochs_ran': float(np.mean(epochs_arr)),
            'std_epochs_ran': float(np.std(epochs_arr)),
        }

    json_path = RESULTS_DIR / 'ablation_modalities_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    fold_csv_path = RESULTS_DIR / 'ablation_modalities_fold_metrics.csv'
    pd.DataFrame(fold_rows).to_csv(fold_csv_path, index=False)

    summary_csv_path = RESULTS_DIR / 'ablation_modalities_overview.csv'
    summary_rows = []
    for mode in ['image', 'tabular', 'multimodal']:
        mode_res = summary['results'][mode]
        summary_rows.append({
            'mode': mode,
            'mean_accuracy_pct': mode_res['mean_accuracy'] * 100.0,
            'std_accuracy_pct': mode_res['std_accuracy'] * 100.0,
            'mean_f1_weighted': mode_res['mean_f1_weighted'],
            'mean_train_minutes': mode_res['mean_train_seconds'] / 60.0,
            'std_train_minutes': mode_res['std_train_seconds'] / 60.0,
            'mean_inference_ms_per_sample': mode_res['mean_inference_ms_per_sample'],
            'mean_epochs_ran': mode_res['mean_epochs_ran'],
        })
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    txt_path = RESULTS_DIR / 'ablation_modalities_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('ABLATION STUDY REPORT: IMAGE vs TABULAR vs MULTIMODAL\n')
        f.write('CONFIGURATION\n')
        for key, value in summary['config'].items():
            f.write(f'{key}: {value}\n')
        f.write('\n')

        f.write('EXECUTION ENVIRONMENT\n')
        for key, value in summary['system_info'].items():
            f.write(f'{key}: {value}\n')
        f.write('\n')

        for mode in ['image', 'tabular', 'multimodal']:
            mode_res = summary['results'][mode]
            f.write(f'{mode.upper()}\n')
            for i, acc in enumerate(mode_res['fold_accuracy'], start=1):
                f.write(f'Fold {i} accuracy: {acc*100:.2f}%\n')
            f.write(f"Mean accuracy: {mode_res['mean_accuracy']*100:.2f}%\n")
            f.write(f"Std accuracy: {mode_res['std_accuracy']*100:.2f}%\n")
            f.write(f"Mean F1-weighted: {mode_res['mean_f1_weighted']:.4f}\n")
            f.write(f"Std F1-weighted: {mode_res['std_f1_weighted']:.4f}\n\n")
            f.write(f"Mean train time: {mode_res['mean_train_seconds']/60.0:.2f} min\n")
            f.write(f"Std train time: {mode_res['std_train_seconds']/60.0:.2f} min\n")
            f.write(f"Mean inference time: {mode_res['mean_inference_seconds']:.4f} s\n")
            f.write(f"Mean inference latency: {mode_res['mean_inference_ms_per_sample']:.2f} ms/sample\n")
            f.write(f"Mean epochs used: {mode_res['mean_epochs_ran']:.2f}\n\n")

    save_ablation_plots(summary, RESULTS_DIR)

    logger.info(f"Resumen JSON guardado en: {json_path}")
    logger.info(f"Informe guardado en: {txt_path}")
    logger.info(f"Métricas de folds guardadas en: {fold_csv_path}")
    logger.info(f"Resumen guardado en: {summary_csv_path}")
    logger.info("Estudio de ablación completado")


if __name__ == '__main__':
    try:
        run_ablation()
    except Exception as exc:
        logger.error(f"Error en el estudio de ablación: {exc}", exc_info=True)
        raise
