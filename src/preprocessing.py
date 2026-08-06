# Utilidades básicas de preprocesamiento para el dataset MRI.
# El flujo intenta ser simple y reproducible.

import os
import sys
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import warnings
import csv
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

TARGET_SIZE = (128, 128)
TEST_SPLIT = 0.1
VAL_SPLIT = 0.2
RANDOM_SEED = 42

CLASSES = ['Non-Demented', 'Very Mild', 'Demented']


# Normaliza la orientación de la imagen al sistema RAS.

def normalize_orientation(nifti_image):
    """Reorienta la imagen a un sistema de referencia más consistente."""
    if nifti_image is None:
        return None
    
    try:
        # Obtener orientación actual
        current_orientation = nib.aff2axcodes(nifti_image.affine)
        target_orientation = 'RAS'
        
        # Si ya está en RAS, retornar datos sin transformar (eficiencia)
        if ''.join(current_orientation) == target_orientation:
            return np.asarray(nifti_image.get_fdata(), dtype=np.float32)
        
        # Calcular transformación de orientación
        current_ornt = nib.io_orientation(nifti_image.affine)
        target_ornt = nib.orientations.axcodes2ornt(target_orientation)
        
        # Calcular el mapeo de ejes (permutación + posibles reflexiones)
        ornt_xfm = nib.orientations.ornt_transform(current_ornt, target_ornt)
        
        # Aplicar reorientación
        reoriented_img = nifti_image.as_reoriented(ornt_xfm)
        
        # Extraer datos de la imagen reorientada
        data = np.asarray(reoriented_img.get_fdata(), dtype=np.float32)
        
        return data
        
    except Exception as e:
        logger.warning(f"Error normalizando orientación {nifti_image.get_filename()}: {str(e)}")
        # Fallback: retornar sin transformar
        try:
            return np.asarray(nifti_image.get_fdata(), dtype=np.float32)
        except:
            return None


# Carga una imagen médica desde NIfTI o ANALYZE.

def load_medical_image(file_path):
    """Carga una imagen médica desde un archivo NIfTI o ANALYZE."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"Archivo no encontrado: {file_path}")
        return None
    
    try:
        # Cargar con nibabel (soporta NIFTI y ANALYZE)
        # Para archivos ANALYZE problemáticos, usar check=False para evitar validación estricta
        try:
            img = nib.load(file_path)
        except Exception as e:
            # Si falla con check estricto, intentar con check=False para ANALYZE
            if file_path.suffix in ['.img', '.hdr']:
                try:
                    img = nib.load(file_path, keep_file_open=False)
                except:
                    logger.warning(f"No se pudo cargar {file_path.name} (ANALYZE): {str(e)}")
                    return None
            else:
                logger.warning(f"Error cargando {file_path.name}: {str(e)}")
                return None
        
        # NUEVO: Normalizar orientación a RAS
        data = normalize_orientation(img)
        
        if data is None:
            return None
        
        # Asegurar que es 3D
        while data.ndim > 3 and data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        
        if data.ndim != 3:
            logger.warning(f"Imagen {file_path.name} no es 3D: {data.shape}")
            return None
        
        return data
        
    except Exception as e:
        logger.warning(f"Error cargando {file_path.name}: {str(e)}")
        return None


# Normaliza la imagen con percentiles para reducir el efecto de valores extremos.

def normalize_image_robust(image_array):
    """Normaliza la imagen usando percentiles para reducir el efecto de valores extremos."""
    if image_array is None:
        return None
    
    # Obtener percentiles
    p2 = np.percentile(image_array, 2)
    p98 = np.percentile(image_array, 98)
    
    # Clipear outliers
    clipped = np.clip(image_array, p2, p98)
    
    # Normalizar a [0, 1]
    if p98 != p2:
        normalized = (clipped - p2) / (p98 - p2)
    else:
        normalized = np.zeros_like(clipped)
    
    return normalized.astype(np.float32)


# Extrae un corte central para trabajar con una imagen 2D.

def extract_central_slice(volume_3d, axis='axial'):
    """Extrae un corte central del volumen para trabajar con una imagen 2D."""
    if volume_3d is None or volume_3d.ndim != 3:
        return None
    
    if axis == 'axial':
        # Corte axial: vista desde arriba
        center_idx = volume_3d.shape[2] // 2
        return volume_3d[:, :, center_idx]
    
    elif axis == 'coronal':
        # Corte coronal: vista frontal
        center_idx = volume_3d.shape[1] // 2
        return volume_3d[:, center_idx, :]
    
    elif axis == 'sagital':
        # Corte sagital: vista lateral
        center_idx = volume_3d.shape[0] // 2
        return volume_3d[center_idx, :, :]
    
    else:
        logger.warning(f"Eje desconocido: {axis}")
        return None


# Extrae varios cortes cercanos al centro para dar algo de contexto espacial.

def extract_multiple_slices(volume_3d, num_slices=3, axis='axial'):
    """Extrae varios cortes cercanos al centro para dar algo de contexto espacial."""
    if volume_3d is None or volume_3d.ndim != 3:
        return None
    
    slices = []
    
    if axis == 'axial':
        center_idx = volume_3d.shape[2] // 2
        depth = volume_3d.shape[2]
    elif axis == 'coronal':
        center_idx = volume_3d.shape[1] // 2
        depth = volume_3d.shape[1]
    elif axis == 'sagital':
        center_idx = volume_3d.shape[0] // 2
        depth = volume_3d.shape[0]
    else:
        logger.warning(f"Eje desconocido: {axis}")
        return None
    
    # Calcular offsets
    offset = num_slices // 2
    indices = []
    for i in range(-offset, offset + 1):
        idx = center_idx + i
        idx = max(0, min(idx, depth - 1))  # Clipear a rango válido
        indices.append(idx)
    
    # Extraer slices
    for idx in indices:
        if axis == 'axial':
            slices.append(volume_3d[:, :, idx])
        elif axis == 'coronal':
            slices.append(volume_3d[:, idx, :])
        elif axis == 'sagital':
            slices.append(volume_3d[idx, :, :])
    
    return slices


# Redimensiona la imagen al tamaño que usa el modelo.

def resize_image(image_2d, target_size=TARGET_SIZE):
    """Redimensiona la imagen al tamaño que usa el modelo."""
    if image_2d is None:
        return None
    
    try:
        resized = cv2.resize(image_2d, (target_size[1], target_size[0]), 
                            interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32)
    except Exception as e:
        logger.warning(f"Error redimensionando: {str(e)}")
        return None


# Lee los metadatos del sujeto para obtener la etiqueta de clase.

def parse_oasis_metadata(subject_dir):
    """Lee el archivo de metadatos del sujeto y devuelve la etiqueta de clase."""
    try:
        # Construir nombre del archivo .txt
        dir_name = subject_dir.name  # ej: "OAS1_0001_MR1"
        txt_file = subject_dir / f"{dir_name}.txt"
        
        if not txt_file.exists():
            logger.warning(f"Archivo .txt no encontrado: {txt_file}")
            return None
        
        # Leer archivo y buscar línea CDR
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('CDR:'):
                    # Extraer valor después de "CDR:"
                    # Formato: "CDR:          0.5" o "CDR:          " (vacío)
                    cdr_value = line.split(':')[1].strip()
                    
                    # Mapear a etiqueta
                    if not cdr_value or cdr_value == '0' or cdr_value == '0.0':
                        return 'Non-Demented'
                    elif cdr_value == '0.5':
                        return 'Very Mild'
                    elif cdr_value == '1' or cdr_value == '1.0' or cdr_value == '2' or cdr_value == '2.0':
                            return 'Demented'  # Combina Mild (CDR 1) + Moderate (CDR 2)
        logger.warning(f"No encontré línea CDR en {txt_file}")
        return None
        
    except Exception as e:
        logger.warning(f"Error parseando metadatos de {subject_dir}: {str(e)}")
        return None


# Extrae datos clínicos básicos del archivo del sujeto.

def extract_demographic_features(subject_dir):
    """Lee los datos clínicos básicos del archivo del sujeto y los devuelve como diccionario."""
    try:
        dir_name = subject_dir.name
        txt_file = subject_dir / f"{dir_name}.txt"
        
        if not txt_file.exists():
            return None
        
        features = {}
        
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extraer género (M/F)
                if 'M/F:' in line:
                    value = line.split(':')[1].strip()
                    features['gender'] = 1 if value.upper() == 'M' else 0
                
                # Extraer edad
                elif 'Age:' in line:
                    value = line.split(':')[1].strip()
                    try:
                        features['age'] = float(value)
                    except:
                        pass
                
                # Extraer MMSE
                elif 'MMSE:' in line:
                    value = line.split(':')[1].strip()
                    try:
                        features['mmse'] = float(value)
                    except:
                        pass
                
                # Extraer nWBV (Normalized Whole Brain Volume)
                elif 'nWBV:' in line:
                    value = line.split(':')[1].strip()
                    try:
                        features['nwbv'] = float(value)
                    except:
                        pass
                
                # Extraer eTIV (Estimated Total Intracranial Volume)
                elif 'eTIV:' in line:
                    value = line.split(':')[1].strip()
                    try:
                        features['etiv'] = float(value)
                    except:
                        pass
                
                # Extraer ASF (Atlas Scaling Factor)
                elif 'ASF:' in line:
                    value = line.split(':')[1].strip()
                    try:
                        features['asf'] = float(value)
                    except:
                        pass
        
        return features if len(features) > 0 else None
        
    except Exception as e:
        logger.warning(f"Error extrayendo demográficos de {subject_dir}: {str(e)}")
        return None


# Busca los sujetos disponibles y les asigna la etiqueta desde los metadatos.

def find_subjects_with_labels(raw_data_path):
    """Busca los sujetos disponibles y les asigna la etiqueta según los metadatos."""
    logger.info(f"Buscando sujetos en: {raw_data_path}")
    
    subjects = []
    found_any = False
    
    # PASO 1: Encontrar todos los directorios de sujetos OAS1_XXXX_MRY
    for subject_dir in sorted(raw_data_path.glob('**/OAS1_*_MR*')):
        if not subject_dir.is_dir():
            continue
        
        found_any = True
        subject_id = subject_dir.name  # ej: "OAS1_0001_MR1"
        
        # PASO 2: Parsear CDR real del archivo .txt
        label = parse_oasis_metadata(subject_dir)
        if label is None:
            continue
        
        # PASO 3: Buscar imágenes dentro de este directorio de sujeto
        # Preferencia: usar imagen promediada procesada (mejor calidad)
        # Fallback: usar RAW si no existe procesada
        
        image_found = False
        
        # Preferencia 1: Imagen promediada en espacio nativo (SUBJ_111)
        for img_path in subject_dir.glob('PROCESSED/MPRAGE/SUBJ_111/*_sbj_111.img'):
            subjects.append((subject_id, img_path, label, subject_dir))
            image_found = True
            break
        
        if image_found:
            continue
        
        # Preferencia 2: Imagen atlas-registrada brain-masked (T88_111)
        for img_path in subject_dir.glob('PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img'):
            subjects.append((subject_id, img_path, label, subject_dir))
            image_found = True
            break
        
        if image_found:
            continue
        
        # Preferencia 3: Imágenes RAW si no hay procesadas
        for img_path in subject_dir.glob('RAW/*_mpr-*.img'):
            subjects.append((subject_id, img_path, label, subject_dir))
            image_found = True
            break
        
        if not image_found:
            logger.warning(f"No se encontraron imágenes en {subject_dir}")
    
    if not found_any:
        logger.warning(f"No se encontraron directorios OAS1 en {raw_data_path}")
    
    logger.info(f"Se encontraron {len(subjects)} sujetos")
    
    if len(subjects) > 0:
        class_counts = {}
        for _, _, label, _ in subjects:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        for class_name in CLASSES:
            count = class_counts.get(class_name, 0)
            logger.info(f"{class_name}: {count}")
    
    return subjects


# Procesa una imagen y la guarda como PNG en la carpeta de la clase.

def process_and_save_image(image_path, subject_id, label, output_dir):
    """Procesa una imagen y la guarda como archivo PNG en la carpeta de la clase."""
    try:
        # 1. Cargar imagen 3D
        volume = load_medical_image(image_path)
        if volume is None:
            return None
        
        # 2. Normalizar
        normalized = normalize_image_robust(volume)
        if normalized is None:
            return None
        
        # 3. Extraer corte central (axial)
        slice_2d = extract_central_slice(normalized, axis='axial')
        if slice_2d is None:
            return None
        
        # 4. Redimensionar
        resized = resize_image(slice_2d, TARGET_SIZE)
        if resized is None:
            return None
        
        # 5. Convertir a uint8 [0, 255]
        image_uint8 = (resized * 255).astype(np.uint8)
        
        # 6. Crear directorio de clase si no existe
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # 7. Guardar como PNG
        # Usar solo el stem de la imagen como nombre (ya incluye subject_id)
        filename = f"{image_path.stem}.png"
        output_path = class_dir / filename
        cv2.imwrite(str(output_path), image_uint8)
        
        return filename  # Retornar nombre del archivo (no ruta completa)
        
    except Exception as e:
        logger.warning(f"Error procesando {subject_id}: {str(e)}")
        return None


# Procesa varios cortes de la imagen y los guarda en disco.

def process_and_save_multiple_slices(image_path, subject_id, label, output_dir, num_slices=3):
    """Procesa varios cortes de la imagen y los guarda en disco."""
    try:
        # 1. Cargar imagen 3D
        volume = load_medical_image(image_path)
        if volume is None:
            return None
        
        # 2. Normalizar
        normalized = normalize_image_robust(volume)
        if normalized is None:
            return None
        
        # 3. Extraer múltiples slices (axial)
        slices_2d = extract_multiple_slices(normalized, num_slices=num_slices, axis='axial')
        if slices_2d is None or len(slices_2d) == 0:
            return None
        
        # Crear directorio de clase si no existe
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        
        filenames = []
        base_stem = image_path.stem
        
        # 4. Procesar cada slice
        for idx, slice_2d in enumerate(slices_2d):
            # Redimensionar
            resized = resize_image(slice_2d, TARGET_SIZE)
            if resized is None:
                continue
            
            # Convertir a uint8
            image_uint8 = (resized * 255).astype(np.uint8)
            
            # Guardar con sufijo
            filename = f"{base_stem}_s{idx}.png"
            output_path = class_dir / filename
            cv2.imwrite(str(output_path), image_uint8)
            filenames.append(filename)
        
        return filenames if filenames else None
        
    except Exception as e:
        logger.warning(f"Error procesando múltiples slices de {subject_id}: {str(e)}")
        return None


# Divide los sujetos en train, validation y test manteniendo la distribución de clases.

def split_subjects(subjects, test_size=TEST_SPLIT, val_size=VAL_SPLIT, 
                  random_state=RANDOM_SEED):
    """Divide los sujetos en train, validation y test manteniendo la distribución de clases."""
    logger.info("Dividiendo el dataset en train, validation y test")
    
    # Extraer etiquetas para split estratificado
    labels = [item[2] for item in subjects]
    
    # Primera división: train vs test
    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Segunda división: train vs validation
    train_labels = [item[2] for item in train_subjects]
    train_split, val_split = train_test_split(
        train_subjects,
        test_size=val_size,
        stratify=train_labels,
        random_state=random_state
    )
    
    logger.info(f"train: {len(train_split)}")
    logger.info(f"validation: {len(val_split)}")
    logger.info(f"test: {len(test_subjects)}")
    
    return {
        'train': train_split,
        'validation': val_split,
        'test': test_subjects
    }


# Ejecuta el pipeline completo para preparar los datos desde los archivos raw.

def run_preprocessing_pipeline():
    """Ejecuta el pipeline completo para preparar los datos desde los archivos raw."""
    
    logger.info("Iniciando el preprocesamiento")
    
    # =========================================================================
    # PASO 1: VALIDAR DIRECTORIOS
    # =========================================================================
    if not RAW_DATA_PATH.exists():
        logger.error(f"Directorio raw no encontrado: {RAW_DATA_PATH}")
        logger.error("Asegúrese de que data/raw/ contiene imágenes NIFTI/ANALYZE")
        sys.exit(1)
    
    # Crear directorio processed
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Carpeta de entrada: {RAW_DATA_PATH}")
    logger.info(f"Carpeta de salida: {PROCESSED_DATA_PATH}")
    
    # =========================================================================
    # PASO 2: BUSCAR SUJETOS
    # =========================================================================
    subjects = find_subjects_with_labels(RAW_DATA_PATH)
    
    if not subjects:
        logger.error("No se encontraron imágenes en data/raw/")
        logger.error("Estructura esperada: data/raw/oasis_cross-sectional_disc*/disc*/*.nii")
        sys.exit(1)
    
    # =========================================================================
    # PASO 3: DIVIDIR DATASET
    # =========================================================================
    split_data = split_subjects(subjects)
    
    # =========================================================================
    # PASO 4: PROCESAR Y GUARDAR
    # =========================================================================
    logger.info("Procesando y guardando imágenes")
    
    stats = {
        'train': {'total': 0, 'success': 0, 'by_class': {}},
        'validation': {'total': 0, 'success': 0, 'by_class': {}},
        'test': {'total': 0, 'success': 0, 'by_class': {}}
    }
    
    for split_name, subjects_to_process in split_data.items():
        logger.info(f"Procesando conjunto: {split_name.upper()}")
        
        output_dir = PROCESSED_DATA_PATH / split_name
        
        # CSV para guardar datos demográficos
        demographics_csv = PROCESSED_DATA_PATH / f"{split_name}_demographics.csv"
        
        with open(demographics_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['subject_id', 'image_filename', 'label', 'gender', 'age', 'mmse', 'nwbv', 'etiv', 'asf']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for subject_id, image_path, label, subject_dir in subjects_to_process:
                stats[split_name]['total'] += 1
                
                # Actualizar conteo por clase
                if label not in stats[split_name]['by_class']:
                    stats[split_name]['by_class'][label] = {'total': 0, 'success': 0}
                stats[split_name]['by_class'][label]['total'] += 1
                
                # Extraer datos demográficos
                demographics = extract_demographic_features(subject_dir) or {}
                
                # Procesar y guardar múltiples slices
                image_filenames = process_and_save_multiple_slices(image_path, subject_id, label, output_dir, num_slices=3)
                
                if image_filenames:
                    stats[split_name]['success'] += len(image_filenames)
                    stats[split_name]['by_class'][label]['success'] += len(image_filenames)
                    
                    # Guardar fila en CSV para cada slice
                    for image_filename in image_filenames:
                        row = {
                            'subject_id': subject_id,
                            'image_filename': image_filename,
                            'label': label,
                            'gender': demographics.get('gender', ''),
                            'age': demographics.get('age', ''),
                            'mmse': demographics.get('mmse', ''),
                            'nwbv': demographics.get('nwbv', ''),
                            'etiv': demographics.get('etiv', ''),
                            'asf': demographics.get('asf', '')
                        }
                        writer.writerow(row)
        
        logger.info(f"{split_name}: {stats[split_name]['success']} imágenes procesadas")
    
    # =========================================================================
    # PASO 5: GENERAR REPORTE
    # =========================================================================
    logger.info("Resumen del preprocesamiento")
    
    for split_name in ['train', 'validation', 'test']:
        logger.info(f"{split_name}: {stats[split_name]['success']} imágenes")
        
        for class_name, class_stats in stats[split_name]['by_class'].items():
            logger.info(f"  {class_name}: {class_stats['success']}")
    
    logger.info(f"Salida en: {PROCESSED_DATA_PATH}")
    
    return stats


# Ejecución directa del script.

if __name__ == "__main__":
    """
    Ejecuta el pipeline de preprocesamiento directamente.
    """
    try:
        stats = run_preprocessing_pipeline()
        
    except Exception as e:
        logger.error(f"Error durante preprocesamiento: {str(e)}", exc_info=True)
        sys.exit(1)
