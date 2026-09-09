# TFM-Alzheimer-XAI

Herramienta software multimodal para la clasificación experimental de estados cognitivos asociados al Alzheimer e interpretación mediante IA explicable.

## Descripción del Proyecto

Este repositorio contiene el código, datos y documentación asociados a un Trabajo de Fin de Master (TFM) que implementa un modelo de deep learning multimodal para clasificar tres estados cognitivos relacionados con el Alzheimer:

- **Non-Demented** (CDR=0): Sin demencia
- **Very Mild** (CDR=0.5): Deterioro cognitivo muy leve  
- **Demented** (CDR≥1): Demencia

El modelo fusiona dos modalidades:
1. **Rama de imagen**: ResNet50 (pretrained en ImageNet) procesando 3 cortes sagitales de MRI
2. **Rama tabular**: Red densa con variables demográficas y clínicas (edad, género, MMSE, nWBV, eTIV, ASF)

**Resultado principal**: 81.67% ± 2.22% accuracy en validación cruzada 5-fold

## Datos

Se utilizó **OASIS-1 Cross-Sectional**:
- **Acceso**: https://www.oasis-brains.org/
- **Muestra**: 436 sujetos (723 Non-Demented, 150 Very Mild, 66 Demented)
- **Formato**: Imágenes NIfTI 3D T1-weighted
- **Preprocesamiento**: Reorientación a RAS, extracción de 3 cortes sagitales, normalización, redimensionado a 128×128

## Instalación

### Requisitos Previos
- **Python**: 3.13.11 (o compatible >= 3.9)
- **Memoria RAM**: Mínimo 16 GB recomendado
- **Almacenamiento**: ~60 GB para datos OASIS-1 crudos + 10 GB procesados

### Pasos

```bash
# 1. Clonar repositorio
git clone https://github.com/Pabsimon/TFM-Alzheimer-XAI.git
cd TFM-Alzheimer-XAI

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar OASIS-1 (IMPORTANTE: No está incluido en el repositorio)
# El dataset debe descargarse manualmente:
# - Acceder a: https://www.oasis-brains.org/
# - Registrarse con email académico
# - Seleccionar "OASIS-1 Cross-Sectional"
# - Descargar todos los discos (disc1 a disc12)
# - Tamaño total: ~50-60 GB
# - Crear estructura: mkdir -p data/raw/oasis_cross-sectional_discX/
# - Mover archivos descargados a data/raw/
```

**Verificar descarga:**
```bash
# Contar volúmenes descargados (debe ser ~436)
find data/raw -name "*.nii.gz" | wc -l
```

## Uso

### Preprocesamiento

```bash
cd src
python preprocessing.py
# Salida: ~1.308 imágenes PNG en data/processed/
#         CSVs de demográficos alineados
```

### Entrenamiento Multimodal (5-Fold)

```bash
python train_kfold_multimodal.py
# Salida: results_kfold.txt (accuracy por fold)
#         models/best_multimodal_model.keras
```

Resultado esperado: **81.67% ± 2.22%**

### Evaluación Detallada

```bash
python evaluation_kfold_detailed_v2.py
# Salida: Matrices de confusión, ROC-AUC por clase, métricas detalladas
```

### Ablación (Comparar modalidades)

```bash
python ablation_modalities_kfold.py
# Compara: imagen-only vs tabular-only vs multimodal
```

### Comparación Single-slice vs Multi-slice

```bash
python compare_single_vs_multi_kfold.py
# Prueba t pareada: mejora significativa con multi-slice (p=0.0465)
```

### Interfaz Interactiva (Streamlit)

```bash
streamlit run streamlit_app.py
# Abre http://localhost:8501 en navegador
# Permite: predicción + Grad-CAM + histogramas
```

## Resultados Principales

| Métrica | Valor |
|---------|-------|
| Accuracy global (5-fold) | 81.67% ± 2.22% |
| Accuracy single-slice | 77.73% ± 0.91% |
| Mejora multi-slice | +3.94 pp (p=0.0465) |
| AUC Non-Demented | 0.983 ± 0.005 |
| AUC Very Mild | 0.709 ± 0.047 |
| AUC Demented | 0.813 ± 0.039 |
| Sensibilidad Demented | 13.33% ⚠️ |

**⚠️ Limitación crítica**: Sensibilidad muy baja en clase Demented (7% de los datos), indicando necesidad de balanceo o técnicas especializadas.

## Estructura del Repositorio

```
TFM-Alzheimer-XAI/
├── src/                              # Scripts principales
│   ├── preprocessing.py              # Preprocesamiento NIfTI → PNG
│   ├── train_kfold_multimodal.py     # Entrenamiento principal
│   ├── evaluation_kfold_detailed_v2.py
│   ├── ablation_modalities_kfold.py
│   ├── compare_single_vs_multi_kfold.py
│   ├── gradcam_final.py              # Interpretabilidad
│   └── streamlit_app.py              # Interfaz web
│
├── data/
│   ├── raw/                          # OASIS-1 descargar de https://www.oasis-brains.org/
│   └── processed/                    # Imágenes + CSVs preprocesados
│       ├── train/, validation/, test/
│       ├── train_demographics.csv
│       ├── validation_demographics.csv
│       └── test_demographics.csv
│
├── models/                           # Modelos entrenados
│   ├── best_multimodal_model.keras
│   └── final_multimodal_model.keras
│
├── results_kfold_analysis/           # Reportes de resultados
├── results_interpretability/         # Visualizaciones Grad-CAM
├── logs/                             # Registros de ejecución
├── notebooks/                        # Análisis exploratorio (Jupyter)
│
├── requirements.txt                  # Dependencias Python (v1.0)
├── README.md                         # Este archivo
└── .git/                             # Control de versiones
```

## Reproducibilidad

La reproducibilidad se garantiza mediante:

1. **Versiones congeladas**: `requirements.txt` con versiones exactas (e.g., tensorflow==2.16.1)
2. **Semilla global**: `random_state=42` en todas las operaciones aleatorias
3. **Partición por sujeto**: StratifiedGroupKFold evita data leakage entre imágenes del mismo sujeto
4. **Verificación automática**: RuntimeError si detecta solapamiento en folds
5. **Git**: Histórico completo; etiqueta `v1.0` en rama `main`

### Validación de Reproducibilidad

Ejecutar nuevamente y verificar que accuracy medio ≈ 81.67% ± 0.5%:

```bash
python train_kfold_multimodal.py
tail results_kfold.txt  # Verifica que línea final coincide con 81.67% ± 2.22%
```

## Configuración Experimental

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| N_SPLITS | 5 | Validación cruzada estratificada |
| Estrategia partición | StratifiedGroupKFold | Evita data leakage por sujeto |
| Arquitectura imagen | ResNet50 (ImageNet) | Transfer learning probado |
| Rama tabular | Dense(64)→Dense(32) | Compresión de 6 variables |
| Fusión | Concatenación + Dense(128) | Simple y eficiente |
| Optimizador | Adam, LR=1e-4 | Convergencia estable |
| Loss | Sparse categorical crossentropy | Clasificación multiclase desbalanceada |
| Dropout | 0.3/0.2/0.5 (imagen/tab/fusión) | Regularización progresiva |
| IMG_SIZE | 128×128 | Balance resolución/memoria |
| Batch size | 32 | Estándar para GPU/CPU |
| Early stopping | patience=15 | Evita overfitting |

## Hardware Requerido

- **Procesador**: AMD Ryzen 5 7600X (12 hilos) o equivalente
- **Memoria RAM**: 32 GB (31.1 GB utilizables)
- **GPU**: Opcional (NVIDIA GeForce + AMD Radeon integrada disponibles)
- **SO**: Windows 11 25H2 (también funciona en Linux con ajustes de rutas)

## Limitaciones Conocidas

1. **Robustez externa**: Accuracy ~70% en datos de otros hospitales (no OASIS-1)
2. **Desbalance de clases**: Clase Demented representa solo 7% → sensibilidad muy baja (13.33%)
3. **No-determinismo**: TensorFlow puede producir variaciones <0.5% entre ejecuciones

## Interpretabilidad

El modelo incluye Grad-CAM para visualizar regiones de decisión:

```bash
python gradcam_final.py  # Genera heatmaps por corte
# Salida: results_interpretability/heatmap_*.png
```

**⚠️ Nota importante**: Grad-CAM es herramienta técnica de investigación, NO es diagnóstico clínico ni validación radiológica.

## Autor y Versión

- **Autor**: Pabsimon
- **Versión**: v1.0 (rama main)
- **Rama de desarrollo**: develop
- **Licencia**: Consultar LICENSE en repositorio

## Cita

Si usas este código, citar:

```bibtex
@thesis{pabsimon2026tfm,
  title={Herramienta software multimodal para la clasificación experimental de estados cognitivos asociados al Alzheimer e interpretación mediante IA explicable},
  author={Pabsimon},
  year={2026},
  school={UNIR},
  url={https://github.com/Pabsimon/TFM-Alzheimer-XAI}
}
```

## Contacto y Soporte

Para preguntas, issues o pull requests: https://github.com/Pabsimon/TFM-Alzheimer-XAI/issues

## Reconocimientos

- Base de datos OASIS-1: Marcus et al. (2007)
- ResNet50: He et al. (2015)
- Grad-CAM: Selvaraju et al. (2016)
- Streamlit: https://streamlit.io/
