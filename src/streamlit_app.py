#!/usr/bin/env python3
"""Aplicación Streamlit para clasificación de Alzheimer con Grad-CAM."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import tempfile
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
from preprocessing import load_medical_image, normalize_image_robust, extract_multiple_slices, resize_image

IMG_SIZE = 128
MODELS_DIR = Path("models")
DATA_DIR = Path("data/processed")
DEMO_FEATURE_COLUMNS = ["gender", "age", "mmse", "nwbv", "etiv", "asf"]

CDR_MAPPING = {
    0: "Non-Demented (CDR=0)",
    1: "Very Mild Dementia (CDR=0.5)",
    2: "Demented (CDR≥1)"
}

CASE_LABELS = ["Non-Demented", "Very Mild", "Demented"]

@st.cache_resource
def load_model():
    """Carga el modelo entrenado."""
    model_path = MODELS_DIR / "best_multimodal_model.keras"
    if not model_path.exists():
        model_path = MODELS_DIR / "final_multimodal_model.keras"
    
    return keras.models.load_model(model_path)


@st.cache_resource
def load_demo_scaler():
    """Ajusta el escalado de demografía igual que en entrenamiento."""
    train_csv = DATA_DIR / "train_demographics.csv"
    df = pd.read_csv(train_csv)
    demo = df[DEMO_FEATURE_COLUMNS].copy().fillna(0).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(demo.values)
    return scaler


@st.cache_data
def load_dataset_demographics(split_name):
    """Carga la demografía de un split."""
    csv_path = DATA_DIR / f"{split_name}_demographics.csv"
    return pd.read_csv(csv_path).fillna(0)


def list_demo_cases(label_name, split_name="validation"):
    """Devuelve los casos disponibles de una clase."""
    df = load_dataset_demographics(split_name)
    subset = df[df["label"] == label_name].copy()
    if subset.empty:
        raise ValueError(f"No hay casos para la clase {label_name} en el split {split_name}.")
    subset = subset.sort_values(["subject_id", "image_filename"])
    subset["case_display"] = subset.apply(
        lambda row: f"{row['subject_id']} | {row['image_filename']}", axis=1
    )
    return subset


def get_three_case_rows(label_name, split_name="validation", case_display=None):
    """Obtiene 3 cortes del mismo sujeto alrededor del corte seleccionado."""
    subset = list_demo_cases(label_name, split_name)

    if case_display is None:
        selected_row = subset.iloc[0]
    else:
        selected = subset[subset["case_display"] == case_display]
        if selected.empty:
            raise ValueError(f"No se encontró el caso {case_display} para la clase {label_name}.")
        selected_row = selected.iloc[0]

    subject_rows = subset[subset["subject_id"] == selected_row["subject_id"]].reset_index(drop=True)
    selected_pos = subject_rows[subject_rows["case_display"] == selected_row["case_display"]].index
    center_idx = int(selected_pos[0]) if len(selected_pos) > 0 else 0

    if len(subject_rows) >= 3:
        start_idx = min(max(center_idx - 1, 0), len(subject_rows) - 3)
        rows_3 = subject_rows.iloc[start_idx:start_idx + 3].copy()
        padded_to_three = False
    else:
        idxs = [center_idx]
        while len(idxs) < 3:
            if idxs[0] > 0:
                idxs.insert(0, idxs[0] - 1)
            else:
                idxs.append(idxs[-1])
        rows_3 = subject_rows.iloc[idxs[:3]].copy()
        padded_to_three = True

    return selected_row, rows_3, padded_to_three


def standardize_demographics(raw_values):
    """Aplica el mismo escalado usado en entrenamiento."""
    scaler = load_demo_scaler()
    return scaler.transform([raw_values]).astype(np.float32)[0]


def build_demo_vector(gender_num, age, mmse, nwbv, etiv, asf):
    """Construye el vector demográfico en el orden esperado por el modelo."""
    return np.array([gender_num, age, mmse, nwbv, etiv, asf], dtype=np.float32)


def load_real_demo_case(label_name, split_name="validation", case_display=None):
    """Carga un caso real del dataset procesado."""
    row, rows_3, padded_to_three = get_three_case_rows(label_name, split_name, case_display)
    image_paths = [DATA_DIR / split_name / r["label"] / r["image_filename"] for _, r in rows_3.iterrows()]

    raw_demo = build_demo_vector(
        float(row["gender"]),
        float(row["age"]) if pd.notna(row["age"]) else 0.0,
        float(row["mmse"]) if pd.notna(row["mmse"]) else 0.0,
        float(row["nwbv"]) if pd.notna(row["nwbv"]) else 0.0,
        float(row["etiv"]) if pd.notna(row["etiv"]) else 0.0,
        float(row["asf"]) if pd.notna(row["asf"]) else 0.0,
    )

    demographics_dict = {
        "age": float(row["age"]) if pd.notna(row["age"]) else 0.0,
        "mmse": float(row["mmse"]) if pd.notna(row["mmse"]) else 0.0,
        "nwbv": float(row["nwbv"]) if pd.notna(row["nwbv"]) else 0.0,
        "etiv": float(row["etiv"]) if pd.notna(row["etiv"]) else 0.0,
        "asf": float(row["asf"]) if pd.notna(row["asf"]) else 0.0,
        "gender": "Male" if int(row["gender"]) == 0 else "Female",
        "subject_id": row["subject_id"],
        "label": row["label"],
        "image_filename": row["image_filename"],
        "image_filenames": rows_3["image_filename"].tolist(),
        "padded_to_three": padded_to_three,
        "split": split_name,
    }

    return image_paths, standardize_demographics(raw_demo), raw_demo, demographics_dict


def load_and_process_image(uploaded_file):
    """Carga y procesa una imagen 2D."""
    try:
        image = Image.open(uploaded_file).convert('L')
        image_np = np.array(image)
        
        resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
        if np.max(resized) > np.min(resized):
            normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
        else:
            normalized = np.zeros_like(resized, dtype=np.float32)
            
        return normalized.astype(np.float32)
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def get_file_extension(filename):
    """Devuelve extensión normalizada, soportando .nii.gz."""
    lower_name = filename.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower_name).suffix


def load_and_process_volume_files(uploaded_files):
    """Procesa un volumen 3D subido usando el mismo pipeline de entrenamiento."""
    if not uploaded_files:
        raise ValueError("No se recibió ningún archivo de volumen.")

    nii_files = []
    hdr_files = []
    img_files = []

    for f in uploaded_files:
        ext = get_file_extension(f.name)
        if ext in {".nii", ".nii.gz"}:
            nii_files.append(f)
        elif ext == ".hdr":
            hdr_files.append(f)
        elif ext == ".img":
            img_files.append(f)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for f in uploaded_files:
            file_dest = tmp_path / Path(f.name).name
            file_dest.write_bytes(f.getbuffer())

        volume_path = None
        source_text = None

        if len(nii_files) == 1 and len(uploaded_files) == 1:
            volume_name = Path(nii_files[0].name).name
            volume_path = tmp_path / volume_name
            source_text = volume_name
        elif len(hdr_files) == 1 and len(img_files) == 1 and len(uploaded_files) == 2:
            hdr_name = Path(hdr_files[0].name).name
            img_name = Path(img_files[0].name).name
            if Path(hdr_name).stem != Path(img_name).stem:
                raise ValueError("El archivo .hdr y el .img deben tener el mismo nombre base.")
            volume_path = tmp_path / hdr_name
            source_text = f"{hdr_name} + {img_name}"
        else:
            raise ValueError(
                "Sube exactamente 1 archivo .nii/.nii.gz o un par .hdr + .img con el mismo nombre base."
            )

        volume = load_medical_image(volume_path)
        if volume is None:
            raise ValueError("No se pudo cargar el volumen 3D. Revisa formato y consistencia del archivo.")

        normalized = normalize_image_robust(volume)
        if normalized is None:
            raise ValueError("No se pudo normalizar el volumen 3D.")

        slices = extract_multiple_slices(normalized, num_slices=3, axis='axial')
        if slices is None or len(slices) != 3:
            raise ValueError("No se pudieron extraer 3 cortes axiales del volumen.")

        mri_slices = []
        for slice_2d in slices:
            resized = resize_image(slice_2d, target_size=(IMG_SIZE, IMG_SIZE))
            if resized is None:
                raise ValueError("No se pudo redimensionar uno de los cortes del volumen.")
            resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
            mri_slices.append(resized)

    return mri_slices, source_text


def get_actual_gradcam(model, img_batch, demo_batch, layer_name="conv5_block3_out"):
    """Calcula un mapa Grad-CAM sobre la rama de imagen."""
    resnet_inside = model.get_layer("resnet50")

    resnet_grad_model = tf.keras.models.Model(
        inputs=resnet_inside.inputs,
        outputs=[resnet_inside.get_layer(layer_name).output, resnet_inside.output]
    )

    with tf.GradientTape() as tape:
        if img_batch.shape[-1] != 3:
            img_batch = np.repeat(img_batch, 3, axis=-1)
            
        conv_outputs, resnet_features = resnet_grad_model(img_batch)
        tape.watch(conv_outputs)

        resnet_features = model.get_layer('global_average_pooling2d')(conv_outputs)
        img_features = model.get_layer('img_dense_1')(resnet_features)
        img_features = model.get_layer('dropout')(img_features)
        img_features = model.get_layer('img_dense_2')(img_features)

        demo_features = model.get_layer('demo_dense_1')(demo_batch)
        demo_features = model.get_layer('dropout_1')(demo_features)
        demo_features = model.get_layer('demo_dense_2')(demo_features)

        merged = tf.keras.layers.concatenate([img_features, demo_features])
        predictions = model.get_layer('classification')(model.get_layer('merged_dense_2')(model.get_layer('dropout_2')(model.get_layer('merged_dense_1')(merged))))
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    output = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(guided_grads, output), axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        
    return heatmap_resized, class_idx.numpy()


def predict_with_model(model, mri_slice, demographics):
    """Realiza la predicción sobre un corte."""
    img_batch = np.expand_dims(np.expand_dims(mri_slice, axis=-1), axis=0)
    demo_batch = np.array([demographics])
    predictions = model.predict([img_batch, demo_batch], verbose=0)
    
    probabilities = predictions[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class] * 100
    
    return predicted_class, confidence, probabilities


def plot_gradcam_single_slice(model, mri_slice, demographics):
    """Muestra la imagen y su Grad-CAM."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')

    slice_input = np.expand_dims(mri_slice, axis=-1)
    slice_rgb = np.repeat(slice_input, 3, axis=-1)
    slice_batch = np.expand_dims(slice_rgb, axis=0)

    heatmap, class_idx = get_actual_gradcam(model, slice_batch, np.array([demographics]))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_PLASMA)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_rgb = np.stack([mri_slice] * 3, axis=-1)
    img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-5) * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_rgb, 0.75, heatmap_color, 0.25, 0)

    axes[0].imshow(img_rgb)
    axes[0].set_title('Corte Axial Original', fontweight='bold', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Atención del Modelo (Grad-CAM)', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    
    plt.suptitle('Mapa de activación local, no segmentación anatómica', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def apply_light_style():
    """Carga los estilos de la interfaz desde el archivo CSS."""
    css_path = Path(__file__).with_name("streamlit_app.css")
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_simple_card(label, value, note=None):
    note_html = f'<div class="simple-card-note">{note}</div>' if note else ''
    st.markdown(
        f"""
        <div class="simple-card">
            <div class="simple-card-label">{label}</div>
            <div class="simple-card-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(
        page_title="Deteccion de Alzheimer",
        layout="wide"
    )

    apply_light_style()

    st.title("Deteccion de Alzheimer")
    st.markdown("Aplicacion de apoyo para mostrar la salida del modelo y su interpretacion visual.")
    st.error(
        "Uso exclusivo academico y experimental. No usar para diagnostico ni soporte a decisiones clinicas."
    )

    st.sidebar.markdown("## Opciones")
    st.sidebar.markdown("Prueba con un caso del dataset o sube datos propios.")
    source_mode = st.sidebar.radio(
        "Fuente de la imagen",
        ["Caso real del dataset", "Subir volumen 3D"],
        index=0
    )
    real_label = None
    
    if source_mode == "Caso real del dataset":
        case_label = st.sidebar.selectbox("Caso de demostración", CASE_LABELS, index=1)
        available_cases = list_demo_cases(case_label)
        case_display = st.sidebar.selectbox(
            "Paciente",
            available_cases["case_display"].tolist(),
            index=0
        )
        st.markdown("### Caso del dataset")
        image_paths, demographics, raw_demo, case_info = load_real_demo_case(case_label, case_display=case_display)

        demographics_dict = {
            'age': case_info['age'],
            'mmse': case_info['mmse'],
            'nwbv': case_info['nwbv'],
            'etiv': case_info['etiv'],
            'asf': case_info['asf'],
            'gender': case_info['gender'],
            'subject_id': case_info['subject_id'],
            'label': case_info['label'],
        }
        real_label = case_info['label']

        col_case_1, col_case_2, col_case_3 = st.columns(3)
        with col_case_1:
            st.metric("Subject", case_info['subject_id'])
        with col_case_2:
            st.metric("Split", case_info['split'])
        with col_case_3:
            st.metric("Label", case_info['label'])

        st.markdown("#### Cortes del dataset utilizados (3-slice)")
        cols_img = st.columns(3)
        mri_slices = []
        for idx, (image_path, col) in enumerate(zip(image_paths, cols_img)):
            with col:
                st.image(
                    str(image_path),
                    caption=f"Corte {idx+1}: {image_path.name}",
                    use_container_width=True
                )
            mri_slices.append(load_and_process_image(image_path))

        if case_info["padded_to_three"]:
            st.warning("Este sujeto tiene menos de 3 cortes disponibles en este split. Se repite el corte más cercano para completar el promedio.")

        st.caption("Caso real del dataset en modo multi-slice (3 cortes). La demografía se normaliza con el mismo criterio usado en entrenamiento.")
    else:
        st.markdown("### Entrada del usuario")
        with st.expander("Datos demograficos", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                age_unknown = st.checkbox(
                    "Edad desconocida",
                    value=False,
                    help="Activa esta opción si no se conoce la edad. Se utilizará 0 internamente."
                )
                age = st.number_input(
                    "Age (years)",
                    min_value=0,
                    max_value=120,
                    value=70,
                    step=1,
                    disabled=age_unknown,
                    help="Usa 0 cuando la edad sea desconocida para mantener consistencia con el preprocesado."
                )
                if age_unknown:
                    age = 0

            with col2:
                mmse = st.number_input("MMSE Score", min_value=0, max_value=30, value=25, step=1,
                                       help="Mini-Mental State Exam (0-30, lower = worse)")

            with col3:
                nwbv = st.number_input("nWBV (Brain Volume)", min_value=0.0, max_value=1.5, value=0.78, step=0.01, format="%.3f",
                                       help="Normalized Whole Brain Volume (lower = atrophy)")

            col4, col5 = st.columns(2)
            with col4:
                etiv = st.number_input("eTIV (Estimated Total Intracranial Volume)", min_value=0, max_value=4000, value=1450, step=10,
                                       help="Estimated Total Intracranial Volume in mm³")

            with col5:
                asf = st.number_input("ASF (Atlas Scaling Factor)", min_value=0.0, max_value=3.0, value=1.21, step=0.01, format="%.3f",
                                      help="Atlas Scaling Factor (unitless metric)")

            col6, col7 = st.columns(2)
            with col6:
                gender = st.selectbox("Gender", ["Male", "Female"])
            with col7:
                gender_num = 0 if gender == "Male" else 1
                st.markdown(f"<br>**Selected Gender**: {gender}", unsafe_allow_html=True)

        if mmse < 0 or mmse > 30:
            st.warning("MMSE fuera de rango permitido (0-30).")
            st.stop()

        if nwbv <= 0:
            st.warning("nWBV debe ser mayor que 0 para una inferencia valida.")
            st.stop()

        if etiv <= 0:
            st.warning("eTIV debe ser mayor que 0 para una inferencia valida.")
            st.stop()

        if age_unknown is False and age <= 0:
            st.warning("Si la edad es desconocida, activa la opcion correspondiente; en caso contrario debe ser > 0.")
            st.stop()

        raw_demo = build_demo_vector(gender_num, age, mmse, nwbv, etiv, asf)
        demographics = standardize_demographics(raw_demo)
        demographics_dict = {
            'age': age,
            'mmse': mmse,
            'nwbv': nwbv,
            'etiv': etiv,
            'asf': asf,
            'gender': gender,
        }

        if source_mode == "Subir volumen 3D":
            st.info(
                "Modo recomendado: usa el mismo pipeline del entrenamiento "
                "(normalización robusta + extracción automática de 3 cortes axiales)."
            )
            uploaded_volume_files = st.file_uploader(
                "Sube 1 archivo .nii/.nii.gz o el par .hdr + .img (mismo nombre base)",
                accept_multiple_files=True,
                key="volume_uploader",
            )

            if uploaded_volume_files is None or len(uploaded_volume_files) == 0:
                st.info("Sube un volumen para poder realizar la predicción y generar la explicación Grad-CAM.")
                st.stop()

            try:
                mri_slices, volume_source = load_and_process_volume_files(uploaded_volume_files)
            except ValueError as e:
                st.warning(str(e))
                st.stop()

            st.markdown("#### Cortes extraídos automáticamente del volumen")
            cols_img = st.columns(3)
            for idx, (mri_slice, col) in enumerate(zip(mri_slices, cols_img)):
                with col:
                    st.image(mri_slice, caption=f"Corte axial {idx+1}", use_container_width=True, clamp=True)

            st.caption(
                f"Volumen procesado: {volume_source}. "
                "Se aplicó el mismo preprocesado del entrenamiento antes de inferir."
            )
    st.divider()
    calculate_prediction = st.button("Calcular predicción", type="primary", use_container_width=True)
    if not calculate_prediction:
        st.info("Revisa los datos y pulsa el botón para ejecutar la predicción.")
        st.stop()

    model = load_model()

    st.markdown("#### Predicción multi-slice (promedio de 3 cortes)")

    # Hacer predicción para cada corte
    probabilities_all = []

    for mri_slice in mri_slices:
        predicted_class_i, confidence_i, all_probs_i = predict_with_model(model, mri_slice, demographics)
        probabilities_all.append(all_probs_i)

    # Promediar probabilidades
    avg_probabilities = np.mean(probabilities_all, axis=0)
    predicted_class = np.argmax(avg_probabilities)
    confidence = avg_probabilities[predicted_class] * 100
    all_probs = avg_probabilities

    col_pred, col_conf = st.columns(2)
    
    with col_pred:
        render_simple_card(
            "Diagnosis",
            CDR_MAPPING[predicted_class],
            "Requires Follow-up" if predicted_class > 0 else "Normal Aging"
        )
    
    with col_conf:
        render_simple_card(
            "Confidence",
            f"{confidence:.1f}%",
            "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
        )

    if real_label is not None:
        result_text = "Correcta" if CDR_MAPPING[predicted_class].startswith(real_label) else "Incorrecta"
        st.info(f"Etiqueta real: {real_label} | Resultado: {result_text}")

    st.markdown("#### Distribucion de probabilidad")
    probs_dict = {CDR_MAPPING[i]: all_probs[i]*100 for i in range(3)}
    
    fig_gauge, ax = plt.subplots(figsize=(10, 2))
    colors = ['#4ECDC4', '#FFE66D', '#FF6B6B']
    bars = ax.barh(list(probs_dict.keys()), list(probs_dict.values()), color=colors, edgecolor='black', linewidth=2)
    ax.set_xlabel('Probability (%)', fontweight='bold')
    ax.set_xlim([0, 100])
    
    for i, (name, val) in enumerate(probs_dict.items()):
        ax.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold')
    
    st.pyplot(fig_gauge, use_container_width=True)

    st.markdown("### Grad-CAM")
    st.info(
        "Lectura rápida: las zonas cálidas indican dónde el modelo encontró más señal útil para decidir; "
        "no son una segmentación exacta ni una lesión dibujada con precisión anatómica."
    )
    st.caption(
        "Úsalo como apoyo interpretativo técnico: las regiones resaltadas explican la predicción del modelo, "
        "pero no validan biomarcadores ni constituyen evidencia diagnóstica."
    )

    st.warning(
        "Limitación importante: este Grad-CAM depende de un modelo 2D entrenado sobre cortes redimensionados. "
        "No reconstruye un volumen 3D ni garantiza precisión anatómica fina en cada caso."
    )
    
    st.markdown("#### Grad-CAM por corte")
    cols_gradcam = st.columns(3)
    for idx, slice_to_explain in enumerate(mri_slices):
        with cols_gradcam[idx]:
            fig_gradcam_i = plot_gradcam_single_slice(model, slice_to_explain, demographics)
            st.pyplot(fig_gradcam_i, use_container_width=True)

    st.markdown("### Interpretacion")
    
    if predicted_class == 0:
        interpretation = """
        **Patrón compatible con ausencia de demencia**
        - Rendimiento cognitivo conservado dentro del contexto del caso analizado
        - Volumen cerebral relativamente preservado en las variables disponibles
        - La decisión del modelo se alinea con un perfil de menor deterioro
        """
        color = "🟢"
    elif predicted_class == 1:
        interpretation = """
        **Patrón compatible con deterioro cognitivo muy leve**
        - Se observan indicios sutiles de afectación cognitiva en el perfil clínico
        - La señal de imagen y las variables demográficas sugieren un estado intermedio
        - La predicción debe interpretarse como apoyo cuantitativo, no como diagnóstico definitivo
        """
        color = "🟡"
    else:
        interpretation = """
        **Patrón compatible con demencia establecida**
        - El perfil del caso refleja mayor probabilidad de deterioro clínicamente relevante
        - La contribución conjunta de imagen y demografía apunta a una clase de mayor severidad
        - Esta salida expresa la estimación del modelo dentro de las limitaciones del estudio
        """
        color = "🔴"
    
    st.markdown(f"{color} Interpretación del modelo")
    st.markdown(interpretation)

    st.markdown("### Informacion del modelo")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        render_simple_card("Model Type", "ResNet50 + Demographics")
    
    with col_info2:
        render_simple_card("Input", "3 cortes axiales + demografía")
    
    st.markdown("""
    **Arquitectura**:
    - MRI image analysis (ResNet50 backbone)
    - Demographic features (Gender, Age, MMSE, nWBV, eTIV, ASF)
    
    **Validacion**: 5-Fold Cross-Validation sobre 939 imagenes de entrenamiento
    
    **Nota**: Esta aplicacion tiene un fin academico y experimental.
    La salida del modelo no debe utilizarse como diagnostico ni como apoyo directo a decisiones medicas.
    """)


if __name__ == "__main__":
    main()
