# Sistema de Reconocimiento Facial con Análisis de Emociones

Sistema de visión artificial para el **registro**, **identificación** y **análisis emocional** de personas en tiempo real. Mediante una interfaz gráfica moderna, el software captura rostros, genera *embeddings* faciales (Facenet512) para el reconocimiento y utiliza modelos de aprendizaje profundo para clasificar **7 emociones básicas** con sus respectivos niveles de confianza.

---

## Tabla de Contenidos

1. [Características](#características)
2. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Módulos Principales](#módulos-principales)
6. [Base de Datos](#base-de-datos)
7. [Contribución](#contribución)

---

## Características

| Característica | Detalle |
|---|---|
| 🎭 Reconocimiento facial | Embeddings con **Facenet512** (DeepFace) |
| 😀 Análisis de emociones | 7 emociones: felicidad, tristeza, enojo, sorpresa, neutral, miedo, disgusto |
| 📝 Registro completo | Nombre, apellido, email con validación de duplicados |
| 📸 Captura múltiple | 5 capturas con indicador de calidad (🔴🟡🟢) |
| 🖥 Interfaz moderna | **CustomTkinter** con tema claro/oscuro |
| 📊 Reportes avanzados | Gráficas por emoción y persona, estadísticas generales |
| 📥 Exportación | Historial de detecciones exportable a CSV |
| 🗄 Persistencia | **SQLite** (por defecto) vía SQLAlchemy |
| ⏱ Métricas | Tiempo de detección y niveles de confianza en vivo |

---

## Arquitectura del Proyecto

```
SistemaReconocimientoFacial/
├── main.py                        # Punto de entrada de la aplicación
├── requirements.txt               # Dependencias de Python
├── .gitignore
│
├── models/                        # Pesos de modelos de IA (descarga automática)
│   └── .gitkeep
│
└── src/
    ├── __init__.py
    │
    ├── gui/                       # Capa de interfaz gráfica (CustomTkinter)
    │   ├── __init__.py
    │   ├── app.py                 # Ventana raíz + navegación por sidebar
    │   ├── registration_screen.py # Registro: nombre, apellido, email + captura múltiple
    │   ├── detection_screen.py    # Detección en vivo: overlay + 7 emociones + confianza
    │   └── reports_screen.py      # Reportes: estadísticas, gráficas, exportación CSV
    │
    ├── logic/                     # Procesamiento de imágenes y modelos de IA
    │   ├── __init__.py
    │   ├── face_recognizer.py     # FaceRecognizer: detección + embeddings + calidad
    │   └── emotion_analyzer.py    # EmotionAnalyzer: clasificación de 7 emociones
    │
    └── database/                  # Capa de persistencia (SQLAlchemy ORM)
        ├── __init__.py
        └── db_manager.py          # DatabaseManager: CRUD + reportes + CSV export
```

---

## Instalación

### Requisitos previos

- Python **3.9 – 3.12** (recomendado 3.10)
- `pip` ≥ 23
- Webcam funcional (para detección en vivo)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Lejayk/SistemaReconocimientoFacial.git
cd SistemaReconocimientoFacial

# 2. Crear y activar un entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** En la primera ejecución, DeepFace descargará automáticamente los pesos del modelo Facenet512 (~90 MB). Asegúrate de tener conexión a internet.

---

## Uso

```bash
# Iniciar la aplicación
python main.py
```

La aplicación abrirá una ventana con tres pantallas accesibles desde la barra lateral:

### 📝 Pantalla de Registro

| Función | Descripción |
|---|---|
| Formulario | Campos para nombre, apellido (obligatorios) y email (opcional, único) |
| Cámara en vivo | Vista previa con detección de rostro en tiempo real |
| Captura múltiple | Hasta 5 capturas con indicador de calidad visual |
| Validación | Rechaza imágenes borrosas o rostros muy pequeños |
| Duplicados | Verifica que no exista un registro con el mismo email |

### 🔍 Pantalla de Detección

| Función | Descripción |
|---|---|
| Video en vivo | Stream con overlay: nombre, emoción y confianza (%) |
| Panel lateral | Persona identificada, emoción dominante, barra de confianza |
| 7 Emociones | Distribución en tiempo real con barras de porcentaje |
| Tiempo | Medición del tiempo de detección en milisegundos |
| Registro | Cada detección se guarda automáticamente en la BD |

### 📊 Pantalla de Reportes

| Pestaña | Descripción |
|---|---|
| 📋 Eventos | Historial paginado de detecciones con filtros |
| 📈 Estadísticas | Tarjetas: total detecciones, personas, top emoción/persona |
| 😀 Por Emoción | Gráfico de barras con conteo por cada emoción |
| 👤 Por Persona | Filtro por persona con gráfico de pie (distribución emocional) |
| 📥 Exportar | Botón para exportar historial completo a CSV |

---

## Módulos Principales

### `FaceRecognizer` (`src/logic/face_recognizer.py`)

```python
from src.logic.face_recognizer import FaceRecognizer

recognizer = FaceRecognizer(model_name="Facenet512", distance_metric="cosine")

# Evaluar calidad de un rostro
quality = recognizer.assess_face_quality(face_img)
# → {"is_valid": True, "score": 85.0, "sharpness": 120.5, ...}

# Registrar una persona con datos completos
recognizer.register_face(
    name="Ana", face_img=frame, db_manager=db,
    apellido="García", email="ana@email.com"
)

# Identificar un rostro
result = recognizer.identify(face_img=face_crop, db_manager=db)
# → {"name": "Ana", "apellido": "García", "distance": 0.21, "person_id": 1}
```

| Método | Descripción |
|---|---|
| `detect_faces(frame)` | Devuelve lista de bounding boxes detectados |
| `get_embedding(face_img)` | Genera el vector de embedding (512-D) |
| `assess_face_quality(face_img)` | Evalúa tamaño y nitidez del rostro |
| `register_face(...)` | Extrae embedding y lo persiste en la BD |
| `register_multiple_faces(...)` | Registra con la mejor captura de múltiples |
| `identify(face_img, db_manager)` | Compara con todos los registros |

---

### `EmotionAnalyzer` (`src/logic/emotion_analyzer.py`)

```python
from src.logic.emotion_analyzer import EmotionAnalyzer

analyzer = EmotionAnalyzer()
result = analyzer.analyze(face_img)
# → {
#     "dominant_emotion": "happy",
#     "dominant_emotion_es": "Felicidad",
#     "scores": {"happy": 0.92, "neutral": 0.05, ...},
#     "scores_es": {"Felicidad": 0.92, "Neutral": 0.05, ...}
#   }
```

Emociones clasificadas: `angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## Base de Datos

### Tablas (SQLAlchemy ORM)

#### `persons`
| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK | Identificador único |
| `name` | VARCHAR(255) | Nombre de la persona |
| `apellido` | VARCHAR(255) | Apellido de la persona |
| `email` | VARCHAR(255) UNIQUE | Correo electrónico (opcional) |
| `embedding_json` | TEXT | Vector de embedding serializado en JSON |
| `registered_at` | DATETIME | Fecha y hora de registro |

#### `detection_logs`
| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK | Identificador único |
| `person_id` | INTEGER FK | Referencia a `persons.id` (nullable) |
| `person_name` | VARCHAR(255) | Nombre en el momento de la detección |
| `dominant_emotion` | VARCHAR(64) | Emoción dominante detectada |
| `confidence` | FLOAT | Nivel de confianza de la emoción |
| `detected_at` | DATETIME | Fecha y hora del evento |

---

## Contribución

1. Haz un *fork* del repositorio.
2. Crea una rama descriptiva: `git checkout -b feature/mi-mejora`.
3. Realiza tus cambios y añade tests si aplica.
4. Abre un *Pull Request* con una descripción clara de los cambios.

---

*Desarrollado con ❤️ usando Python · OpenCV · DeepFace · CustomTkinter · SQLAlchemy*
