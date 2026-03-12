# Sistema de Reconocimiento Facial con Análisis de Emociones

Sistema de visión artificial para el **registro**, **identificación** y **análisis emocional** de personas en tiempo real. Mediante una interfaz gráfica, el software captura rostros, genera *embeddings* faciales para el reconocimiento y utiliza modelos de aprendizaje profundo para clasificar siete emociones básicas con sus respectivos niveles de confianza.

---

## Tabla de Contenidos

1. [Características](#características)
2. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
3. [Instalación](#instalación)
4. [Configuración](#configuración)
5. [Uso](#uso)
6. [Módulos Principales](#módulos-principales)
7. [Base de Datos](#base-de-datos)
8. [Contribución](#contribución)

---

## Características

| Característica | Detalle |
|---|---|
| 🎭 Reconocimiento facial | Embeddings con **Facenet512** (DeepFace) |
| 😀 Análisis de emociones | 7 emociones básicas: felicidad, tristeza, enojo, sorpresa, neutral, miedo, disgusto |
| 🖥 Interfaz moderna | **CustomTkinter** con tema claro/oscuro |
| 🗄 Persistencia | **SQLite** (por defecto) o **PostgreSQL** vía SQLAlchemy |
| 📊 Reportes | Gráficas por emoción y por persona con Matplotlib |

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
    │   ├── app.py                 # Ventana raíz + navegación entre pantallas
    │   ├── registration_screen.py # Pantalla de registro de personas
    │   ├── detection_screen.py    # Pantalla de detección en vivo
    │   └── reports_screen.py      # Pantalla de reportes y estadísticas
    │
    ├── logic/                     # Procesamiento de imágenes y modelos de IA
    │   ├── __init__.py
    │   ├── face_recognizer.py     # FaceRecognizer: detección + embeddings + identificación
    │   └── emotion_analyzer.py    # EmotionAnalyzer: clasificación de 7 emociones
    │
    └── database/                  # Capa de persistencia (SQLAlchemy ORM)
        ├── __init__.py
        └── db_manager.py          # DatabaseManager: CRUD de personas y logs
```

---

## Instalación

### Requisitos previos

- Python **3.9 – 3.11** (recomendado 3.10)
- `pip` ≥ 23
- Webcam funcional (para detección en vivo)
- *(Opcional)* PostgreSQL si se desea usar como backend de base de datos

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Lejayk/SistemaReconocimientoFacial.git
cd SistemaReconocimientoFacial

# 2. Crear y activar un entorno virtual
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** En la primera ejecución, DeepFace descargará automáticamente los pesos del modelo seleccionado (≈ 90 MB para Facenet512). Asegúrate de tener conexión a internet.

---

## Configuración

La configuración se maneja mediante variables de entorno. Crea un archivo `.env` en la raíz del proyecto:

```dotenv
# URL de conexión a la base de datos (por defecto: SQLite local)
DATABASE_URL=sqlite:///facial_recognition.db

# Para PostgreSQL:
# DATABASE_URL=postgresql+psycopg2://usuario:contraseña@localhost:5432/facial_db
```

---

## Uso

```bash
# Iniciar la aplicación
python main.py
```

La aplicación abrirá una ventana con tres pantallas accesibles desde la barra lateral:

| Pantalla | Descripción |
|---|---|
| 📝 **Registro** | Captura un rostro desde la webcam e ingresa el nombre para registrar a la persona. |
| 🔍 **Detección** | Inicia el stream de la cámara; detecta, reconoce y analiza emociones en tiempo real. |
| 📊 **Reportes** | Muestra el historial de detecciones y gráficas de emociones y personas. |

---

## Módulos Principales

### `FaceRecognizer` (`src/logic/face_recognizer.py`)

```python
from src.logic.face_recognizer import FaceRecognizer

recognizer = FaceRecognizer(model_name="Facenet512", distance_metric="cosine")

# Registrar una persona
recognizer.register_face(name="Ana García", face_img=frame, db_manager=db)

# Identificar un rostro
result = recognizer.identify(face_img=face_crop, db_manager=db)
# → {"name": "Ana García", "distance": 0.21, "person_id": 1}
```

| Método | Descripción |
|---|---|
| `detect_faces(frame)` | Devuelve lista de bounding boxes detectados |
| `get_embedding(face_img)` | Genera el vector de embedding (512-D) |
| `register_face(name, face_img, db_manager)` | Extrae embedding y lo persiste en la BD |
| `identify(face_img, db_manager)` | Compara con todos los registros y devuelve el mejor match |
| `compare_embedding(q, s)` | Calcula distancia coseno/euclidiana entre dos embeddings |

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

