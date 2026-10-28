# Vision Lab AI

Plataforma web **autohospedada** (Docker) para cargar imágenes, entrenar modelos de **clasificación de imágenes** con transfer learning (TensorFlow/Keras), evaluar los resultados y descargar los modelos en **5 formatos**: Keras, HDF5, TensorFlow SavedModel, TensorFlow Lite y ONNX.

Sin licencias ni servicios de pago, sin GPU obligatoria y con un presupuesto de RAM configurable (por defecto ≈ 18,9 GB, máximo 20 GB).

| Capa | Tecnología |
|---|---|
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS (español, responsive, modo claro/oscuro) |
| Backend | Python 3.11 + FastAPI + SQLAlchemy + Alembic |
| ML | TensorFlow/Keras 2.15 (11 modelos base: MobileNetV3 Small/Large, MobileNetV2, EfficientNetB0/B3, EfficientNetV2-B0/S, ResNet50V2, DenseNet121, ConvNeXt Tiny/Small), tf2onnx, ONNX Runtime |
| Datos | PostgreSQL 16 + volúmenes persistentes |
| Trabajos | Redis + Celery (worker de entrenamiento `solo`, worker liviano, beat) |
| Tiempo real | Server-Sent Events (SSE) sobre Redis pub/sub |
| Zona horaria | `America/Santiago` (respeta los cambios de horario de Chile) |

## Puesta en marcha

Requisitos: Docker con Compose v2 y ~8 GB de disco libre para las imágenes.

```bash
# 1. Generar .env con secretos aleatorios (no se versiona)
./scripts/init-env.sh            # Windows: .\scripts\init-env.ps1

# 2. Levantar todo
docker compose up -d --build

# 3. Crear el administrador inicial (la contraseña se pide por consola, sin eco)
docker compose exec api python -m app.cli create-admin --username admin --email admin@tu-dominio.cl
```

Abre <http://localhost:8180> (puerto `HTTP_PORT` del `.env`; cámbialo si choca con otro contenedor). Los usuarios se registran desde «Solicita acceso» y **no pueden usar la plataforma hasta que un administrador los apruebe**.

Alternativa para el administrador inicial: define `ADMIN_USERNAME`, `ADMIN_EMAIL` y `ADMIN_PASSWORD` en `.env` antes del primer arranque; se crea automáticamente sólo si todavía no existe ningún administrador (conviene borrar `ADMIN_PASSWORD` de `.env` después). El código no incluye contraseñas predeterminadas.

Los pesos preentrenados de ImageNet se descargan durante `docker build` y se guardan en la imagen; si el build no tuvo internet, se descargan al primer entrenamiento.

## Registro y aprobación

Estados: `PENDING` → `APPROVED` / `REJECTED`; `APPROVED` ⇄ `SUSPENDED`.

- El estado se comprueba **en cada petición protegida del backend**, leyéndolo de la base de datos (dependencia `require_approved`). Pendientes, rechazados y suspendidos reciben `403`/`401` en proyectos, imágenes, entrenamientos, modelos y descargas.
- Un usuario pendiente o rechazado sólo puede iniciar sesión para ver el estado de su solicitud (pantalla `/estado`). Los suspendidos no pueden iniciar sesión.
- **Suspender o rechazar invalida las sesiones activas** (contador `token_version` dentro del JWT en cookie `HttpOnly`).
- Nadie puede cambiar su propio estado o rol; los endpoints `/api/admin/*` exigen rol administrador en el servidor.
- No se puede suspender, degradar ni eliminar a la última cuenta administrativa activa.
- Historial de cada cambio (quién, cuándo, motivo) y registro de auditoría general.
- Aprobación individual o múltiple (con diálogo de confirmación y confirmación obligatoria en la API).
- Limitación de intentos de registro (por IP) e inicio de sesión (por IP+usuario y por IP), en Redis.
- Contraseñas con **Argon2id**.
- Notificaciones internas siempre activas; correo SMTP opcional (`SMTP_*` en `.env`). Sin SMTP todo funciona igual.

## Calendario semanal (America/Santiago)

| Momento | Regla |
|---|---|
| Lunes–viernes | Se permite cargar, modificar y eliminar imágenes y categorías (si no hay entrenamientos activos) |
| Sábado 00:00 | Se bloquean las modificaciones |
| Sábado 00:05 | `celery beat` dispara la cola semanal: se congela un **snapshot** (hardlinks + manifiesto) de cada dataset y se encolan los entrenamientos |
| Sábado y domingo | Todo bloqueado, aunque el entrenamiento termine antes |
| Lunes 00:00 | Se reabre **sólo si** no quedan entrenamientos activos y ya se publicaron los resultados |
| Al publicar los resultados | **Se desmarcan** todos los proyectos que participaron en la ejecución semanal (casilla «Incluir en el entrenamiento semanal automático»); sus propietarios y editores reciben un aviso |
| Lunes–viernes siguiente | Cada persona revisa los resultados, ajusta el modelo y, si quiere que se entrene otra vez, **vuelve a marcar la casilla y guarda**: la plataforma le avisa la fecha de ejecución (el sábado siguiente a las 00:05, hora de Chile) |

### Entrenamiento nocturno (lunes a viernes, 00:00–08:00)

Además del sábado, cada noche hábil se entrenan los modelos cortos que caben en la ventana:

- **Qué se considera**: proyectos marcados para entrenar con al menos 2 categorías válidas, y sólo los modelos cuyo dataset o configuración cambió desde su última versión.
- **Duración estimada** (`services/estimates.py`): se calcula como imágenes × épocas máximas × (resolución/224)², multiplicado por una tasa en segundos, más un costo fijo de exportación. La tasa sale de la mediana de los últimos trabajos del mismo proyecto y modelo, o del mismo modelo en cualquier proyecto. Sin historial se usa una tasa de referencia según la velocidad del modelo, con un margen ×1,5.
- **Qué entra**: sólo los modelos de hasta `NIGHT_MAX_JOB_HOURS` (4 h por defecto). Se reparte el tiempo que queda hasta las 08:00 empezando por los más cortos, para que quepan todos los posibles (p. ej. dos de 3 h 30). Lo que no cabe se entrena el sábado.
- **Bloqueo sólo del proyecto que se entrena**: los demás siguen abiertos. La API responde `423 PROJECT_TRAINING` para ese proyecto, y se desbloquea apenas se publican sus resultados. Entrenar de noche no desmarca el proyecto.
- **Pregunta antes de congelar**: si alguien con permiso de edición modificó el proyecto en los últimos `NIGHT_ACTIVITY_MINUTES` (15) minutos, se le pregunta «¿Todavía te faltan cargas?» con un aviso en la plataforma y una notificación.
  - **«Sí, sigo cargando»**: el proyecto no se entrena esa noche.
  - **«No, ya terminé»**: entra en la siguiente pasada del planificador, en menos de un minuto.
  - **Sin respuesta en `NIGHT_ASK_TIMEOUT_MINUTES` (20)**: se entrena sólo si dejó de cargar.
- **Antes de iniciar cada modelo** se comprueba que su estimación quepa antes de las 08:00. Si no cabe, queda «Pospuesto para el fin de semana». Un modelo que ya empezó nunca se corta, aunque se atrase.
- El planificador corre dentro de `vision.tick` y no hace nada si hay un entrenamiento semanal o manual en curso. Los administradores ven el plan de la noche, con estimaciones y decisiones, en Administración (`GET /api/admin/night-plan`).
- **Resumen de cada noche**: en la primera pasada de cada noche hábil se guarda qué se planificó, también cuando no se entrena nada. En Administración → Cola de entrenamiento, «Últimas noches» muestra qué se entrenó o por qué no (`GET /api/admin/night-reports`). Una noche hábil sin resumen aparece como «Sin registro», señal de que el servidor o el worker estaban detenidos.
- **Aviso al dueño**: si un modelo no cabe de noche (supera el máximo o no queda tiempo), el dueño del proyecto recibe una notificación de que se entrenará el sábado. Se avisa una vez por semana por modelo.

**Orden de la cola del sábado (y de los manuales)**: del modelo que más tarda al que menos (`training_jobs.queue_order`).

Configuración: `NIGHT_TRAINING_ENABLED`, `NIGHT_WINDOW_START`, `NIGHT_WINDOW_END`, `NIGHT_MAX_JOB_HOURS`, `NIGHT_ACTIVITY_MINUTES`, `NIGHT_ASK_TIMEOUT_MINUTES` (migración `0005`).

Los bloqueos se verifican en el backend (`423 UPLOADS_LOCKED`), no sólo en la interfaz. Un latido (`vision.tick`, cada minuto) cubre un `beat` caído a las 00:05: la ejecución semanal es idempotente por semana (clave única). Los proyectos sin al menos 2 categorías con `MIN_IMAGES_PER_CLASS` imágenes se omiten (con aviso al propietario), y los que no cambiaron desde la última versión no se reentrenan.

**Ciclo semanal de marcado.** Cada semana empieza «desmarcada»: al publicarse los resultados de una ejecución automática, la semanal del sábado o una nocturna de lunes a viernes (`reset_weekly_optin` en `services/training_queue.py`), los proyectos que participaron quedan con la casilla apagada, aunque no se hayan entrenado (por ejemplo, omitidos por falta de imágenes) o hayan terminado con error. Las ejecuciones manuales de un administrador no tocan la casilla, ni los proyectos que se marcaron después de arrancar la ejecución. El aviso de fecha incluye la ventana nocturna (los modelos cortos pueden entrenarse antes del sábado, de noche) cuando está activa (`night_window` en el proyecto). Al volver a guardar un proyecto marcado, la respuesta incluye `next_training_start`, se muestra la fecha en la pestaña «Entrenamiento» y se envía una notificación («se ejecutaría el sábado 31-10-2026 a las 00:05»); las tarjetas de «Proyectos» indican si entra o no esta semana. Los proyectos nuevos se crean marcados. Cada desmarcado queda en la auditoría (`project.weekly_optin_reset`).

## Proyectos de audio

Al crear un proyecto se elige el tipo **Imágenes** o **Audio** (no se cambia después). En los proyectos de audio:

- Se aceptan WAV, MP3, OGG, FLAC, M4A, AAC, OPUS y WEBM (hasta `MAX_AUDIO_UPLOAD_MB`, 50 MB por defecto); cada archivo se valida decodificándolo con ffmpeg. La galería muestra el espectrograma, la duración y un reproductor.
- Para entrenar, cada audio (mono, 16 kHz) se corta en clips de «Duración de cada clip» segundos (1–10, por defecto 5; como máximo `max_clips_per_audio` = 20 clips por archivo) y cada clip se convierte en un espectrograma log-mel (128 bandas) que se entrena con los mismos modelos base. La división train/val/test se hace **por archivo**, para que trozos del mismo audio no queden en entrenamiento y prueba a la vez.
- Aumento de datos propio de audio (SpecAugment: desplazamiento temporal y enmascarado de bandas de frecuencia y de tiempo).
- Exportación a los mismos 5 formatos. El paquete incluye `audio_features.py` (el mismo código de espectrograma usado al entrenar) y un `inference.py` que analiza un audio por segmentos y promedia.
- La prueba de modelos muestra el resultado global y el de cada segmento.

## Proyectos de datos tabulares (CSV)

El tercer tipo de proyecto clasifica **filas de una tabla**: por ejemplo, predecir si un Pokémon es legendario a partir de sus estadísticas, o el tipo de cliente a partir de sus compras. No usa imágenes ni redes de visión: entrena modelos de scikit-learn.

- **Carga:** archivos CSV o TSV (hasta `MAX_TABULAR_UPLOAD_MB` = 50 MB cada uno, 500 000 filas entre todos, 300 columnas), con los nombres de columna en la primera línea; el separador (coma, punto y coma, tabulación o barra vertical) y la codificación (UTF-8 o Windows) se detectan solos. Si se suben varios archivos se unen en una sola tabla (deben compartir columnas). Cada CSV se valida leyéndolo de verdad y los duplicados se detectan por contenido.
- **Columnas:** en la pestaña «Datos» se elige la columna a predecir (sus valores pasan a ser las categorías; entre 2 y 100, y cada una necesita al menos `MIN_IMAGES_PER_CLASS` filas) y las columnas de entrada. Por defecto se sugieren todas salvo las que parecen identificadores (nombres, números correlativos), las constantes y las casi vacías. El tipo de cada columna (número o texto/categoría) se detecta solo; los valores vacíos se completan con la mediana (números) o «sin dato» (texto), y una categoría que el modelo no vio al entrenar cuenta como «otra».
- **Modelos:** Random Forest, Gradient Boosting, Regresión logística y Red neuronal (MLP), con las clases balanceadas. No hay épocas ni data augmentation: cada modelo entrena en segundos, con la misma división estratificada 70/15/15 (train/val/test) que el resto, y entra en el entrenamiento nocturno y semanal igual que los demás. Las métricas son las de siempre (exactitud, F1, matriz de confusión) más la **referencia** (acertar siempre la categoría más común), qué **columnas pesan más** (importancia por permutación) y avisos si el modelo no supera la referencia o parece sobreajustado. Sólo clasificación; predecir un número continuo no está soportado.
- **Exportación y paquete:** `model.onnx` (preprocesado incluido, para ONNX Runtime) y `model.joblib`, ambos validados contra el modelo original, más `labels.json`, `metadata.json` (columnas, tipos y categorías conocidas), `inference.py` y `tabular_features.py` (la misma conversión de filas usada al entrenar).
- **Prueba:** un formulario generado con las columnas del modelo o un CSV con hasta `max_tabular_predict_rows` (1 000) filas, con la tabla de resultados descargable. `POST /api/models/{id}/predict` también acepta un objeto JSON `{columna: valor}` o una lista de ellos; «lo más parecido» no aplica a las tablas.
- **Dependencias:** `pandas`, `scikit-learn`, `scipy` y `skl2onnx` (en `backend/requirements.txt`): hay que reconstruir las imágenes `api`, `worker-train` y `worker-light` para usarlo.

Sirve para clasificar **sonidos** (motores, aeronaves, alarmas, animales, fallas de máquinas). No es reconocimiento de voz.

## Compartir proyectos

Los proyectos son privados por defecto. El propietario puede compartirlos con otros usuarios aprobados desde la pestaña «Compartir» (por nombre de usuario o correo); la persona recibe una notificación y lo ve en su lista de proyectos.

| Permiso | Qué puede hacer |
|---|---|
| **Propietario** (y administradores) | Todo: eliminar el proyecto o una categoría completa, cambiar nombre y descripción, gestionar quién tiene acceso |
| **Editor** | Cargar imágenes/audios, crear y renombrar categorías, mover o borrar imágenes, ajustar parámetros de entrenamiento (incluida la configuración avanzada por modelo) |
| **Sólo lectura** | Ver el proyecto, las imágenes, los trabajos y los modelos; probar y descargar modelos |

Quien no tiene acceso sigue recibiendo `404`; quien lo tiene pero no el permiso suficiente recibe `403 PROJECT_FORBIDDEN`. El calendario semanal (`423 UPLOADS_LOCKED`) aplica igual a los editores. Las cargas cuentan en la cuota del propietario. Cada miembro puede abandonar el proyecto, y compartir/quitar/cambiar permisos queda en la auditoría (`project.shared`, `project.unshared`, `project.share_role`). Los resultados de entrenamiento se notifican al propietario y a todos los miembros, y el progreso en vivo llega a todos ellos.

Endpoints: `GET/POST /api/projects/{id}/members`, `PATCH/DELETE /api/projects/{id}/members/{user_id}`. La migración `0003` crea la tabla `project_members` (se aplica sola al arrancar el contenedor).

## Tokens de API personales

Cada usuario aprobado puede usar la API desde un script o aplicación sin su contraseña, con un **token personal** (`Authorization: Bearer vl_...`). Un token **siempre actúa como su dueño**: ve sus proyectos propios y los que le compartieron, con el mismo rol (propietario, editor o sólo lectura) y sujeto al calendario semanal. Nunca da más acceso que la cuenta.

| Tipo | Quién lo crea | Qué permite |
|---|---|---|
| **Lectura** | El propio usuario, en *Cuenta* (hasta 5 activos) | Sólo consultas `GET`: proyectos, progreso de entrenamientos (`/api/training/jobs`, `/logs`, `/api/training/stream`), modelos y descargas. Cualquier otro método responde `403 TOKEN_READ_ONLY` |
| **Lectura y escritura** | El usuario lo **solicita** en *Cuenta* y un **administrador lo aprueba** (otro administrador, si quien pide lo es); luego el usuario lo genera | Además cargar imágenes, crear categorías, ajustar parámetros, etc., pero únicamente donde su cuenta ya puede hacerlo (un lector sigue siendo lector, sólo el propietario elimina) |

- El token se muestra **una sola vez** al crearlo; en la base de datos sólo queda su hash SHA-256 y un prefijo para reconocerlo. Se puede revocar en cualquier momento (el usuario o un administrador) y vence (lectura: 30 días–2 años o sin vencimiento; escritura: lo fija el administrador al aprobar, 90 días por defecto, máximo 365).
- Ningún token llega a `/api/admin/*` ni puede crear o aprobar otros tokens: eso sólo se hace con la sesión web.
- Suspender o rechazar una cuenta revoca todos sus tokens (reactivarla no los revive). Cambiar el rol o quitar el acceso a un proyecto se refleja de inmediato, porque los permisos se leen de la base de datos en cada petición.
- Límites: 300 peticiones por minuto por token y 30 tokens inválidos por IP en 15 min (`API_TOKEN_*` en `app/config.py`). Todo queda en la auditoría (`apitoken.*`); el administrador ve las solicitudes y todos los tokens en *Administración → Tokens de API*.
- Con `Authorization: Bearer` no hace falta la cabecera CSRF `X-Requested-With`, y la cookie de sesión se ignora.

```bash
curl -H "Authorization: Bearer vl_TU_TOKEN" http://localhost:8180/api/training/jobs?project_id=1
curl -N -H "Authorization: Bearer vl_TU_TOKEN" http://localhost:8180/api/training/stream   # progreso en vivo (SSE)
```

La migración `0004` crea las tablas `api_tokens` y `api_token_requests` (se aplica sola al arrancar el contenedor).

## Entrenamiento

- Transfer learning sobre 11 modelos base de Keras Applications con pesos ImageNet (la normalización de cada arquitectura va **dentro** del modelo: entrada float32 0‑255).
- **Resolución configurable por proyecto**: 160, 224 (predeterminada), 260, 300 o 384 px. Más resolución distingue mejor detalles finos pero el tiempo crece con el número de píxeles (384 px ≈ 3× de 224 px). El tamaño elegido queda en `metadata.json` y la inferencia lo usa automáticamente.
- Cada proyecto puede tener **uno o varios modelos base seleccionados**: se congela un único snapshot del dataset y se encola un entrenamiento por modelo; la pestaña «Modelos y pruebas» los compara (exactitud, top‑5, F1, tiempo, tamaño).
- **Configuración avanzada por modelo**: en «Entrenamiento» cada modelo base seleccionado puede tener sus propios parámetros (épocas, ajuste fino, batch size, learning rate, paciencia, augmentation, resolución y, en audio, duración del clip). Lo que se deja vacío hereda el valor general del proyecto. El botón «Usar ajuste recomendado» rellena esos campos con el ajuste pensado para ese modelo (`TUNING` en `app/ml/architectures.py`: los lentos, como ConvNeXt, EfficientNetB3 o V2‑S, llevan menos épocas de ajuste fino y menos paciencia; los livianos, más). Se guarda en `config.model_overrides`; al encolar, cada trabajo recibe la configuración efectiva (general + ajustes de su modelo), así que un cambio en un solo modelo reentrena sólo ese modelo.
- División estratificada train/val/test (70/15/15), data augmentation, class weights, `EarlyStopping`, `ModelCheckpoint`, épocas y batch size configurables por proyecto.
- Ajuste fino de las últimas capas del modelo base activado por defecto en proyectos nuevos (15 épocas + 5 de ajuste fino).
- Métricas: exactitud, top-5, loss, F1 por clase y matriz de confusión, más sugerencias automáticas (el modelo seguía mejorando, posible sobreajuste, falta de ajuste fino).
- Lectura progresiva: cada imagen se decodifica una vez (en paralelo) a un `memmap` en disco y los lotes se leen desde ahí; nunca se carga el dataset completo en RAM. Esos temporales viven en el volumen interno `work` (rápido), aunque `DATA_PATH` apunte a otro disco.
- **Un único entrenamiento intensivo a la vez**: worker Celery con pool `solo`, más un *advisory lock* de PostgreSQL.
- Recuperación tras reinicios: latido por trabajo; si el worker muere, `tick` devuelve el trabajo a la cola (hasta `MAX_JOB_ATTEMPTS`) y Keras `BackupAndRestore` reanuda desde la última época. Errores de memoria se reportan con un mensaje accionable.
- Se registran dataset (hash y conteos), configuración, métricas (accuracy, loss, F1, matriz de confusión), fechas y versión del modelo.

## Exportación y prueba

Por cada entrenamiento se genera `model.keras`, `model.h5`, `saved_model.zip`, `model.tflite`, `model.onnx`, además de `labels.json`, `metadata.json`, `metrics.json`, `inference.py`, `README.md` y `bundle.zip`. **Cada formato se valida** recargándolo y comparando sus probabilidades con las del modelo Keras sobre muestras reales de prueba (misma clase predicha y diferencia máxima < 0,02); sólo entonces queda marcado como válido. Un formato que falla no se ofrece para descarga.

La página «Modelos» muestra **una tarjeta por proyecto** con cuántos entrenamientos tiene y solo su **mejor modelo** (mayor exactitud; a igualdad, el más reciente). Al hacer clic se despliegan todos sus entrenamientos y, en el detalle de cada uno, los **parámetros con que se entrenó** (también en el detalle de cada trabajo de entrenamiento). Desde ahí se pueden **borrar** modelos ya entrenados, uno a uno o varios con las casillas (también desde la pestaña «Modelos» de cada proyecto). Sólo puede borrar el **propietario** del proyecto o un administrador (`DELETE /api/models/{id}`; con token de API hace falta uno con escritura). Se eliminan el registro, las exportaciones, los archivos de `DATA_DIR/models/<proyecto>/v<N>` y su espacio en la cuota; el trabajo de entrenamiento queda en el historial y la acción en la auditoría (`model.deleted`). Si borras la versión más reciente de un modelo base y el dataset no cambió, el siguiente entrenamiento semanal lo vuelve a generar, porque ya no hay una versión que coincida; además, la versión borrada puede reutilizar su número en el próximo entrenamiento.

La pestaña «Modelos y pruebas» permite subir una imagen, ver la puntuación de todas las categorías, la versión del modelo y una advertencia cuando la confianza es menor que `LOW_CONFIDENCE_THRESHOLD`.

## Probar un modelo desde el celular («Probar»)

La página **Probar** (`/probar`) está pensada para el teléfono: eliges un proyecto y una versión del modelo (por defecto la mejor), **tomas la foto o grabas el audio en el momento** (o subes un archivo) y ves qué reconoce el modelo, con su confianza, y **lo más parecido del proyecto** (las 4 imágenes o audios del dataset más cercanos, con su categoría y porcentaje de parecido; al tocarlos se amplían).

- **Cámara y micrófono:** «Tomar foto» abre la cámara nativa del teléfono y funciona siempre; «Cámara en vivo» (con cambio de cámara frontal/trasera) y «Grabar audio» necesitan abrir la plataforma por **HTTPS** (p. ej. el dominio de Cloudflare); por IP local en HTTP la página avisa y ofrece subir el archivo. Las fotos se reducen en el teléfono (máx. 1 600 px) antes de enviarlas.
- **Nada se guarda:** lo que capturas se analiza en memoria y se descarta; no cuenta en la cuota ni entra al dataset.
- **Cómo se calcula el parecido:** se usa el vector de características del propio modelo (la salida de su pooling global, justo antes de la capa de clasificación). Se obtiene recortando `model.onnx` en `embedder.onnx`, sin reentrenar y también para modelos ya entrenados, y se compara por coseno con el índice `similar.npz` guardado en la carpeta del modelo. En pruebas con tus imágenes, el vecino más cercano cae en la misma categoría ~90 % de las veces.
- **Índice:** se calcula en segundo plano (worker liviano) al terminar cada entrenamiento y se actualiza solo con lo que cambió (archivos nuevos o borrados); es reanudable, guarda avance cada 2 000 archivos y, mientras se prepara, la clasificación igual responde y la página muestra el progreso. Proyectos grandes tardan (≈ 25 imágenes/s: unos 30 min para 48 000). Para precalcularlo: `docker compose exec api python -m app.cli build-similarity` (mejor modelo de cada proyecto; `--model ID` para uno concreto, `--now` para calcular en el momento).
- **API:** `POST /api/models/{id}/predict?similar=N` (N de 0 a 8) añade `similar` y `similar_status` (`ready`, `updating`, `building`, `unavailable`); `GET /api/models/{id}/similar-status` informa el avance. Un fallo en esta parte nunca impide devolver la clasificación.

## Tipos de dato (modalidades)

Lo que cambia de un tipo de dato a otro (validar y guardar el archivo, convertirlo en muestras, predecir, exportar el código de entrada y calcular el vector de parecido) vive en una clase `Modality` registrada en `backend/app/modalities/`. Hoy hay `image`, `audio` y `tabular`; la interfaz lee sus textos y formatos de `GET /api/modalities`. Para añadir otro tipo basta una subclase de `Modality` y una línea en `REGISTRY`, siempre que se pueda convertir en muestras N×N×3 para las redes de visión (como el audio, con espectrogramas). Los datos tabulares son la excepción: su modalidad marca `tabular=True` y tiene su propio flujo de entrenamiento (`ml/tabular_runner.py`, `services/tabular.py`) sin TensorFlow.

## Recursos y RAM

Límites por servicio en `docker-compose.yml` (variables `*_MEM_LIMIT`): entrenamiento 14 g, API 2 g, DB 1 g, Redis 0,5 g, worker liviano 1 g, beat 0,25 g, web 0,125 g (suma ≈ 18,9 GB). Ajusta `TRAIN_THREADS` para limitar la CPU de TensorFlow. El panel de administración muestra CPU/RAM del host y del contenedor, y cada trabajo muestra su consumo.

## Acceso público (Cloudflare Tunnel)

1. En Cloudflare Zero Trust → Networks → Tunnels, crea un túnel (o usa uno existente) y copia su token.
2. En ese túnel agrega un **Public Hostname**: `vision.tu-dominio` → tipo **HTTP** → URL **`web:80`**.
3. En `.env`: `CLOUDFLARE_TUNNEL_TOKEN=<token>`, `PUBLIC_URL=https://vision.tu-dominio`, `COOKIE_SECURE=true`.
4. `docker compose --profile tunnel up -d tunnel`

Nginx toma la IP real del visitante de `CF-Connecting-IP` (necesario para el límite de intentos), y las cargas se envían en lotes de ≤ 40 MB porque Cloudflare rechaza peticiones de más de 100 MB. Con `COOKIE_SECURE=true` el acceso local debe hacerse por `http://localhost:8180` o por el dominio HTTPS (no por la IP de la red local).

## Actualizar / desplegar cambios

```bash
./scripts/deploy.sh          # Windows: .\scripts\deploy.ps1
```

Reconstruye y reinicia sólo si no hay entrenamientos en cola o en proceso (reiniciar `worker-train` corta el entrenamiento en curso). Con `--force` (`-Force`) despliega igualmente; los trabajos interrumpidos se reintentan solos.

## Respaldos

```bash
./scripts/backup.sh ./backups                       # BD (pg_dump) + volumen de datos
./scripts/restore.sh backups/db-*.dump backups/data-*.tar.gz   # restauración (destructiva, pide confirmación)
```

Detalles y recomendaciones en [docs/operacion.md](docs/operacion.md).

## Estructura del repositorio

```text
.
├── backend/            API FastAPI, workers Celery, modelos SQLAlchemy, migraciones Alembic y pruebas
│   ├── app/
│   ├── alembic/
│   └── tests/
├── frontend/           SPA React + TypeScript + Vite + Tailwind (se sirve con Nginx)
├── scripts/            init-env, deploy, backup y restore (versiones .sh y .ps1)
├── docs/               Guía de operación (respaldos, recuperación, mantenimiento)
├── docker-compose.yml  Orquestación: db, redis, api, worker-train, worker-light, beat, web, tunnel (opcional)
├── .env.example        Plantilla de configuración (copiar a .env; el .env real nunca se versiona)
└── README.md
```

## Pruebas

```bash
docker compose run --rm --no-deps --entrypoint "" api sh -c \
  "COOKIE_SECURE=false DATABASE_URL=sqlite:////tmp/t.db REDIS_URL= python -m pytest tests -q"
```

Cubren registro/aprobación/rechazo, suspensión de sesiones, permisos por usuario y proyecto, cargas de imágenes (formatos, duplicados, cuotas), calendario (incluidos cambios de horario de Chile y reapertura del lunes), recuperación de trabajos, y un **entrenamiento real de extremo a extremo** (MobileNetV3 y EfficientNetB0) con exportación y validación de los 5 formatos, descargas e inferencia. Las imágenes sintéticas existen sólo como fixtures de prueba; la plataforma no incluye datos de entrenamiento falsos.

## Alcance

Primera versión: **clasificación de imágenes completas**. Detección de objetos y segmentación quedan como ampliaciones futuras.

Limitaciones conocidas: sin GPU el entrenamiento es lento en datasets grandes; la cuota de usuario cuenta imágenes originales (los modelos se muestran aparte en el panel); el SMTP se envía en un hilo sin cola de reintentos.

## Licencia

Distribuido bajo la licencia [MIT](LICENSE). Puedes usar, copiar, modificar y distribuir este proyecto libremente, manteniendo el aviso de copyright.
