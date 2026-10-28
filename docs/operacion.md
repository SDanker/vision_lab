# Operación de Vision Lab AI

## Servicios

| Servicio | Función | Healthcheck |
|---|---|---|
| `db` | PostgreSQL (volumen `pgdata`) | `pg_isready` |
| `redis` | Broker Celery, pub/sub de progreso, contadores de rate limit (volumen `redisdata`) | `redis-cli ping` |
| `api` | FastAPI; aplica migraciones (`alembic upgrade head`) al iniciar | `GET /api/health` |
| `worker-train` | Entrenamiento y exportación, un trabajo a la vez (pool `solo`) | proceso activo |
| `worker-light` | Tareas livianas | `celery inspect ping` |
| `beat` | Planificador: cola semanal sáb 00:05, latido por minuto, limpieza diaria | proceso activo |
| `web` | Nginx: SPA + proxy `/api` (SSE sin búfer) | `GET /healthz` |

Volúmenes: `pgdata` (usuarios, metadatos), `data` (imágenes, miniaturas, snapshots, modelos, checkpoints, caché de pesos), `redisdata`.

## Respaldos

- **Base de datos**: `pg_dump --format=custom` (incluido en `scripts/backup.sh`).
- **Archivos**: `/data` del volumen `data`, excluyendo `work/` (temporales regenerables) y `keras/` (caché de pesos).
- Programa el respaldo diario, por ejemplo: `0 2 * * * cd /opt/vision && ./scripts/backup.sh /mnt/respaldos`. Copia los archivos fuera del servidor y prueba la restauración periódicamente.
- Los snapshots de datasets se conservan 2 por proyecto (limpieza automática); los modelos y checkpoints publicados se conservan siempre.

## Administrador y credenciales

- Crear: `docker compose exec api python -m app.cli create-admin --username ... --email ...`.
- Recuperar acceso si se pierde la contraseña: crear otro administrador con el mismo comando y, desde el panel, eliminar o degradar el anterior.
- Rotar `SECRET_KEY` cierra todas las sesiones. Cambiar `POSTGRES_PASSWORD` requiere actualizar también la contraseña dentro de PostgreSQL.

## Importación de datasets desde Hugging Face

Dentro de un proyecto de imágenes, «Importar dataset» descarga un dataset del Hub (carpetas con imágenes o tablas parquet con columna de imagen y etiqueta) y crea una categoría por carpeta/etiqueta. Corre en segundo plano en `worker-light` y se valida igual que una carga manual (formato real, duplicados por contenido, cuota del propietario, calendario semanal).

- **Datasets públicos**: no necesitan cuenta. **Privados o restringidos**: cada usuario conecta su propio token en «Mi cuenta» (sólo lectura basta); quien importa usa siempre su cuenta, nunca la del propietario del proyecto.
- **Token**: se guarda cifrado (Fernet) en `hf_connections`; nunca se devuelve por la API ni se escribe en registros. La clave es `HF_ENCRYPTION_KEY` (`init-env` la genera). **Inclúyela en el respaldo del `.env`**: si se pierde, los usuarios deben reconectar su cuenta (aparece «Reconectar»). Si no está definida se deriva de `SECRET_KEY`, y las dos se aceptan al descifrar, así que definirla más adelante no pierde conexiones.
- **Datasets sin etiquetas** (muy comunes: fotos con una descripción o con columnas de datos). En el diálogo, «¿De dónde sale la categoría?» permite: lo que declare el dataset (ClassLabel o carpetas), **cualquier columna** que parezca de categorías (pocos valores que se repiten), **un texto** (lo que sigue a una frase, por ejemplo «of type» en «… of type Grass …», con las siguientes N palabras o hasta un delimitador) o ninguna (todo a una categoría de destino). La plataforma lee una muestra de ~2 000 filas (sólo las columnas de texto, sin bajar imágenes), **sugiere** las frases que se repiten con pocos valores y muestra el reparto antes de importar; ese reparto es de la muestra, el total puede variar. Las reglas de texto no usan expresiones regulares (tiempo lineal: una regla no puede colgar el worker).
- **Filtros y máximos** (opcionales): desmarcar categorías (el resto se omite o se junta en «otros»), máximo de imágenes en total (tamaño y cuota se calculan sobre esa parte; sirve para probar con una muestra) y máximo por categoría (si todas las categorías elegidas se completan, la importación termina sola sin leer el resto). En carpetas, lo filtrado no se descarga. El contador «fuera por filtros o máximos» (`skipped`) aparece en el historial.
- **Límites**: `HF_IMPORT_MAX_GB` por importación (5 GB; también se vigila el tamaño real mientras se guarda), una importación activa por usuario, y el calendario: si se cierra a mitad de camino (fin de semana o entrenamiento) la importación se detiene con lo ya importado y puede repetirse (se omiten las repetidas).
- **Red y disco**: los contenedores deben poder salir a `huggingface.co` (y a su CDN). Las descargas pasan por `/data/work/hf-<id>` (volumen `work`) y se borran al terminar; un parquet se baja de a uno.
- **Atascos**: una importación sin señal de vida durante 30 min (worker reiniciado) se marca como error sola. Registros: `docker compose logs worker-light`. Auditoría: `hf.connected`, `hf.import_started`, `hf.import_finished`, `hf.import_cancelled`.
- Primer despliegue: `docker compose build` y recrear `api`, `worker-light` y `web` (dependencias nuevas: `huggingface_hub`, `pyarrow`, `cryptography`); `api` aplica las migraciones `0007` y `0008` al iniciar. `worker-train` no ejecuta nada de esto: se puede dejar entrenando y recrearlo en otro momento (usa la misma imagen, pero no hace falta reiniciarlo).

## HTTPS

La plataforma escucha HTTP en `HTTP_PORT`. Para publicarla, coloca un proxy inverso con TLS (Caddy, Traefik, Nginx) delante del servicio `web`, conserva `X-Forwarded-For`/`-Proto` y define `COOKIE_SECURE=true` y `PUBLIC_URL=https://...`.

## Resolución de problemas

| Síntoma | Qué revisar |
|---|---|
| Entrenamiento en «error» por memoria | Reducir `batch_size` del proyecto o subir `TRAIN_MEM_LIMIT`; el panel «Cola de entrenamiento» permite **Reintentar** |
| Trabajo «pendiente» que no avanza | `docker compose logs worker-train`; el latido (`tick`) reencola trabajos huérfanos |
| Cargas bloqueadas un lunes | Hay un entrenamiento activo o resultados sin publicar: ver «Cola de entrenamiento». Un administrador puede cancelar el trabajo |
| Formato exportado «Falló» | Ver el mensaje de validación en la pestaña «Modelos» y los registros del trabajo |
| No llegan correos | Revisar `SMTP_*`; las notificaciones internas funcionan sin SMTP |
| Reinicio durante un entrenamiento | Se recupera solo; si agotó `MAX_JOB_ATTEMPTS` queda en error y puede reintentarse |
| «Probar» dice «preparando las comparaciones» | Es el índice de «lo más parecido» del modelo, que se calcula en el worker liviano (`docker compose logs worker-light`). Es reanudable; para adelantarlo: `docker compose exec api python -m app.cli build-similarity` (`--now` lo calcula en el momento). Si dice «no disponible», ver el último error en `worker-light` |
| Importación de Hugging Face «En cola» que no avanza | `docker compose logs worker-light`; el worker liviano debe estar en pie. Pasada 1 h sin que nadie la recoja se marca como error sola |
| Importación de Hugging Face: «No se pudo conectar con Hugging Face» | Salida a internet del contenedor (`docker compose exec worker-light python -c "import urllib.request;print(urllib.request.urlopen('https://huggingface.co').status)"`) |
| «Reconectar» en Mi cuenta | Cambió `HF_ENCRYPTION_KEY` (o `SECRET_KEY` sin clave dedicada): cada usuario vuelve a pegar su token |
| «Probar»: la cámara/micrófono en vivo no aparece | Hay que abrir la plataforma por HTTPS; por HTTP solo funciona «Tomar foto» y subir archivos |

## Seguridad (resumen)

Cookie de sesión `HttpOnly` + `SameSite=Lax`; cabecera obligatoria `X-Requested-With` en peticiones que modifican datos (defensa CSRF); Argon2id; límite de intentos; validación de archivos con Pillow (formato real, tamaño, píxeles, integridad); autorización por proyecto en cada endpoint de datos; auditoría de acciones sensibles; el estado de la cuenta se lee de la base de datos en cada petición. No hay secretos en el repositorio: `.env` se genera localmente y está en `.gitignore`.
