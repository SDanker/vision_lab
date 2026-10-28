export type UserStatus = "PENDING" | "APPROVED" | "REJECTED" | "SUSPENDED";

export interface User {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  username: string;
  role: "USER" | "ADMIN";
  status: UserStatus;
  reason: string | null;
  quota_mb: number | null;
  created_at: string;
  reviewed_at: string | null;
  review_note: string | null;
  reviewed_by: string | null;
  can_use_platform?: boolean;
  status_message?: string;
  storage_bytes?: number;
  quota_bytes?: number;
  projects?: number;
}

export interface Category {
  id: number;
  name: string;
  image_count: number;
}

/** Parámetros de entrenamiento de un modelo concreto; lo que falta se hereda de los valores generales. */
export type ModelOverride = Partial<Omit<TrainingConfig, "model_overrides">>;

export interface TrainingConfig {
  epochs: number;
  fine_tune_epochs: number;
  batch_size: number;
  learning_rate: number;
  patience: number;
  augmentation: boolean;
  image_size?: number;
  clip_seconds?: number;
  /** Datos tabulares: columna a predecir y columnas de entrada (sin definir = las sugeridas). */
  tabular_target?: string | null;
  tabular_features?: string[] | null;
  /** Ajuste fino (sólo por modelo): capas finales que se descongelan y tasa = learning rate × factor. */
  unfreeze_layers?: number;
  fine_tune_lr_factor?: number;
  /** Configuración avanzada: clave = id del modelo base. */
  model_overrides?: Record<string, ModelOverride>;
}

export type ProjectRole = "owner" | "editor" | "viewer";

export interface ProjectMember {
  user_id: number;
  username: string;
  name: string;
  role: "editor" | "viewer";
  added_at: string;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  base_model: string;
  base_models: string[];
  config: TrainingConfig;
  auto_train: boolean;
  /** Cuándo se ejecutaría el próximo entrenamiento semanal (sólo si el proyecto está marcado). */
  next_training_start?: string | null;
  /** Ventana del entrenamiento nocturno (lun–vie), p. ej. «00:00–08:00»; null si está desactivado. */
  night_window?: string | null;
  modality?: string;
  created_at: string;
  owner: string;
  owner_id: number;
  /** Mi rol en el proyecto: propietario, editor o sólo lectura. */
  my_role?: ProjectRole | null;
  members?: ProjectMember[];
  categories: Category[];
  image_count: number;
  min_images_per_class?: number;
  size_bytes: number;
  latest_model: { id: number; version: number; accuracy: number | null } | null;
  /** De noche se bloquea sólo el proyecto que se está entrenando. */
  training_lock?: { kind: "night"; message: string; remaining_seconds: number } | null;
}

export interface ImageItem {
  id: number;
  project_id: number;
  category_id: number;
  name: string;
  size_bytes: number;
  width: number;
  height: number;
  format: string;
  duration_s?: number | null;
  created_at: string;
}

export type JobStatus = "pending" | "preparing" | "training" | "validating" | "exporting" | "completed" | "error" | "cancelled";

export interface HistoryRow {
  epoch: number;
  loss?: number;
  accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;
}

export interface Job {
  id: number;
  project_id: number;
  project_name: string;
  owner_id: number;
  status: JobStatus;
  phase: string | null;
  base_model: string;
  trigger: string;
  config?: Partial<TrainingConfig> & { modality?: string };
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  epoch: number;
  total_epochs: number;
  percent: number;
  eta_seconds: number | null;
  /** Duración estimada al encolar (se usa para decidir qué cabe de noche y el orden del fin de semana). */
  estimated_seconds?: number | null;
  elapsed_seconds: number | null;
  metrics: Record<string, number> | null;
  history: HistoryRow[];
  cpu_percent: number | null;
  ram_mb: number | null;
  error: string | null;
  queue_position: number | null;
  cancel_requested?: boolean;
  model_version_id: number | null;
  snapshot?: { dataset_hash: string; class_counts: Record<string, number>; total_images: number; excluded: Record<string, number> | null };
}

export interface LogRow {
  id: number;
  level: string;
  message: string;
  created_at: string;
}

export interface ModelExportInfo {
  format: string;
  status: "ok" | "failed";
  filename: string | null;
  size_bytes: number | null;
  validation: { max_abs_diff: number; argmax_match: boolean } | null;
  error: string | null;
}

export interface ModelInfo {
  id: number;
  project_id: number;
  project_name?: string;
  project_owner?: string;
  my_role?: "owner" | "editor" | "viewer" | null;
  can_delete?: boolean; // sólo el propietario del proyecto (o un administrador)
  version: number;
  base_model: string;
  labels: string[];
  created_at: string;
  accuracy: number | null;
  top5?: number | null;
  macro_f1?: number | null;
  epochs_run?: number | null;
  train_seconds?: number | null;
  dataset_hash: string;
  modality?: string;
  exports: ModelExportInfo[];
  config?: Partial<TrainingConfig> & { modality?: string };
  /** Sólo en el detalle: entrada del modelo (imagen: forma; tabla: las columnas con las que se arma el formulario). */
  input_spec?: { kind?: string; target?: string | null; features?: TabularFeature[] };
  metrics?: {
    feature_importance?: { feature: string; importance: number }[];
    majority_baseline?: number;
    unit?: string;
    test_accuracy: number;
    test_top5_accuracy?: number;
    top_k?: number;
    hints?: string[];
    test_loss: number;
    macro_f1: number;
    val_accuracy: number | null;
    epochs_run: number;
    train_images: number;
    val_images: number;
    test_images: number;
    per_class: { label: string; precision: number; recall: number; f1: number; support: number }[];
    confusion_matrix: number[][];
  };
}

/** Columna de entrada de un modelo tabular. */
export interface TabularFeature {
  name: string;
  kind: "numeric" | "categorical";
  median?: number;
  min?: number;
  max?: number;
  categories?: string[];
  default?: string;
}

/** Columna de la tabla de un proyecto (GET /projects/{id}/tabular). */
export interface TabularColumn {
  name: string;
  kind: "numeric" | "categorical";
  unique: number;
  missing: number;
  missing_pct: number;
  examples: string[];
  min: number | null;
  max: number | null;
  mean: number | null;
  values: { value: string; count: number }[] | null;
  constant: boolean;
  id_like: boolean;
  target_ok: boolean;
}

export interface TabularSummary {
  n_files: number;
  n_rows: number;
  n_columns: number;
  columns: TabularColumn[];
  preview: { columns: string[]; rows: string[][] } | null;
  target: string | null;
  features: string[];
  suggested_features: string[];
  classes: { label: string; count: number; used: boolean }[] | null;
  min_per_class: number;
  max_classes: number;
  problems: string[];
  ready: boolean;
}

export interface Prediction {
  prediction: string;
  modality?: string;
  /** Datos tabulares: filas consultadas, columnas que faltaron y resultado de cada fila. */
  n_rows?: number;
  missing_fields?: string[];
  inputs?: Record<string, unknown>;
  rows?: { row: number; prediction: string; confidence: number; top: { label: string; score: number }[] }[];
  duration_s?: number;
  segments?: { start: number; end: number; label: string; confidence: number }[];
  confidence: number;
  scores: { label: string; score: number }[];
  low_confidence: boolean;
  warning: string | null;
  model_version: number;
  engine: string;
  /** Con ?similar=N: lo más parecido del proyecto. */
  similar?: SimilarItem[];
  similar_status?: "ready" | "updating" | "building" | "unavailable";
}

export interface Schedule {
  open: boolean;
  reason: "OPEN" | "WEEKEND_LOCK" | "TRAINING_ACTIVE";
  message: string;
  now: string;
  next_lock_at: string;
  next_training_start: string;
}

export interface NotificationItem {
  id: number;
  kind: string;
  title: string;
  body: string;
  read: boolean;
  created_at: string;
}

export interface ApiTokenItem {
  id: number;
  name: string;
  prefix: string;
  scope: "read" | "write";
  state: "active" | "expired" | "revoked";
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
  last_used_ip: string | null;
  revoked_at: string | null;
  owner?: { id: number; username: string; name: string };
}

export interface TokenRequestItem {
  id: number;
  name: string;
  reason: string;
  status: "pending" | "approved" | "rejected" | "cancelled";
  expires_days: number | null;
  created_at: string;
  reviewed_at: string | null;
  review_note: string | null;
  reviewed_by: string | null;
  token_id: number | null;
  can_generate: boolean;
  owner?: { id: number; username: string; name: string; role: string };
}

/** Respuesta al crear un token: es la única vez que llega el valor en claro. */
export type NewApiToken = ApiTokenItem & { token: string };

/** Cuenta de Hugging Face del usuario (el token nunca llega al navegador). */
export interface HfStatus {
  connected: boolean;
  username?: string;
  hint?: string;
  connected_at?: string;
  last_used_at?: string | null;
  needs_reconnect?: boolean;
}

/** Regla para sacar la categoría de un texto: lo que sigue a una frase (N palabras o hasta un delimitador). */
export interface HfPhrase {
  column: string;
  after: string;
  words: number | null;
  until: string | null;
}

export interface HfValueCount {
  value: string;
  count: number;
}

/** Frase que se repite en los textos y va seguida de pocos valores distintos (candidata a categoría). */
export interface HfSuggestion {
  column: string;
  after: string;
  words: number;
  coverage: number;
  distinct: number;
  values: HfValueCount[];
}

/** Reparto de categorías: exacto en carpetas, sobre una muestra de filas en tablas. */
export interface HfLabelPreview {
  exact: boolean;
  sample_rows: number;
  matched: number;
  unmatched: number;
  distinct: number;
  values: HfValueCount[];
}

/** Revisión de un dataset de Hugging Face antes de importarlo (POST /projects/{id}/hf/inspect). */
export interface HfPreview {
  label_source: "dataset" | "column" | "phrase" | "folders" | "none";
  label_phrase: HfPhrase | null;
  text_columns: string[];
  suggestions: HfSuggestion[];
  label_preview: HfLabelPreview | null;
  effective_bytes: number;
  repo_id: string;
  revision: string;
  private: boolean;
  gated: boolean;
  layout: "files" | "parquet";
  files: number;
  items: number | null;
  total_bytes: number;
  categories: { name: string; count: number }[] | null;
  category_count: number | null;
  class_names: string[] | null;
  unlabeled: number;
  image_columns: string[];
  label_columns: string[];
  image_column: string | null;
  label_column: string | null;
  blockers: string[];
  warnings: string[];
  can_import: boolean;
}

export type HfImportStatus = "pending" | "running" | "completed" | "error" | "cancelled";

export interface HfImportItem {
  id: number;
  project_id: number;
  repo_id: string;
  revision: string;
  status: HfImportStatus;
  phase: string | null;
  percent: number | null;
  total: number;
  processed: number;
  imported: number;
  duplicates: number;
  rejected: number;
  unlabeled: number;
  skipped: number;
  rejected_sample: { name: string; reason: string }[];
  total_bytes: number;
  cancel_requested: boolean;
  error: string | null;
  user: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface SimilarItem {
  image_id: number;
  name: string;
  category: string | null;
  similarity: number; // 0–1 (coseno)
  kind: "image" | "audio" | string;
  duration_s: number | null;
}

/** Pregunta nocturna: «¿todavía te faltan cargas?» antes de congelar un proyecto en el que alguien trabaja. */
export interface NightCheck {
  id: number;
  project_id: number;
  project_name: string | null;
  night: string;
  status: "asking" | "go" | "skip";
  reason: string | null;
  created_at: string;
  answered_at: string | null;
  timeout_minutes: number;
}

export interface NightPlan {
  enabled: boolean;
  in_window: boolean;
  night: string;
  window: string;
  capacity_seconds: number;
  max_job_seconds: number;
  items: {
    project_id: number;
    project: string;
    base_model: string;
    model: string;
    estimated_seconds: number;
    source: string;
    decision: "tonight" | "too_long" | "no_room" | "asking" | "skipped_by_user";
  }[];
  not_ready: { project_id: number; project: string; reason: string }[];
}

export interface NightReport {
  night: string;
  missing: boolean;
  in_progress: boolean;
  tone: "ok" | "running" | "error" | "idle" | "missing";
  created_at: string | null;
  capacity_seconds: number;
  headline: string;
  items: (Omit<NightPlan["items"][number], "decision" | "source"> & {
    source: string | null;
    decision: NightPlan["items"][number]["decision"] | "not_queued";
    job?: { id: number; status: string; phase: string | null };
  })[];
  not_ready: { project_id: number; project: string; reason: string }[];
}
