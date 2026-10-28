const TZ = "America/Santiago";

export const fmtDate = (iso: string | null | undefined) =>
  iso
    ? new Date(iso).toLocaleString("es-CL", { timeZone: TZ, day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" })
    : "—";

export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function fmtDuration(sec: number | null | undefined): string {
  if (sec == null) return "—";
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h} h ${m} min`;
  if (m) return `${m} min ${r} s`;
  return `${r} s`;
}

export const pct = (v: number | null | undefined, digits = 1) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);

export const STATUS_LABEL: Record<string, string> = {
  PENDING: "Pendiente",
  APPROVED: "Aprobado",
  REJECTED: "Rechazado",
  SUSPENDED: "Suspendido",
  pending: "Pendiente",
  preparing: "Preparando",
  running: "Importando",
  training: "Entrenando",
  validating: "Validando",
  exporting: "Exportando",
  completed: "Completado",
  error: "Error",
  cancelled: "Cancelado",
};

export const FORMAT_LABEL: Record<string, string> = {
  keras: "Keras (.keras)",
  h5: "HDF5 (.h5)",
  savedmodel: "SavedModel (.zip)",
  tflite: "TensorFlow Lite (.tflite)",
  onnx: "ONNX (.onnx)",
  joblib: "scikit-learn (.joblib)",
};

export const BASE_LABEL: Record<string, string> = {
  mobilenetv3small: "MobileNetV3 Small",
  mobilenetv3: "MobileNetV3 Large",
  mobilenetv2: "MobileNetV2",
  efficientnetb0: "EfficientNetB0",
  efficientnetv2b0: "EfficientNetV2 B0",
  efficientnetb3: "EfficientNetB3",
  resnet50v2: "ResNet50V2",
  densenet121: "DenseNet121",
  efficientnetv2s: "EfficientNetV2 S",
  convnext_tiny: "ConvNeXt Tiny",
  convnext_small: "ConvNeXt Small",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
  logistic_regression: "Regresión logística",
  mlp: "Red neuronal (MLP)",
};
export const baseLabels = (ids: string[] | undefined) => (ids ?? []).map((m) => BASE_LABEL[m] ?? m).join(" + ");

export const AUDIO_ACCEPT = ".wav,.mp3,.ogg,.flac,.m4a,.aac,.opus,.webm,audio/*";
export const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp";
export const AUDIO_RE = /\.(wav|mp3|ogg|flac|m4a|aac|opus|webm)$/i;
export const IMAGE_RE = /\.(jpe?g|png|webp)$/i;
export const fmtSeconds = (s: number | null | undefined) => {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
};

/** «sábado 31-10-2026, 00:05» (hora de Chile): cuándo se ejecuta el próximo entrenamiento semanal. */
export const fmtWhen = (iso: string | null | undefined) =>
  iso
    ? new Date(iso).toLocaleString("es-CL", { timeZone: TZ, weekday: "long", day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" })
    : "—";

/** «sáb 31-10» */
export const fmtDayShort = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleDateString("es-CL", { timeZone: TZ, weekday: "short", day: "2-digit", month: "2-digit" }).replace(",", "").replace(".", "") : "—";

/** « Los modelos cortos pueden entrenarse antes, de noche (00:00–08:00).» cuando hay entrenamiento nocturno. */
export const nightNote = (window: string | null | undefined) =>
  window ? `; los modelos cortos pueden entrenarse antes, de lunes a viernes de noche (${window})` : "";
