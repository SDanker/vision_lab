import { BASE_LABEL } from "../lib/format";
import { useModality } from "../lib/modalities";
import type { TrainingConfig } from "../lib/types";

type Cfg = Partial<TrainingConfig> & { modality?: string };

/** Resumen de una línea: «15 + 5 épocas · lote 32 · LR 0.001 · paciencia 3 · 224×224 px · con aumento». */
export function TrainingParamsInline({ config }: { config?: Cfg | null }) {
  const mod = useModality(config?.modality);
  if (!config || config.epochs == null) return null;
  const parts = mod.tabular ? [
    config.tabular_target ? `predice «${config.tabular_target}»` : null,
    config.tabular_features?.length ? `${config.tabular_features.length} columnas de entrada` : null,
  ].filter(Boolean) : [
    config.fine_tune_epochs ? `${config.epochs} + ${config.fine_tune_epochs} épocas (ajuste fino)` : `${config.epochs} épocas`,
    config.batch_size != null && `lote ${config.batch_size}`,
    config.learning_rate != null && `LR ${config.learning_rate}`,
    config.patience != null && `paciencia ${config.patience}`,
    config.image_size && `${config.image_size}×${config.image_size} px`,
    config.augmentation != null && (config.augmentation ? "con aumento de datos" : "sin aumento de datos"),
    mod.windowed && config.clip_seconds != null && `clips de ${config.clip_seconds} s`,
  ].filter(Boolean);
  return (
    <div className="mt-2 flex flex-wrap gap-1.5" aria-label="Parámetros del entrenamiento">
      {parts.map((p) => (
        <span key={String(p)} className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700 dark:bg-slate-800 dark:text-slate-300">{p}</span>
      ))}
    </div>
  );
}

/** Parámetros con los que se entrenó (o se está entrenando) un modelo. */
export default function TrainingParams({ config, baseModel, epochsRun }: { config?: Cfg | null; baseModel?: string; epochsRun?: number | null }) {
  const mod = useModality(config?.modality);
  if (!config) return null;
  const size = config.image_size;
  const planned = config.epochs != null ? config.epochs + (config.fine_tune_epochs ?? 0) : null;
  const rows: [string, string | null][] = mod.tabular ? [
    ["Modelo base", baseModel ? (BASE_LABEL[baseModel] ?? baseModel) : null],
    ["Columna a predecir", config.tabular_target ?? null],
    ["Columnas de entrada", config.tabular_features?.length ? String(config.tabular_features.length) : "las sugeridas"],
  ] : [
    ["Modelo base", baseModel ? (BASE_LABEL[baseModel] ?? baseModel) : null],
    ["Épocas", config.epochs != null ? String(config.epochs) : null],
    ["Ajuste fino", config.fine_tune_epochs != null ? (config.fine_tune_epochs > 0 ? `${config.fine_tune_epochs} épocas` : "No") : null],
    ["Épocas ejecutadas", epochsRun != null ? `${epochsRun}${planned ? ` de ${planned}` : ""}${planned && epochsRun < planned ? " (se detuvo antes)" : ""}` : null],
    ["Tamaño de lote", config.batch_size != null ? String(config.batch_size) : null],
    ["Learning rate", config.learning_rate != null ? String(config.learning_rate) : null],
    ["Paciencia (early stopping)", config.patience != null ? `${config.patience} épocas` : null],
    ["Aumento de datos", config.augmentation != null ? (config.augmentation ? "Sí" : "No") : null],
    [mod.resolution_label, size ? `${size}×${size} px` : null],
    ["Duración de cada clip", mod.windowed && config.clip_seconds != null ? `${config.clip_seconds} s` : null],
  ];
  const shown = rows.filter((r): r is [string, string] => r[1] !== null);
  if (shown.length === 0) return null;
  return (
    <div>
      <h4 className="mb-2 text-sm font-semibold">Parámetros del entrenamiento</h4>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 rounded-lg border border-slate-200 p-3 text-sm sm:grid-cols-3 lg:grid-cols-5 dark:border-slate-800">
        {shown.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt className="text-xs text-slate-500">{label}</dt>
            <dd className="font-medium">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
