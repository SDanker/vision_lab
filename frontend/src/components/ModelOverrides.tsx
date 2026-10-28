import { api } from "../lib/api";
import { BASE_LABEL } from "../lib/format";
import type { ModalityInfo } from "../lib/modalities";
import type { ModelOverride, TrainingConfig } from "../lib/types";
import { useAsync } from "./ui";

type NumKey = "epochs" | "fine_tune_epochs" | "batch_size" | "learning_rate" | "patience" | "clip_seconds" | "unfreeze_layers" | "fine_tune_lr_factor";

// Sólo existen por modelo (no hay valor «general»): vacío = el que trae la arquitectura.
const PER_MODEL_ONLY: NumKey[] = ["unfreeze_layers", "fine_tune_lr_factor"];

const NUMERIC: { key: NumKey; label: string; min: number; max: number; step?: string; windowedOnly?: boolean }[] = [
  { key: "epochs", label: "Épocas máximas", min: 1, max: 100 },
  { key: "fine_tune_epochs", label: "Épocas de ajuste fino", min: 0, max: 50 },
  { key: "unfreeze_layers", label: "Capas a descongelar", min: 1, max: 1000 },
  { key: "fine_tune_lr_factor", label: "Tasa del ajuste fino (× learning rate)", min: 0.001, max: 1, step: "any" },
  { key: "batch_size", label: "Batch size", min: 2, max: 64 },
  { key: "learning_rate", label: "Learning rate", min: 0.00001, max: 0.1, step: "any" },
  { key: "patience", label: "Paciencia", min: 1, max: 20 },
  { key: "clip_seconds", label: "Duración del clip (s)", min: 1, max: 10, step: "0.5", windowedOnly: true },
];

const SIZES = [160, 224, 260, 300, 384];

/**
 * Configuración avanzada: cada modelo base seleccionado puede tener sus propios parámetros.
 * Un campo vacío significa «usar el valor general», que se muestra como sugerencia gris.
 */
export default function ModelOverrides({
  models, cfg, onChange, modality, disabled,
}: {
  models: string[];
  cfg: TrainingConfig;
  onChange: (v: Record<string, ModelOverride>) => void;
  modality: ModalityInfo;
  disabled?: boolean;
}) {
  const all = cfg.model_overrides ?? {};
  const { data: catalog } = useAsync(() => api<{ items: { id: string; tuning?: ModelOverride }[] }>("/architectures"), []);
  const tuningOf = (model: string) => catalog?.items.find((a) => a.id === model)?.tuning;
  const set =(model: string, patch: ModelOverride) => {
    const next: ModelOverride = { ...(all[model] ?? {}), ...patch };
    for (const k of Object.keys(next) as (keyof ModelOverride)[]) if (next[k] === undefined) delete next[k];
    const out = { ...all };
    if (Object.keys(next).length) out[model] = next;
    else delete out[model];
    onChange(out);
  };
  const setNum = (model: string, key: NumKey, raw: string) => set(model, { [key]: raw === "" ? undefined : Number(raw) } as ModelOverride);
  const used = models.filter((m) => Object.keys(all[m] ?? {}).length > 0).length;

  return (
    <details className="rounded-lg border border-slate-200 p-3 dark:border-slate-700" open={used > 0}>
      <summary className="cursor-pointer select-none text-sm font-medium">
        Configuración avanzada por modelo
        {used > 0 && <span className="ml-2 rounded-full bg-brand-600 px-2 py-0.5 text-xs text-white">{used} personalizado{used === 1 ? "" : "s"}</span>}
      </summary>
      <p className="mt-2 text-xs text-slate-500">
        Cada modelo base se entrena con los valores generales de arriba. Aquí puedes darle a un modelo parámetros distintos, por ejemplo más épocas o mayor resolución para uno pesado y menos para uno liviano. Deja un campo vacío para heredar el valor general.
      </p>
      <div className="mt-3 space-y-3">
        {models.map((m) => {
          const ov = all[m] ?? {};
          const size = ov.image_size ?? cfg.image_size ?? 224;
          const batch = ov.batch_size ?? cfg.batch_size;
          const count = Object.keys(ov).length;
          return (
            <fieldset key={m} disabled={disabled} className="rounded-lg bg-slate-50 p-3 dark:bg-slate-900">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <legend className="float-left text-sm font-semibold">{BASE_LABEL[m] ?? m}{count > 0 && <span className="ml-2 text-xs font-normal text-brand-600">{count} ajuste{count === 1 ? "" : "s"} propio{count === 1 ? "" : "s"}</span>}</legend>
                {!disabled && (
                  <div className="flex flex-wrap gap-2">
                    {tuningOf(m) && (
                      <button type="button" className="btn btn-secondary btn-sm" title="Épocas, ajuste fino, paciencia, batch, learning rate y resolución pensados para este modelo" onClick={() => set(m, tuningOf(m)!)}>
                        Usar ajuste recomendado
                      </button>
                    )}
                    {count > 0 && (
                      <button type="button" className="btn btn-secondary btn-sm" onClick={() => set(m, Object.fromEntries(Object.keys(ov).map((k) => [k, undefined])) as ModelOverride)}>
                        Volver a los valores generales
                      </button>
                    )}
                  </div>
                )}
              </div>
              <div className="grid clear-both gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {NUMERIC.filter((f) => !f.windowedOnly || modality.windowed).map((f) => (
                  <label key={f.key} className="block text-xs">
                    <span className="label">{f.label}</span>
                    <input
                      className="w-full"
                      type="number"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={ov[f.key] ?? ""}
                      placeholder={PER_MODEL_ONLY.includes(f.key) ? `Del modelo: ${tuningOf(m)?.[f.key] ?? "—"}` : `General: ${cfg[f.key] ?? "—"}`}
                      onChange={(e) => setNum(m, f.key, e.target.value)}
                    />
                  </label>
                ))}
                <label className="block text-xs">
                  <span className="label">{modality.resolution_label}</span>
                  <select className="w-full" value={ov.image_size ?? ""} onChange={(e) => set(m, { image_size: e.target.value === "" ? undefined : Number(e.target.value) })}>
                    <option value="">General ({cfg.image_size ?? 224} px)</option>
                    {SIZES.map((s) => <option key={s} value={s}>{s} × {s}</option>)}
                  </select>
                </label>
                <label className="block text-xs">
                  <span className="label">Data augmentation</span>
                  <select className="w-full" value={ov.augmentation === undefined ? "" : ov.augmentation ? "1" : "0"} onChange={(e) => set(m, { augmentation: e.target.value === "" ? undefined : e.target.value === "1" })}>
                    <option value="">General ({cfg.augmentation ? "activado" : "desactivado"})</option>
                    <option value="1">Activado</option>
                    <option value="0">Desactivado</option>
                  </select>
                </label>
              </div>
              {size >= 300 && batch > 16 && (
                <p className="mt-2 text-xs text-amber-700 dark:text-amber-300">Con {size} px conviene un batch size de 16 o menos para este modelo.</p>
              )}
            </fieldset>
          );
        })}
      </div>
    </details>
  );
}
