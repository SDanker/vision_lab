import { api } from "../lib/api";
import { useAsync } from "./ui";

interface Arch { id: string; label: string; description: string; size_mb: number; speed: string; modality?: "image" | "tabular" }

/** Selección de uno o más modelos base. Cada modelo marcado genera su propio entrenamiento y versión.
 *  `tabular` muestra los modelos de scikit-learn (datos tabulares); si no, las redes de visión. */
export default function ModelPicker({ value, onChange, disabled, tabular = false }: { value: string[]; onChange: (v: string[]) => void; disabled?: boolean; tabular?: boolean }) {
  const { data } = useAsync(() => api<{ default: string; items: Arch[] }>("/architectures"), []);
  const items = data?.items.filter((a) => (a.modality ?? "image") === (tabular ? "tabular" : "image"));
  const toggle = (id: string) => {
    const next = value.includes(id) ? value.filter((v) => v !== id) : [...value, id];
    if (next.length) onChange(next);
  };
  return (
    <div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {items?.map((a) => (
          <label key={a.id} className={`flex cursor-pointer gap-2 rounded-lg border p-3 text-sm ${value.includes(a.id) ? "border-brand-500 bg-brand-50 dark:bg-brand-700/10" : "border-slate-200 dark:border-slate-700"}`}>
            <input type="checkbox" className="mt-0.5" checked={value.includes(a.id)} disabled={disabled} onChange={() => toggle(a.id)} />
            <span>
              <span className="font-medium">{a.label}</span>
              <span className="ml-1 text-xs text-slate-500">· {a.speed} · ~{a.size_mb} MB</span>
              <span className="block text-xs text-slate-500">{a.description}</span>
            </span>
          </label>
        ))}
      </div>
      {value.length > 1 && (
        <p className="mt-2 text-xs text-amber-700 dark:text-amber-300">
          {tabular
            ? `Seleccionaste ${value.length} modelos: se entrenarán con los mismos datos (tardan segundos) y podrás comparar sus resultados.`
            : `Seleccionaste ${value.length} modelos: se entrenarán uno tras otro con el mismo dataset (${value.length} veces el tiempo) y podrás comparar sus resultados.`}
        </p>
      )}
    </div>
  );
}
