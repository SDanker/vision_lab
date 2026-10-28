import { useRef, useState } from "react";
import { api, errMsg } from "../lib/api";
import { BASE_LABEL, FORMAT_LABEL, fmtBytes, fmtDate, pct } from "../lib/format";
import { extensionRegex, useModality } from "../lib/modalities";
import type { ModelInfo, Prediction } from "../lib/types";
import { ScoreBars, Segments } from "./PredictionResult";
import TabularTest from "./TabularTest";
import TrainingParams from "./TrainingParams";
import { Alert, Badge } from "./ui";

const EXTRAS: [string, string][] = [
  ["labels", "labels.json"],
  ["metadata", "metadata.json"],
  ["metrics", "metrics.json"],
  ["inference", "inference.py"],
];

export function Downloads({ model }: { model: ModelInfo }) {
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-sm font-semibold">Descargas</h4>
        <a className="btn btn-primary btn-sm" href={`/api/models/${model.id}/download/all`}>⬇ ZIP completo</a>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {model.exports.map((e) => (
          <div key={e.format} className="flex min-w-0 items-center justify-between gap-2 rounded-lg border border-slate-200 p-3 text-sm dark:border-slate-800">
            <div>
              <div className="font-medium">{FORMAT_LABEL[e.format] ?? e.format}</div>
              <div className="text-xs text-slate-500">
                {e.status === "ok" ? `${fmtBytes(e.size_bytes)} · validado (Δ ${e.validation?.max_abs_diff})` : "no disponible"}
              </div>
              {e.status !== "ok" && e.error && <div className="mt-1 break-words text-xs text-red-600">{e.error.slice(0, 120)}</div>}
            </div>
            {e.status === "ok" ? (
              <a className="btn btn-secondary btn-sm" href={`/api/models/${model.id}/download/${e.format}`}>⬇</a>
            ) : (
              <Badge status="error">Falló</Badge>
            )}
          </div>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {EXTRAS.map(([k, label]) => (
          <a key={k} className="btn btn-secondary btn-sm" href={`/api/models/${model.id}/download/${k}`}>{label}</a>
        ))}
      </div>
    </div>
  );
}

type TestItem = { file: File; url: string; result?: Prediction; error?: string; loading?: boolean };
const MAX_TEST = 10;

export function TestModel({ model }: { model: ModelInfo }) {
  const mod = useModality(model.modality);
  return mod.tabular ? <TabularTest model={model} /> : <TestFiles model={model} />;
}

function TestFiles({ model }: { model: ModelInfo }) {
  const mod = useModality(model.modality);
  const isAudio = mod.preview === "audio";
  const noun = mod.unit;
  const [items, setItems] = useState<TestItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const choose = (list: FileList | null) => {
    items.forEach((i) => URL.revokeObjectURL(i.url));
    const all = Array.from(list ?? []).filter((f) => extensionRegex(mod).test(f.name));
    setNote(all.length > MAX_TEST ? `Se seleccionaron ${all.length} imágenes; se probarán las primeras ${MAX_TEST}.` : null);
    setItems(all.slice(0, MAX_TEST).map((file) => ({ file, url: URL.createObjectURL(file) })));
  };
  const patch = (i: number, p: Partial<TestItem>) => setItems((prev) => prev.map((it, k) => (k === i ? { ...it, ...p } : it)));
  const run = async () => {
    setBusy(true);
    for (let i = 0; i < items.length; i++) {
      patch(i, { loading: true, error: undefined, result: undefined });
      try {
        const form = new FormData();
        form.append("file", items[i].file);
        patch(i, { result: await api<Prediction>(`/models/${model.id}/predict`, { form }), loading: false });
      } catch (e) {
        patch(i, { error: errMsg(e), loading: false });
      }
    }
    setBusy(false);
  };
  const done = items.filter((i) => i.result);
  const counts = done.reduce<Record<string, number>>((a, i) => ({ ...a, [i.result!.prediction]: (a[i.result!.prediction] ?? 0) + 1 }), {});
  return (
    <div>
      <h4 className="mb-2 text-sm font-semibold">Probar el modelo v{model.version} con hasta {MAX_TEST} {noun}</h4>
      <p className="mb-3 text-xs text-slate-500">Sube {noun} nuevos (que no estén en el entrenamiento) para ver qué reconoce el modelo y con cuánta confianza.{mod.test_hint}</p>
      <div className="flex flex-wrap items-center gap-3">
        <input ref={input} type="file" multiple accept={mod.accept} onChange={(e) => choose(e.target.files)} />
        <button className="btn btn-primary" disabled={!items.length || busy} onClick={run}>{busy ? "Clasificando…" : `Clasificar ${items.length || ""} ${mod.unit}`}</button>
      </div>
      {note && <div className="mt-2"><Alert tone="warning">{note}</Alert></div>}
      {done.length > 0 && (
        <p className="mt-3 text-sm">
          Resumen ({done.length} de {items.length}): {Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(" · ")}
        </p>
      )}
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((it, i) => (
          <div key={i} className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
            <div className="flex gap-3">
              {isAudio ? (
                <div className="grid h-24 w-24 shrink-0 place-items-center rounded-lg bg-slate-100 text-3xl dark:bg-slate-800" aria-hidden="true">🎵</div>
              ) : (
                <img src={it.url} alt={it.file.name} className="h-24 w-24 shrink-0 rounded-lg object-cover" />
              )}
              <div className="min-w-0 text-sm">
                <div className="truncate text-xs text-slate-500" title={it.file.name}>{it.file.name}</div>
                {it.loading && <div className="mt-1">Clasificando…</div>}
                {it.error && <div className="mt-1 text-red-600">{it.error}</div>}
                {it.result && (
                  <>
                    <div className="mt-1 font-bold">{it.result.prediction}</div>
                    <div className="text-slate-500">{pct(it.result.confidence)} de confianza</div>
                    {it.result.low_confidence && <div className="mt-1 text-xs text-amber-700 dark:text-amber-300">⚠ Puntuación insuficiente: no es confiable</div>}
                  </>
                )}
              </div>
            </div>
            {it.result && <div className="mt-2"><ScoreBars scores={it.result.scores} /></div>}
            {isAudio && <audio controls src={it.url} className="mt-2 h-8 w-full" />}
            <Segments segments={it.result?.segments} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function Metrics({ model }: { model: ModelInfo }) {
  const m = model.metrics;
  const [sort, setSort] = useState<"label" | "worst" | "best">("worst");
  if (!m) return null;
  const labels = model.labels;
  const many = labels.length > 12;
  const rows = [...m.per_class].sort((a, b) =>
    sort === "label" ? a.label.localeCompare(b.label) : sort === "worst" ? a.f1 - b.f1 : b.f1 - a.f1,
  );
  // confusiones más frecuentes (fuera de la diagonal)
  const confusions: { real: string; pred: string; n: number; share: number }[] = [];
  m.confusion_matrix.forEach((row, i) => {
    const total = row.reduce((a, v) => a + v, 0) || 1;
    row.forEach((v, j) => {
      if (i !== j && v > 0) confusions.push({ real: labels[i], pred: labels[j], n: v, share: v / total });
    });
  });
  confusions.sort((a, b) => b.n - a.n);
  const cellTone = (i: number, j: number, v: number, rowTotal: number) => {
    if (!v) return "";
    const share = v / (rowTotal || 1);
    if (i === j) return share > 0.7 ? "bg-emerald-300 dark:bg-emerald-700/70" : share > 0.4 ? "bg-emerald-200 dark:bg-emerald-800/50" : "bg-emerald-100 dark:bg-emerald-900/30";
    return share > 0.2 ? "bg-red-300 dark:bg-red-700/70" : share > 0.08 ? "bg-red-200 dark:bg-red-800/50" : "bg-red-100 dark:bg-red-900/30";
  };
  const matrix = (
    <table className={`${many ? "text-[11px]" : "text-sm"} border-collapse`}>
      <thead className="sticky top-0 z-10 bg-white dark:bg-slate-900">
        <tr>
          <th className="sticky left-0 z-20 bg-white px-1 dark:bg-slate-900" />
          {labels.map((l) => (
            <th key={l} title={l} className="max-w-[3.5rem] truncate px-1 py-1 text-left font-medium text-slate-500">{l}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {m.confusion_matrix.map((row, i) => {
          const total = row.reduce((a, v) => a + v, 0);
          return (
            <tr key={i}>
              <th title={labels[i]} className="sticky left-0 max-w-[8rem] truncate bg-white px-1 py-0.5 text-left font-medium text-slate-500 dark:bg-slate-900">{labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} title={`Real: ${labels[i]} → Predicho: ${labels[j]}: ${v}`} className={`min-w-[1.75rem] border border-slate-100 px-1 py-0.5 text-center dark:border-slate-800 ${cellTone(i, j, v, total)}`}>{v || ""}</td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
  return (
    <div className="min-w-0 space-y-4">
      {m.hints?.map((h, i) => <Alert key={i} tone="warning">💡 {h}</Alert>)}
      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-3 lg:grid-cols-5">
        <div><dt className="text-xs text-slate-500">Exactitud (test)</dt><dd className="text-lg font-bold">{pct(m.test_accuracy)}</dd></div>
        {m.test_top5_accuracy != null && m.top_k != null && m.top_k > 1 && m.top_k < labels.length && ( // con tantas categorías como el top-k siempre da 100 %
          <div><dt className="text-xs text-slate-500">Top-{m.top_k} (test)</dt><dd className="text-lg font-bold">{pct(m.test_top5_accuracy)}</dd></div>
        )}
        <div><dt className="text-xs text-slate-500">Loss (test)</dt><dd className="text-lg font-bold">{m.test_loss.toFixed(4)}</dd></div>
        <div><dt className="text-xs text-slate-500">F1 macro</dt><dd className="text-lg font-bold">{pct(m.macro_f1)}</dd></div>
        <div><dt className="text-xs text-slate-500">{m.unit === "filas" ? "Filas" : "Imágenes"} train / val / test</dt><dd className="text-lg font-bold">{m.train_images} / {m.val_images} / {m.test_images}</dd></div>
        {m.majority_baseline != null && (
          <div title="Exactitud que se lograría respondiendo siempre la categoría más común"><dt className="text-xs text-slate-500">Referencia (siempre la más común)</dt><dd className="text-lg font-bold">{pct(m.majority_baseline)}</dd></div>
        )}
      </dl>

      {!!m.feature_importance?.length && (
        <div className="min-w-0">
          <div className="mb-1 text-sm font-medium">Columnas que más pesan en la decisión</div>
          <p className="mb-2 text-xs text-slate-500">Cuánto baja la exactitud al desordenar cada columna (sobre las filas de validación). Una columna que no aparece aporta poco o nada.</p>
          <div className="space-y-1">
            {m.feature_importance.map((f) => {
              const top = m.feature_importance?.[0]?.importance || 1;
              return (
                <div key={f.feature}>
                  <div className="flex justify-between text-xs"><span className="truncate">{f.feature}</span><span>{pct(f.importance)}</span></div>
                  <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800"><div className="h-1.5 rounded-full bg-brand-500" style={{ width: `${(f.importance / top) * 100}%` }} /></div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="grid min-w-0 gap-4 lg:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
            <span className="text-sm font-medium">Resultados por categoría ({labels.length})</span>
            <select className="py-1 text-xs" value={sort} onChange={(e) => setSort(e.target.value as typeof sort)} aria-label="Ordenar categorías">
              <option value="worst">Peores primero</option>
              <option value="best">Mejores primero</option>
              <option value="label">Por nombre</option>
            </select>
          </div>
          <div className="max-h-80 overflow-auto rounded-lg border border-slate-200 dark:border-slate-800">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800"><tr><th className="th">Categoría</th><th className="th">Precisión</th><th className="th">Recall</th><th className="th">F1</th><th className="th">N</th></tr></thead>
              <tbody>
                {rows.map((c) => (
                  <tr key={c.label} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="td max-w-[12rem] truncate font-medium" title={c.label}>{c.label}</td>
                    <td className="td">{pct(c.precision, 0)}</td><td className="td">{pct(c.recall, 0)}</td>
                    <td className={`td font-medium ${c.f1 < 0.5 ? "text-red-600 dark:text-red-400" : c.f1 > 0.85 ? "text-emerald-600 dark:text-emerald-400" : ""}`}>{pct(c.f1, 0)}</td>
                    <td className="td">{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-sm font-medium">Confusiones más frecuentes</div>
          {confusions.length === 0 ? (
            <p className="text-sm text-slate-500">Sin confusiones en el conjunto de prueba.</p>
          ) : (
            <ul className="max-h-80 space-y-1 overflow-auto rounded-lg border border-slate-200 p-2 text-sm dark:border-slate-800">
              {confusions.slice(0, 15).map((c, k) => (
                <li key={k} className="flex items-center justify-between gap-2">
                  <span className="min-w-0 truncate" title={`${c.real} → ${c.pred}`}><strong>{c.real}</strong> <span className="text-slate-400">confundido con</span> {c.pred}</span>
                  <span className="shrink-0 text-xs text-slate-500">{c.n} ({pct(c.share, 0)})</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {many ? (
        <details className="rounded-lg border border-slate-200 dark:border-slate-800">
          <summary className="cursor-pointer px-3 py-2 text-sm font-medium">Ver matriz de confusión completa ({labels.length}×{labels.length})</summary>
          <p className="px-3 text-xs text-slate-500">Filas = categoría real, columnas = predicha. Verde = aciertos, rojo = confusiones; pasa el mouse sobre una celda para ver el detalle.</p>
          <div className="max-h-[32rem] overflow-auto p-3">{matrix}</div>
        </details>
      ) : (
        <div className="min-w-0">
          <div className="mb-1 text-sm font-medium">Matriz de confusión (filas = real, columnas = predicho)</div>
          <div className="overflow-x-auto">{matrix}</div>
        </div>
      )}
    </div>
  );
}

export default function ModelPanel({ model, onDelete }: { model: ModelInfo; onDelete?: () => void }) {
  const mod = useModality(model.modality);
  return (
    <div className="min-w-0 space-y-6">
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-500">
        {model.can_delete && onDelete && (
          <button className="btn btn-danger btn-sm order-last ml-auto" onClick={onDelete} title="Borra el modelo y sus archivos">🗑 Borrar modelo</button>
        )}
        <span>Versión <strong className="text-slate-900 dark:text-slate-100">v{model.version}</strong></span>
        <span>{BASE_LABEL[model.base_model] ?? model.base_model}</span>
        <span title={model.labels.join(", ")}>{model.labels.length} categorías: {model.labels.slice(0, 6).join(", ")}{model.labels.length > 6 ? ` y ${model.labels.length - 6} más` : ""}</span>
        <span>
          {mod.tabular
            ? `${mod.icon} ${mod.label} · predice «${model.config?.tabular_target ?? model.input_spec?.target ?? "—"}»${model.input_spec?.features ? ` con ${model.input_spec.features.length} columnas` : ""}`
            : mod.preview !== "image"
              ? `${mod.icon} ${mod.label} · ${mod.resolution_label.toLowerCase()} ${model.config?.image_size ?? 224}×${model.config?.image_size ?? 224}`
              : `Entrada: ${model.config?.image_size ?? 224}×${model.config?.image_size ?? 224}×3 float32 (0-255)`}
        </span>
        <span>{fmtDate(model.created_at)}</span>
      </div>
      <TrainingParams config={model.config} baseModel={model.base_model} epochsRun={model.epochs_run ?? model.metrics?.epochs_run} />
      <Metrics model={model} />
      <Downloads model={model} />
      <TestModel model={model} />
    </div>
  );
}
