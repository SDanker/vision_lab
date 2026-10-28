import { useMemo, useRef, useState } from "react";
import { api, errMsg } from "../lib/api";
import { pct } from "../lib/format";
import type { ModelInfo, Prediction, TabularFeature } from "../lib/types";
import { ScoreBars } from "./PredictionResult";
import { Alert, Spinner, useAsync } from "./ui";

const typical = (f: TabularFeature) => (f.kind === "numeric" ? (f.median != null ? String(f.median) : "") : (f.default ?? f.categories?.[0] ?? ""));

/** Prueba de un modelo de datos tabulares: un formulario con sus columnas, o un CSV con varias filas. */
export default function TabularTest({ model }: { model: ModelInfo }) {
  // la lista de modelos no trae las columnas de entrada; el detalle sí
  const detail = useAsync(() => (model.input_spec ? Promise.resolve(model) : api<ModelInfo>(`/models/${model.id}`)), [model.id]);
  const features = useMemo(() => detail.data?.input_spec?.features ?? [], [detail.data]);
  const target = detail.data?.input_spec?.target;
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [batchName, setBatchName] = useState<string | null>(null);
  const csv = useRef<HTMLInputElement>(null);

  const send = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      setResult(await api<Prediction>(`/models/${model.id}/predict`, { form }));
    } catch (e) {
      setResult(null);
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  const predictRow = () => {
    const row: Record<string, string | number> = {};
    for (const f of features) {
      const v = (values[f.name] ?? "").trim();
      if (v === "") continue; // vacío = el modelo lo completa (mediana / «sin dato»)
      row[f.name] = f.kind === "numeric" ? Number(v.replace(",", ".")) : v;
    }
    setBatchName(null);
    void send(new File([JSON.stringify(row)], "fila.json", { type: "application/json" }));
  };
  const predictCsv = (list: FileList | null) => {
    const f = list?.[0];
    if (!f) return;
    setBatchName(f.name);
    void send(f);
    if (csv.current) csv.current.value = "";
  };
  const download = () => {
    if (!result?.rows) return;
    const lines = ["fila,prediccion,confianza", ...result.rows.map((r) => `${r.row},"${r.prediction.replace(/"/g, '""')}",${r.confidence.toFixed(4)}`)];
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = `predicciones-v${model.version}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (detail.loading && !detail.data) return <Spinner />;
  if (!features.length) return <p className="text-sm text-slate-500">Este modelo no tiene las columnas de entrada disponibles.</p>;
  const batch = !!batchName && !!result && (result.n_rows ?? 1) > 1;
  const counts = (result?.rows ?? []).reduce<Record<string, number>>((a, r) => ({ ...a, [r.prediction]: (a[r.prediction] ?? 0) + 1 }), {});

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold">Probar el modelo v{model.version}</h4>
        <p className="text-xs text-slate-500">
          Completa los valores de una fila para ver qué {target ? <>valor de <strong>{target}</strong> </> : ""}predice el modelo. Lo que dejes vacío se completa con un valor típico.
        </p>
      </div>
      {error && <Alert tone="error">{error}</Alert>}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {features.map((f) => (
          <label key={f.name} className="block min-w-0">
            <span className="label truncate" title={f.name}>{f.name}</span>
            {f.kind === "numeric" ? (
              <input className="w-full" type="number" step="any" inputMode="decimal" value={values[f.name] ?? ""} placeholder={f.median != null ? `típico: ${f.median}` : ""}
                onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
            ) : f.categories?.length ? (
              <select className="w-full" value={values[f.name] ?? ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}>
                <option value="">(sin dato)</option>
                {f.categories.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : (
              <input className="w-full" value={values[f.name] ?? ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
            )}
            {f.kind === "numeric" && f.min != null && <span className="mt-0.5 block text-xs text-slate-400">visto al entrenar: {f.min} … {f.max}</span>}
          </label>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn btn-primary" disabled={busy} onClick={predictRow}>{busy && !batchName ? "Analizando…" : "Predecir"}</button>
        <button className="btn btn-secondary" disabled={busy} onClick={() => setValues(Object.fromEntries(features.map((f) => [f.name, typical(f)])))}>Rellenar con valores típicos</button>
        <button className="btn btn-secondary" disabled={busy} onClick={() => setValues({})}>Limpiar</button>
        <span className="mx-1 text-xs text-slate-400">o</span>
        <button className="btn btn-secondary" disabled={busy} onClick={() => csv.current?.click()}>📄 Probar con un CSV (varias filas)</button>
        <input ref={csv} type="file" accept=".csv,.tsv,.json,text/csv,application/json" className="hidden" onChange={(e) => predictCsv(e.target.files)} />
      </div>

      {busy && <div className="flex items-center gap-2 text-sm"><Spinner /> Analizando…</div>}

      {result && !batch && (
        <div className="space-y-3 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-slate-500">El modelo predice{target ? ` · ${target}` : ""}</div>
            <div className="text-3xl font-bold leading-tight">{result.prediction}</div>
            <div className="text-sm text-slate-500">{pct(result.confidence)} de confianza</div>
          </div>
          {result.low_confidence && <Alert tone="warning">⚠ {result.warning ?? "Puntuación insuficiente: la clasificación no es confiable."}</Alert>}
          {!!result.missing_fields?.length && <p className="text-xs text-slate-500">No indicaste: {result.missing_fields.join(", ")} (se usó un valor típico).</p>}
          <ScoreBars scores={result.scores} />
        </div>
      )}

      {result && batch && (
        <div className="space-y-3 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm"><strong>{batchName}</strong>: {result.n_rows} filas — {Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>
            <button className="btn btn-secondary btn-sm" onClick={download}>⬇ Descargar predicciones</button>
          </div>
          {!!result.missing_fields?.length && <Alert tone="warning">Al archivo le faltan columnas del modelo: {result.missing_fields.join(", ")}. Se usó un valor típico.</Alert>}
          <div className="max-h-80 overflow-auto rounded-lg border border-slate-200 dark:border-slate-800">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800"><tr><th className="th">Fila</th><th className="th">Predicción</th><th className="th">Confianza</th><th className="th">Otras opciones</th></tr></thead>
              <tbody>
                {(result.rows ?? []).slice(0, 200).map((r) => (
                  <tr key={r.row} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="td">{r.row}</td>
                    <td className="td font-medium">{r.prediction}</td>
                    <td className={`td ${r.confidence < 0.6 ? "text-amber-700 dark:text-amber-300" : ""}`}>{pct(r.confidence, 0)}</td>
                    <td className="td text-xs text-slate-500">{r.top.slice(1).map((t) => `${t.label} ${pct(t.score, 0)}`).join(" · ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(result.rows?.length ?? 0) > 200 && <p className="text-xs text-slate-500">Se muestran las primeras 200 filas; descarga el CSV para ver todas.</p>}
        </div>
      )}
    </div>
  );
}
