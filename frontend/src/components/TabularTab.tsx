import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { api, apiUpload, errMsg } from "../lib/api";
import { fmtBytes, fmtDate, pct } from "../lib/format";
import { useJobs } from "../lib/jobs";
import { useModality } from "../lib/modalities";
import type { ImageItem, Project, TabularColumn, TabularSummary } from "../lib/types";
import { Alert, Progress, Spinner, useAsync } from "./ui";

type UploadResult = { uploaded: unknown[]; duplicates: { filename: string; reason: string }[]; rejected: { filename: string; reason: string }[] };

const KIND_LABEL = { numeric: "número", categorical: "texto / categoría" } as const;

/** Datos tabulares: archivos CSV del proyecto y elección de la columna a predecir y de las columnas de entrada. */
export default function TabularTab({ project, reload }: { project: Project; reload: () => void }) {
  const { schedule } = useJobs();
  const mod = useModality(project.modality);
  const scheduleLocked = (schedule ? !schedule.open : false) || !!project.training_lock;
  const readOnly = project.my_role === "viewer";
  const locked = scheduleLocked || readOnly;

  const summary = useAsync(() => api<TabularSummary>(`/projects/${project.id}/tabular`), [project.id]);
  const files = useAsync(() => api<{ items: ImageItem[] }>(`/projects/${project.id}/images?page_size=200`), [project.id]);
  const s = summary.data;

  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState<{ loaded: number; total: number } | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  // selección de columnas en edición (se guarda con el botón)
  const [target, setTarget] = useState<string>("");
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    if (!s) return;
    setTarget(s.target ?? "");
    setChosen(new Set(s.features));
  }, [s]);

  const refresh = useCallback(() => {
    summary.reload();
    files.reload();
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload]);

  const upload = async (list: File[]) => {
    const ok = list.filter((f) => /\.(csv|tsv)$/i.test(f.name));
    if (!ok.length) {
      setError("Sólo se aceptan archivos CSV o TSV.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const agg: UploadResult = { uploaded: [], duplicates: [], rejected: [] };
      const BATCH_BYTES = 40 * 1024 * 1024; // el servidor limita el tamaño de cada petición
      let batch: File[] = [];
      let bytes = 0;
      const batches: File[][] = [];
      for (const f of ok) {
        if (batch.length && (batch.length >= 5 || bytes + f.size > BATCH_BYTES)) {
          batches.push(batch);
          batch = [];
          bytes = 0;
        }
        batch.push(f);
        bytes += f.size;
      }
      if (batch.length) batches.push(batch);
      for (const b of batches) {
        const form = new FormData();
        b.forEach((f) => form.append("files", f));
        const r = await apiUpload<UploadResult>(`/projects/${project.id}/images`, form, (loaded, total) => setSent({ loaded, total }));
        agg.uploaded.push(...r.uploaded);
        agg.duplicates.push(...r.duplicates);
        agg.rejected.push(...r.rejected);
      }
      setResult(agg);
      refresh();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
      setSent(null);
      if (input.current) input.current.value = "";
    }
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (!locked && !busy) void upload(Array.from(e.dataTransfer.files));
  };
  const removeFile = async (f: ImageItem) => {
    if (!confirm(`¿Eliminar «${f.name}»? Sus filas dejarán de usarse para entrenar.`)) return;
    try {
      await api(`/images/${f.id}`, { method: "DELETE" });
      refresh();
    } catch (e) {
      setError(errMsg(e));
    }
  };

  const save = async (nextTarget: string, features: string[] | null) => {
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      await api(`/projects/${project.id}/tabular/columns`, { method: "PUT", body: { target: nextTarget, features } });
      setSaved(true);
      refresh();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setSaving(false);
    }
  };
  const changeTarget = (name: string) => {
    setTarget(name);
    if (name) void save(name, null); // al cambiar la columna a predecir, las de entrada vuelven a las sugeridas
  };
  const toggle = (name: string) => {
    setSaved(false);
    setChosen((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };
  const dirty = !!s && (target !== (s.target ?? "") || chosen.size !== s.features.length || s.features.some((f) => !chosen.has(f)));
  const cols = s?.columns ?? [];
  const inputs = cols.filter((c) => c.name !== target);

  const note = (c: TabularColumn) =>
    c.constant ? "siempre el mismo valor" : c.id_like ? "parece un identificador" : c.missing_pct >= 0.6 ? "casi todo vacío" : "";

  return (
    <div className="space-y-6">
      {error && <Alert tone="error">{error}</Alert>}

      <div
        className={`card border-2 border-dashed ${drag ? "border-brand-500 bg-brand-50 dark:bg-brand-700/10" : ""} ${locked ? "opacity-60" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <div className="flex flex-col items-center gap-3 py-3 text-center">
          <div className="text-4xl" aria-hidden="true">📊</div>
          <p className="font-medium">Arrastra aquí tus archivos CSV</p>
          <p className="max-w-xl text-xs text-slate-500">{mod.upload_help} Cada fila es un ejemplo; una de sus columnas será la categoría que el modelo aprende a predecir.</p>
          <button type="button" className="btn btn-primary" disabled={locked || busy} onClick={() => input.current?.click()}>Agregar CSV</button>
          <input ref={input} type="file" multiple accept={mod.accept} className="hidden" onChange={(e) => e.target.files && void upload(Array.from(e.target.files))} />
          {busy && <Spinner />}
        </div>
        {sent && (
          <div className="mt-2 space-y-1" aria-live="polite">
            <Progress value={(sent.loaded / Math.max(sent.total, 1)) * 100} />
            <p className="text-xs text-slate-500">{fmtBytes(sent.loaded)} de {fmtBytes(sent.total)}{sent.loaded >= sent.total ? " · validando en el servidor…" : ""}</p>
          </div>
        )}
        {readOnly ? (
          <p className="mt-2 text-center text-sm text-amber-700 dark:text-amber-300">Tienes acceso de sólo lectura: no puedes cargar ni modificar datos en este proyecto.</p>
        ) : (
          scheduleLocked && <p className="mt-2 text-center text-sm text-amber-700 dark:text-amber-300">{project.training_lock ? "🌙 Este proyecto se está entrenando esta noche: podrás cargar cuando termine." : "Cargas bloqueadas por el calendario semanal."}</p>
        )}
        {result && (
          <div className="mt-3 space-y-1 text-sm">
            <Alert tone="success">{result.uploaded.length} archivo(s) cargado(s).</Alert>
            {result.duplicates.length > 0 && <Alert tone="warning">{result.duplicates.length} duplicado(s) omitido(s): {result.duplicates.slice(0, 5).map((d) => d.filename).join(", ")}</Alert>}
            {result.rejected.length > 0 && (
              <Alert tone="error">
                {result.rejected.length} rechazado(s):
                <ul className="ml-4 list-disc">{result.rejected.slice(0, 8).map((r, i) => <li key={i}>{r.filename}: {r.reason}</li>)}</ul>
              </Alert>
            )}
          </div>
        )}
      </div>

      {files.data && files.data.items.length > 0 && (
        <div className="card overflow-x-auto p-0">
          <div className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">Archivos ({files.data.items.length})</div>
          <table className="w-full">
            <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Archivo</th><th className="th">Filas</th><th className="th">Columnas</th><th className="th">Tamaño</th><th className="th">Subido</th><th className="th" /></tr></thead>
            <tbody>
              {files.data.items.map((f) => (
                <tr key={f.id} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="td max-w-[16rem] truncate font-medium" title={f.name}>{f.name}</td>
                  <td className="td">{f.height.toLocaleString("es-CL")}</td>
                  <td className="td">{f.width}</td>
                  <td className="td">{fmtBytes(f.size_bytes)}</td>
                  <td className="td whitespace-nowrap">{fmtDate(f.created_at)}</td>
                  <td className="td text-right">
                    <a className="btn btn-secondary btn-sm mr-1" href={`/api/images/${f.id}/file`} download={f.name} title="Descargar">⬇</a>
                    {!readOnly && <button className="btn btn-danger btn-sm" disabled={locked} onClick={() => void removeFile(f)} aria-label={`Eliminar ${f.name}`}>✕</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {summary.loading && !s && <Spinner />}

      {s && s.n_files > 0 && cols.length > 0 && (
        <div className="card space-y-5">
          <div>
            <h3 className="font-semibold">Columnas de la tabla</h3>
            <p className="text-sm text-slate-500">{s.n_rows.toLocaleString("es-CL")} filas · {s.n_columns} columnas. Elige qué quieres predecir y con qué columnas.</p>
          </div>

          <label className="block max-w-md">
            <span className="label">¿Qué quieres predecir?</span>
            <select className="w-full" value={target} disabled={locked || saving} onChange={(e) => changeTarget(e.target.value)}>
              <option value="">— elige una columna —</option>
              {cols.map((c) => (
                <option key={c.name} value={c.name}>{c.name} · {c.values ? `${c.unique} valores` : "demasiados valores"}{c.target_ok ? "" : " (poco adecuada)"}</option>
              ))}
            </select>
            <span className="mt-1 block text-xs text-slate-500">El modelo aprende a clasificar cada fila en uno de los valores de esta columna (entre 2 y {s.max_classes}).</span>
          </label>

          {s.classes && (
            <div>
              <div className="mb-1 text-sm font-medium">Categorías a predecir ({s.classes.filter((c) => c.used).length})</div>
              <div className="flex flex-wrap gap-1.5">
                {s.classes.map((c) => (
                  <span
                    key={c.label}
                    className={`rounded-full px-2 py-0.5 text-xs ${c.used ? "bg-brand-50 text-brand-700 dark:bg-brand-700/20 dark:text-brand-100" : "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"}`}
                    title={c.used ? `${c.count} filas` : `Menos de ${s.min_per_class} filas: no se usa al entrenar`}
                  >
                    {c.label} · {c.count.toLocaleString("es-CL")}{c.used ? "" : " (muy pocas)"}
                  </span>
                ))}
              </div>
              {s.classes.some((c) => !c.used) && <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">Los valores con menos de {s.min_per_class} filas no se usan al entrenar.</p>}
            </div>
          )}

          {target && (
            <div>
              <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-medium">Columnas de entrada ({chosen.size} de {inputs.length})</span>
                <span className="flex flex-wrap gap-2">
                  <button type="button" className="btn btn-secondary btn-sm" disabled={locked} onClick={() => { setChosen(new Set(s.suggested_features.filter((n) => n !== target))); setSaved(false); }}>Usar las sugeridas</button>
                  <button type="button" className="btn btn-secondary btn-sm" disabled={locked} onClick={() => { setChosen(new Set(inputs.map((c) => c.name))); setSaved(false); }}>Todas</button>
                </span>
              </div>
              <div className="max-h-96 overflow-auto rounded-lg border border-slate-200 dark:border-slate-800">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800"><tr><th className="th w-10" /><th className="th">Columna</th><th className="th">Tipo</th><th className="th">Distintos</th><th className="th">Vacíos</th><th className="th">Ejemplos</th></tr></thead>
                  <tbody>
                    {inputs.map((c) => (
                      <tr key={c.name} className={`border-t border-slate-100 dark:border-slate-800 ${chosen.has(c.name) ? "" : "text-slate-400"}`}>
                        <td className="td"><input type="checkbox" checked={chosen.has(c.name)} disabled={locked} onChange={() => toggle(c.name)} aria-label={`Usar ${c.name}`} /></td>
                        <td className="td max-w-[12rem] truncate font-medium" title={c.name}>{c.name}{note(c) && <span className="ml-2 text-xs font-normal text-amber-700 dark:text-amber-300">⚠ {note(c)}</span>}</td>
                        <td className="td whitespace-nowrap">{KIND_LABEL[c.kind]}</td>
                        <td className="td">{c.unique.toLocaleString("es-CL")}</td>
                        <td className="td">{c.missing ? pct(c.missing_pct, 0) : "—"}</td>
                        <td className="td max-w-[14rem] truncate text-xs text-slate-500" title={c.examples.join(" · ")}>{c.kind === "numeric" && c.min != null ? `${c.min} … ${c.max}` : c.examples.join(" · ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-1 text-xs text-slate-500">Las columnas que parecen identificadores (nombres, números correlativos) se omiten por defecto: el modelo las memorizaría en vez de aprender. Los valores vacíos se completan solos.</p>
            </div>
          )}

          {s.problems.length > 0 && target && s.problems.map((p) => <Alert key={p} tone="warning">{p}</Alert>)}
          {s.ready && <Alert tone="success">Listo para entrenar: {s.classes?.filter((c) => c.used).length} categorías, {s.features.length} columnas de entrada y {s.classes?.filter((c) => c.used).reduce((a, c) => a + c.count, 0).toLocaleString("es-CL")} filas. Ve a «Entrenamiento» para elegir los modelos.</Alert>}
          {saved && !dirty && <p className="text-sm text-emerald-700 dark:text-emerald-300">✔ Columnas guardadas.</p>}

          {!readOnly && target && (
            <div className="flex flex-wrap gap-2">
              <button className="btn btn-primary" disabled={locked || saving || !dirty || chosen.size === 0} onClick={() => void save(target, [...chosen])}>{saving ? "Guardando…" : "Guardar columnas"}</button>
              {chosen.size === 0 && <span className="self-center text-xs text-amber-700 dark:text-amber-300">Marca al menos una columna de entrada.</span>}
            </div>
          )}

          {s.preview && (
            <details className="rounded-lg border border-slate-200 dark:border-slate-800">
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium">Vista previa de las primeras filas</summary>
              <div className="max-h-72 overflow-auto p-3">
                <table className="text-xs">
                  <thead><tr>{s.preview.columns.map((c) => <th key={c} className="th whitespace-nowrap">{c}</th>)}</tr></thead>
                  <tbody>
                    {s.preview.rows.map((r, i) => (
                      <tr key={i} className="border-t border-slate-100 dark:border-slate-800">{r.map((v, j) => <td key={j} className="td whitespace-nowrap">{v}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      )}

      {s && s.n_files > 0 && cols.length === 0 && s.problems.length > 0 && <Alert tone="error">{s.problems.join(" ")}</Alert>}
    </div>
  );
}
