import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, errMsg } from "../lib/api";
import { fmtBytes } from "../lib/format";
import type { HfPreview, HfStatus, HfSuggestion, Project } from "../lib/types";
import { Alert, Badge, Field, Modal, Spinner, useAsync } from "./ui";

/** Todo lo que el usuario puede decidir al importar. Se manda completo en cada revisión y al importar. */
interface Cfg {
  sel: string; // auto | none | phrase | col:<columna>
  phCol: string;
  phAfter: string;
  phMode: "words" | "until";
  phWords: number;
  phUntil: string;
  imageCol: string;
  defaultCat: string;
  excluded: string[]; // categorías desmarcadas en la vista previa
  rest: "skip" | "other";
  otherLabel: string;
  maxItems: string;
  maxPer: string;
}

const EMPTY: Cfg = {
  sel: "auto", phCol: "", phAfter: "", phMode: "words", phWords: 1, phUntil: "", imageCol: "", defaultCat: "",
  excluded: [], rest: "skip", otherLabel: "otros", maxItems: "", maxPer: "",
};

const posInt = (s: string) => {
  const n = Math.floor(Number(s));
  return Number.isFinite(n) && n > 0 ? n : null;
};

function requestBody(source: string, cfg: Cfg, preview: HfPreview | null) {
  const body: Record<string, unknown> = {
    source,
    image_column: cfg.imageCol || null,
    default_category: cfg.defaultCat.trim() || null,
    max_items: posInt(cfg.maxItems),
    max_per_category: posInt(cfg.maxPer),
    rest: cfg.rest,
    other_label: cfg.otherLabel.trim() || null,
    label_source: "auto",
  };
  if (cfg.sel === "none") body.label_source = "none";
  else if (cfg.sel.startsWith("col:")) Object.assign(body, { label_source: "column", label_column: cfg.sel.slice(4) });
  else if (cfg.sel === "phrase" && cfg.phCol && cfg.phAfter.trim()) {
    Object.assign(body, {
      label_source: "phrase",
      label_phrase: {
        column: cfg.phCol,
        after: cfg.phAfter.trim(),
        words: cfg.phMode === "words" ? cfg.phWords : null,
        until: cfg.phMode === "until" && cfg.phUntil ? cfg.phUntil : null,
      },
    });
  }
  // sólo se manda la lista de incluidas si el usuario desmarcó alguna; así lo que no salga en la muestra sigue entrando
  const values = preview?.label_preview?.values ?? [];
  if (cfg.excluded.length > 0) body.include = values.map((v) => v.value).filter((v) => !cfg.excluded.includes(v));
  return body;
}

export default function ImportDialog({ project, onClose, onStarted }: { project: Project; onClose: () => void; onStarted: () => void }) {
  const { data: account } = useAsync(() => api<HfStatus>("/integrations/huggingface"), []);
  const [source, setSource] = useState("");
  const [cfg, setCfg] = useState<Cfg>(EMPTY);
  const [preview, setPreview] = useState<HfPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const lastRule = useRef("");
  const inspect = async (next: Cfg) => {
    // si cambió de dónde sale la categoría, las categorías desmarcadas antes ya no significan nada
    const rule = JSON.stringify([next.sel, next.phCol, next.phAfter, next.phMode, next.phWords, next.phUntil, next.imageCol]);
    if (rule !== lastRule.current) next = { ...next, excluded: [] };
    lastRule.current = rule;
    setCfg(next);
    setBusy(true);
    setError(null);
    try {
      const p = await api<HfPreview>(`/projects/${project.id}/hf/inspect`, { body: requestBody(source, next, preview) });
      setPreview(p);
      if (p.layout === "parquet" && !next.phCol && p.text_columns.length) setCfg((c) => ({ ...c, phCol: p.text_columns[0] }));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };
  const change = (patch: Partial<Cfg>, reinspect = true) => {
    const next = { ...cfg, ...patch };
    if (reinspect) void inspect(next);
    else setCfg(next);
  };

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      await api(`/projects/${project.id}/hf/imports`, { body: requestBody(source, cfg, preview) });
      onStarted();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const applySuggestion = (s: HfSuggestion) => change({ sel: "phrase", phCol: s.column, phAfter: s.after, phMode: "words", phWords: s.words, phUntil: "" });
  const toggle = (value: string) =>
    change({ excluded: cfg.excluded.includes(value) ? cfg.excluded.filter((v) => v !== value) : [...cfg.excluded, value] }, false);

  const isTable = preview?.layout === "parquet";
  const lp = preview?.label_preview ?? null;
  const unlabeledPossible = !!preview && ((isTable && preview.label_source === "none") || (!isTable && preview.unlabeled > 0) || (lp !== null && lp.unmatched > 0));
  const maxCount = Math.max(1, ...(lp?.values.map((v) => v.count) ?? [1]));
  const keptValues = lp ? lp.values.filter((v) => !cfg.excluded.includes(v.value)).length : 0;
  const size = preview ? preview.effective_bytes || preview.total_bytes : 0;

  return (
    <Modal title="Importar desde Hugging Face" onClose={onClose} wide>
      <div className="space-y-4">
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setPreview(null);
            void inspect(EMPTY);
          }}
        >
          <div className="min-w-[16rem] flex-1">
            <Field label="Dataset" hint="Escribe «usuario/nombre» o pega la dirección de huggingface.co/datasets/…">
              <input className="w-full" value={source} onChange={(e) => setSource(e.target.value)} placeholder="uoft-cs/cifar10" maxLength={300} required autoFocus />
            </Field>
          </div>
          <button className="btn btn-secondary" disabled={busy || !source.trim()}>{busy && !preview ? "Revisando…" : "Revisar"}</button>
        </form>
        {account && !account.connected && (
          <p className="text-xs text-slate-500">¿Es privado o restringido? <Link className="text-brand-600 underline" to="/cuenta" onClick={onClose}>Conecta tu cuenta de Hugging Face</Link> antes de revisarlo.</p>
        )}
        {error && <Alert tone="error">{error}</Alert>}
        {busy && !preview && !error && <Spinner />}

        {preview && (
          <div className={`space-y-4 ${busy ? "opacity-70" : ""}`} aria-busy={busy}>
            <div className="flex flex-wrap items-center gap-2">
              <strong>{preview.repo_id}</strong>
              {preview.private && <Badge status="PENDING">Privado</Badge>}
              {preview.gated && <Badge status="PENDING">Restringido</Badge>}
              <span className="font-mono text-xs text-slate-500">versión {preview.revision.slice(0, 7)}</span>
              {busy && <Spinner />}
            </div>
            <div className="grid gap-2 text-sm sm:grid-cols-3">
              <div className="rounded-lg bg-slate-100 p-3 dark:bg-slate-800">
                <div className="text-xs text-slate-500">Tamaño a descargar</div>
                <div className="font-semibold">{fmtBytes(size)}</div>
                {size < preview.total_bytes && <div className="text-xs text-slate-500">de {fmtBytes(preview.total_bytes)} (por tu máximo de imágenes)</div>}
              </div>
              <div className="rounded-lg bg-slate-100 p-3 dark:bg-slate-800"><div className="text-xs text-slate-500">Imágenes</div><div className="font-semibold">{preview.items != null ? preview.items.toLocaleString("es-CL") : "se contarán al descargar"}</div></div>
              <div className="rounded-lg bg-slate-100 p-3 dark:bg-slate-800"><div className="text-xs text-slate-500">Categorías</div><div className="font-semibold">{lp ? lp.distinct.toLocaleString("es-CL") : preview.category_count ?? "—"}{lp && !lp.exact && <span className="text-xs font-normal text-slate-500"> (en la muestra)</span>}</div></div>
            </div>
            <p className="text-xs text-slate-500">
              {isTable ? "Tablas parquet: cada fila trae su imagen y, si hay, datos que pueden dar la categoría." : "Carpetas con imágenes: cada carpeta será una categoría (train/test se juntan)."}
            </p>

            {isTable && preview.image_columns.length > 1 && (
              <Field label="Columna de imagen">
                <select value={preview.image_column ?? ""} disabled={busy} onChange={(e) => change({ imageCol: e.target.value })}>
                  {preview.image_columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </Field>
            )}

            {isTable && (
              <section className="space-y-3 rounded-lg border border-slate-200 p-3 dark:border-slate-700" aria-label="Origen de la categoría">
                <h4 className="text-sm font-semibold">¿De dónde sale la categoría de cada imagen?</h4>
                <select className="w-full sm:w-auto" value={cfg.sel} disabled={busy} onChange={(e) => change({ sel: e.target.value })} aria-label="Origen de la categoría">
                  <option value="auto">Lo que declare el dataset{preview.label_source === "dataset" && preview.label_column ? ` (columna «${preview.label_column}»)` : " (no declara ninguna)"}</option>
                  {preview.label_columns.map((c) => <option key={c} value={`col:${c}`}>Columna «{c}»</option>)}
                  {preview.text_columns.length > 0 && <option value="phrase">Extraída de un texto…</option>}
                  <option value="none">Ninguna: todas a una misma categoría</option>
                </select>

                {cfg.sel === "phrase" && (
                  <div className="space-y-2">
                    <div className="grid gap-2 sm:grid-cols-2">
                      <Field label="Columna de texto">
                        <select value={cfg.phCol} onChange={(e) => change({ phCol: e.target.value }, false)}>
                          {preview.text_columns.map((c) => <option key={c} value={c}>{c}</option>)}
                        </select>
                      </Field>
                      <Field label="Tomar lo que sigue a la frase" hint="Por ejemplo «of type» en «… of type Grass …». No distingue mayúsculas.">
                        <input className="w-full" value={cfg.phAfter} maxLength={80} placeholder="of type" onChange={(e) => change({ phAfter: e.target.value }, false)} />
                      </Field>
                    </div>
                    <div className="flex flex-wrap items-end gap-3 text-sm">
                      <label className="flex items-center gap-1.5"><input type="radio" checked={cfg.phMode === "words"} onChange={() => change({ phMode: "words" }, false)} /> las siguientes
                        <input className="w-16" type="number" min={1} max={6} value={cfg.phWords} onChange={(e) => change({ phWords: Math.min(6, Math.max(1, Number(e.target.value) || 1)) }, false)} /> palabra(s)</label>
                      <label className="flex items-center gap-1.5"><input type="radio" checked={cfg.phMode === "until"} onChange={() => change({ phMode: "until" }, false)} /> todo hasta
                        <input className="w-40" value={cfg.phUntil} maxLength={40} placeholder=" with the" disabled={cfg.phMode !== "until"} onChange={(e) => change({ phUntil: e.target.value }, false)} /></label>
                      <button className="btn btn-secondary btn-sm" disabled={busy || !cfg.phAfter.trim() || !cfg.phCol || (cfg.phMode === "until" && !cfg.phUntil)} onClick={() => change({})}>Aplicar</button>
                    </div>
                  </div>
                )}

                {preview.suggestions.length > 0 && (cfg.sel === "phrase" || preview.label_source === "none") && (
                  <div className="space-y-1.5">
                    <div className="text-xs font-medium text-slate-600 dark:text-slate-400">Sugerencias encontradas en los textos (haz clic para usar una):</div>
                    <div className="flex flex-wrap gap-1.5">
                      {preview.suggestions.map((s) => (
                        <button key={`${s.column}:${s.after}`} className="rounded-lg border border-slate-300 px-2.5 py-1 text-left text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800" disabled={busy} onClick={() => applySuggestion(s)}>
                          <span className="font-mono">«{s.after}»</span> en <em>{s.column}</em> · {s.distinct} valores
                          <span className="block text-slate-500">{s.values.slice(0, 4).map((v) => v.value).join(", ")}{s.distinct > 4 ? "…" : ""}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            {lp && lp.values.length > 0 && (
              <section className="space-y-2" aria-label="Reparto de categorías">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h4 className="text-sm font-semibold">Categorías que se crearán</h4>
                  <span className="text-xs text-slate-500">
                    {lp.exact ? "Reparto exacto del dataset" : `Reparto en una muestra de ${lp.sample_rows.toLocaleString("es-CL")} filas: el total puede variar`}
                    {lp.unmatched > 0 && ` · ${lp.unmatched.toLocaleString("es-CL")} sin categoría`}
                  </span>
                </div>
                <div className="max-h-60 overflow-auto rounded-lg border border-slate-200 dark:border-slate-700">
                  {lp.values.map((v) => (
                    <label key={v.value} className="flex cursor-pointer items-center gap-2 border-b border-slate-100 px-3 py-1.5 text-sm last:border-0 dark:border-slate-800">
                      <input type="checkbox" checked={!cfg.excluded.includes(v.value)} onChange={() => toggle(v.value)} />
                      <span className="w-40 shrink-0 truncate" title={v.value}>{v.value}</span>
                      <span className="h-2 flex-1 overflow-hidden rounded bg-slate-100 dark:bg-slate-800"><span className="block h-full rounded bg-brand-500" style={{ width: `${(v.count / maxCount) * 100}%` }} /></span>
                      <span className="w-16 shrink-0 text-right tabular-nums text-xs text-slate-500">{v.count.toLocaleString("es-CL")}</span>
                    </label>
                  ))}
                </div>
                {lp.distinct > lp.values.length && <p className="text-xs text-slate-500">Se muestran las {lp.values.length} más frecuentes de {lp.distinct}. Si desmarcas alguna, las que no aparecen aquí también quedan fuera.</p>}
                {cfg.excluded.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 text-sm">
                    <span>{keptValues} marcada(s). Las demás:</span>
                    <select value={cfg.rest} onChange={(e) => change({ rest: e.target.value as "skip" | "other" }, false)}>
                      <option value="skip">no se importan</option>
                      <option value="other">se juntan en una categoría</option>
                    </select>
                    {cfg.rest === "other" && <input className="w-40" value={cfg.otherLabel} maxLength={80} onChange={(e) => change({ otherLabel: e.target.value }, false)} aria-label="Nombre de la categoría para el resto" />}
                    <button className="btn btn-secondary btn-sm" disabled={busy} onClick={() => change({})}>Comprobar</button>
                  </div>
                )}
              </section>
            )}

            <section className="space-y-2" aria-label="Límites">
              <h4 className="text-sm font-semibold">Límites (opcionales)</h4>
              <div className="flex flex-wrap gap-3">
                <Field label="Máximo de imágenes en total" hint="Útil para probar con una muestra antes de bajarlo todo.">
                  <input className="w-40" type="number" min={1} value={cfg.maxItems} placeholder="sin límite" onChange={(e) => change({ maxItems: e.target.value }, false)} onBlur={() => void inspect(cfg)} />
                </Field>
                <Field label="Máximo por categoría" hint="Evita que una categoría domine el dataset.">
                  <input className="w-40" type="number" min={1} value={cfg.maxPer} placeholder="sin límite" onChange={(e) => change({ maxPer: e.target.value }, false)} onBlur={() => void inspect(cfg)} />
                </Field>
              </div>
            </section>

            {unlabeledPossible && (
              <Field label="Categoría de destino" hint={isTable && preview.label_source === "none" ? "Sin categoría de origen: todas las imágenes irán a esta categoría." : "Las imágenes que no tengan categoría irán a esta (si la dejas vacía se omiten)."}>
                <input className="w-full" value={cfg.defaultCat} maxLength={80} onChange={(e) => change({ defaultCat: e.target.value }, false)} onBlur={() => void inspect(cfg)} placeholder="Nombre de la categoría" />
              </Field>
            )}

            {preview.warnings.map((w) => <Alert key={w} tone="warning">{w}</Alert>)}
            {preview.blockers.map((b) => <Alert key={b} tone="error">{b}</Alert>)}
            <p className="text-xs text-slate-500">Las imágenes se validan igual que una carga manual; las repetidas (mismo contenido) se omiten. Puedes cancelar en cualquier momento y lo ya importado se conserva.</p>
            <div className="flex flex-wrap justify-end gap-2">
              <button className="btn btn-secondary" onClick={onClose}>Cerrar</button>
              <button className="btn btn-primary" disabled={busy || !preview.can_import} onClick={() => void start()}>{busy ? "Un momento…" : `Importar ${fmtBytes(size)}`}</button>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}
