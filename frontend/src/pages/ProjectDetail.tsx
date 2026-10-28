import { useCallback, useEffect, useRef, useState, type DragEvent, type FormEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, apiUpload, errMsg } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useJobs } from "../lib/jobs";
import { BASE_LABEL, baseLabels, fmtBytes, fmtDate, fmtDuration, fmtSeconds, fmtWhen, nightNote, pct } from "../lib/format";
import { extensionRegex, useModality } from "../lib/modalities";
import { HfImport } from "../components/HuggingFace";
import ModelPicker from "../components/ModelPicker";
import ModelOverrides from "../components/ModelOverrides";
import ShareTab, { ROLE_LABEL } from "../components/ShareTab";
import TabularTab from "../components/TabularTab";
import { deleteModels, deleteSummary } from "../lib/models";
import type { ImageItem, ModelInfo, Project, TrainingConfig } from "../lib/types";
import JobCard from "../components/JobCard";
import ModelPanel from "../components/ModelPanel";
import { Alert, Badge, Field, Modal, PageTitle, Progress, Spinner, Tabs, useAsync } from "../components/ui";

type Item = { file: File; path: string };
type UploadResult = { uploaded: unknown[]; duplicates: { filename: string; reason: string }[]; rejected: { filename: string; reason: string }[] };

function ImagesTab({ project, reload }: { project: Project; reload: () => void }) {
  const { schedule } = useJobs();
  // bloqueo general (calendario semanal) o sólo de este proyecto (se está entrenando esta noche)
  const scheduleLocked = (schedule ? !schedule.open : false) || !!project.training_lock;
  const readOnly = project.my_role === "viewer";
  const isOwner = project.my_role === "owner";
  // se bloquea todo lo que modifica datos: por el calendario semanal o porque el acceso es de sólo lectura
  const locked = scheduleLocked || readOnly;
  const [filter, setFilter] = useState<number | "all">("all");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<{ total: number; items: ImageItem[]; page_size: number } | null>(null);
  const [target, setTarget] = useState<number | "">(project.categories[0]?.id ?? "");
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [newCat, setNewCat] = useState("");
  const [selected, setSelected] = useState<ImageItem | null>(null);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<{ files: number; total: number; sent: number; totalBytes: number; speed: number; eta: number | null; processing: boolean } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const [staged, setStaged] = useState<Item[]>([]);
  const [catQuery, setCatQuery] = useState("");
  const [catSort, setCatSort] = useState<"name" | "few" | "many">("name");
  const [showAllCats, setShowAllCats] = useState(false);
  const CAT_LIMIT = 60;
  const matchedCats = project.categories
    .filter((c) => c.name.toLowerCase().includes(catQuery.trim().toLowerCase()))
    .sort((a, b) => (catSort === "name" ? a.name.localeCompare(b.name) : catSort === "few" ? a.image_count - b.image_count : b.image_count - a.image_count));
  const visibleCats = showAllCats ? matchedCats : matchedCats.slice(0, CAT_LIMIT);
  const hiddenCats = matchedCats.length - visibleCats.length;
  const minPerClass = project.min_images_per_class ?? 10;
  const lowCats = project.categories.filter((c) => c.image_count < minPerClass).length;

  const load = useCallback(() => {
    const q = new URLSearchParams({ page: String(page), page_size: "48" });
    if (filter !== "all") q.set("category_id", String(filter));
    api<typeof data>(`/projects/${project.id}/images?${q}`).then(setData).catch((e) => setError(errMsg(e)));
  }, [project.id, filter, page]);
  useEffect(load, [load]);
  useEffect(() => {
    if (target === "" && project.categories.length) setTarget(project.categories[0].id);
  }, [project.categories, target]);

  const uploadGroups = async (groups: { catId: number; files: File[] }[]) => {
    // lotes de una sola categoría, en orden, con progreso global
    const batches: { catId: number; files: File[] }[] = [];
    const BATCH = 20;
    // lotes de una sola categoría: máx. 20 archivos y ~40 MB (Cloudflare rechaza peticiones > 100 MB)
    const MAX_BATCH_BYTES = 40 * 1024 * 1024;
    for (const g of groups) {
      let cur: File[] = [];
      let bytes = 0;
      for (const f of g.files) {
        if (cur.length && (cur.length >= BATCH || bytes + f.size > MAX_BATCH_BYTES)) {
          batches.push({ catId: g.catId, files: cur });
          cur = [];
          bytes = 0;
        }
        cur.push(f);
        bytes += f.size;
      }
      if (cur.length) batches.push({ catId: g.catId, files: cur });
    }
    const all = groups.flatMap((g) => g.files);
    if (!all.length) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const agg: UploadResult = { uploaded: [], duplicates: [], rejected: [] };
      const totalBytes = all.reduce((a, f) => a + f.size, 0);
      const t0 = performance.now();
      let doneBytes = 0;
      let doneFiles = 0;
      const report =(sentInBatch: number, filesInBatch: number, batchBytes: number, processing: boolean) => {
        const sent = doneBytes + sentInBatch;
        const secs = Math.max((performance.now() - t0) / 1000, 0.1);
        const speed = sent / secs;
        const fraction = batchBytes ? sentInBatch / batchBytes : 1;
        setProgress({
          files: Math.min(all.length, doneFiles + Math.floor(filesInBatch * fraction)),
          total: all.length,
          sent,
          totalBytes,
          speed,
          eta: speed > 0 ? (totalBytes - sent) / speed : null,
          processing,
        });
      };
      report(0, 0, 0, false);
      for (const { catId, files: chunk } of batches) {
        const batchBytes = chunk.reduce((a, f) => a + f.size, 0);
        const form = new FormData();
        form.append("category_id", String(catId));
        chunk.forEach((f) => form.append("files", f));
        const r = await apiUpload<UploadResult>(`/projects/${project.id}/images`, form, (loaded, total) =>
          report(Math.min(loaded, batchBytes), chunk.length, batchBytes || total, loaded >= total),
        );
        doneBytes += batchBytes;
        doneFiles += chunk.length;
        report(0, 0, 0, false);
        agg.uploaded.push(...r.uploaded);
        agg.duplicates.push(...r.duplicates);
        agg.rejected.push(...r.rejected);
      }
      setResult(agg);
      load();
      reload();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
      setProgress(null);
      if (fileInput.current) fileInput.current.value = "";
      if (folderInput.current) folderInput.current.value = "";
    }
  };
  const mod = useModality(project.modality);
  const IMG = extensionRegex(mod);
  const noun = mod.unit;
  const resetInputs = () => {
    if (fileInput.current) fileInput.current.value = "";
    if (folderInput.current) folderInput.current.value = "";
  };

  /** Caja única: archivos sueltos van a la categoría elegida (o se pide una); carpeta = categoría;
   *  carpeta con subcarpetas = una categoría por subcarpeta. */
  const handleItems = async (items: Item[]) => {
    if (locked || busy) return;
    const byCat = new Map<string, File[]>();
    let ignored = 0;
    let loose: File[] = [];
    for (const { file, path } of items) {
      if (!IMG.test(file.name)) {
        ignored++;
        continue;
      }
      const parts = path.split("/").filter(Boolean);
      if (parts.length <= 1) loose.push(file);
      else {
        const name = (parts.length >= 3 ? parts[1] : parts[0]).trim();
        byCat.set(name, [...(byCat.get(name) ?? []), file]);
      }
    }
    if (loose.length) {
      let name = project.categories.find((c) => c.id === target)?.name;
      if (!name) {
        name = (prompt(`Hay ${loose.length} ${noun} sueltos. ¿En qué categoría los guardo? (se creará si no existe)`) ?? "").trim();
        if (!name) {
          resetInputs();
          return;
        }
      }
      byCat.set(name, [...(byCat.get(name) ?? []), ...loose]);
      loose = [];
    }
    if (!byCat.size) {
      setError("No se encontraron imágenes JPG, PNG o WEBP.");
      resetInputs();
      return;
    }
    const existing = new Map(project.categories.map((c) => [c.name.toLowerCase(), c.id]));
    setError(null);
    try {
      const groups: { catId: number; files: File[] }[] = [];
      for (const [name, files] of byCat) {
        let id = existing.get(name.toLowerCase());
        if (id === undefined) {
          const c = await api<{ id: number }>(`/projects/${project.id}/categories`, { body: { name } });
          id = c.id;
          existing.set(name.toLowerCase(), id);
        }
        groups.push({ catId: id, files });
      }
      reload();
      await uploadGroups(groups);
      setStaged([]);
    } catch (e) {
      setError(errMsg(e));
      resetInputs();
    }
  };

  // Lee archivos y carpetas arrastrados (recorre las carpetas recursivamente)
  const readDropped = async (dt: DataTransfer): Promise<Item[]> => {
    const out: Item[] = [];
    const walk = async (entry: FileSystemEntry): Promise<void> => {
      if (entry.isFile) {
        const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
        out.push({ file, path: entry.fullPath.replace(/^\//, "") });
      } else if (entry.isDirectory) {
        const reader = (entry as FileSystemDirectoryEntry).createReader();
        for (;;) {
          const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
          if (!batch.length) break;
          for (const e of batch) await walk(e);
        }
      }
    };
    const entries = Array.from(dt.items).map((i) => i.webkitGetAsEntry?.()).filter((e): e is FileSystemEntry => !!e);
    if (!entries.length) return Array.from(dt.files).map((file) => ({ file, path: file.name }));
    for (const e of entries) await walk(e);
    return out;
  };
  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (locked) return;
    const items = await readDropped(e.dataTransfer);
    addStaged(items);
  };
  const addStaged = (items: Item[]) => {
    setError(null);
    setResult(null);
    setStaged((prev) => {
      const seen = new Set(prev.map((i) => `${i.path}|${i.file.size}|${i.file.lastModified}`));
      return [...prev, ...items.filter((i) => !seen.has(`${i.path}|${i.file.size}|${i.file.lastModified}`))];
    });
    resetInputs();
  };
  const fromInput = (list: FileList | null) =>
    list && addStaged(Array.from(list).map((file) => ({ file, path: file.webkitRelativePath || file.name })));
  // resumen de la lista de espera: categoría -> cantidad
  const stagedSummary = (() => {
    const m = new Map<string, number>();
    let loose = 0;
    let ignored = 0;
    for (const { file, path } of staged) {
      if (!IMG.test(file.name)) { ignored++; continue; }
      const parts = path.split("/").filter(Boolean);
      if (parts.length <= 1) loose++;
      else { const n = (parts.length >= 3 ? parts[1] : parts[0]).trim(); m.set(n, (m.get(n) ?? 0) + 1); }
    }
    return { m, loose, ignored, total: staged.length - ignored };
  })();
  const addCategory = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api(`/projects/${project.id}/categories`, { body: { name: newCat } });
      setNewCat("");
      reload();
    } catch (err) {
      setError(errMsg(err));
    }
  };
  const removeCategory = async (id: number, name: string) => {
    if (!confirm(`¿Eliminar la categoría «${name}» y todas sus imágenes?`)) return;
    try {
      await api(`/projects/${project.id}/categories/${id}`, { method: "DELETE" });
      setFilter("all");
      reload();
      load();
    } catch (err) {
      setError(errMsg(err));
    }
  };
  const removeImage = async (img: ImageItem) => {
    if (!confirm("¿Eliminar esta imagen?")) return;
    try {
      await api(`/images/${img.id}`, { method: "DELETE" });
      setSelected(null);
      load();
      reload();
    } catch (err) {
      setError(errMsg(err));
    }
  };
  const move = async (img: ImageItem, category_id: number) => {
    try {
      await api(`/images/${img.id}`, { method: "PATCH", body: { category_id } });
      setSelected(null);
      load();
      reload();
    } catch (err) {
      setError(errMsg(err));
    }
  };
  const catName = (id: number) => project.categories.find((c) => c.id === id)?.name ?? "";
  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  return (
    <div className="space-y-6">
      {error && <Alert tone="error">{error}</Alert>}
      <div className="card">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-semibold">Categorías ({project.categories.length})</h3>
          {project.categories.length > 12 && (
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <input type="search" placeholder="Buscar categoría…" value={catQuery} onChange={(e) => { setCatQuery(e.target.value); setShowAllCats(false); }} aria-label="Buscar categoría" />
              <select value={catSort} onChange={(e) => setCatSort(e.target.value as "name" | "few" | "many")} aria-label="Ordenar categorías">
                <option value="name">Nombre</option>
                <option value="few">Menos imágenes primero</option>
                <option value="many">Más imágenes primero</option>
              </select>
            </div>
          )}
        </div>
        {lowCats > 0 && <p className="mb-2 text-xs text-amber-700 dark:text-amber-300">{lowCats} categoría(s) tienen menos de {minPerClass} imágenes y no se usarán al entrenar.</p>}
        <div className="flex flex-wrap gap-2">
          <button className={`btn btn-sm ${filter === "all" ? "btn-primary" : "btn-secondary"}`} onClick={() => { setFilter("all"); setPage(1); }}>Todas · {project.image_count}</button>
          {visibleCats.map((c) => (
            <span key={c.id} className="inline-flex">
              <button className={`btn btn-sm ${isOwner ? "rounded-r-none" : ""} ${filter === c.id ? "btn-primary" : "btn-secondary"} ${c.image_count < minPerClass ? "text-amber-700 dark:text-amber-300" : ""}`} onClick={() => { setFilter(c.id); setPage(1); }}>{c.name} · {c.image_count}</button>
              {isOwner && <button className="btn btn-secondary btn-sm rounded-l-none border-l-0" disabled={locked} title="Eliminar categoría" aria-label={`Eliminar categoría ${c.name}`} onClick={() => removeCategory(c.id, c.name)}>✕</button>}
            </span>
          ))}
          {hiddenCats > 0 && <button className="btn btn-secondary btn-sm" onClick={() => setShowAllCats(true)}>Mostrar {hiddenCats} más…</button>}
          {matchedCats.length === 0 && <span className="text-sm text-slate-500">Ninguna categoría coincide.</span>}
        </div>
        {!readOnly && (
          <form className="mt-3 flex gap-2" onSubmit={addCategory}>
            <input value={newCat} onChange={(e) => setNewCat(e.target.value)} placeholder="Nueva categoría" disabled={locked} maxLength={80} required />
            <button className="btn btn-secondary" disabled={locked}>Añadir</button>
          </form>
        )}
      </div>

      {!readOnly && (project.modality ?? "image") === "image" && <HfImport project={project} locked={locked} onChanged={() => { load(); reload(); }} />}

      <div
        className={`card border-2 border-dashed ${drag ? "border-brand-500 bg-brand-50 dark:bg-brand-700/10" : ""} ${locked ? "opacity-60" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => void onDrop(e)}
      >
        <div className="flex flex-col items-center gap-3 py-4 text-center">
          <div className="text-4xl" aria-hidden="true">📁</div>
          <p className="font-medium">Arrastra aquí {noun} o carpetas</p>
          {mod.upload_help && <p className="text-xs text-slate-500">{mod.upload_help}</p>}
          <p className="max-w-xl text-xs text-slate-500">
            Una carpeta se carga como categoría con el nombre de la carpeta (se crea si no existe). Puedes agregar varias carpetas (arrastrando varias a la vez o con «Agregar carpeta» repetidas veces) y subirlas juntas. Si una carpeta tiene subcarpetas, cada subcarpeta será una categoría.
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            <button type="button" className="btn btn-primary" disabled={locked || busy} onClick={() => fileInput.current?.click()}>Agregar {noun}</button>
            <button type="button" className="btn btn-secondary" disabled={locked || busy} onClick={() => folderInput.current?.click()}>Agregar carpeta</button>
          </div>
          {project.categories.length > 0 && (
            <label className="flex items-center gap-2 text-xs text-slate-500">
              Archivos sueltos van a:
              <select value={target} onChange={(e) => setTarget(Number(e.target.value))} disabled={locked || busy}>
                {project.categories.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            </label>
          )}
          {staged.length > 0 && (
            <div className="w-full max-w-xl rounded-lg border border-slate-200 bg-white p-3 text-left text-sm dark:border-slate-700 dark:bg-slate-900">
              <div className="mb-2 font-medium">Listo para subir: {stagedSummary.total} {noun}</div>
              <ul className="mb-2 space-y-0.5">
                {Array.from(stagedSummary.m, ([n, c]) => (
                  <li key={n}>📁 {n}: {c} {noun} {project.categories.some((x) => x.name.toLowerCase() === n.toLowerCase()) ? "" : <span className="text-xs text-emerald-600">(categoría nueva)</span>}</li>
                ))}
                {stagedSummary.loose > 0 && <li>🖼 {stagedSummary.loose} {noun} sueltos → {project.categories.find((c) => c.id === target)?.name ?? "se pedirá la categoría"}</li>}
              </ul>
              {stagedSummary.ignored > 0 && <p className="mb-2 text-xs text-slate-500">{stagedSummary.ignored} archivos que no son JPG/PNG/WEBP se ignorarán.</p>}
              <div className="flex flex-wrap gap-2">
                <button type="button" className="btn btn-primary" disabled={busy || locked || stagedSummary.total === 0} onClick={() => void handleItems(staged)}>Subir todo</button>
                <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => setStaged([])}>Vaciar lista</button>
              </div>
            </div>
          )}
          <input ref={fileInput} type="file" multiple accept={mod.accept} className="hidden" onChange={(e) => { fromInput(e.target.files); }} />
          <input
            ref={folderInput}
            type="file"
            // @ts-expect-error atributo no estándar soportado por Chrome, Edge y Firefox
            webkitdirectory=""
            directory=""
            multiple
            className="hidden"
            onChange={(e) => { fromInput(e.target.files); }}
          />
          {busy && <Spinner />}
        </div>
        <p className="text-center text-xs text-slate-500">Se detectan duplicados por contenido y se valida cada archivo en el servidor.</p>
        {progress && (
          <div className="mt-4 space-y-2" aria-live="polite">
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
              <strong>{progress.files} de {progress.total} imágenes cargadas</strong>
              <span>{Math.round((progress.sent / Math.max(progress.totalBytes, 1)) * 100)}%</span>
            </div>
            <Progress value={(progress.sent / Math.max(progress.totalBytes, 1)) * 100} />
            <div className="flex flex-wrap gap-x-5 text-xs text-slate-500">
              <span>Velocidad: {fmtBytes(progress.speed)}/s</span>
              <span>{fmtBytes(progress.sent)} de {fmtBytes(progress.totalBytes)}</span>
              <span>Restante: {progress.eta != null ? fmtDuration(progress.eta) : "—"}</span>
              {progress.processing && <span>Validando en el servidor…</span>}
            </div>
          </div>
        )}
        {readOnly ? (
          <p className="mt-2 text-center text-sm text-amber-700 dark:text-amber-300">Tienes acceso de sólo lectura: no puedes cargar ni modificar datos en este proyecto.</p>
        ) : (
          scheduleLocked && <p className="mt-2 text-sm text-amber-700 dark:text-amber-300">{project.training_lock ? "🌙 Este proyecto se está entrenando esta noche: podrás cargar cuando termine." : "Cargas bloqueadas por el calendario semanal."}</p>
        )}
        {result && (
          <div className="mt-3 space-y-1 text-sm">
            <Alert tone="success">{result.uploaded.length} imagen(es) cargada(s).</Alert>
            {result.duplicates.length > 0 && <Alert tone="warning">{result.duplicates.length} duplicada(s) omitida(s): {result.duplicates.slice(0, 5).map((d) => d.filename).join(", ")}</Alert>}
            {result.rejected.length > 0 && (
              <Alert tone="error">
                {result.rejected.length} rechazada(s):
                <ul className="ml-4 list-disc">{result.rejected.slice(0, 8).map((r, i) => <li key={i}>{r.filename}: {r.reason}</li>)}</ul>
              </Alert>
            )}
          </div>
        )}
      </div>

      <div>
        <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
          {data?.items.map((img) => (
            <button key={img.id} onClick={() => setSelected(img)} className="group relative aspect-square overflow-hidden rounded-lg border border-slate-200 dark:border-slate-800" title={img.name}>
              <img src={`/api/images/${img.id}/thumb`} alt={img.name} loading="lazy" className="h-full w-full object-cover transition group-hover:scale-105" />
              {img.duration_s != null && <span className="absolute right-1 top-1 rounded bg-black/60 px-1 text-[10px] text-white">🎵 {fmtSeconds(img.duration_s)}</span>}
              <span className="absolute bottom-0 left-0 right-0 truncate bg-black/55 px-1 text-[10px] text-white">{catName(img.category_id)}</span>
            </button>
          ))}
        </div>
        {data && data.items.length === 0 && <p className="py-8 text-center text-sm text-slate-500">No hay imágenes en esta vista.</p>}
        {pages > 1 && (
          <div className="mt-4 flex items-center justify-center gap-3 text-sm">
            <button className="btn btn-secondary btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>Anterior</button>
            <span>Página {page} de {pages}</span>
            <button className="btn btn-secondary btn-sm" disabled={page >= pages} onClick={() => setPage(page + 1)}>Siguiente</button>
          </div>
        )}
      </div>

      {selected && (
        <Modal title={selected.name} onClose={() => setSelected(null)} wide>
          {mod.preview === "audio" ? (
            <div className="space-y-3">
              <img src={`/api/images/${selected.id}/thumb`} alt="Espectrograma" className="mx-auto max-h-64 w-full rounded-lg object-fill" />
              <audio controls src={`/api/images/${selected.id}/file`} className="w-full" />
            </div>
          ) : (
            <img src={`/api/images/${selected.id}/file`} alt={selected.name} className="mx-auto max-h-[55vh] rounded-lg object-contain" />
          )}
          <p className="mt-3 text-sm text-slate-500">{selected.duration_s != null ? `${fmtSeconds(selected.duration_s)} min` : `${selected.width}×${selected.height}`} · {selected.format.toUpperCase()} · {fmtBytes(selected.size_bytes)} · {fmtDate(selected.created_at)}</p>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <select value={selected.category_id} disabled={locked} onChange={(e) => move(selected, Number(e.target.value))} aria-label="Mover a categoría">
              {project.categories.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <button className="btn btn-danger" disabled={locked} onClick={() => removeImage(selected)}>Eliminar imagen</button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function TrainingTab({ project, reload }: { project: Project; reload: () => void }) {
  const { user } = useAuth();
  const { jobs, schedule, reload: reloadJobs } = useJobs();
  const mine = jobs.filter((j) => j.project_id === project.id);
  const mod = useModality(project.modality);
  const [cfg, setCfg] = useState<TrainingConfig>(project.config);
  const [models, setModels] = useState<string[]>(project.base_models?.length ? project.base_models : [project.base_model]);
  const [auto, setAuto] = useState(project.auto_train);
  useEffect(() => setAuto(project.auto_train), [project.auto_train]); // p. ej. se desmarcó solo al terminar la semana
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const readOnly = project.my_role === "viewer";
  const num = (k: keyof TrainingConfig) => (e: { target: { value: string } }) => setCfg({ ...cfg, [k]: Number(e.target.value) });
  const save = async (e: FormEvent) => {
    e.preventDefault();
    try {
      // los ajustes avanzados de modelos que ya no están seleccionados se descartan
      const overrides = Object.fromEntries(Object.entries(cfg.model_overrides ?? {}).filter(([m]) => models.includes(m)));
      const saved = await api<Project>(`/projects/${project.id}`, { method: "PATCH", body: { base_models: models, config: { ...cfg, model_overrides: overrides }, auto_train: auto } });
      setMsg(
        saved.auto_train
          ? { tone: "success", text: `Configuración guardada. Este proyecto se entrenará el ${fmtWhen(saved.next_training_start)} (hora de Chile)${nightNote(saved.night_window)}.` }
          : { tone: "success", text: "Configuración guardada. Este proyecto NO está marcado, así que no se entrenará automáticamente: márcalo y guarda para incluirlo en el próximo entrenamiento." },
      );
      reload();
    } catch (err) {
      setMsg({ tone: "error", text: errMsg(err) });
    }
  };
  const trainNow = async () => {
    if (!confirm("Esto congelará el dataset ahora, bloqueará las cargas de todos los usuarios hasta que termine el entrenamiento y lo pondrá en la cola. ¿Continuar?")) return;
    try {
      await api("/admin/training/enqueue", { body: { project_ids: [project.id] } });
      reloadJobs();
      setMsg({ tone: "success", text: "Entrenamiento encolado." });
    } catch (err) {
      setMsg({ tone: "error", text: errMsg(err) });
    }
  };
  return (
    <div className="space-y-6">
      <form className="card space-y-4" onSubmit={save}>
        <h3 className="font-semibold">Configuración de entrenamiento</h3>
        {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}
        {project.auto_train ? (
          <Alert tone="success">Este proyecto está marcado: se entrenará el <strong>{fmtWhen(project.next_training_start)}</strong> (hora de Chile){nightNote(project.night_window)}.</Alert>
        ) : (
          <Alert tone="warning">
            <strong>Este proyecto no está marcado para el entrenamiento semanal.</strong> Revisa los resultados, ajusta el modelo y, si quieres que se entrene, marca «Incluir en el entrenamiento semanal automático» y guarda: te avisaremos la fecha de ejecución (la próxima ventana es el {fmtWhen(schedule?.next_training_start)}).
          </Alert>
        )}
        {readOnly && <Alert tone="info">Tienes acceso de sólo lectura: puedes ver la configuración, pero sólo el propietario o un editor puede cambiarla.</Alert>}
        {project.my_role === "editor" && <Alert tone="info">Eres editor de este proyecto de @{project.owner}: los cambios que guardes aquí también los ve el propietario.</Alert>}
        <fieldset disabled={readOnly} className="space-y-4">
        <div>
          <span className="label">Modelos base</span>
          <ModelPicker value={models} onChange={setModels} disabled={readOnly} tabular={mod.tabular} />
        </div>
        {mod.tabular && (
          <Alert tone="info">
            {cfg.tabular_target
              ? <>Columna a predecir: <strong>{cfg.tabular_target}</strong>{cfg.tabular_features?.length ? ` · ${cfg.tabular_features.length} columnas de entrada` : " · columnas de entrada sugeridas"}. Se cambian en la pestaña «Datos».</>
              : <>Todavía no elegiste qué columna predecir: hazlo en la pestaña «Datos».</>}
            {" "}Estos modelos no usan épocas ni imágenes: cada uno entrena en segundos y se evalúa con el 15 % de las filas que nunca vio.
          </Alert>
        )}
        {!mod.tabular && (<>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Épocas máximas"><input className="w-full" type="number" min={1} max={100} value={cfg.epochs} onChange={num("epochs")} /></Field>
          <Field label="Épocas de ajuste fino" hint="0 = sólo cabeza de clasificación"><input className="w-full" type="number" min={0} max={50} value={cfg.fine_tune_epochs} onChange={num("fine_tune_epochs")} /></Field>
          <Field label="Batch size" hint="Redúcelo si hay falta de memoria"><input className="w-full" type="number" min={2} max={64} value={cfg.batch_size} onChange={num("batch_size")} /></Field>
          <Field label="Learning rate"><input className="w-full" type="number" step="any" min={0.00001} max={0.1} value={cfg.learning_rate} onChange={num("learning_rate")} /></Field>
          <Field label="Paciencia (early stopping)"><input className="w-full" type="number" min={1} max={20} value={cfg.patience} onChange={num("patience")} /></Field>
          {mod.windowed && (
            <Field label={mod.window_label} hint={mod.window_hint}>
              <input className="w-full" type="number" min={1} max={10} step={0.5} value={cfg.clip_seconds ?? 5} onChange={num("clip_seconds")} />
            </Field>
          )}
          <Field label={mod.resolution_label} hint={mod.resolution_hint}>
            <select className="w-full" value={cfg.image_size ?? 224} onChange={num("image_size")}>
              <option value={160}>160 × 160 — más rápido (~0,5× tiempo)</option>
              <option value={224}>224 × 224 — estándar</option>
              <option value={260}>260 × 260 (~1,4× tiempo)</option>
              <option value={300}>300 × 300 (~1,8× tiempo)</option>
              <option value={384}>384 × 384 — máximo detalle (~3× tiempo)</option>
            </select>
          </Field>
        </div>
        {(cfg.image_size ?? 224) >= 300 && cfg.batch_size > 16 && (
          <Alert tone="warning">Con resolución {cfg.image_size} px conviene un batch size de 16 o menos para no quedarse sin memoria.</Alert>
        )}
        </>)}
        <div className="flex flex-wrap gap-6 text-sm">
          {!mod.tabular && <label className="flex items-center gap-2"><input type="checkbox" checked={cfg.augmentation} onChange={(e) => setCfg({ ...cfg, augmentation: e.target.checked })} /> Data augmentation</label>}
          <label className="flex items-center gap-2"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> Incluir en el entrenamiento semanal automático</label>
          <span className="text-xs text-slate-500">Al terminar cada entrenamiento (el del sábado o uno nocturno) el proyecto se desmarca: revisa los resultados, ajusta el modelo y vuelve a marcarlo y guardar para que entre en el siguiente.</span>
        </div>
        {!mod.tabular && (
          <ModelOverrides
            models={models}
            cfg={cfg}
            modality={mod}
            disabled={readOnly}
            onChange={(model_overrides) => setCfg({ ...cfg, model_overrides })}
          />
        )}
        </fieldset>
        <div className="flex flex-wrap gap-2">
          {!readOnly && <button className="btn btn-primary">Guardar configuración</button>}
          {user?.role === "ADMIN" && <button type="button" className="btn btn-secondary" onClick={trainNow}>Entrenar ahora (admin)</button>}
        </div>
      </form>
      <h3 className="font-semibold">Entrenamientos del proyecto</h3>
      {mine.length === 0 && (
        <p className="text-sm text-slate-500">
          Todavía no hay entrenamientos. Marca «Incluir en el entrenamiento semanal automático» y guarda: se entrenará el sábado a las 00:05 si
          {mod.tabular ? " ya elegiste la columna a predecir y hay al menos 2 de sus valores con suficientes filas." : " hay al menos 2 categorías con suficientes imágenes."}
        </p>
      )}
      <div className="space-y-4">{mine.slice(0, 10).map((j, i) => <JobCard key={j.id} job={j} expanded={i === 0} />)}</div>
    </div>
  );
}

function ModelsTab({ project }: { project: Project }) {
  const { data, loading, reload } = useAsync(() => api<ModelInfo[]>(`/projects/${project.id}/models`), [project.id, project.latest_model?.id]);
  const [sel, setSel] = useState<number | null>(null);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const { data: detail } = useAsync(() => (sel ? api<ModelInfo>(`/models/${sel}`) : Promise.resolve(null)), [sel]);
  useEffect(() => {
    if (!data) return;
    if (sel === null || !data.some((m) => m.id === sel)) setSel(data[0]?.id ?? null);
  }, [data, sel]);
  const remove = async (m: ModelInfo) => {
    const r = await deleteModels([{ ...m, project_name: project.name }]);
    if (!r) return;
    setMsg(deleteSummary(r));
    reload();
  };
  if (loading && !data) return <Spinner />;
  if (!data?.length) return <div className="space-y-3">{msg && <Alert tone={msg.tone}>{msg.text}</Alert>}<p className="text-sm text-slate-500">Aún no hay modelos entrenados para este proyecto.</p></div>;
  const best = data.reduce((a, m) => ((m.accuracy ?? -1) > (a.accuracy ?? -1) ? m : a), data[0]);
  const tabular = project.modality === "tabular";
  const sizeOf = (m: ModelInfo) => (m.exports.find((e) => e.format === "keras") ?? m.exports.find((e) => e.status === "ok"))?.size_bytes ?? null;
  return (
    <div className="space-y-4">
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}
      <div className="card overflow-x-auto p-0">
        <div className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">Comparación de modelos (haz clic en una fila para ver el detalle y probarlo)</div>
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-800">
              <th className="th">Versión</th><th className="th">Modelo base</th><th className="th">Exactitud</th>{!tabular && <th className="th">Top-5</th>}<th className="th">F1 macro</th>{!tabular && <th className="th">Épocas</th>}<th className="th">Tiempo</th><th className="th">Tamaño</th><th className="th">Formatos</th><th className="th">Fecha</th>
            </tr>
          </thead>
          <tbody>
            {data.map((m) => (
              <tr key={m.id} onClick={() => setSel(m.id)} className={`cursor-pointer border-b border-slate-100 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/50 ${sel === m.id ? "bg-brand-50 dark:bg-brand-700/10" : ""}`}>
                <td className="td font-medium">v{m.version}{m.id === best.id && data.length > 1 && <span className="ml-1 rounded bg-emerald-100 px-1.5 text-xs text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">mejor</span>}</td>
                <td className="td">{BASE_LABEL[m.base_model] ?? m.base_model}</td>
                <td className="td font-semibold">{pct(m.accuracy)}</td>
                {!tabular && <td className="td">{pct(m.top5)}</td>}
                <td className="td">{pct(m.macro_f1)}</td>
                {!tabular && <td className="td">{m.epochs_run ?? "—"}</td>}
                <td className="td whitespace-nowrap">{fmtDuration(m.train_seconds)}</td>
                <td className="td">{fmtBytes(sizeOf(m))}</td>
                <td className="td">{m.exports.filter((e) => e.status === "ok").length}/{m.exports.length}</td>
                <td className="td whitespace-nowrap">{fmtDate(m.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card min-w-0">
        {detail && detail.id === sel ? <ModelPanel model={detail} onDelete={() => void remove(detail)} /> : <Spinner />}
      </div>
    </div>
  );
}

export default function ProjectDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const { schedule } = useJobs();
  const { data: project, error, loading, reload } = useAsync(() => api<Project>(`/projects/${id}`), [id]);
  const [tab, setTab] = useState<"images" | "training" | "models" | "share">("images");
  const mod = useModality(project?.modality);
  if (loading && !project) return <Spinner />;
  if (error || !project) return <Alert tone="error">{error ?? "Proyecto no encontrado."}</Alert>;
  const remove = async () => {
    if (!confirm(`¿Eliminar el proyecto «${project.name}» con ${mod.tabular ? "todos sus archivos" : "todas sus imágenes"} y modelos? Esta acción no se puede deshacer.`)) return;
    try {
      await api(`/projects/${project.id}`, { method: "DELETE" });
      nav("/proyectos");
    } catch (e) {
      alert(errMsg(e));
    }
  };
  return (
    <div>
      <PageTitle title={project.name}>
        <Badge status="completed">{baseLabels(project.base_models)}</Badge>
        {project.my_role && project.my_role !== "owner" && (
          <Badge status="exporting">Compartido por @{project.owner} · {ROLE_LABEL[project.my_role]}</Badge>
        )}
        {project.my_role === "owner" && <button className="btn btn-danger btn-sm" disabled={schedule ? !schedule.open : false} onClick={remove}>Eliminar proyecto</button>}
      </PageTitle>
      {project.description && <p className="-mt-3 mb-5 text-sm text-slate-500">{project.description}</p>}
      {project.training_lock && (
        <div className="mb-5"><Alert tone="info">🌙 {project.training_lock.message} Tiempo restante estimado: {fmtDuration(project.training_lock.remaining_seconds)}. Tus otros proyectos siguen abiertos.</Alert></div>
      )}
      <Tabs
        tabs={[
          { id: "images", label: mod.tabular ? `Datos (${project.image_count})` : `${mod.unit.charAt(0).toUpperCase()}${mod.unit.slice(1)} (${project.image_count})` },
          { id: "training", label: "Entrenamiento" },
          { id: "models", label: "Modelos y pruebas" },
          { id: "share", label: "Compartir", badge: project.members?.length || undefined },
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === "images" && (mod.tabular ? <TabularTab project={project} reload={reload} /> : <ImagesTab project={project} reload={reload} />)}
      {tab === "training" && <TrainingTab project={project} reload={reload} />}
      {tab === "models" && <ModelsTab project={project} />}
      {tab === "share" && <ShareTab project={project} reload={reload} />}
    </div>
  );
}
