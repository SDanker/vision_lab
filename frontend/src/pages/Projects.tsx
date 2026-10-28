import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, errMsg } from "../lib/api";
import { baseLabels, fmtBytes, fmtDayShort, pct } from "../lib/format";
import ModelPicker from "../components/ModelPicker";
import { ROLE_LABEL } from "../components/ShareTab";
import { useModalities } from "../lib/modalities";
import type { Project } from "../lib/types";
import { Alert, Field, Modal, PageTitle, Spinner, useAsync } from "../components/ui";

function NewProject({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [models, setModels] = useState<string[]>(["mobilenetv3"]);
  const modalities = useModalities();
  const [modality, setModality] = useState<string>("image");
  const tabular = !!modalities.find((m) => m.id === modality)?.tabular;
  const pickModality = (id: string) => {
    setModality(id);
    setModels([modalities.find((m) => m.id === id)?.tabular ? "random_forest" : "mobilenetv3"]); // los modelos dependen del tipo
  };
  const [cats, setCats] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api("/projects", {
        body: { name, description, modality, base_models: models, categories: tabular ? [] : cats.split(/[\n,]/).map((c) => c.trim()).filter(Boolean) },
      });
      onCreated();
    } catch (err) {
      setError(errMsg(err));
      setBusy(false);
    }
  };
  return (
    <Modal title="Nuevo proyecto" onClose={onClose} wide>
      <form className="space-y-4" onSubmit={submit}>
        {error && <Alert tone="error">{error}</Alert>}
        <Field label="Nombre"><input className="w-full" value={name} onChange={(e) => setName(e.target.value)} required maxLength={120} autoFocus /></Field>
        <Field label="Descripción"><textarea className="w-full" rows={2} value={description} onChange={(e) => setDescription(e.target.value)} /></Field>
        <div>
          <span className="label">Tipo de datos</span>
          <div className="grid gap-2 sm:grid-cols-2">
            {modalities.map((m) => [m.id, `${m.icon} ${m.label}`, m.description] as const).map(([id, title, desc]) => (
              <label key={id} className={`flex cursor-pointer gap-2 rounded-lg border p-3 text-sm ${modality === id ? "border-brand-500 bg-brand-50 dark:bg-brand-700/10" : "border-slate-200 dark:border-slate-700"}`}>
                <input type="radio" name="modality" checked={modality === id} onChange={() => pickModality(id)} />
                <span><span className="font-medium">{title}</span><span className="block text-xs text-slate-500">{desc}</span></span>
              </label>
            ))}
          </div>
          <p className="mt-1 text-xs text-slate-500">No se puede cambiar después de crear el proyecto.</p>
        </div>
        <div>
          <span className="label">Modelos base (puedes elegir uno o varios)</span>
          <ModelPicker value={models} onChange={setModels} tabular={tabular} />
        </div>
        {tabular ? (
          <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-600 dark:bg-slate-800/50 dark:text-slate-300">
            Las categorías salen de la tabla: después de subir tu CSV eliges la columna que quieres predecir y sus valores pasan a ser las categorías.
          </p>
        ) : (
          <Field label="Categorías (separadas por coma o salto de línea)" hint="Necesitas al menos 2 categorías con suficientes imágenes para entrenar."><textarea className="w-full" rows={3} value={cats} onChange={(e) => setCats(e.target.value)} placeholder="gatos, perros" /></Field>
        )}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn btn-secondary" onClick={onClose}>Cancelar</button>
          <button className="btn btn-primary" disabled={busy}>Crear proyecto</button>
        </div>
      </form>
    </Modal>
  );
}

export default function Projects() {
  const { data, loading, error, reload } = useAsync(() => api<Project[]>("/projects"), []);
  const mods = useModalities();
  const infoOf = (id?: string) => mods.find((m) => m.id === (id ?? "image")) ?? mods[0];
  const [creating, setCreating] = useState(false);
  return (
    <div>
      <PageTitle title="Proyectos">
        <button className="btn btn-primary" onClick={() => setCreating(true)}>+ Nuevo proyecto</button>
      </PageTitle>
      {error && <Alert tone="error">{error}</Alert>}
      {loading && <Spinner />}
      {data && data.length === 0 && (
        <div className="card text-center text-slate-500">Aún no tienes proyectos. Crea el primero para subir imágenes y entrenar un modelo.</div>
      )}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {data?.map((p) => (
          <Link key={p.id} to={`/proyectos/${p.id}`} className="card transition hover:border-brand-500">
            <div className="flex items-start justify-between gap-2">
              <h3 className="font-semibold">{p.modality && p.modality !== "image" ? `${infoOf(p.modality).icon} ` : ""}{p.name}</h3>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs dark:bg-slate-800">{baseLabels(p.base_models)}</span>
            </div>
            {p.my_role && p.my_role !== "owner" && (
              <p className="mt-1 text-xs font-medium text-indigo-700 dark:text-indigo-300">
                🤝 Compartido por @{p.owner} · {ROLE_LABEL[p.my_role]}
              </p>
            )}
            {p.my_role === "owner" && !!p.members?.length && (
              <p className="mt-1 text-xs text-slate-500">🤝 Compartido con {p.members.map((m) => `@${m.username}`).join(", ")}</p>
            )}
            <p className="mt-1 line-clamp-2 text-sm text-slate-500">{p.description || "Sin descripción"}</p>
            {infoOf(p.modality).tabular ? (
              <div className="mt-3 text-xs font-medium text-slate-600 dark:text-slate-300">
                {p.config.tabular_target ? <>Predice: <strong>{p.config.tabular_target}</strong></> : <span className="font-normal text-amber-700 dark:text-amber-300">Falta elegir la columna a predecir</span>}
              </div>
            ) : (
              <>
                <div className="mt-3 text-xs font-medium text-slate-600 dark:text-slate-300">
                  {p.categories.length} categoría{p.categories.length === 1 ? "" : "s"}
                  {(() => {
                    const min = p.min_images_per_class ?? 10;
                    const low = p.categories.filter((c) => c.image_count < min).length;
                    return low > 0 ? <span className="ml-1 font-normal text-amber-700 dark:text-amber-300">· {low} con menos de {min} imágenes</span> : null;
                  })()}
                </div>
                <div className="mt-1.5 flex max-h-14 flex-wrap gap-1 overflow-hidden">
                  {[...p.categories].sort((a, b) => b.image_count - a.image_count).slice(0, 6).map((c) => (
                    <span key={c.id} className="max-w-[10rem] truncate rounded-full bg-brand-50 px-2 py-0.5 text-xs text-brand-700 dark:bg-brand-700/20 dark:text-brand-100" title={`${c.name}: ${c.image_count} imágenes`}>{c.name} · {c.image_count}</span>
                  ))}
                  {p.categories.length > 6 && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">+{p.categories.length - 6} más</span>}
                </div>
              </>
            )}
            <div className="mt-2">
              {p.auto_train ? (
                <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300" title="Marcado para el entrenamiento semanal automático">
                  🗓 Entrena el {fmtDayShort(p.next_training_start)}
                </span>
              ) : (
                <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-300" title="Entra a la ventana de entrenamiento sólo si lo marcas en la pestaña Entrenamiento">
                  ⏸ Sin entrenar esta semana
                </span>
              )}
            </div>
            <div className="mt-3 flex justify-between text-xs text-slate-500">
              <span>{p.image_count} {infoOf(p.modality).unit} · {fmtBytes(p.size_bytes)}</span>
              <span>{p.latest_model ? `Modelo v${p.latest_model.version} · ${pct(p.latest_model.accuracy)}` : "Sin modelo"}</span>
            </div>
          </Link>
        ))}
      </div>
      {creating && <NewProject onClose={() => setCreating(false)} onCreated={() => { setCreating(false); reload(); }} />}
    </div>
  );
}
