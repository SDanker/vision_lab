import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { BASE_LABEL, fmtDate, fmtDuration, pct } from "../lib/format";
import { useJobs } from "../lib/jobs";
import { useModalityBadge } from "../lib/modalities";
import { deleteModels, deleteSummary } from "../lib/models";
import type { ModelInfo } from "../lib/types";
import JobCard from "../components/JobCard";
import ModelPanel from "../components/ModelPanel";
import { Alert, Badge, PageTitle, Spinner, useAsync } from "../components/ui";

const ACTIVE = ["pending", "preparing", "training", "validating", "exporting"];

export function TrainingPage() {
  const { jobs, schedule, connected } = useJobs();
  const active = jobs.filter((j) => ACTIVE.includes(j.status));
  const recent = jobs.filter((j) => !ACTIVE.includes(j.status)).slice(0, 5);
  return (
    <div>
      <PageTitle title="Entrenamiento" />
      <p className="mb-6 text-sm text-slate-500">
        Noches de lunes a viernes (00:00–08:00): se entrenan los modelos marcados que tardan menos de 4 horas y caben en la noche, primero los más cortos. Cola semanal: sábado 00:05 (America/Santiago), del modelo que más tarda al que menos; próximo inicio: {fmtDate(schedule?.next_training_start)}. Se procesa un único entrenamiento intensivo a la vez.
        {connected ? " Progreso en tiempo real activo." : " Reconectando con el servidor…"}
      </p>
      <h2 className="mb-3 text-lg font-semibold">En cola y en ejecución</h2>
      {active.length === 0 ? <p className="mb-8 text-sm text-slate-500">No hay entrenamientos en cola.</p> : <div className="mb-8 space-y-4">{active.map((j) => <JobCard key={j.id} job={j} expanded />)}</div>}
      <h2 className="mb-3 text-lg font-semibold">Recientes</h2>
      <div className="space-y-4">{recent.map((j) => <JobCard key={j.id} job={j} />)}</div>
    </div>
  );
}

const ROLE_NOTE = { editor: "editor", viewer: "sólo lectura" } as const;

interface ProjectGroup {
  id: number;
  name: string;
  owner?: string;
  role?: ModelInfo["my_role"];
  modality?: string;
  models: ModelInfo[]; // versión más nueva primero
  best: ModelInfo; // el de mayor exactitud (a igualdad, el más nuevo)
  latest: string;
}

const score = (m: ModelInfo) => m.accuracy ?? -1;
const bestOf = (list: ModelInfo[]) => list.reduce((a, m) => (score(m) > score(a) || (score(m) === score(a) && m.version > a.version) ? m : a), list[0]);
const plural = (n: number) => `${n} ${n === 1 ? "entrenamiento" : "entrenamientos"}`;

function RoleNote({ g }: { g: ProjectGroup }) {
  if (!g.role || g.role === "owner") return null;
  return <div className="text-xs text-slate-500">de @{g.owner} · tienes acceso de {ROLE_NOTE[g.role]}</div>;
}

export function ModelsPage() {
  const { data, loading, reload } = useAsync(() => api<ModelInfo[]>("/models"), []);
  const [open, setOpen] = useState<number | null>(null); // proyecto desplegado
  const [sel, setSel] = useState<number | null>(null); // modelo cuyo detalle se muestra
  const badge = useModalityBadge();
  const [checked, setChecked] = useState<number[]>([]);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const { data: detail } = useAsync(() => (sel ? api<ModelInfo>(`/models/${sel}`) : Promise.resolve(null)), [sel]);

  // Una burbuja por proyecto, el de entrenamiento más reciente primero.
  const groups = useMemo(() => {
    const map = new Map<number, ModelInfo[]>();
    for (const m of data ?? []) map.set(m.project_id, [...(map.get(m.project_id) ?? []), m]);
    return [...map.values()].map((list): ProjectGroup => {
      const models = [...list].sort((a, b) => b.version - a.version);
      const first = models[0];
      return {
        id: first.project_id,
        name: first.project_name ?? `Proyecto ${first.project_id}`,
        owner: first.project_owner,
        modality: first.modality,
        role: first.my_role,
        models,
        best: bestOf(models),
        latest: models.reduce((a, m) => (m.created_at > a ? m.created_at : a), first.created_at),
      };
    });
  }, [data]);

  useEffect(() => {
    if (!data) return;
    setChecked((c) => c.filter((id) => data.some((m) => m.id === id)));
    if (open === null) return;
    const g = groups.find((x) => x.id === open);
    if (!g) {
      setOpen(null);
      setSel(null);
    } else if (sel === null || !g.models.some((m) => m.id === sel)) {
      setSel(g.best.id);
    }
  }, [data, groups, open, sel]);

  const show = (g: ProjectGroup | null) => {
    setOpen(g ? g.id : null);
    setSel(g ? g.best.id : null); // al desplegar se ve primero el mejor modelo
    setChecked([]);
  };
  const toggle = (id: number) => setChecked((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id]));
  const remove = async (list: ModelInfo[]) => {
    const r = await deleteModels(list);
    if (!r) return;
    setMsg(deleteSummary(r));
    setChecked([]);
    reload();
  };
  const current = detail && detail.id === sel ? detail : null;

  return (
    <div>
      <PageTitle title="Modelos, descargas y pruebas" />
      {loading && !data && <Spinner />}
      {msg && <div className="mb-4"><Alert tone={msg.tone}>{msg.text}</Alert></div>}
      {data && data.length === 0 && <div className="card text-center text-slate-500">Todavía no hay modelos entrenados. Aparecerán aquí tras el primer entrenamiento semanal.</div>}
      {groups.length > 0 && (
        <>
          {open === null && <p className="mb-4 text-sm text-slate-500">Cada tarjeta es un proyecto y muestra su mejor modelo. Haz clic para ver todos sus entrenamientos.</p>}
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {groups.map((g) => {
              if (g.id !== open) {
                return (
                  <button key={g.id} onClick={() => show(g)} aria-expanded={false} className="card flex flex-col gap-3 text-left transition hover:border-brand-500 hover:shadow-md">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="min-w-0 truncate font-semibold" title={g.name}>{badge(g.modality)}{g.name}</h3>
                      <span className="shrink-0 rounded-full bg-brand-50 px-2.5 py-0.5 text-xs font-medium text-brand-700 dark:bg-brand-700/20 dark:text-brand-100">{plural(g.models.length)}</span>
                    </div>
                    <RoleNote g={g} />
                    <div>
                      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Mejor modelo</div>
                      <div className="flex items-baseline justify-between gap-2">
                        <div className="min-w-0 truncate text-sm">v{g.best.version} · {BASE_LABEL[g.best.base_model] ?? g.best.base_model}</div>
                        <div className="shrink-0 text-2xl font-bold text-emerald-600 dark:text-emerald-400">{pct(g.best.accuracy)}</div>
                      </div>
                    </div>
                    <div className="mt-auto flex items-center justify-between text-xs">
                      <span className="text-slate-400">Último: {fmtDate(g.latest)}</span>
                      <span className="font-medium text-brand-600">{g.models.length > 1 ? `Ver los ${g.models.length} ▸` : "Ver detalle ▸"}</span>
                    </div>
                  </button>
                );
              }
              const deletable = g.models.filter((m) => m.can_delete);
              const selected = g.models.filter((m) => checked.includes(m.id));
              return (
                <section key={g.id} aria-label={`Entrenamientos de ${g.name}`} className="card col-span-full space-y-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="truncate text-lg font-semibold">{badge(g.modality)}{g.name}</h3>
                      <p className="text-xs text-slate-500">{plural(g.models.length)} · mejor: v{g.best.version} con {pct(g.best.accuracy)}</p>
                      <RoleNote g={g} />
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      {selected.length > 0 && (
                        <>
                          <span className="text-sm">{selected.length} seleccionado(s)</span>
                          <button className="btn btn-secondary btn-sm" onClick={() => setChecked([])}>Quitar selección</button>
                          <button className="btn btn-danger btn-sm" onClick={() => void remove(selected)}>🗑 Borrar</button>
                        </>
                      )}
                      <button className="btn btn-secondary btn-sm" onClick={() => show(null)} aria-expanded>▴ Contraer</button>
                    </div>
                  </div>

                  <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-slate-200 dark:border-slate-800">
                          <th className="th w-8">
                            {deletable.length > 0 && (
                              <input type="checkbox" checked={deletable.every((m) => checked.includes(m.id))} aria-label={`Seleccionar todos los modelos de ${g.name}`}
                                onChange={() => setChecked(deletable.every((m) => checked.includes(m.id)) ? [] : deletable.map((m) => m.id))} />
                            )}
                          </th>
                          <th className="th">Versión</th><th className="th">Modelo base</th><th className="th">Exactitud</th><th className="th">F1 macro</th>{g.modality !== "tabular" && <th className="th">Épocas</th>}<th className="th">Fecha</th><th className="th" />
                        </tr>
                      </thead>
                      <tbody>
                        {g.models.map((m) => (
                          <tr key={m.id} onClick={() => setSel(m.id)} className={`cursor-pointer border-b border-slate-100 last:border-0 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/50 ${sel === m.id ? "bg-brand-50 dark:bg-brand-700/10" : ""}`}>
                            <td className="td" onClick={(e) => e.stopPropagation()}>
                              {m.can_delete && <input type="checkbox" checked={checked.includes(m.id)} onChange={() => toggle(m.id)} aria-label={`Seleccionar v${m.version} de ${g.name}`} />}
                            </td>
                            <td className="td font-medium">v{m.version}{m.id === g.best.id && g.models.length > 1 && <span className="ml-1 rounded bg-emerald-100 px-1.5 text-xs text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">mejor</span>}</td>
                            <td className="td">{BASE_LABEL[m.base_model] ?? m.base_model}</td>
                            <td className="td font-semibold">{pct(m.accuracy)}</td>
                            <td className="td">{pct(m.macro_f1)}</td>
                            {g.modality !== "tabular" && <td className="td">{m.epochs_run ?? "—"}</td>}
                            <td className="td whitespace-nowrap">{fmtDate(m.created_at)}</td>
                            <td className="td text-right" onClick={(e) => e.stopPropagation()}>
                              {m.can_delete && <button className="btn btn-secondary btn-sm" onClick={() => void remove([m])} title="Borrar este modelo" aria-label={`Borrar v${m.version} de ${g.name}`}>🗑</button>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="min-w-0 border-t border-slate-200 pt-4 dark:border-slate-800">
                    {current ? <ModelPanel model={current} onDelete={() => void remove([{ ...current, project_name: g.name }])} /> : <Spinner />}
                  </div>
                </section>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export function HistoryPage() {
  const { jobs } = useJobs();
  const done = jobs.filter((j) => !ACTIVE.includes(j.status));
  const badge = useModalityBadge();
  return (
    <div>
      <PageTitle title="Historial de entrenamientos" />
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">#</th><th className="th">Proyecto</th><th className="th">Modelo base</th><th className="th">Estado</th><th className="th">Accuracy test</th><th className="th">Épocas</th><th className="th">Duración</th><th className="th">Inicio</th></tr></thead>
          <tbody>
            {done.map((j) => (
              <tr key={j.id} className="border-b border-slate-100 dark:border-slate-800">
                <td className="td">{j.id}</td>
                <td className="td font-medium">{badge(j.config?.modality)}{j.project_name}</td>
                <td className="td">{BASE_LABEL[j.base_model] ?? j.base_model}</td>
                <td className="td"><Badge status={j.status} />{j.error && <div className="mt-1 max-w-xs text-xs text-red-600">{j.error}</div>}</td>
                <td className="td">{pct(j.metrics?.test_accuracy)}</td>
                <td className="td">{j.config?.modality === "tabular" ? "—" : `${j.epoch}/${j.total_epochs}`}</td>
                <td className="td">{fmtDuration(j.elapsed_seconds)}</td>
                <td className="td">{fmtDate(j.started_at ?? j.created_at)}</td>
              </tr>
            ))}
            {done.length === 0 && <tr><td className="td text-center text-slate-500" colSpan={8}>Sin historial todavía.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
