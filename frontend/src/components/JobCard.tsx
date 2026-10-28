import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib/api";
import { BASE_LABEL, fmtDate, fmtDuration, pct } from "../lib/format";
import type { Job, LogRow } from "../lib/types";
import { useAuth } from "../lib/auth";
import { useJobs } from "../lib/jobs";
import { useModalityBadge } from "../lib/modalities";
import TrainingParams, { TrainingParamsInline } from "./TrainingParams";
import { Badge, Progress } from "./ui";

const ACTIVE = ["pending", "preparing", "training", "validating", "exporting"];
export const TRIGGER_LABEL: Record<string, string> = { schedule: "cola semanal", night: "🌙 nocturno", manual: "manual" };

export function TrainingCharts({ history }: { history: Job["history"] }) {
  if (!history.length) return <p className="text-sm text-slate-500">Aún no hay épocas completadas.</p>;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {[
        { title: "Pérdida (loss)", keys: [["loss", "Entrenamiento", "#6366f1"], ["val_loss", "Validación", "#f59e0b"]] },
        { title: "Exactitud (accuracy)", keys: [["accuracy", "Entrenamiento", "#10b981"], ["val_accuracy", "Validación", "#ef4444"]] },
      ].map((chart) => (
        <div key={chart.title}>
          <div className="mb-1 text-sm font-medium">{chart.title}</div>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={history} margin={{ left: -10, right: 10, top: 5 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={chart.title.startsWith("Exactitud") ? [0, 1] : ["auto", "auto"]} />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {chart.keys.map(([k, name, color]) => (
                  <Line key={k} type="monotone" dataKey={k} name={name} stroke={color} dot={history.length < 30} strokeWidth={2} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}
    </div>
  );
}

function Logs({ jobId, status }: { jobId: number; status: string }) {
  const [rows, setRows] = useState<LogRow[]>([]);
  useEffect(() => {
    let alive = true;
    const load = () => api<LogRow[]>(`/training/jobs/${jobId}/logs`).then((r) => alive && setRows(r)).catch(() => undefined);
    void load();
    if (!ACTIVE.includes(status)) return () => { alive = false; };
    const t = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [jobId, status]);
  return (
    <pre className="max-h-64 overflow-auto rounded-lg bg-slate-900 p-3 text-xs leading-relaxed text-slate-200" aria-label="Registros de ejecución">
      {rows.length === 0 ? "Sin registros todavía." : rows.map((r) => `${fmtDate(r.created_at)} [${r.level}] ${r.message}`).join("\n")}
    </pre>
  );
}

export default function JobCard({ job, expanded = false }: { job: Job; expanded?: boolean }) {
  const [open, setOpen] = useState(expanded);
  const { user } = useAuth();
  const { reload } = useJobs();
  const [cancelErr, setCancelErr] = useState<string | null>(null);
  const badge = useModalityBadge();
  const active = ACTIVE.includes(job.status);
  const cancel = async () => {
    const msg = job.status === "pending"
      ? `¿Quitar de la cola el entrenamiento de «${job.project_name}»?`
      : `¿Cancelar el entrenamiento en curso de «${job.project_name}»? Se perderá el progreso de este intento.`;
    if (!confirm(msg)) return;
    try {
      await api(`/admin/training/${job.id}/cancel`, { method: "POST" });
      setCancelErr(null);
      reload();
    } catch (e) {
      setCancelErr(e instanceof Error ? e.message : "No se pudo cancelar.");
    }
  };
  const m = job.metrics ?? {};
  return (
    <div className="card">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold">{badge(job.config?.modality)}{job.project_name}</h3>
            <Badge status={job.status} />
          </div>
          <div className="text-xs text-slate-500">
            Trabajo #{job.id} · {BASE_LABEL[job.base_model] ?? job.base_model} · {TRIGGER_LABEL[job.trigger] ?? job.trigger} · creado {fmtDate(job.created_at)}
          </div>
          <TrainingParamsInline config={job.config} />
        </div>
        <div className="flex gap-2">
          {user?.role === "ADMIN" && active && (
            <button className="btn btn-danger btn-sm" disabled={job.cancel_requested} onClick={cancel}>
              {job.cancel_requested ? "Cancelando…" : job.status === "pending" ? "Quitar de la cola" : "Cancelar entrenamiento"}
            </button>
          )}
          <button className="btn btn-secondary btn-sm" onClick={() => setOpen(!open)}>{open ? "Ocultar detalle" : "Ver detalle"}</button>
        </div>
      </div>

      {cancelErr && <p className="mt-2 text-sm text-red-600">{cancelErr}</p>}
      {job.status === "pending" && (
        <p className="mt-3 text-sm">
          Posición en la cola: <strong>{job.queue_position ?? "—"}</strong>
          {job.estimated_seconds != null && <> · duración estimada <strong>{fmtDuration(job.estimated_seconds)}</strong></>}. Se entrena un modelo a la vez
          {job.trigger === "night" ? ", primero los más cortos para que quepan en la noche." : ", primero los que más tardan."}
        </p>
      )}

      {(active || job.status === "completed") && job.status !== "pending" && (
        <div className="mt-4 space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span>{job.phase ?? ""}</span>
            <strong>{job.percent.toFixed(0)}%</strong>
          </div>
          <Progress value={job.percent} tone={job.status === "completed" ? "green" : "brand"} />
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4 lg:grid-cols-7">
            {job.config?.modality !== "tabular" && <Metric label="Época" value={job.total_epochs ? `${job.epoch} / ${job.total_epochs}` : "—"} />}
            <Metric label="Accuracy" value={pct(m.accuracy)} />
            <Metric label="Loss" value={m.loss != null ? m.loss.toFixed(4) : "—"} />
            <Metric label="Transcurrido" value={fmtDuration(job.elapsed_seconds)} />
            <Metric label="Restante est." value={active ? fmtDuration(job.eta_seconds) : "—"} />
            <Metric label="Duración prevista" value={job.estimated_seconds != null ? fmtDuration(job.estimated_seconds) : "—"} />
            <Metric label="CPU / RAM" value={job.cpu_percent != null ? `${job.cpu_percent.toFixed(0)}% · ${(job.ram_mb ?? 0).toFixed(0)} MB` : "—"} />
          </dl>
        </div>
      )}

      {job.error && (
        <p className={`mt-3 rounded-lg p-3 text-sm ${job.status === "error" ? "bg-red-50 text-red-800 dark:bg-red-950/40 dark:text-red-200" : ""}`}>{job.error}</p>
      )}

      {open && (
        <div className="mt-5 space-y-5">
          <TrainingParams
            config={job.config}
            baseModel={job.base_model}
            epochsRun={["completed", "error", "cancelled"].includes(job.status) && job.history.length ? job.history.length : null}
          />
          {job.snapshot && (
            <div className="text-sm text-slate-600 dark:text-slate-400">
              Dataset congelado: {job.snapshot.total_images} {job.config?.modality === "tabular" ? "filas" : "imágenes"} ·{" "}
              {Object.keys(job.snapshot.class_counts).length} categorías ({Object.entries(job.snapshot.class_counts).slice(0, 5).map(([k, v]) => `${k}: ${v}`).join(", ")}{Object.keys(job.snapshot.class_counts).length > 5 ? ", …" : ""}) · hash {job.snapshot.dataset_hash.slice(0, 12)}
            </div>
          )}
          {job.config?.modality === "tabular" ? (
            job.history[0] && (
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Exactitud con las filas de entrenamiento: <strong>{pct(job.history[0].accuracy)}</strong> · con las de validación: <strong>{pct(job.history[0].val_accuracy)}</strong>.
                Si la de entrenamiento es mucho mayor, el modelo memorizó los datos en vez de aprender de ellos.
              </p>
            )
          ) : (
            <TrainingCharts history={job.history} />
          )}
          <div>
            <div className="mb-1 text-sm font-medium">Registros de ejecución</div>
            <Logs jobId={job.id} status={job.status} />
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}
