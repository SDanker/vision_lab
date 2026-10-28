import { useEffect, useState } from "react";
import { api, errMsg } from "../lib/api";
import { fmtBytes, fmtDate, fmtDuration, STATUS_LABEL } from "../lib/format";
import { useJobs } from "../lib/jobs";
import type { ApiTokenItem, Job, NightPlan, NightReport, Project, TokenRequestItem, User } from "../lib/types";
import JobCard from "../components/JobCard";
import { Alert, Badge, Modal, PageTitle, Spinner, Stat, Tabs, useAsync } from "../components/ui";
import { REQUEST_STATE, SCOPE_LABEL, TOKEN_STATE } from "./Account";

interface Dash {
  users: { total: number; pending: number; approved: number; rejected: number; suspended: number };
  projects: number;
  images: number;
  storage: { images_bytes: number; models_bytes: number };
  training: { queued: number; active: number; completed: number; failed: number; cancelled: number };
  models: number;
  token_requests_pending: number;
  lock: { open: boolean; message: string };
  last_run: { week_key: string; status: string; started_at: string; published_at: string | null } | null;
  smtp_configured: boolean;
  resources: Resources;
}
interface Resources {
  cpu_percent: number;
  cpu_count: number;
  host_ram_total_gb: number;
  host_ram_used_gb: number;
  host_ram_percent: number;
  container_ram_used_gb: number | null;
  container_ram_limit_gb: number | null;
  platform_ram_budget_gb: number;
}
interface HistoryItem { id: number; from_status: string | null; to_status: string; note: string | null; actor: string | null; created_at: string }

function Overview() {
  const { data, loading, reload } = useAsync(() => api<Dash>("/admin/dashboard"), []);
  useEffect(() => {
    const t = setInterval(reload, 8000);
    return () => clearInterval(t);
  }, [reload]);
  if (loading && !data) return <Spinner />;
  if (!data) return null;
  const r = data.resources;
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Usuarios registrados" value={data.users.total} sub={`${data.users.approved} aprobados`} />
        <Stat label="Pendientes de aprobación" value={data.users.pending} />
        <Stat label="Rechazados / suspendidos" value={`${data.users.rejected} / ${data.users.suspended}`} />
        <Stat label="Proyectos" value={data.projects} sub={`${data.images} archivos de datos`} />
        <Stat label="Almacenamiento" value={fmtBytes(data.storage.images_bytes + data.storage.models_bytes)} sub={`datos ${fmtBytes(data.storage.images_bytes)} · modelos ${fmtBytes(data.storage.models_bytes)}`} />
        <Stat label="En cola / activos" value={`${data.training.queued} / ${data.training.active}`} />
        <Stat label="Completados / fallidos" value={`${data.training.completed} / ${data.training.failed}`} />
        <Stat label="Modelos publicados" value={data.models} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card space-y-3">
          <h3 className="font-semibold">Recursos del servidor</h3>
          <div className="text-sm">CPU: <strong>{r.cpu_percent.toFixed(0)}%</strong> ({r.cpu_count} núcleos)</div>
          <div className="text-sm">RAM host: <strong>{r.host_ram_used_gb} / {r.host_ram_total_gb} GB</strong> ({r.host_ram_percent}%)</div>
          <div className="text-sm">RAM del contenedor API: {r.container_ram_used_gb ?? "—"} GB{r.container_ram_limit_gb ? ` de ${r.container_ram_limit_gb} GB` : ""}</div>
          <div className="text-sm text-slate-500">Presupuesto de RAM de la plataforma: {r.platform_ram_budget_gb} GB (límites por servicio en docker-compose).</div>
        </div>
        <div className="card space-y-2 text-sm">
          <h3 className="font-semibold">Estado del sistema</h3>
          <div>Cargas: {data.lock.open ? <Badge status="APPROVED">Abiertas</Badge> : <Badge status="SUSPENDED">Bloqueadas</Badge>}</div>
          <div className="text-slate-500">{data.lock.message}</div>
          <div>Última ejecución semanal: {data.last_run ? `${data.last_run.week_key} (${data.last_run.status === "published" ? "publicada" : "en curso"})` : "ninguna"}</div>
          <div>Correo SMTP: {data.smtp_configured ? "configurado" : "no configurado (sólo notificaciones internas)"}</div>
        </div>
      </div>
    </div>
  );
}

function Registrations() {
  const [status, setStatus] = useState("");
  const { data, loading, error, reload } = useAsync(() => api<User[]>(`/admin/registrations${status ? `?status=${status}` : ""}`), [status]);
  const [selected, setSelected] = useState<number[]>([]);
  const [action, setAction] = useState<{ user: User; kind: "reject" | "suspend" } | null>(null);
  const [note, setNote] = useState("");
  const [bulk, setBulk] = useState(false);
  const [history, setHistory] = useState<{ user: User; items: HistoryItem[] } | null>(null);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);

  const run = async (u: User, kind: string, n?: string) => {
    try {
      await api(`/admin/users/${u.id}/${kind}`, { body: { note: n || null } });
      setMsg({ tone: "success", text: `Acción «${kind}» aplicada a @${u.username}.` });
      setAction(null);
      setNote("");
      reload();
    } catch (e) {
      setMsg({ tone: "error", text: errMsg(e) });
    }
  };
  const approveBulk = async () => {
    try {
      const r = await api<{ approved: number; results: { ok: boolean }[] }>("/admin/users/bulk-approve", { body: { user_ids: selected, confirm: true } });
      setMsg({ tone: "success", text: `${r.approved} de ${r.results.length} usuarios aprobados.` });
      setSelected([]);
      setBulk(false);
      reload();
    } catch (e) {
      setMsg({ tone: "error", text: errMsg(e) });
    }
  };
  const showHistory = async (u: User) => setHistory({ user: u, items: await api<HistoryItem[]>(`/admin/users/${u.id}/history`) });
  const pendingSel = selected.filter((id) => data?.find((u) => u.id === id)?.status === "PENDING");
  const toggle = (id: number) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <select value={status} onChange={(e) => { setStatus(e.target.value); setSelected([]); }} aria-label="Filtrar por estado">
          <option value="">Todos los estados</option>
          {["PENDING", "APPROVED", "REJECTED", "SUSPENDED"].map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}
        </select>
        <button className="btn btn-primary" disabled={pendingSel.length === 0} onClick={() => setBulk(true)}>Aprobar seleccionados ({pendingSel.length})</button>
      </div>
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}
      {error && <Alert tone="error">{error}</Alert>}
      {loading && !data && <Spinner />}
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-800">
              <th className="th" />
              <th className="th">Nombre</th><th className="th">Correo</th><th className="th">Usuario</th><th className="th">Solicitud</th><th className="th">Motivo</th><th className="th">Estado</th><th className="th">Revisión</th><th className="th">Revisó</th><th className="th">Acciones</th>
            </tr>
          </thead>
          <tbody>
            {data?.map((u) => (
              <tr key={u.id} className="border-b border-slate-100 dark:border-slate-800">
                <td className="td"><input type="checkbox" disabled={u.status !== "PENDING"} checked={selected.includes(u.id)} onChange={() => toggle(u.id)} aria-label={`Seleccionar ${u.username}`} /></td>
                <td className="td font-medium">{u.first_name} {u.last_name}</td>
                <td className="td">{u.email}</td>
                <td className="td">@{u.username}</td>
                <td className="td whitespace-nowrap">{fmtDate(u.created_at)}</td>
                <td className="td max-w-xs text-slate-600 dark:text-slate-400">{u.reason ?? "—"}</td>
                <td className="td"><Badge status={u.status} /></td>
                <td className="td whitespace-nowrap">{fmtDate(u.reviewed_at)}{u.review_note && <div className="max-w-[12rem] text-xs text-slate-500">{u.review_note}</div>}</td>
                <td className="td">{u.reviewed_by ? `@${u.reviewed_by}` : "—"}</td>
                <td className="td">
                  <div className="flex flex-wrap gap-1">
                    {(u.status === "PENDING" || u.status === "REJECTED") && <button className="btn btn-primary btn-sm" onClick={() => run(u, "approve")}>Aprobar</button>}
                    {u.status === "PENDING" && <button className="btn btn-secondary btn-sm" onClick={() => setAction({ user: u, kind: "reject" })}>Rechazar</button>}
                    {u.status === "APPROVED" && u.role !== "ADMIN" && <button className="btn btn-secondary btn-sm" onClick={() => setAction({ user: u, kind: "suspend" })}>Suspender</button>}
                    {u.status === "SUSPENDED" && <button className="btn btn-primary btn-sm" onClick={() => run(u, "reactivate")}>Reactivar</button>}
                    <button className="btn btn-secondary btn-sm" onClick={() => showHistory(u)}>Historial</button>
                  </div>
                </td>
              </tr>
            ))}
            {data?.length === 0 && <tr><td className="td text-center text-slate-500" colSpan={10}>No hay solicitudes.</td></tr>}
          </tbody>
        </table>
      </div>

      {action && (
        <Modal title={action.kind === "reject" ? `Rechazar a @${action.user.username}` : `Suspender a @${action.user.username}`} onClose={() => setAction(null)}>
          <label className="label" htmlFor="note">Motivo (opcional, se envía al usuario)</label>
          <textarea id="note" className="w-full" rows={3} maxLength={500} value={note} onChange={(e) => setNote(e.target.value)} />
          <div className="mt-4 flex justify-end gap-2">
            <button className="btn btn-secondary" onClick={() => setAction(null)}>Cancelar</button>
            <button className="btn btn-danger" onClick={() => run(action.user, action.kind, note)}>{action.kind === "reject" ? "Rechazar solicitud" : "Suspender usuario"}</button>
          </div>
        </Modal>
      )}
      {bulk && (
        <Modal title="Confirmar aprobación múltiple" onClose={() => setBulk(false)}>
          <p className="text-sm">Se aprobarán <strong>{pendingSel.length}</strong> cuentas y podrán usar la plataforma de inmediato:</p>
          <ul className="my-3 ml-5 list-disc text-sm">{pendingSel.map((id) => { const u = data?.find((x) => x.id === id); return <li key={id}>{u?.first_name} {u?.last_name} (@{u?.username})</li>; })}</ul>
          <div className="flex justify-end gap-2">
            <button className="btn btn-secondary" onClick={() => setBulk(false)}>Cancelar</button>
            <button className="btn btn-primary" onClick={approveBulk}>Confirmar aprobación</button>
          </div>
        </Modal>
      )}
      {history && (
        <Modal title={`Historial de @${history.user.username}`} onClose={() => setHistory(null)}>
          <ul className="space-y-3 text-sm">
            {history.items.map((h) => (
              <li key={h.id} className="border-l-2 border-brand-500 pl-3">
                <div><strong>{h.from_status ? `${STATUS_LABEL[h.from_status]} → ` : ""}{STATUS_LABEL[h.to_status]}</strong> · {fmtDate(h.created_at)}</div>
                <div className="text-slate-500">{h.actor ? `por @${h.actor}` : "automático"}{h.note ? ` — ${h.note}` : ""}</div>
              </li>
            ))}
          </ul>
        </Modal>
      )}
    </div>
  );
}

function Users() {
  const [q, setQ] = useState("");
  const { data, loading, error, reload } = useAsync(() => api<User[]>(`/admin/users?q=${encodeURIComponent(q)}`), [q]);
  const [msg, setMsg] = useState<string | null>(null);
  const patch = async (u: User, body: object) => {
    try {
      await api(`/admin/users/${u.id}`, { method: "PATCH", body });
      reload();
    } catch (e) {
      setMsg(errMsg(e));
    }
  };
  const remove = async (u: User) => {
    if (!confirm(`¿Eliminar definitivamente a @${u.username} junto con sus proyectos, datos y modelos?`)) return;
    try {
      await api(`/admin/users/${u.id}`, { method: "DELETE" });
      reload();
    } catch (e) {
      setMsg(errMsg(e));
    }
  };
  return (
    <div className="space-y-4">
      <input className="w-full max-w-sm" placeholder="Buscar por nombre, usuario o correo" value={q} onChange={(e) => setQ(e.target.value)} />
      {(msg || error) && <Alert tone="error">{msg ?? error}</Alert>}
      {loading && !data && <Spinner />}
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Usuario</th><th className="th">Estado</th><th className="th">Rol</th><th className="th">Proyectos</th><th className="th">Almacenamiento / cuota</th><th className="th">Acciones</th></tr></thead>
          <tbody>
            {data?.map((u) => (
              <tr key={u.id} className="border-b border-slate-100 dark:border-slate-800">
                <td className="td"><div className="font-medium">{u.first_name} {u.last_name}</div><div className="text-xs text-slate-500">@{u.username} · {u.email}</div></td>
                <td className="td"><Badge status={u.status} /></td>
                <td className="td">{u.role === "ADMIN" ? "Administrador" : "Usuario"}</td>
                <td className="td">{u.projects}</td>
                <td className="td">{fmtBytes(u.storage_bytes)} / {fmtBytes(u.quota_bytes)}{u.quota_mb != null && <span className="ml-1 text-xs text-slate-500">(personalizada)</span>}</td>
                <td className="td">
                  <div className="flex flex-wrap gap-1">
                    <button className="btn btn-secondary btn-sm" onClick={() => { const v = prompt("Nueva cuota en MB (vacío = predeterminada)", String(u.quota_mb ?? "")); if (v === null) return; void patch(u, v.trim() === "" ? { clear_quota: true } : { quota_mb: Number(v) }); }}>Cuota</button>
                    {u.status === "APPROVED" && <button className="btn btn-secondary btn-sm" onClick={() => patch(u, { role: u.role === "ADMIN" ? "USER" : "ADMIN" })}>{u.role === "ADMIN" ? "Quitar admin" : "Hacer admin"}</button>}
                    <button className="btn btn-danger btn-sm" onClick={() => remove(u)}>Eliminar</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Queue() {
  const { reload: reloadJobs } = useJobs();
  const { data, reload } = useAsync(() => api<{ jobs: Job[]; runs: { id: number; week_key: string; trigger: string; status: string; started_at: string; published_at: string | null; summary: { queued?: number; completed?: number; failed?: number; skipped?: { project: string; reason: string }[] } | null }[] }>("/admin/queue"), []);
  const projects = useAsync(() => api<Project[]>("/projects?all=true"), []);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const [pid, setPid] = useState<number | "">("");
  useEffect(() => {
    const t = setInterval(reload, 6000);
    return () => clearInterval(t);
  }, [reload]);
  const act = async (id: number, kind: "cancel" | "retry") => {
    try {
      await api(`/admin/training/${id}/${kind}`, { method: "POST" });
      reload();
      reloadJobs();
    } catch (e) {
      setMsg({ tone: "error", text: errMsg(e) });
    }
  };
  const enqueue = async (all = false) => {
    if (!all && pid === "") return;
    const what = all ? "TODOS los proyectos" : "el proyecto seleccionado";
    if (!confirm(`Se congelará el dataset de ${what} y se bloquearán las cargas de todos los usuarios hasta publicar resultados. ¿Iniciar el entrenamiento ahora?`)) return;
    try {
      await api("/admin/training/enqueue", { body: all ? { all: true } : { project_ids: [pid] } });
      setMsg({ tone: "success", text: "Entrenamiento encolado." });
      reload();
      reloadJobs();
    } catch (e) {
      setMsg({ tone: "error", text: errMsg(e) });
    }
  };
  return (
    <div className="space-y-6">
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}
      <div className="card flex flex-wrap items-end gap-3">
        <div>
          <label className="label" htmlFor="proj">Iniciar entrenamiento manual (cuando quieras, sin esperar al sábado)</label>
          <select id="proj" value={pid} onChange={(e) => setPid(e.target.value ? Number(e.target.value) : "")}>
            <option value="">Selecciona un proyecto…</option>
            {projects.data?.map((p) => <option key={p.id} value={p.id}>{p.name} (@{p.owner})</option>)}
          </select>
        </div>
        <button className="btn btn-primary" disabled={pid === ""} onClick={() => enqueue(false)}>Entrenar ahora</button>
        <button className="btn btn-secondary" onClick={() => enqueue(true)}>Entrenar todos los proyectos</button>
      </div>
      <NightPlanCard />
      <NightReportsCard />
      <div>
        <h3 className="mb-2 font-semibold">Ejecuciones</h3>
        <div className="card overflow-x-auto p-0">
          <table className="w-full">
            <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Semana / clave</th><th className="th">Origen</th><th className="th">Estado</th><th className="th">Inicio</th><th className="th">Publicada</th><th className="th">Resumen</th></tr></thead>
            <tbody>
              {data?.runs.map((r) => (
                <tr key={r.id} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="td">{r.week_key}</td><td className="td">{{ schedule: "Semanal", night: "🌙 Nocturna", manual: "Manual" }[r.trigger] ?? r.trigger}</td>
                  <td className="td"><Badge status={r.status === "published" ? "completed" : "training"}>{r.status === "published" ? "Publicada" : "En curso"}</Badge></td>
                  <td className="td">{fmtDate(r.started_at)}</td><td className="td">{fmtDate(r.published_at)}</td>
                  <td className="td text-xs">{r.summary ? `${r.summary.queued ?? 0} encolados · ${r.summary.completed ?? 0} ok · ${r.summary.failed ?? 0} con error` : ""}{r.summary?.skipped?.map((s, i) => <div key={i} className="text-slate-500">Omitido {s.project}: {s.reason}</div>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div>
        <h3 className="mb-2 font-semibold">Trabajos</h3>
        <div className="space-y-3">
          {data?.jobs.slice(0, 20).map((j) => (
            <div key={j.id}>
              <JobCard job={j} />
              <div className="mt-1 flex gap-2">
                {["error", "cancelled"].includes(j.status) && <button className="btn btn-secondary btn-sm" onClick={() => act(j.id, "retry")}>Reintentar</button>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ApiTokens({ onChange }: { onChange: () => void }) {
  const reqs = useAsync(() => api<TokenRequestItem[]>("/admin/token-requests"), []);
  const toks = useAsync(() => api<ApiTokenItem[]>("/admin/api-tokens"), []);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const [review, setReview] = useState<{ req: TokenRequestItem; kind: "approve" | "reject" } | null>(null);
  const [note, setNote] = useState("");
  const [days, setDays] = useState("90");

  const refresh = () => {
    reqs.reload();
    toks.reload();
    onChange();
  };
  const submit = async () => {
    if (!review) return;
    try {
      const body = review.kind === "approve" ? { note: note || null, expires_days: Number(days) || null } : { note: note || null };
      await api(`/admin/token-requests/${review.req.id}/${review.kind}`, { body });
      setMsg({ tone: "success", text: `Solicitud de @${review.req.owner?.username} ${review.kind === "approve" ? "aprobada: ya puede generar su token" : "rechazada"}.` });
      setReview(null);
      setNote("");
      refresh();
    } catch (e) {
      setMsg({ tone: "error", text: errMsg(e) });
    }
  };
  const revoke = async (t: ApiTokenItem) => {
    if (!confirm(`¿Revocar el token «${t.name}» de @${t.owner?.username}? Se le avisará.`)) return;
    try {
      await api(`/admin/api-tokens/${t.id}/revoke`, { method: "POST" });
      refresh();
    } catch (e) {
      setMsg({ tone: "error", text: errMsg(e) });
    }
  };

  return (
    <div className="space-y-6">
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}
      {(reqs.error || toks.error) && <Alert tone="error">{reqs.error ?? toks.error}</Alert>}
      <Alert>
        Un token con escritura actúa siempre dentro de los privilegios de su dueño (sus proyectos y permisos) y respeta el calendario semanal; nunca llega al panel de administración. Al aprobar eliges cuánto dura desde que el usuario lo genera.
      </Alert>
      <div>
        <h3 className="mb-2 font-semibold">Solicitudes de tokens con escritura</h3>
        {reqs.loading && !reqs.data && <Spinner />}
        <div className="card overflow-x-auto p-0">
          <table className="w-full">
            <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Usuario</th><th className="th">Token</th><th className="th">Motivo</th><th className="th">Fecha</th><th className="th">Estado</th><th className="th">Resolución</th><th className="th">Acciones</th></tr></thead>
            <tbody>
              {reqs.data?.map((r) => (
                <tr key={r.id} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="td"><div className="font-medium">{r.owner?.name}</div><div className="text-xs text-slate-500">@{r.owner?.username}</div></td>
                  <td className="td">{r.name}</td>
                  <td className="td max-w-xs text-slate-600 dark:text-slate-400">{r.reason}</td>
                  <td className="td whitespace-nowrap">{fmtDate(r.created_at)}</td>
                  <td className="td"><Badge status={REQUEST_STATE[r.status]?.[0] ?? "pending"}>{REQUEST_STATE[r.status]?.[1] ?? r.status}</Badge>{r.token_id != null && <div className="mt-1 text-xs text-slate-500">token generado</div>}</td>
                  <td className="td text-xs text-slate-500">{r.reviewed_by ? `@${r.reviewed_by} · ${fmtDate(r.reviewed_at)}` : "—"}{r.expires_days ? ` · ${r.expires_days} días` : ""}{r.review_note && <div>{r.review_note}</div>}</td>
                  <td className="td">
                    {r.status === "pending" && (
                      <div className="flex flex-wrap gap-1">
                        <button className="btn btn-primary btn-sm" onClick={() => { setReview({ req: r, kind: "approve" }); setDays("90"); setNote(""); }}>Aprobar</button>
                        <button className="btn btn-secondary btn-sm" onClick={() => { setReview({ req: r, kind: "reject" }); setNote(""); }}>Rechazar</button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
              {reqs.data?.length === 0 && <tr><td className="td text-center text-slate-500" colSpan={7}>No hay solicitudes.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
      <div>
        <h3 className="mb-2 font-semibold">Todos los tokens</h3>
        <div className="card overflow-x-auto p-0">
          <table className="w-full">
            <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Usuario</th><th className="th">Nombre</th><th className="th">Token</th><th className="th">Permiso</th><th className="th">Estado</th><th className="th">Vence</th><th className="th">Último uso</th><th className="th" /></tr></thead>
            <tbody>
              {toks.data?.map((t) => (
                <tr key={t.id} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="td">@{t.owner?.username}</td>
                  <td className="td font-medium">{t.name}</td>
                  <td className="td font-mono text-xs">{t.prefix}…</td>
                  <td className="td">{SCOPE_LABEL[t.scope]}</td>
                  <td className="td"><Badge status={TOKEN_STATE[t.state]?.[0] ?? "pending"}>{TOKEN_STATE[t.state]?.[1] ?? t.state}</Badge></td>
                  <td className="td whitespace-nowrap">{t.expires_at ? fmtDate(t.expires_at) : "Nunca"}</td>
                  <td className="td whitespace-nowrap">{fmtDate(t.last_used_at)}{t.last_used_ip && <div className="text-xs text-slate-500">{t.last_used_ip}</div>}</td>
                  <td className="td text-right">{t.state === "active" && <button className="btn btn-danger btn-sm" onClick={() => void revoke(t)}>Revocar</button>}</td>
                </tr>
              ))}
              {toks.data?.length === 0 && <tr><td className="td text-center text-slate-500" colSpan={8}>Nadie ha creado tokens todavía.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      {review && (
        <Modal title={review.kind === "approve" ? `Aprobar token con escritura de @${review.req.owner?.username}` : `Rechazar la solicitud de @${review.req.owner?.username}`} onClose={() => setReview(null)}>
          <p className="mb-3 text-sm"><strong>{review.req.name}</strong>: {review.req.reason}</p>
          {review.kind === "approve" && (
            <>
              <label className="label" htmlFor="days">Vigencia del token (días, máximo 365)</label>
              <input id="days" type="number" min={1} max={365} className="mb-3 w-32" value={days} onChange={(e) => setDays(e.target.value)} />
            </>
          )}
          <label className="label" htmlFor="tnote">Nota (opcional, se envía al usuario)</label>
          <textarea id="tnote" className="w-full" rows={3} maxLength={500} value={note} onChange={(e) => setNote(e.target.value)} />
          <div className="mt-4 flex justify-end gap-2">
            <button className="btn btn-secondary" onClick={() => setReview(null)}>Cancelar</button>
            <button className={`btn ${review.kind === "approve" ? "btn-primary" : "btn-danger"}`} onClick={() => void submit()}>{review.kind === "approve" ? "Aprobar" : "Rechazar"}</button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function Audit() {
  const [page, setPage] = useState(1);
  const [action, setAction] = useState("");
  const { data, loading } = useAsync(() => api<{ total: number; items: { id: number; actor: string | null; action: string; target_type: string | null; target_id: string | null; detail: unknown; ip: string | null; created_at: string }[] }>(`/admin/audit?page=${page}${action ? `&action=${encodeURIComponent(action)}` : ""}`), [page, action]);
  const pages = data ? Math.max(1, Math.ceil(data.total / 50)) : 1;
  return (
    <div className="space-y-4">
      <input className="w-full max-w-xs" placeholder="Filtrar por acción (ej. user.)" value={action} onChange={(e) => { setAction(e.target.value); setPage(1); }} />
      {loading && !data && <Spinner />}
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Fecha</th><th className="th">Actor</th><th className="th">Acción</th><th className="th">Objetivo</th><th className="th">Detalle</th><th className="th">IP</th></tr></thead>
          <tbody>
            {data?.items.map((a) => (
              <tr key={a.id} className="border-b border-slate-100 dark:border-slate-800">
                <td className="td whitespace-nowrap">{fmtDate(a.created_at)}</td><td className="td">{a.actor ? `@${a.actor}` : "—"}</td><td className="td font-mono text-xs">{a.action}</td>
                <td className="td">{a.target_type ? `${a.target_type} #${a.target_id ?? ""}` : "—"}</td>
                <td className="td max-w-xs truncate font-mono text-xs" title={JSON.stringify(a.detail)}>{a.detail ? JSON.stringify(a.detail) : ""}</td><td className="td">{a.ip}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-center gap-3 text-sm">
        <button className="btn btn-secondary btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>Anterior</button>
        <span>Página {page} de {pages}</span>
        <button className="btn btn-secondary btn-sm" disabled={page >= pages} onClick={() => setPage(page + 1)}>Siguiente</button>
      </div>
    </div>
  );
}

export default function Admin() {
  const [tab, setTab] = useState<"overview" | "requests" | "users" | "tokens" | "queue" | "audit">("overview");
  const pending = useAsync(() => api<Dash>("/admin/dashboard"), []);
  return (
    <div>
      <PageTitle title="Administración" />
      <Tabs
        tabs={[
          { id: "overview", label: "Resumen" },
          { id: "requests", label: "Solicitudes de registro", badge: pending.data?.users.pending },
          { id: "users", label: "Usuarios y cuotas" },
          { id: "tokens", label: "Tokens de API", badge: pending.data?.token_requests_pending },
          { id: "queue", label: "Cola de entrenamiento" },
          { id: "audit", label: "Auditoría" },
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === "overview" && <Overview />}
      {tab === "requests" && <Registrations />}
      {tab === "users" && <Users />}
      {tab === "tokens" && <ApiTokens onChange={pending.reload} />}
      {tab === "queue" && <Queue />}
      {tab === "audit" && <Audit />}
    </div>
  );
}

const NIGHT_DECISION: Record<NightPlan["items"][number]["decision"], [string, string]> = {
  tonight: ["Esta noche", "completed"],
  asking: ["Esperando respuesta", "preparing"],
  skipped_by_user: ["Hoy no (siguen cargando)", "cancelled"],
  too_long: ["Fin de semana (más de 4 h)", "pending"],
  no_room: ["Fin de semana (no cabe)", "pending"],
};

/** Qué modelos se entrenarían esta noche (o la próxima), con su duración estimada. */
function NightPlanCard() {
  const { data, loading, reload } = useAsync(() => api<NightPlan>("/admin/night-plan"), []);
  if (loading && !data) return <Spinner />;
  if (!data) return null;
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold">🌙 Plan nocturno {data.in_window ? "de esta noche" : `para la noche del ${data.night}`}</h3>
        <button className="btn btn-secondary btn-sm" onClick={reload}>Actualizar</button>
      </div>
      <p className="mb-2 text-sm text-slate-500">
        {data.enabled
          ? <>Ventana {data.window} de lunes a viernes. {data.in_window ? "Tiempo disponible ahora" : "Tiempo disponible"}: <strong>{fmtDuration(data.capacity_seconds)}</strong>. Sólo modelos de hasta {fmtDuration(data.max_job_seconds)}; los más cortos primero. El resto se entrena el sábado, del más largo al más corto.</>
          : "El entrenamiento nocturno está desactivado (NIGHT_TRAINING_ENABLED=false)."}
      </p>
      <div className="card overflow-x-auto p-0">
        <table className="w-full">
          <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Proyecto</th><th className="th">Modelo</th><th className="th">Duración estimada</th><th className="th">Según</th><th className="th">Decisión</th></tr></thead>
          <tbody>
            {data.items.map((i) => (
              <tr key={`${i.project_id}-${i.base_model}`} className="border-b border-slate-100 dark:border-slate-800">
                <td className="td">{i.project}</td><td className="td">{i.model}</td>
                <td className="td whitespace-nowrap">{fmtDuration(i.estimated_seconds)}</td>
                <td className="td text-xs text-slate-500">{i.source === "fórmula" ? "fórmula (sin historial)" : `historial del ${i.source}`}</td>
                <td className="td"><Badge status={NIGHT_DECISION[i.decision][1]}>{NIGHT_DECISION[i.decision][0]}</Badge></td>
              </tr>
            ))}
            {data.items.length === 0 && <tr><td className="td text-slate-500" colSpan={5}>No hay modelos pendientes: ningún proyecto marcado tiene cambios desde su última versión.</td></tr>}
          </tbody>
        </table>
      </div>
      {data.not_ready.length > 0 && (
        <p className="mt-2 text-xs text-slate-500">Sin datos suficientes: {data.not_ready.map((n) => n.project).join(", ")}.</p>
      )}
    </div>
  );
}

const NIGHT_TONE: Record<NightReport["tone"], [string, string]> = {
  ok: ["Entrenó", "completed"],
  running: ["En curso", "training"],
  error: ["Con errores", "error"],
  idle: ["Sin entrenamiento", "pending"],
  missing: ["Sin registro", "error"],
};

const REPORT_DECISION: Record<NightReport["items"][number]["decision"], [string, string]> = {
  ...NIGHT_DECISION,
  not_queued: ["No se encoló", "cancelled"],
};

function nightLabel(night: string) {
  return new Date(`${night}T12:00:00`).toLocaleDateString("es-CL", { weekday: "long", day: "numeric", month: "short" });
}

/** Qué pasó cada noche hábil: qué se entrenó o por qué no se entrenó nada (y las noches sin registro). */
function NightReportsCard() {
  const { data, loading, reload } = useAsync(() => api<NightReport[]>("/admin/night-reports"), []);
  if (loading && !data) return <Spinner />;
  if (!data) return null;
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold">🌙 Últimas noches</h3>
        <button className="btn btn-secondary btn-sm" onClick={reload}>Actualizar</button>
      </div>
      <div className="card divide-y divide-slate-100 p-0 dark:divide-slate-800">
        {data.length === 0 && <p className="p-4 text-sm text-slate-500">Todavía no hay resúmenes: el primero se guarda al empezar la próxima noche hábil.</p>}
        {data.map((r) => (
          <details key={r.night} className="group px-4 py-3" open={r.in_progress}>
            <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-3 gap-y-1">
              <span className="shrink-0 text-sm font-medium capitalize sm:w-36">{nightLabel(r.night)}</span>
              <Badge status={NIGHT_TONE[r.tone][1]}>{NIGHT_TONE[r.tone][0]}</Badge>
              <span className="w-full text-sm text-slate-600 sm:w-auto sm:min-w-0 sm:flex-1 dark:text-slate-300">{r.headline}</span>
              {r.items.length > 0 && <span className="text-xs text-slate-400 group-open:hidden">Ver detalle ▾</span>}
            </summary>
            {r.items.length > 0 && (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full">
                  <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Proyecto</th><th className="th">Modelo</th><th className="th">Duración estimada</th><th className="th">Resultado</th></tr></thead>
                  <tbody>
                    {r.items.map((i) => {
                      const postponed = i.job?.status === "cancelled" && i.job.phase?.startsWith("Pospuesto");
                      return (
                        <tr key={`${i.project_id}-${i.base_model}`} className="border-b border-slate-100 last:border-0 dark:border-slate-800">
                          <td className="td">{i.project}</td><td className="td">{i.model}</td>
                          <td className="td whitespace-nowrap">{fmtDuration(i.estimated_seconds)}</td>
                          <td className="td">
                            {i.job
                              ? <Badge status={postponed ? "pending" : i.job.status}>{postponed ? "Pospuesto al sábado" : undefined}</Badge>
                              : <Badge status={REPORT_DECISION[i.decision][1]}>{REPORT_DECISION[i.decision][0]}</Badge>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {r.not_ready.length > 0 && <p className="mt-2 text-xs text-slate-500">Sin datos suficientes: {r.not_ready.map((n) => n.project).join(", ")}.</p>}
          </details>
        ))}
      </div>
    </div>
  );
}
