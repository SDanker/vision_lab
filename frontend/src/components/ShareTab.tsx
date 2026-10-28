import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api, errMsg } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtDate } from "../lib/format";
import type { Project, ProjectMember } from "../lib/types";
import { Alert, Spinner, useAsync } from "./ui";

export const ROLE_LABEL = { owner: "Propietario", editor: "Editor", viewer: "Sólo lectura" } as const;

interface Members {
  owner: { user_id: number; username: string; name: string };
  members: ProjectMember[];
  my_role: Project["my_role"];
}

const ROLE_HELP: Record<"editor" | "viewer", string> = {
  editor: "Puede cargar datos (imágenes, audios o CSV), crear categorías, mover o borrar archivos y ajustar los parámetros de entrenamiento.",
  viewer: "Sólo puede ver el proyecto, sus modelos y probarlos. No cambia nada.",
};

/** Compartir el proyecto con otros usuarios de la plataforma. */
export default function ShareTab({ project, reload }: { project: Project; reload: () => void }) {
  const nav = useNavigate();
  const { data, loading, reload: reloadMembers } = useAsync(() => api<Members>(`/projects/${project.id}/members`), [project.id]);
  const [identifier, setIdentifier] = useState("");
  const [role, setRole] = useState<"editor" | "viewer">("editor");
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const { user } = useAuth();
  const isOwner = project.my_role === "owner";
  const mine = data?.members.find((m) => m.user_id === user?.id);

  const refresh = () => {
    reloadMembers();
    reload();
  };
  const add = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api(`/projects/${project.id}/members`, { body: { identifier, role } });
      setMsg({ tone: "success", text: `Proyecto compartido con ${identifier.trim()}. Le llegó una notificación.` });
      setIdentifier("");
      refresh();
    } catch (err) {
      setMsg({ tone: "error", text: errMsg(err) });
    } finally {
      setBusy(false);
    }
  };
  const changeRole = async (m: ProjectMember, next: "editor" | "viewer") => {
    try {
      await api(`/projects/${project.id}/members/${m.user_id}`, { method: "PATCH", body: { role: next } });
      setMsg(null);
      refresh();
    } catch (err) {
      setMsg({ tone: "error", text: errMsg(err) });
    }
  };
  const remove = async (m: ProjectMember, leaving: boolean) => {
    const q = leaving ? `¿Abandonar el proyecto «${project.name}»? Dejarás de verlo en tu lista.` : `¿Quitar el acceso de ${m.username}?`;
    if (!confirm(q)) return;
    try {
      await api(`/projects/${project.id}/members/${m.user_id}`, { method: "DELETE" });
      if (leaving) nav("/proyectos");
      else refresh();
    } catch (err) {
      setMsg({ tone: "error", text: errMsg(err) });
    }
  };

  if (loading && !data) return <Spinner />;
  return (
    <div className="space-y-6">
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}

      {isOwner && (
        <form className="card space-y-3" onSubmit={add}>
          <h3 className="font-semibold">Compartir con otro usuario</h3>
          <div className="flex flex-wrap items-end gap-2">
            <label className="block min-w-[14rem] flex-1">
              <span className="label">Usuario o correo</span>
              <input className="w-full" value={identifier} onChange={(e) => setIdentifier(e.target.value)} placeholder="nombre.usuario" required maxLength={255} />
            </label>
            <label className="block">
              <span className="label">Permiso</span>
              <select value={role} onChange={(e) => setRole(e.target.value as "editor" | "viewer")}>
                <option value="editor">Editor</option>
                <option value="viewer">Sólo lectura</option>
              </select>
            </label>
            <button className="btn btn-primary" disabled={busy || !identifier.trim()}>Compartir</button>
          </div>
          <p className="text-xs text-slate-500">{ROLE_HELP[role]}</p>
          <p className="text-xs text-slate-500">
            Los archivos que cargue un editor cuentan en la cuota de almacenamiento del propietario. Sólo el propietario puede eliminar el proyecto o una categoría completa y gestionar quién tiene acceso. Los entrenamientos siguen el calendario semanal de siempre.
          </p>
        </form>
      )}

      <div className="card overflow-x-auto p-0">
        <div className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">Personas con acceso</div>
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-800">
              <th className="th">Usuario</th><th className="th">Permiso</th><th className="th">Desde</th><th className="th" />
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-100 dark:border-slate-800">
              <td className="td font-medium">{data?.owner.name} <span className="font-normal text-slate-500">@{data?.owner.username}</span></td>
              <td className="td">{ROLE_LABEL.owner}</td>
              <td className="td">—</td>
              <td className="td" />
            </tr>
            {data?.members.map((m) => (
              <tr key={m.user_id} className="border-b border-slate-100 dark:border-slate-800">
                <td className="td font-medium">
                  {m.name} <span className="font-normal text-slate-500">@{m.username}</span>
                  {m.user_id === user?.id && <span className="ml-2 rounded bg-slate-100 px-1.5 text-xs font-normal dark:bg-slate-800">tú</span>}
                </td>
                <td className="td">
                  {isOwner ? (
                    <select value={m.role} onChange={(e) => void changeRole(m, e.target.value as "editor" | "viewer")} aria-label={`Permiso de ${m.username}`}>
                      <option value="editor">Editor</option>
                      <option value="viewer">Sólo lectura</option>
                    </select>
                  ) : (
                    ROLE_LABEL[m.role]
                  )}
                </td>
                <td className="td whitespace-nowrap">{fmtDate(m.added_at)}</td>
                <td className="td text-right">
                  {isOwner && <button className="btn btn-secondary btn-sm" onClick={() => void remove(m, false)}>Quitar</button>}
                </td>
              </tr>
            ))}
            {data && data.members.length === 0 && (
              <tr><td className="td text-slate-500" colSpan={4}>El proyecto todavía no está compartido con nadie.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {!isOwner && (
        <div className="card flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-slate-500">Tienes acceso de <strong>{project.my_role ? ROLE_LABEL[project.my_role] : ""}</strong> a este proyecto de @{project.owner}.</p>
          {mine && <button className="btn btn-secondary btn-sm" onClick={() => void remove(mine, true)}>Abandonar proyecto</button>}
        </div>
      )}
    </div>
  );
}
