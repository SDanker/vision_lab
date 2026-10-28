import { useState, type FormEvent } from "react";
import { api, errMsg } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtDate } from "../lib/format";
import type { ApiTokenItem, NewApiToken, TokenRequestItem } from "../lib/types";
import { HfAccountCard } from "../components/HuggingFace";
import { Alert, Badge, Field, PageTitle, Spinner, useAsync } from "../components/ui";

interface Tokens {
  tokens: ApiTokenItem[];
  requests: TokenRequestItem[];
  limits: { read: number; write: number };
}

export const TOKEN_STATE: Record<string, [string, string]> = {
  active: ["APPROVED", "Activo"],
  expired: ["SUSPENDED", "Vencido"],
  revoked: ["REJECTED", "Revocado"],
};
export const REQUEST_STATE: Record<string, [string, string]> = {
  pending: ["PENDING", "Pendiente de aprobación"],
  approved: ["APPROVED", "Aprobada"],
  rejected: ["REJECTED", "Rechazada"],
  cancelled: ["cancelled", "Cancelada"],
};
export const SCOPE_LABEL = { read: "Lectura", write: "Lectura y escritura" } as const;

function StateBadge({ map, value }: { map: Record<string, [string, string]>; value: string }) {
  const [tone, label] = map[value] ?? ["pending", value];
  return <Badge status={tone}>{label}</Badge>;
}

/** Muestra el token recién creado. Es la única vez que se puede leer. */
function FreshToken({ token, onClose }: { token: NewApiToken; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(token.token);
      setCopied(true);
    } catch {
      setCopied(false); // sin HTTPS el navegador no permite copiar: se puede seleccionar el texto a mano
    }
  };
  return (
    <div className="card space-y-3 border-emerald-300 dark:border-emerald-800">
      <h3 className="font-semibold">Token «{token.name}» creado ({SCOPE_LABEL[token.scope].toLowerCase()})</h3>
      <Alert tone="warning">
        <strong>Cópialo ahora.</strong> Por seguridad no se vuelve a mostrar; si lo pierdes, revócalo y crea otro. Trátalo como una contraseña.
      </Alert>
      <div className="flex flex-wrap gap-2">
        <input className="min-w-[16rem] flex-1 font-mono text-sm" readOnly value={token.token} onFocus={(e) => e.currentTarget.select()} aria-label="Valor del token" />
        <button className="btn btn-primary" onClick={() => void copy()}>{copied ? "Copiado ✓" : "Copiar"}</button>
        <button className="btn btn-secondary" onClick={onClose}>Ya lo guardé</button>
      </div>
      {token.expires_at && <p className="text-xs text-slate-500">Vence el {fmtDate(token.expires_at)}.</p>}
    </div>
  );
}

export default function Account() {
  const { user } = useAuth();
  const { data, loading, error, reload } = useAsync(() => api<Tokens>("/tokens"), []);
  const [fresh, setFresh] = useState<NewApiToken | null>(null);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const [name, setName] = useState("");
  const [days, setDays] = useState("90");
  const [wName, setWName] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setMsg(null);
    try {
      await fn();
    } catch (e) {
      setMsg({ tone: "error", text: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const createRead = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      setFresh(await api<NewApiToken>("/tokens", { body: { name, expires_days: days ? Number(days) : null } }));
      setName("");
      reload();
    });
  };
  const requestWrite = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      await api("/tokens/requests", { body: { name: wName, reason } });
      setMsg({ tone: "success", text: "Solicitud enviada. Un administrador la revisará y te avisaremos con una notificación." });
      setWName("");
      setReason("");
      reload();
    });
  };
  const generate = (r: TokenRequestItem) =>
    run(async () => {
      setFresh(await api<NewApiToken>(`/tokens/requests/${r.id}/generate`, { method: "POST" }));
      reload();
    });
  const cancel = (r: TokenRequestItem) =>
    run(async () => {
      await api(`/tokens/requests/${r.id}`, { method: "DELETE" });
      reload();
    });
  const revoke = (t: ApiTokenItem) => {
    if (!confirm(`¿Revocar el token «${t.name}»? Los programas que lo usen dejarán de funcionar de inmediato.`)) return;
    void run(async () => {
      await api(`/tokens/${t.id}`, { method: "DELETE" });
      reload();
    });
  };

  if (loading && !data) return <Spinner />;
  const tokens = data?.tokens ?? [];
  const requests = data?.requests ?? [];
  const activeRead = tokens.filter((t) => t.scope === "read" && t.state === "active").length;
  const hasWriteSlot = tokens.some((t) => t.scope === "write" && t.state === "active") || requests.some((r) => r.status === "pending" || r.can_generate);
  const origin = window.location.origin;

  return (
    <div className="space-y-6">
      <PageTitle title="Cuenta e integraciones" />
      <HfAccountCard />

      <div className="card space-y-2 text-sm">
        <p>
          Un <strong>token de API</strong> permite que un programa o script consulte la plataforma como si fueras tú (@{user?.username}), sin usar tu contraseña.
          Siempre tiene <strong>los mismos permisos que tu cuenta</strong>: ve tus proyectos y los que te compartieron (propietario, editor o sólo lectura), y nunca más.
        </p>
        <ul className="ml-5 list-disc text-slate-600 dark:text-slate-400">
          <li><strong>Lectura:</strong> lo creas tú. Sirve para ver proyectos, el progreso de los entrenamientos y los modelos.</li>
          <li><strong>Lectura y escritura:</strong> lo pides aquí y un administrador debe aprobarlo. Puede cargar datos (imágenes, audios o CSV), crear categorías o ajustar parámetros, pero sólo donde tu cuenta ya puede hacerlo y respetando el calendario semanal.</li>
          <li>Ningún token puede usar el panel de administración ni crear otros tokens.</li>
        </ul>
      </div>

      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}
      {error && <Alert tone="error">{error}</Alert>}
      {fresh && <FreshToken token={fresh} onClose={() => setFresh(null)} />}

      <form className="card space-y-3" onSubmit={createRead}>
        <h3 className="font-semibold">Crear token de lectura</h3>
        <div className="flex flex-wrap items-end gap-2">
          <div className="min-w-[14rem] flex-1">
            <Field label="Nombre (para reconocerlo)">
              <input className="w-full" value={name} onChange={(e) => setName(e.target.value)} placeholder="Panel de la oficina" maxLength={80} required />
            </Field>
          </div>
          <Field label="Vigencia">
            <select value={days} onChange={(e) => setDays(e.target.value)}>
              <option value="30">30 días</option>
              <option value="90">90 días</option>
              <option value="365">1 año</option>
              <option value="">Sin vencimiento</option>
            </select>
          </Field>
          <button className="btn btn-primary" disabled={busy || !name.trim() || activeRead >= (data?.limits.read ?? 5)}>Crear token</button>
        </div>
        <p className="text-xs text-slate-500">Puedes tener hasta {data?.limits.read ?? 5} tokens de lectura activos ({activeRead} en uso).</p>
      </form>

      <div className="card space-y-3">
        <h3 className="font-semibold">Token con escritura (requiere aprobación)</h3>
        {!hasWriteSlot && (
          <form className="space-y-3" onSubmit={requestWrite}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Nombre">
                <input className="w-full" value={wName} onChange={(e) => setWName(e.target.value)} placeholder="Carga automática de datos" maxLength={80} required />
              </Field>
              <Field label="¿Para qué lo necesitas?" hint="El administrador lo lee para decidir (5 a 500 caracteres).">
                <textarea className="w-full" rows={2} value={reason} onChange={(e) => setReason(e.target.value)} minLength={5} maxLength={500} required />
              </Field>
            </div>
            <button className="btn btn-primary" disabled={busy || !wName.trim() || reason.trim().length < 5}>Solicitar token con escritura</button>
          </form>
        )}
        {hasWriteSlot && <p className="text-sm text-slate-500">Ya tienes un token con escritura o una solicitud en curso. Revoca el token o cancela la solicitud para pedir otro.</p>}
        {requests.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Solicitud</th><th className="th">Estado</th><th className="th">Fecha</th><th className="th">Respuesta</th><th className="th" /></tr></thead>
              <tbody>
                {requests.map((r) => (
                  <tr key={r.id} className="border-b border-slate-100 dark:border-slate-800">
                    <td className="td"><div className="font-medium">{r.name}</div><div className="max-w-xs text-xs text-slate-500">{r.reason}</div></td>
                    <td className="td"><StateBadge map={REQUEST_STATE} value={r.status} />{r.token_id != null && <div className="mt-1 text-xs text-slate-500">token generado</div>}</td>
                    <td className="td whitespace-nowrap">{fmtDate(r.created_at)}</td>
                    <td className="td text-xs text-slate-500">
                      {r.reviewed_by ? `@${r.reviewed_by} · ${fmtDate(r.reviewed_at)}` : "—"}
                      {r.status === "approved" && r.expires_days && r.can_generate && <div>Vigencia al generarlo: {r.expires_days} días</div>}
                      {r.review_note && <div>{r.review_note}</div>}
                    </td>
                    <td className="td text-right">
                      <div className="flex justify-end gap-1">
                        {r.can_generate && <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => void generate(r)}>Generar token</button>}
                        {(r.status === "pending" || r.can_generate) && <button className="btn btn-secondary btn-sm" disabled={busy} onClick={() => void cancel(r)}>Cancelar</button>}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card overflow-x-auto p-0">
        <div className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">Mis tokens</div>
        <table className="w-full">
          <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Nombre</th><th className="th">Token</th><th className="th">Permiso</th><th className="th">Estado</th><th className="th">Vence</th><th className="th">Último uso</th><th className="th" /></tr></thead>
          <tbody>
            {tokens.map((t) => (
              <tr key={t.id} className="border-b border-slate-100 dark:border-slate-800">
                <td className="td font-medium">{t.name}</td>
                <td className="td font-mono text-xs">{t.prefix}…</td>
                <td className="td">{SCOPE_LABEL[t.scope]}</td>
                <td className="td"><StateBadge map={TOKEN_STATE} value={t.state} /></td>
                <td className="td whitespace-nowrap">{t.expires_at ? fmtDate(t.expires_at) : "Nunca"}</td>
                <td className="td whitespace-nowrap">{fmtDate(t.last_used_at)}{t.last_used_ip && <div className="text-xs text-slate-500">{t.last_used_ip}</div>}</td>
                <td className="td text-right">{t.state === "active" && <button className="btn btn-secondary btn-sm" disabled={busy} onClick={() => revoke(t)}>Revocar</button>}</td>
              </tr>
            ))}
            {tokens.length === 0 && <tr><td className="td text-center text-slate-500" colSpan={7}>Todavía no tienes tokens.</td></tr>}
          </tbody>
        </table>
      </div>

      <div className="card space-y-2 text-sm">
        <h3 className="font-semibold">Cómo usarlo</h3>
        <p className="text-slate-600 dark:text-slate-400">Envía el token en la cabecera <code>Authorization</code>. Por ejemplo, para ver el progreso de tus entrenamientos:</p>
        <pre className="overflow-x-auto rounded-lg bg-slate-100 p-3 text-xs dark:bg-slate-800">{`curl -H "Authorization: Bearer vl_TU_TOKEN" ${origin}/api/training/jobs`}</pre>
        <p className="text-slate-600 dark:text-slate-400">
          Otros endpoints útiles: <code>/api/projects</code>, <code>/api/training/jobs?project_id=ID</code>, <code>/api/training/jobs/ID/logs</code>, <code>/api/models</code> y el flujo en vivo <code>/api/training/stream</code> (Server-Sent Events, <code>curl -N</code>).
          Con un token de lectura sólo funcionan las consultas (GET); las demás responden <code>403 TOKEN_READ_ONLY</code>.
        </p>
      </div>
    </div>
  );
}
