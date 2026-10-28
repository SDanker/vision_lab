import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { api, errMsg } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtBytes, fmtDate } from "../lib/format";
import type { HfImportItem, HfStatus, Project } from "../lib/types";
import ImportDialog from "./HfImportDialog";
import { Alert, Badge, Field, Progress, Spinner, useAsync } from "./ui";

const HF_TOKENS_URL = "https://huggingface.co/settings/tokens";

/** Tarjeta de «Mi cuenta»: conectar la cuenta de Hugging Face para bajar datasets privados o restringidos. */
export function HfAccountCard() {
  const { data, loading, error, reload } = useAsync(() => api<HfStatus>("/integrations/huggingface"), []);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);

  const connect = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    try {
      const s = await api<HfStatus>("/integrations/huggingface", { method: "PUT", body: { token: token.trim() } });
      setToken(""); // el valor deja de estar en la pantalla en cuanto el servidor lo guardó
      setMsg({ tone: "success", text: `Cuenta @${s.username} conectada.` });
      reload();
    } catch (err) {
      setMsg({ tone: "error", text: errMsg(err) });
    } finally {
      setBusy(false);
    }
  };
  const disconnect = async () => {
    if (!confirm("¿Desconectar tu cuenta de Hugging Face? Se borra el token guardado; no podrás importar datasets privados o restringidos hasta volver a conectarla.")) return;
    setBusy(true);
    setMsg(null);
    try {
      await api("/integrations/huggingface", { method: "DELETE" });
      reload();
    } catch (err) {
      setMsg({ tone: "error", text: errMsg(err) });
    } finally {
      setBusy(false);
    }
  };

  const connected = !!data?.connected;
  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold">🤗 Hugging Face</h3>
        {loading && !data ? <Spinner /> : connected && <Badge status={data?.needs_reconnect ? "SUSPENDED" : "APPROVED"}>{data?.needs_reconnect ? "Reconectar" : "Conectada"}</Badge>}
      </div>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Los datasets públicos se importan sin cuenta. Conecta la tuya para importar los <strong>privados</strong> y los <strong>restringidos</strong> (los que piden aceptar condiciones). Cada persona usa su propia cuenta.
      </p>
      {error && <Alert tone="error">{error}</Alert>}
      {msg && <Alert tone={msg.tone}>{msg.text}</Alert>}

      {connected ? (
        <div className="space-y-2 text-sm">
          {data?.needs_reconnect && <Alert tone="warning">No se pudo leer el token guardado. Vuelve a pegarlo abajo para reconectar.</Alert>}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              Conectada como <strong>@{data?.username}</strong> <span className="font-mono text-xs text-slate-500">({data?.hint})</span>
              <div className="text-xs text-slate-500">Conectada el {fmtDate(data?.connected_at)} · último uso: {fmtDate(data?.last_used_at)}</div>
            </div>
            <button className="btn btn-secondary btn-sm" disabled={busy} onClick={() => void disconnect()}>Desconectar</button>
          </div>
        </div>
      ) : null}

      {(!connected || data?.needs_reconnect) && (
        <form className="space-y-2" onSubmit={(e) => void connect(e)}>
          <Field label="Token de acceso de Hugging Face" hint="Basta un token de sólo lectura. Se guarda cifrado en el servidor y no se vuelve a mostrar.">
            <input
              className="w-full font-mono text-sm"
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="hf_…"
              maxLength={200}
              required
            />
          </Field>
          <div className="flex flex-wrap items-center gap-3">
            <button className="btn btn-primary" disabled={busy || token.trim().length < 10}>{busy ? "Comprobando…" : "Conectar cuenta"}</button>
            <a className="text-sm text-brand-600 underline" href={HF_TOKENS_URL} target="_blank" rel="noreferrer">Crear un token en huggingface.co</a>
          </div>
        </form>
      )}
      <details className="text-xs text-slate-500">
        <summary className="cursor-pointer">¿Cómo se protege mi token?</summary>
        <ul className="ml-5 mt-1 list-disc space-y-0.5">
          <li>Se guarda cifrado; nadie lo ve en pantalla, ni siquiera los administradores.</li>
          <li>Sólo se usa para descargar los datasets que tú pides importar, y únicamente se envía a huggingface.co.</li>
          <li>Los demás usuarios, incluidos los editores de tus proyectos, usan su propia cuenta, nunca la tuya.</li>
          <li>Al desconectar se borra de inmediato. Si crees que se filtró, revócalo también en huggingface.co.</li>
        </ul>
      </details>
    </div>
  );
}

const ACTIVE = ["pending", "running"];

/** Importar un dataset de Hugging Face al proyecto + historial de importaciones. */
export function HfImport({ project, locked, onChanged }: { project: Project; locked: boolean; onChanged: () => void }) {
  const { user } = useAuth();
  const [items, setItems] = useState<HfImportItem[]>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wasActive = useRef(false);
  const onChangedRef = useRef(onChanged);
  onChangedRef.current = onChanged;

  const load = useCallback(() => {
    api<HfImportItem[]>(`/projects/${project.id}/hf/imports`).then(setItems).catch((e) => setError(errMsg(e)));
  }, [project.id]);
  useEffect(load, [load]);

  const active = items.some((i) => ACTIVE.includes(i.status));
  useEffect(() => {
    if (!active) {
      if (wasActive.current) onChangedRef.current(); // terminó: se refrescan categorías e imágenes
      wasActive.current = false;
      return;
    }
    wasActive.current = true;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [active, load]);

  const cancel = async (i: HfImportItem) => {
    if (!confirm("¿Cancelar la importación? Lo que ya se importó se conserva.")) return;
    try {
      await api(`/hf/imports/${i.id}`, { method: "DELETE" });
      load();
    } catch (e) {
      setError(errMsg(e));
    }
  };

  const canCancel = (i: HfImportItem) => ACTIVE.includes(i.status) && !i.cancel_requested && (i.user === user?.username || project.my_role === "owner");

  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-semibold">🤗 Importar desde Hugging Face</h3>
          <p className="text-sm text-slate-600 dark:text-slate-400">Descarga un dataset del Hub directamente al proyecto. Cada carpeta o etiqueta pasa a ser una categoría.</p>
        </div>
        <button className="btn btn-primary" disabled={locked || active} onClick={() => setOpen(true)} title={active ? "Ya hay una importación en curso" : undefined}>
          Importar dataset
        </button>
      </div>
      {error && <Alert tone="error">{error}</Alert>}
      {items.length > 0 && (
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {items.slice(0, active ? 5 : 3).map((i) => (
            <li key={i.id} className="space-y-1.5 py-2.5 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <a className="font-medium text-brand-600 underline" href={`https://huggingface.co/datasets/${i.repo_id}`} target="_blank" rel="noreferrer">{i.repo_id}</a>
                  <Badge status={i.status} />
                  <span className="text-xs text-slate-500">por @{i.user} · {fmtDate(i.created_at)}</span>
                </div>
                {canCancel(i) && <button className="btn btn-secondary btn-sm" onClick={() => void cancel(i)}>Cancelar</button>}
              </div>
              {ACTIVE.includes(i.status) && (
                <div className="space-y-1" aria-live="polite">
                  {i.percent != null ? <Progress value={i.percent} /> : <div className="text-xs text-slate-500">{i.phase ?? "En cola"}…</div>}
                  <div className="text-xs text-slate-500">
                    {i.phase}{i.total > 0 && ` · ${i.processed.toLocaleString("es-CL")} de ${i.total.toLocaleString("es-CL")}`} · {i.imported.toLocaleString("es-CL")} importadas
                  </div>
                </div>
              )}
              {!ACTIVE.includes(i.status) && (
                <div className="text-xs text-slate-500">
                  {i.imported.toLocaleString("es-CL")} importadas
                  {i.duplicates > 0 && ` · ${i.duplicates.toLocaleString("es-CL")} repetidas omitidas`}
                  {i.rejected > 0 && ` · ${i.rejected.toLocaleString("es-CL")} rechazadas`}
                  {i.unlabeled > 0 && ` · ${i.unlabeled.toLocaleString("es-CL")} sin categoría omitidas`}
                  {i.skipped > 0 && ` · ${i.skipped.toLocaleString("es-CL")} fuera por los filtros o máximos`}
                  {i.imported > 0 && ` · ${fmtBytes(i.total_bytes)}`}
                </div>
              )}
              {i.error && <Alert tone="error">{i.error}</Alert>}
              {!ACTIVE.includes(i.status) && i.rejected_sample.length > 0 && (
                <details className="text-xs text-slate-500">
                  <summary className="cursor-pointer">Ver motivos de rechazo</summary>
                  <ul className="ml-5 list-disc">{i.rejected_sample.map((r, k) => <li key={k}>{r.name}: {r.reason}</li>)}</ul>
                </details>
              )}
            </li>
          ))}
        </ul>
      )}
      {open && <ImportDialog project={project} onClose={() => setOpen(false)} onStarted={() => { setOpen(false); load(); }} />}
    </div>
  );
}
