import { useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { api, errMsg } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ThemeButton } from "../components/Layout";
import { Alert, Field } from "../components/ui";

function Shell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="grid min-h-screen place-items-center p-4">
      <div className="absolute right-4 top-4"><ThemeButton /></div>
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-xl bg-brand-600 text-xl font-bold text-white">V</div>
          <h1 className="text-2xl font-bold">Vision Lab AI</h1>
          <p className="text-sm text-slate-500">{title}</p>
        </div>
        <div className="card">{children}</div>
      </div>
    </div>
  );
}

export function Login() {
  const { user, login } = useAuth();
  const nav = useNavigate();
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to="/" replace />;
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(identifier, password);
      nav("/");
    } catch (err) {
      setError(errMsg(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Shell title="Inicia sesión para continuar">
      <form onSubmit={submit} className="space-y-4">
        {error && <Alert tone="error">{error}</Alert>}
        <Field label="Usuario o correo"><input className="w-full" value={identifier} onChange={(e) => setIdentifier(e.target.value)} autoComplete="username" required autoFocus /></Field>
        <Field label="Contraseña"><input className="w-full" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required /></Field>
        <button className="btn btn-primary w-full" disabled={busy}>{busy ? "Ingresando…" : "Ingresar"}</button>
      </form>
      <p className="mt-4 text-center text-sm text-slate-500">¿No tienes cuenta? <Link className="text-brand-600 hover:underline" to="/registro">Solicita acceso</Link></p>
    </Shell>
  );
}

export function Register() {
  const [form, setForm] = useState({ first_name: "", last_name: "", email: "", username: "", password: "", password_confirm: "", reason: "" });
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) => setForm({ ...form, [k]: e.target.value });
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (form.password !== form.password_confirm) return setError("Las contraseñas no coinciden.");
    setBusy(true);
    try {
      const r = await api<{ message: string }>("/auth/register", { body: { ...form, reason: form.reason || null } });
      setDone(r.message);
    } catch (err) {
      setError(errMsg(err));
    } finally {
      setBusy(false);
    }
  };
  if (done)
    return (
      <Shell title="Solicitud enviada">
        <Alert tone="success">{done}</Alert>
        <Link className="btn btn-primary mt-4 w-full" to="/login">Ir a iniciar sesión</Link>
      </Shell>
    );
  return (
    <Shell title="Solicita acceso a la plataforma">
      <form onSubmit={submit} className="space-y-3">
        {error && <Alert tone="error">{error}</Alert>}
        <div className="grid grid-cols-2 gap-3">
          <Field label="Nombre"><input className="w-full" value={form.first_name} onChange={set("first_name")} required maxLength={80} /></Field>
          <Field label="Apellido"><input className="w-full" value={form.last_name} onChange={set("last_name")} required maxLength={80} /></Field>
        </div>
        <Field label="Correo electrónico"><input className="w-full" type="email" value={form.email} onChange={set("email")} required autoComplete="email" /></Field>
        <Field label="Nombre de usuario" hint="3-32 caracteres: letras, números, punto, guion y guion bajo."><input className="w-full" value={form.username} onChange={set("username")} required pattern="[A-Za-z0-9_.\-]{3,32}" autoComplete="username" /></Field>
        <Field label="Contraseña" hint="Mínimo 10 caracteres, con letras y números."><input className="w-full" type="password" value={form.password} onChange={set("password")} required minLength={10} autoComplete="new-password" /></Field>
        <Field label="Confirmar contraseña"><input className="w-full" type="password" value={form.password_confirm} onChange={set("password_confirm")} required autoComplete="new-password" /></Field>
        <Field label="Motivo por el que deseas utilizar la plataforma (opcional)"><textarea className="w-full" rows={3} maxLength={500} value={form.reason} onChange={set("reason")} /></Field>
        <button className="btn btn-primary w-full" disabled={busy}>{busy ? "Enviando…" : "Solicitar acceso"}</button>
      </form>
      <p className="mt-4 text-center text-sm text-slate-500">¿Ya tienes cuenta? <Link className="text-brand-600 hover:underline" to="/login">Inicia sesión</Link></p>
    </Shell>
  );
}

/** Única pantalla disponible para cuentas pendientes o rechazadas. */
export function ApprovalStatus() {
  const { user, refresh, logout } = useAuth();
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);
  if (!user) return <Navigate to="/login" replace />;
  if (user.status === "APPROVED") return <Navigate to="/" replace />;
  const pending = user.status === "PENDING";
  return (
    <Shell title="Estado de tu solicitud">
      <div className="space-y-4">
        <Alert tone={pending ? "info" : "error"}>{user.status_message}</Alert>
        <dl className="text-sm">
          <div className="flex justify-between py-1"><dt className="text-slate-500">Usuario</dt><dd>@{user.username}</dd></div>
          <div className="flex justify-between py-1"><dt className="text-slate-500">Correo</dt><dd>{user.email}</dd></div>
          <div className="flex justify-between py-1"><dt className="text-slate-500">Estado</dt><dd className="font-medium">{pending ? "Pendiente de aprobación" : "Rechazada"}</dd></div>
        </dl>
        <div className="flex gap-2">
          <button className="btn btn-secondary flex-1" disabled={busy} onClick={async () => { setBusy(true); await refresh(); setBusy(false); }}>Actualizar estado</button>
          <button className="btn btn-secondary flex-1" onClick={async () => { await logout(); nav("/login"); }}>Salir</button>
        </div>
      </div>
    </Shell>
  );
}
