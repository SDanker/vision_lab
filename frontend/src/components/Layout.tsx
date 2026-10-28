import { useEffect, useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useJobs } from "../lib/jobs";
import { api } from "../lib/api";
import type { NotificationItem } from "../lib/types";
import { fmtDate } from "../lib/format";
import { Alert } from "./ui";

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      /* almacenamiento no disponible */
    }
  };
  return { dark, toggle };
}

export function ThemeButton() {
  const { dark, toggle } = useTheme();
  return (
    <button className="btn btn-secondary btn-sm" onClick={toggle} aria-label="Cambiar entre modo claro y oscuro" title="Modo claro / oscuro">
      {dark ? "☀️ Claro" : "🌙 Oscuro"}
    </button>
  );
}

function Bell() {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<{ unread: number; items: NotificationItem[] }>({ unread: 0, items: [] });
  const { jobs } = useJobs();
  const load = () => api<typeof data>("/notifications").then(setData).catch(() => undefined);
  useEffect(() => {
    void load();
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    void load();
  }, [jobs.length]);
  const readAll = async () => {
    await api("/notifications/read-all", { method: "POST" });
    void load();
  };
  return (
    <div className="relative">
      <button className="btn btn-secondary btn-sm" onClick={() => setOpen(!open)} aria-label="Notificaciones">
        🔔{data.unread > 0 && <span className="rounded-full bg-red-600 px-1.5 text-xs text-white">{data.unread}</span>}
      </button>
      {open && (
        <div className="card absolute right-0 z-40 mt-2 max-h-96 w-80 overflow-auto p-3">
          <div className="mb-2 flex items-center justify-between">
            <strong className="text-sm">Notificaciones</strong>
            <button className="text-xs text-brand-600 hover:underline" onClick={readAll}>Marcar todas como leídas</button>
          </div>
          {data.items.length === 0 && <p className="py-4 text-center text-sm text-slate-500">Sin notificaciones.</p>}
          {data.items.map((n) => (
            <div key={n.id} className={`border-t border-slate-100 py-2 text-sm dark:border-slate-800 ${n.read ? "opacity-60" : ""}`}>
              <div className="font-medium">{n.title}</div>
              <div className="text-slate-600 dark:text-slate-400">{n.body}</div>
              <div className="text-xs text-slate-400">{fmtDate(n.created_at)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function LockBanner() {
  const { schedule } = useJobs();
  if (!schedule || schedule.open) return null;
  return (
    <div className="mb-5">
      <Alert tone="warning">
        <strong>Modificaciones bloqueadas.</strong> {schedule.message} Se vuelve a evaluar automáticamente; el entrenamiento semanal inicia el sábado a las 00:05 (hora de Chile).
      </Alert>
    </div>
  );
}

/** «¿Todavía te faltan cargas?»: el entrenamiento nocturno pregunta antes de congelar un proyecto en uso. */
export function NightCheckBanner() {
  const { nightChecks, answerNight } = useJobs();
  const [busy, setBusy] = useState<number | null>(null);
  if (!nightChecks.length) return null;
  const answer = async (id: number, still: boolean) => {
    setBusy(id);
    try {
      await answerNight(id, still);
    } catch {
      /* ya se había decidido: el aviso desaparece igual */
    } finally {
      setBusy(null);
    }
  };
  return (
    <div className="mb-5 space-y-2" aria-live="polite">
      {nightChecks.map((c) => (
        <div key={c.id} className="rounded-xl border border-indigo-200 bg-indigo-50 p-4 text-sm text-indigo-950 dark:border-indigo-900 dark:bg-indigo-950/50 dark:text-indigo-100">
          <div className="font-semibold">🌙 Esta noche se va a entrenar «{c.project_name}». ¿Todavía te faltan cargas?</div>
          <p className="mt-1 text-indigo-900/80 dark:text-indigo-200/80">
            Vimos que estás trabajando en este proyecto. Si sigues cargando, no lo congelamos y se entrena otra noche o el fin de semana.
            Si no respondes en {c.timeout_minutes} minutos, se entrena sólo si ya dejaste de cargar.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button className="btn btn-primary btn-sm" disabled={busy === c.id} onClick={() => void answer(c.id, true)}>Sí, sigo cargando</button>
            <button className="btn btn-secondary btn-sm" disabled={busy === c.id} onClick={() => void answer(c.id, false)}>No, ya terminé: entrénalo</button>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Layout() {
  const { user, logout } = useAuth();
  const { connected } = useJobs();
  const nav = useNavigate();
  const [menu, setMenu] = useState(false);
  const items = [
    ["/", "Inicio"],
    ["/proyectos", "Proyectos"],
    ["/entrenamiento", "Entrenamiento"],
    ["/modelos", "Modelos"],
    ["/probar", "Probar"],
    ["/historial", "Historial"],
    ["/cuenta", "Cuenta"],
    ...(user?.role === "ADMIN" ? [["/admin", "Administración"]] : []),
  ];
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
          <button className="btn btn-secondary btn-sm md:hidden" onClick={() => setMenu(!menu)} aria-label="Menú">☰</button>
          <div className="flex items-center gap-2 text-lg font-bold">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600 text-white">V</span>
            <span className="hidden sm:inline">Vision Lab AI</span>
          </div>
          <nav className="hidden flex-1 gap-1 md:flex" aria-label="Principal">
            {items.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `rounded-lg px-3 py-1.5 text-sm font-medium ${isActive ? "bg-brand-50 text-brand-700 dark:bg-brand-700/20 dark:text-brand-100" : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"}`}>
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <span title={connected ? "Progreso en tiempo real conectado" : "Reconectando…"} className={`h-2.5 w-2.5 rounded-full ${connected ? "bg-emerald-500" : "bg-amber-500"}`} />
            <Bell />
            <ThemeButton />
            <span className="hidden text-sm text-slate-500 lg:inline">@{user?.username}</span>
            <button className="btn btn-secondary btn-sm" onClick={async () => { await logout(); nav("/login"); }}>Salir</button>
          </div>
        </div>
        {menu && (
          <nav className="border-t border-slate-200 p-2 md:hidden dark:border-slate-800" aria-label="Móvil">
            {items.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"} onClick={() => setMenu(false)} className="block rounded-lg px-3 py-2 text-sm font-medium hover:bg-slate-100 dark:hover:bg-slate-800">
                {label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>
      <main className="mx-auto min-w-0 max-w-7xl px-4 py-6">
        <NightCheckBanner />
        <LockBanner />
        <Outlet />
      </main>
    </div>
  );
}
