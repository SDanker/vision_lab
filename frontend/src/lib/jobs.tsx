import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "./api";
import { useAuth } from "./auth";
import type { Job, NightCheck, Schedule } from "./types";

interface JobsCtx {
  jobs: Job[];
  schedule: Schedule | null;
  connected: boolean;
  reload: () => void;
  /** Preguntas nocturnas pendientes para mí («¿todavía te faltan cargas?»). */
  nightChecks: NightCheck[];
  answerNight: (id: number, stillUploading: boolean) => Promise<void>;
}

const Ctx = createContext<JobsCtx | null>(null);

/** Estado compartido de entrenamientos: carga inicial + eventos SSE en tiempo real. */
export function JobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [connected, setConnected] = useState(false);
  const [nightChecks, setNightChecks] = useState<NightCheck[]>([]);
  const lastLock = useRef<string | null>(null);
  const { user } = useAuth();
  const me = useRef<number | undefined>(user?.id);
  me.current = user?.id;

  const loadChecks = () => api<NightCheck[]>("/night-checks/mine").then(setNightChecks).catch(() => undefined);
  const reload = () => {
    api<Job[]>("/training/jobs").then(setJobs).catch(() => undefined);
    api<Schedule>("/schedule").then(setSchedule).catch(() => undefined);
    loadChecks();
  };
  const answerNight = async (id: number, stillUploading: boolean) => {
    try {
      await api(`/night-checks/${id}/answer`, { body: { still_uploading: stillUploading } });
    } finally {
      setNightChecks((prev) => prev.filter((c) => c.id !== id));
    }
  };

  useEffect(() => {
    reload();
    const poll = setInterval(() => {
      api<Schedule>("/schedule").then(setSchedule).catch(() => undefined);
      loadChecks();
    }, 30_000);
    const es = new EventSource("/api/training/stream");
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.addEventListener("job", (ev) => {
      const payload = JSON.parse((ev as MessageEvent).data) as { job: Job };
      setJobs((prev) => {
        const old = prev.find((j) => j.id === payload.job.id);
        const merged = { ...old, ...payload.job, queue_position: payload.job.queue_position ?? old?.queue_position ?? null };
        const rest = prev.filter((j) => j.id !== payload.job.id);
        return [merged, ...rest].sort((a, b) => b.id - a.id);
      });
      // los cambios de estado pueden abrir/cerrar el bloqueo semanal y alterar posiciones en cola
      const key = `${payload.job.status}`;
      if (key !== lastLock.current) {
        lastLock.current = key;
        api<Schedule>("/schedule").then(setSchedule).catch(() => undefined);
        if (["completed", "error", "cancelled", "preparing"].includes(key)) api<Job[]>("/training/jobs").then(setJobs).catch(() => undefined);
      }
    });
    es.addEventListener("night_check", (ev) => {
      const payload = JSON.parse((ev as MessageEvent).data) as { user_ids: number[]; check: NightCheck };
      if (me.current === undefined || !payload.user_ids.includes(me.current)) return; // los admin reciben todo
      setNightChecks((prev) => {
        const rest = prev.filter((c) => c.id !== payload.check.id);
        return payload.check.status === "asking" ? [...rest, payload.check] : rest;
      });
    });
    return () => {
      clearInterval(poll);
      es.close();
    };
  }, []);

  return <Ctx.Provider value={{ jobs, schedule, connected, reload, nightChecks, answerNight }}>{children}</Ctx.Provider>;
}

export function useJobs(): JobsCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("JobsProvider ausente");
  return c;
}
