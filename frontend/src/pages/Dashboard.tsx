import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { fmtBytes, fmtDate } from "../lib/format";
import { useAuth } from "../lib/auth";
import { useJobs } from "../lib/jobs";
import type { Project } from "../lib/types";
import JobCard from "../components/JobCard";
import Guide from "../components/guide/Guide";
import { Alert, PageTitle, Spinner, Stat, useAsync } from "../components/ui";

export default function Dashboard() {
  const { user } = useAuth();
  const { jobs, schedule } = useJobs();
  const { data: projects, loading } = useAsync(() => api<Project[]>("/projects"), []);
  const active = jobs.filter((j) => ["pending", "preparing", "training", "validating", "exporting"].includes(j.status));
  const files = projects?.reduce((a, p) => a + p.image_count, 0) ?? 0; // imágenes, audios y CSV
  const bytes = projects?.reduce((a, p) => a + p.size_bytes, 0) ?? 0;
  return (
    <div>
      <PageTitle title={`Hola, ${user?.first_name}`}>
        <Link to="/proyectos" className="btn btn-primary">Mis proyectos</Link>
      </PageTitle>
      {loading && <Spinner />}
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Proyectos" value={projects?.length ?? "…"} />
        <Stat label="Archivos de datos" value={files} sub={fmtBytes(bytes)} />
        <Stat label="Entrenamientos activos" value={active.length} />
        <Stat label="Estado de cargas" value={schedule ? (schedule.open ? "Abiertas" : "Bloqueadas") : "…"} sub={schedule ? `Próximo bloqueo: ${fmtDate(schedule.next_lock_at)}` : undefined} />
      </div>
      <Alert tone="info">
        Calendario semanal (hora de Chile): carga y edita tus datos (imágenes, audios o tablas) de lunes a viernes. Las noches de lunes a viernes (00:00–08:00) se entrenan los modelos de los proyectos marcados que tardan menos de 4 horas y caben en la noche; sólo se bloquea el proyecto que se entrena, y si estás cargando a esa hora te preguntamos antes. El sábado a las 00:05 se congelan los datasets y se entrena el resto, del modelo que más tarda al que menos. Las cargas se reabren el lunes, una vez publicados los resultados, y los proyectos quedan desmarcados: revisa los resultados, ajusta y vuelve a marcar el proyecto para la semana siguiente.
      </Alert>
      <h2 className="mb-3 mt-8 text-lg font-semibold">Entrenamientos en curso</h2>
      {active.length === 0 ? (
        <p className="text-sm text-slate-500">No hay entrenamientos activos. El próximo inicio automático es el {fmtDate(schedule?.next_training_start)}.</p>
      ) : (
        <div className="space-y-4">{active.map((j) => <JobCard key={j.id} job={j} />)}</div>
      )}
      <Guide projects={projects} />
    </div>
  );
}
