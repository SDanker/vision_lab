import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./lib/auth";
import { JobsProvider } from "./lib/jobs";
import { ModalitiesProvider } from "./lib/modalities";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import { ApprovalStatus, Login, Register } from "./pages/AuthPages";
import Dashboard from "./pages/Dashboard";
import Projects from "./pages/Projects";
import ProjectDetail from "./pages/ProjectDetail";
import { HistoryPage, ModelsPage, TrainingPage } from "./pages/Training";
import Admin from "./pages/Admin";
import Account from "./pages/Account";
import TryModel from "./pages/TryModel";

/** Sólo usuarios aprobados ven la plataforma; el resto es redirigido a la pantalla de estado. */
function ApprovedOnly() {
  const { user, loading } = useAuth();
  if (loading) return <div className="grid min-h-screen place-items-center"><Spinner /></div>;
  if (!user) return <Navigate to="/login" replace />;
  if (user.status !== "APPROVED") return <Navigate to="/estado" replace />;
  return (
    <ModalitiesProvider>
      <JobsProvider>
        <Layout />
      </JobsProvider>
    </ModalitiesProvider>
  );
}

function AdminOnly() {
  const { user } = useAuth();
  return user?.role === "ADMIN" ? <Outlet /> : <Navigate to="/" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/registro" element={<Register />} />
          <Route path="/estado" element={<ApprovalStatus />} />
          <Route element={<ApprovedOnly />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/proyectos" element={<Projects />} />
            <Route path="/proyectos/:id" element={<ProjectDetail />} />
            <Route path="/entrenamiento" element={<TrainingPage />} />
            <Route path="/modelos" element={<ModelsPage />} />
            <Route path="/historial" element={<HistoryPage />} />
            <Route path="/probar" element={<TryModel />} />
            <Route path="/cuenta" element={<Account />} />
            <Route element={<AdminOnly />}>
              <Route path="/admin" element={<Admin />} />
            </Route>
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
