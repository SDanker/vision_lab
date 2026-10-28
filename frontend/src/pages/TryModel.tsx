import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errMsg } from "../lib/api";
import { BASE_LABEL, pct } from "../lib/format";
import { extensionRegex, useModality, useModalityBadge } from "../lib/modalities";
import type { ModelInfo, Prediction, SimilarItem } from "../lib/types";
import { ScoreBars, Segments } from "../components/PredictionResult";
import TabularTest from "../components/TabularTest";
import { Alert, Modal, PageTitle, Spinner, useAsync } from "../components/ui";

// ----------------------------------------------------------------------------- utilidades
const store = {
  get: (k: string) => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set: (k: string, v: string) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* almacenamiento no disponible */
    }
  },
};

const score = (m: ModelInfo) => m.accuracy ?? -1;
const bestOf = (list: ModelInfo[]) => list.reduce((a, m) => (score(m) > score(a) || (score(m) === score(a) && m.version > a.version) ? m : a), list[0]);

/** Cámara y micrófono en vivo sólo existen en HTTPS (o localhost). */
const canCapture = () => typeof window !== "undefined" && window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;

function mediaError(e: unknown): string {
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "Permiso denegado. Habilita la cámara/micrófono para este sitio en el navegador y vuelve a intentarlo.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No se encontró una cámara o micrófono en este dispositivo.";
  if (name === "NotReadableError") return "La cámara o el micrófono está en uso por otra aplicación.";
  return `No se pudo acceder al dispositivo: ${errMsg(e)}`;
}

/** Reduce fotos grandes (los celulares sacan 12 MP o más): sube más rápido y respeta el tamaño máximo. */
async function downscaleImage(file: Blob, max = 1600): Promise<Blob> {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, max / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size < 2_000_000) {
      bmp.close();
      return file;
    }
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * scale);
    c.height = Math.round(bmp.height * scale);
    c.getContext("2d")?.drawImage(bmp, 0, 0, c.width, c.height);
    bmp.close();
    const out = await new Promise<Blob | null>((res) => c.toBlob(res, "image/jpeg", 0.88));
    return out ?? file;
  } catch {
    return file; // navegador sin soporte: el servidor corrige la orientación y redimensiona igual
  }
}

// -------------------------------------------------------------------------- cámara en vivo
function LiveCamera({ onShot, onClose, onError }: { onShot: (f: File) => void; onClose: () => void; onError: (m: string) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setReady(false);
    (async () => {
      try {
        stream.current?.getTracks().forEach((t) => t.stop());
        const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: facing }, width: { ideal: 1920 } }, audio: false });
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream.current = s;
        if (video.current) {
          video.current.srcObject = s;
          await video.current.play().catch(() => undefined);
        }
        setReady(true);
      } catch (e) {
        onError(mediaError(e));
        onClose();
      }
    })();
    return () => {
      cancelled = true;
      stream.current?.getTracks().forEach((t) => t.stop());
      stream.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facing]);

  const shoot = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const scale = Math.min(1, 1600 / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(v.videoWidth * scale);
    c.height = Math.round(v.videoHeight * scale);
    c.getContext("2d")?.drawImage(v, 0, 0, c.width, c.height);
    c.toBlob((b) => b && onShot(new File([b], "foto.jpg", { type: "image/jpeg" })), "image/jpeg", 0.9);
  };

  return (
    <div className="space-y-3">
      <div className="overflow-hidden rounded-xl bg-black">
        <video ref={video} playsInline muted className="max-h-[60vh] w-full object-contain" aria-label="Vista previa de la cámara" />
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        <button className="btn btn-primary px-5 py-3 text-base" disabled={!ready} onClick={shoot}>📸 Capturar</button>
        <button className="btn btn-secondary py-3" onClick={() => setFacing((f) => (f === "environment" ? "user" : "environment"))}>🔄 Girar cámara</button>
        <button className="btn btn-secondary py-3" onClick={onClose}>Cerrar</button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------ grabadora de audio
const MIMES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
const extOf = (mime: string) => (mime.includes("mp4") ? "m4a" : mime.includes("ogg") ? "ogg" : "webm");
const MAX_RECORD_S = 60;

function AudioRecorder({ onDone, onError }: { onDone: (f: File) => void; onError: (m: string) => void }) {
  const [recording, setRecording] = useState(false);
  const [secs, setSecs] = useState(0);
  const rec = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const stream = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    if (rec.current?.state === "recording") rec.current.stop();
  }, []);

  useEffect(() => {
    if (!recording) return;
    const t = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [recording]);
  useEffect(() => {
    if (recording && secs >= MAX_RECORD_S) stop();
  }, [recording, secs, stop]);
  useEffect(
    () => () => {
      if (rec.current?.state === "recording") rec.current.stop();
      stream.current?.getTracks().forEach((t) => t.stop());
    },
    [],
  );

  const start = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = s;
      const type = MIMES.find((t) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t));
      const r = new MediaRecorder(s, type ? { mimeType: type } : undefined);
      chunks.current = [];
      r.ondataavailable = (e) => e.data.size > 0 && chunks.current.push(e.data);
      r.onstop = () => {
        s.getTracks().forEach((t) => t.stop());
        const mime = r.mimeType || type || "audio/webm";
        const blob = new Blob(chunks.current, { type: mime });
        setRecording(false);
        if (blob.size > 0) onDone(new File([blob], `grabacion.${extOf(mime)}`, { type: mime }));
      };
      r.start();
      rec.current = r;
      setSecs(0);
      setRecording(true);
    } catch (e) {
      onError(mediaError(e));
    }
  };

  return recording ? (
    <button className="btn btn-danger w-full py-4 text-base" onClick={stop}>
      <span className="h-3 w-3 animate-pulse rounded-full bg-white" aria-hidden="true" /> Detener ({secs} s / {MAX_RECORD_S} s)
    </button>
  ) : (
    <button className="btn btn-primary w-full py-4 text-base" onClick={() => void start()}>🎙 Grabar audio</button>
  );
}

// --------------------------------------------------------------------------------- parecidos
function SimilarGrid({ items, onOpen }: { items: SimilarItem[]; onOpen: (s: SimilarItem) => void }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((s) => (
        <button key={s.image_id} onClick={() => onOpen(s)} className="rounded-lg border border-slate-200 p-2 text-left transition hover:border-brand-500 dark:border-slate-800">
          <img src={`/api/images/${s.image_id}/thumb`} alt={s.name} loading="lazy" className="aspect-square w-full rounded-md object-cover" />
          <div className="mt-1 truncate text-sm font-medium" title={s.category ?? ""}>{s.kind === "audio" ? "🎵 " : ""}{s.category ?? "—"}</div>
          <div className="text-xs text-slate-500">{pct(s.similarity, 0)} parecido</div>
        </button>
      ))}
    </div>
  );
}

// ----------------------------------------------------------------------------------- página
interface Captured {
  file: Blob;
  url: string;
  name: string;
}

export default function TryModel() {
  const { data: all, loading } = useAsync(() => api<ModelInfo[]>("/models"), []);
  const projects = useMemo(() => {
    const map = new Map<number, { id: number; name: string; models: ModelInfo[] }>();
    for (const m of all ?? []) {
      const g = map.get(m.project_id) ?? { id: m.project_id, name: m.project_name ?? `Proyecto ${m.project_id}`, models: [] };
      g.models.push(m);
      map.set(m.project_id, g);
    }
    return [...map.values()].map((g) => ({ ...g, models: [...g.models].sort((a, b) => b.version - a.version) }));
  }, [all]);

  const [pid, setPid] = useState<number | null>(() => Number(store.get("try.project")) || null);
  const [mid, setMid] = useState<number | null>(null);
  const project = projects.find((p) => p.id === pid) ?? projects[0];
  const model = project ? (project.models.find((m) => m.id === mid) ?? bestOf(project.models)) : undefined;
  const mod = useModality(model?.modality);
  const badge = useModalityBadge();

  const [captured, setCaptured] = useState<Captured | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState(false);
  const [zoom, setZoom] = useState<SimilarItem | null>(null);
  const [progress, setProgress] = useState<{ done: number | null; total: number | null } | null>(null);
  const req = useRef(0);
  const cameraInput = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const secure = canCapture();
  const isImage = mod.preview === "image";
  const isAudio = mod.preview === "audio";

  const clear = useCallback(() => {
    req.current++;
    setCaptured((c) => {
      if (c) URL.revokeObjectURL(c.url);
      return null;
    });
    setResult(null);
    setError(null);
    setBusy(false);
    setProgress(null);
  }, []);
  useEffect(() => () => clear(), [clear]);

  const run = useCallback(
    async (file: Blob, name: string, quiet = false, modelId?: number) => {
      if (!model) return;
      const token = ++req.current;
      if (!quiet) {
        setResult(null);
        setError(null);
        setBusy(true);
      }
      try {
        const form = new FormData();
        form.append("file", file, name);
        const r = await api<Prediction>(`/models/${modelId ?? model.id}/predict?similar=4`, { form });
        if (token === req.current) setResult(r);
      } catch (e) {
        if (token === req.current) setError(errMsg(e));
      } finally {
        if (token === req.current) setBusy(false);
      }
    },
    [model],
  );

  const onFile = useCallback(
    async (f: File | Blob, name?: string) => {
      const fileName = name ?? (f instanceof File ? f.name : "captura");
      const prepared = isImage ? await downscaleImage(f) : f;
      const url = URL.createObjectURL(prepared);
      setCaptured((c) => {
        if (c) URL.revokeObjectURL(c.url);
        return { file: prepared, url, name: fileName };
      });
      setLive(false);
      void run(prepared, prepared instanceof File ? prepared.name : fileName);
    },
    [isImage, run],
  );

  // Si lo más parecido aún se está preparando, se consulta el avance y, al terminar, se repite la consulta con la misma captura.
  useEffect(() => {
    if (!model || !captured || result?.similar_status !== "building") return;
    let alive = true;
    const t = setInterval(async () => {
      try {
        const st = await api<{ state: string; done: number | null; total: number | null }>(`/models/${model.id}/similar-status`);
        if (!alive) return;
        setProgress({ done: st.done, total: st.total });
        if (st.state === "ready") void run(captured.file, captured.name, true);
      } catch {
        /* se reintenta en el siguiente ciclo */
      }
    }, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [model, captured, result?.similar_status, run]);

  // Pegar una imagen desde el portapapeles (escritorio).
  useEffect(() => {
    if (!isImage) return;
    const h = (e: ClipboardEvent) => {
      const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith("image/"));
      if (f) void onFile(f);
    };
    window.addEventListener("paste", h);
    return () => window.removeEventListener("paste", h);
  }, [isImage, onFile]);

  const choose = (list: FileList | null) => {
    const f = Array.from(list ?? []).find((x) => extensionRegex(mod).test(x.name) || (isImage && x.type.startsWith("image/")));
    if (f) void onFile(f);
    else if (list?.length) setError(`Ese archivo no es válido para este proyecto. Formatos: ${mod.extensions.join(", ")}.`);
    if (cameraInput.current) cameraInput.current.value = "";
    if (fileInput.current) fileInput.current.value = "";
  };

  const pickProject = (id: number) => {
    setPid(id);
    setMid(null);
    store.set("try.project", String(id));
    clear();
  };
  const pickModel = (id: number) => {
    setMid(id);
    if (captured) void run(captured.file, captured.name, false, id); // la misma captura con otra versión del modelo
  };

  if (loading && !all) return <Spinner />;
  if (!project || !model) {
    return (
      <div>
        <PageTitle title="Probar un modelo" />
        <div className="card text-center text-slate-500">Todavía no hay modelos entrenados para probar. Aparecerán aquí tras el primer entrenamiento.</div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <PageTitle title="Probar un modelo" />

      <div className="card grid gap-3 sm:grid-cols-2">
        <label className="block">
          <span className="label">Proyecto</span>
          <select className="w-full" value={project.id} onChange={(e) => pickProject(Number(e.target.value))}>
            {projects.map((p) => <option key={p.id} value={p.id}>{badge(p.models[0]?.modality)}{p.name}</option>)}
          </select>
        </label>
        <label className="block">
          <span className="label">Versión del modelo</span>
          <select className="w-full" value={model.id} onChange={(e) => pickModel(Number(e.target.value))}>
            {project.models.map((m) => (
              <option key={m.id} value={m.id}>v{m.version} · {BASE_LABEL[m.base_model] ?? m.base_model} · {pct(m.accuracy)}{m.id === bestOf(project.models).id && project.models.length > 1 ? " (mejor)" : ""}</option>
            ))}
          </select>
        </label>
      </div>

      {error && <Alert tone="error">{error}</Alert>}

      {mod.tabular ? (
        <div className="card"><TabularTest key={model.id} model={model} /></div>
      ) : (<>
      {!captured && !live && (
        <div
          className="card space-y-3"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            choose(e.dataTransfer.files);
          }}
        >
          <p className="text-sm text-slate-600 dark:text-slate-400">
            {isImage ? "Toma una foto o sube una imagen para ver qué reconoce el modelo y con qué se parece más." : isAudio ? "Graba un sonido o sube un audio para ver qué reconoce el modelo y con qué se parece más." : `Sube un archivo (${mod.extensions.join(", ")}).`}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {isImage && (
              <>
                <button className="btn btn-primary py-4 text-base" onClick={() => cameraInput.current?.click()}>📷 Tomar foto</button>
                {secure && <button className="btn btn-secondary py-4 text-base" onClick={() => setLive(true)}>🎥 Cámara en vivo</button>}
              </>
            )}
            {isAudio && (secure ? <AudioRecorder onDone={(f) => void onFile(f)} onError={setError} /> : null)}
            <button className="btn btn-secondary py-4 text-base" onClick={() => fileInput.current?.click()}>{isAudio ? "📁 Subir audio" : "🖼 Elegir de la galería"}</button>
          </div>
          {!secure && (
            <p className="text-xs text-slate-500">
              La {isAudio ? "grabación" : "cámara"} en vivo requiere abrir la plataforma por HTTPS.{isImage ? " «Tomar foto» sí funciona: abre la cámara de tu teléfono." : " Mientras tanto puedes subir un audio ya grabado."}
            </p>
          )}
          <p className="hidden text-xs text-slate-400 md:block">También puedes arrastrar un archivo aquí{isImage ? " o pegar una imagen (Ctrl+V)" : ""}.</p>
          <input ref={cameraInput} type="file" accept="image/*" capture="environment" className="hidden" onChange={(e) => choose(e.target.files)} />
          <input ref={fileInput} type="file" accept={mod.accept} className="hidden" onChange={(e) => choose(e.target.files)} />
        </div>
      )}

      {live && <div className="card"><LiveCamera onShot={(f) => void onFile(f)} onClose={() => setLive(false)} onError={setError} /></div>}

      {captured && (
        <div className="card space-y-4">
          <div className="flex flex-col gap-4 sm:flex-row">
            <div className="sm:w-56 sm:shrink-0">
              {isAudio ? (
                <div className="space-y-2">
                  <div className="grid h-24 place-items-center rounded-lg bg-slate-100 text-4xl dark:bg-slate-800" aria-hidden="true">🎵</div>
                  <audio controls src={captured.url} className="w-full" />
                </div>
              ) : (
                <img src={captured.url} alt="Tu captura" className="max-h-72 w-full rounded-lg object-contain sm:max-h-56" />
              )}
            </div>
            <div className="min-w-0 flex-1">
              {busy && <div className="flex items-center gap-2 text-sm"><Spinner /> Analizando…</div>}
              {result && (
                <div className="space-y-3">
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-slate-500">El modelo reconoce</div>
                    <div className="text-3xl font-bold leading-tight">{result.prediction}</div>
                    <div className="text-sm text-slate-500">{pct(result.confidence)} de confianza</div>
                  </div>
                  {result.low_confidence && <Alert tone="warning">⚠ {result.warning ?? "Puntuación insuficiente: la clasificación no es confiable."}</Alert>}
                  <ScoreBars scores={result.scores} />
                  <Segments segments={result.segments} />
                </div>
              )}
            </div>
          </div>
          <button className="btn btn-secondary w-full py-3 sm:w-auto" onClick={clear}>↺ Probar otra</button>
        </div>
      )}

      {captured && result && (
        <div className="card space-y-3">
          <h3 className="font-semibold">Lo más parecido en el proyecto</h3>
          {result.similar && result.similar.length > 0 && <SimilarGrid items={result.similar} onOpen={setZoom} />}
          {result.similar_status === "building" && (
            <p className="flex items-center gap-2 text-sm text-slate-500"><Spinner /> Preparando las comparaciones de este modelo por primera vez{progress?.total ? ` (${progress.done ?? 0} de ${progress.total})` : ""}. Se mostrarán solas cuando estén listas.</p>
          )}
          {result.similar_status === "updating" && <p className="text-xs text-slate-500">Se agregaron o quitaron archivos en el proyecto; las comparaciones se están actualizando.</p>}
          {result.similar_status === "unavailable" && <p className="text-sm text-slate-500">La comparación no está disponible para este modelo.</p>}
          {result.similar_status === "ready" && result.similar?.length === 0 && <p className="text-sm text-slate-500">El proyecto aún no tiene archivos para comparar.</p>}
        </div>
      )}

      {zoom && (
        <Modal title={`${zoom.category ?? ""} · ${pct(zoom.similarity, 0)} parecido`} onClose={() => setZoom(null)} wide>
          <div className="space-y-3">
            {zoom.kind === "audio" ? (
              <>
                <img src={`/api/images/${zoom.image_id}/thumb`} alt="Espectrograma" className="mx-auto max-h-64 w-full rounded-lg object-fill" />
                <audio controls src={`/api/images/${zoom.image_id}/file`} className="w-full" />
              </>
            ) : (
              <img src={`/api/images/${zoom.image_id}/file`} alt={zoom.name} className="mx-auto max-h-[60vh] rounded-lg object-contain" />
            )}
            <p className="text-xs text-slate-500">{zoom.name}</p>
          </div>
        </Modal>
      )}
      </>)}
    </div>
  );
}
