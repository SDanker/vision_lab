import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { Project } from "../../lib/types";
import { AugmentationDemo, BatchDemo, EpochsPatienceDemo, FineTuneDemo, LearningRateDemo, ModelsDemo, ResolutionDemo } from "./Demos";
import { useSampleImage } from "./sample";
import { TabularColumnsDemo, TabularResultsDemo } from "./TabularDemos";

type Kind = "image" | "audio" | "tabular";
const KINDS: { id: Kind; label: string }[] = [
  { id: "image", label: "🖼 Imágenes" },
  { id: "audio", label: "🎵 Audio" },
  { id: "tabular", label: "📊 Datos tabulares" },
];

interface Step { icon: string; title: string; text: string }

// Lo que no depende del tipo de dato
const AUTO: Step = { icon: "🗓", title: "Entrenamiento automático", text: "De lunes a viernes cargas y editas. Cada noche (00:00–08:00) se entrenan los modelos marcados que tardan menos de 4 horas, todos los que quepan; si estás cargando a esa hora te preguntamos antes. El sábado a las 00:05 se congela el dataset y se entrena el resto, del modelo más largo al más corto. Al publicar los resultados las cargas se reabren y los proyectos se desmarcan: el lunes revisas, ajustas y, si quieres que se entrene de nuevo, vuelves a marcar el proyecto y guardas (te avisamos la fecha)." };
const TUNE: Step = { icon: "🎛", title: "Ajusta el entrenamiento", text: "En la pestaña «Entrenamiento» del proyecto defines épocas, resolución, learning rate, etc. Con la configuración avanzada, cada modelo puede tener sus propios valores." };
const TEAM = (what: string): Step => ({ icon: "🤝", title: "Trabaja en equipo", text: `Comparte un proyecto desde su pestaña «Compartir»: un editor puede ${what} y ajustar parámetros; un lector sólo mira y prueba.` });

const GUIDE: Record<Kind, { intro: string; steps: Step[] }> = {
  image: {
    intro: "Vision Lab entrena modelos que reconocen tus propias categorías a partir de ejemplos. No hace falta programar: cargas imágenes etiquetadas por carpeta o categoría, la plataforma aprovecha un modelo ya entrenado con millones de fotos (transfer learning) y lo adapta a tus categorías. El resultado es un modelo descargable que puedes usar en otra aplicación.",
    steps: [
      { icon: "📁", title: "Crea un proyecto", text: "Elige «Imágenes», define las categorías (p. ej. «F-16», «F-18») y uno o varios modelos base." },
      { icon: "⬆", title: "Carga ejemplos", text: "Sube fotos a cada categoría, o arrastra carpetas: cada carpeta se convierte en una categoría. Se necesitan al menos 2 categorías con suficientes imágenes." },
      TUNE, AUTO,
      { icon: "📊", title: "Revisa, prueba y descarga", text: "En «Modelos y pruebas» comparas exactitud y F1, ves la matriz de confusión, pruebas con una imagen nueva y descargas el modelo en 5 formatos (Keras, H5, SavedModel, TFLite, ONNX)." },
      TEAM("cargar imágenes, crear categorías"),
    ],
  },
  audio: {
    intro: "Para sonidos, cada audio se corta en clips y cada clip se convierte en un espectrograma (una «foto» del sonido); desde ahí se entrena igual que con imágenes, con los mismos modelos base. El resultado es un modelo descargable que clasifica grabaciones nuevas.",
    steps: [
      { icon: "📁", title: "Crea un proyecto", text: "Elige «Audio», define las categorías (p. ej. «motor», «alarma») y uno o varios modelos base." },
      { icon: "⬆", title: "Carga audios", text: "Sube WAV, MP3, OGG, FLAC, M4A, AAC, OPUS o WEBM a cada categoría, o arrastra carpetas. Los audios largos se cortan en clips de la duración que elijas (1–10 s)." },
      { ...TUNE, text: "En la pestaña «Entrenamiento» defines la duración de cada clip, el tamaño del espectrograma, épocas, learning rate, etc. Cada modelo puede tener sus propios valores en la configuración avanzada." },
      AUTO,
      { icon: "📊", title: "Revisa, prueba y descarga", text: "En «Modelos y pruebas» comparas exactitud y F1, ves la matriz de confusión, pruebas grabando un sonido o subiendo un audio (con el resultado de cada segmento) y descargas el modelo en 5 formatos." },
      TEAM("cargar audios, crear categorías"),
    ],
  },
  tabular: {
    intro: "Para tablas (CSV), cada fila es un ejemplo y una de sus columnas es la categoría que el modelo aprende a predecir a partir de las demás: por ejemplo, si un Pokémon es legendario según sus estadísticas, o el tipo de cliente según sus compras. No usa imágenes: entrena algoritmos clásicos (Random Forest, Gradient Boosting, regresión logística y red neuronal) en segundos.",
    steps: [
      { icon: "📁", title: "Crea un proyecto", text: "Elige «Datos tabulares» y uno o varios modelos base. No escribes categorías: salen de los valores de tu tabla." },
      { icon: "⬆", title: "Sube tu CSV", text: "En la pestaña «Datos» arrastra uno o varios archivos CSV o TSV con los nombres de columna en la primera línea (si subes varios, deben tener las mismas columnas). El separador y la codificación se detectan solos." },
      { icon: "🎯", title: "Elige qué predecir", text: "Escoge la columna a predecir (sus valores serán las categorías: entre 2 y 100, con al menos 10 filas cada una) y las columnas de entrada. Los identificadores, como nombres o números correlativos, se descartan solos." },
      { ...AUTO, text: `${AUTO.text} Los modelos tabulares tardan segundos, así que siempre caben en la noche.` },
      { icon: "📊", title: "Revisa, prueba y descarga", text: "En «Modelos y pruebas» comparas exactitud y F1 contra una referencia, ves qué columnas pesan más, pruebas con un formulario (una fila) o con un CSV de varias filas, y descargas el modelo en ONNX y joblib con un script de inferencia." },
      TEAM("subir CSV, elegir columnas"),
    ],
  },
};

const IMAGE_TABS = [
  { id: "resolution", label: "Resolución", hint: "image_size" },
  { id: "augmentation", label: "Data augmentation", hint: "augmentation" },
  { id: "lr", label: "Learning rate", hint: "learning_rate" },
  { id: "epochs", label: "Épocas y paciencia", hint: "epochs · patience" },
  { id: "finetune", label: "Ajuste fino", hint: "fine_tune_epochs" },
  { id: "batch", label: "Batch size", hint: "batch_size" },
  { id: "models", label: "Modelo base", hint: "base_models" },
] as const;
const TABULAR_TABS = [
  { id: "models", label: "Modelo base", hint: "base_models" },
  { id: "columns", label: "Columnas", hint: "columna a predecir · entradas" },
  { id: "results", label: "Leer los resultados", hint: "exactitud · referencia" },
] as const;
type ImageTab = (typeof IMAGE_TABS)[number]["id"];
type TabularTab = (typeof TABULAR_TABS)[number]["id"];

const STORE_KEY = "vision-guide-open";
const readOpen = () => {
  try {
    return localStorage.getItem(STORE_KEY) !== "0";
  } catch {
    return true;
  }
};

/** Guía de la página de inicio: cómo funciona la plataforma con cada tipo de dato y un laboratorio interactivo de los ajustes. */
export default function Guide({ projects }: { projects: Project[] | null }) {
  const [open, setOpen] = useState(readOpen);
  const [kind, setKind] = useState<Kind>("image");
  const chosen = useRef(false); // si la persona ya eligió un tipo, no se lo cambiamos
  const [tab, setTab] = useState<ImageTab>("resolution");
  const [ttab, setTtab] = useState<TabularTab>("models");
  const [sample, setUseOwn, hasOwn] = useSampleImage(projects);

  // Primera vez: se abre el tipo de dato con el que la persona ya trabaja (p. ej. sólo tiene proyectos tabulares).
  useEffect(() => {
    if (chosen.current || !projects?.length) return;
    const kinds = new Set(projects.map((p) => (p.modality ?? "image") as Kind));
    if (!kinds.has("image")) setKind(kinds.has("tabular") ? "tabular" : "audio");
  }, [projects]);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    try {
      localStorage.setItem(STORE_KEY, next ? "1" : "0");
    } catch {
      /* sin almacenamiento: sólo dura esta visita */
    }
  };
  const pickKind = (k: Kind) => {
    chosen.current = true;
    setKind(k);
  };
  const g = GUIDE[kind];
  const tabular = kind === "tabular";
  const tabs = tabular ? TABULAR_TABS : IMAGE_TABS;
  const current = tabular ? ttab : tab;
  const select = (id: string) => (tabular ? setTtab(id as TabularTab) : setTab(id as ImageTab));

  return (
    <section className="mt-8" aria-labelledby="guia-titulo">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 id="guia-titulo" className="text-lg font-semibold">Guía: cómo funciona Vision Lab</h2>
        <button type="button" className="btn btn-secondary btn-sm" onClick={toggle} aria-expanded={open}>{open ? "Ocultar guía" : "Mostrar guía"}</button>
      </div>
      {open && (
        <div className="space-y-6">
          <div className="card">
            <div className="mb-4 flex flex-wrap items-center gap-2" role="tablist" aria-label="Tipo de dato">
              <span className="text-sm text-slate-500">¿Qué vas a entrenar?</span>
              {KINDS.map((k) => (
                <button
                  key={k.id} type="button" role="tab" aria-selected={kind === k.id} onClick={() => pickKind(k.id)}
                  className={`btn btn-sm ${kind === k.id ? "btn-primary" : "btn-secondary"}`}
                >
                  {k.label}
                </button>
              ))}
            </div>
            <p className="mb-4 text-sm text-slate-600 dark:text-slate-400">{g.intro}</p>
            <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {g.steps.map((s, i) => (
                <li key={s.title} className="flex gap-3 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-brand-50 text-lg dark:bg-brand-700/20" aria-hidden="true">{s.icon}</div>
                  <div>
                    <div className="text-sm font-semibold"><span className="text-slate-400">{i + 1}.</span> {s.title}</div>
                    <p className="mt-0.5 text-xs text-slate-600 dark:text-slate-400">{s.text}</p>
                  </div>
                </li>
              ))}
            </ol>
            <div className="mt-4 flex flex-wrap gap-2">
              <Link to="/proyectos" className="btn btn-primary btn-sm">Ir a mis proyectos</Link>
              <Link to="/modelos" className="btn btn-secondary btn-sm">Ver modelos entrenados</Link>
            </div>
          </div>

          <div className="card">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-semibold">Laboratorio de ajustes{tabular ? " · datos tabulares" : ""}</h3>
              {hasOwn && !tabular && (
                <label className="flex items-center gap-2 text-xs text-slate-500">
                  <input type="checkbox" checked={sample.own} onChange={(e) => setUseOwn(e.target.checked)} /> Usar una imagen de mis proyectos
                </label>
              )}
            </div>
            <p className="mb-4 text-sm text-slate-500">
              Mueve cada control y mira qué cambia. Son simulaciones para entender el efecto de cada parámetro; los valores reales dependen de tus datos.{" "}
              {tabular
                ? "Épocas, resolución, learning rate y data augmentation son de las redes de imágenes y no aplican a tablas: aquí lo que importa es elegir bien las columnas y leer los resultados."
                : "Estos ajustes se configuran en la pestaña «Entrenamiento» de cada proyecto."}
            </p>
            <div className="mb-5 flex gap-1 overflow-x-auto border-b border-slate-200 dark:border-slate-800" role="tablist">
              {tabs.map((t) => (
                <button
                  key={t.id} type="button" role="tab" aria-selected={current === t.id} onClick={() => select(t.id)}
                  className={`whitespace-nowrap border-b-2 px-3 py-2 text-left text-sm font-medium ${current === t.id ? "border-brand-600 text-brand-600" : "border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"}`}
                >
                  {t.label}
                  <span className="block font-mono text-[10px] font-normal text-slate-400">{t.hint}</span>
                </button>
              ))}
            </div>
            <div role="tabpanel">
              {tabular ? (
                <>
                  {ttab === "models" && <ModelsDemo kind="tabular" />}
                  {ttab === "columns" && <TabularColumnsDemo />}
                  {ttab === "results" && <TabularResultsDemo />}
                </>
              ) : (
                <>
                  {tab === "resolution" && <ResolutionDemo sample={sample} />}
                  {tab === "augmentation" && <AugmentationDemo sample={sample} />}
                  {tab === "lr" && <LearningRateDemo />}
                  {tab === "epochs" && <EpochsPatienceDemo />}
                  {tab === "finetune" && <FineTuneDemo />}
                  {tab === "batch" && <BatchDemo sample={sample} />}
                  {tab === "models" && <ModelsDemo kind="vision" />}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
