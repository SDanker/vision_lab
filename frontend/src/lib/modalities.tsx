import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "./api";

/** Tipo de dato que se puede entrenar; viene de GET /api/modalities (registro del backend). */
export interface ModalityInfo {
  id: string;
  label: string;
  icon: string;
  description: string;
  unit: string; // plural
  unit_one: string;
  extensions: string[];
  accept: string;
  upload_help: string;
  preview: "image" | "audio" | "table" | string;
  windowed: boolean;
  /** Filas de una tabla: se entrenan con scikit-learn, sin imágenes ni categorías por carpeta. */
  tabular: boolean;
  window_label: string;
  window_hint: string;
  resolution_label: string;
  resolution_hint: string;
  test_hint: string;
}

// Respaldo mientras carga el listado (o si falla): los tipos que trae la plataforma.
const FALLBACK: ModalityInfo[] = [
  {
    id: "image", label: "Imágenes", icon: "🖼", description: "Clasifica fotos (JPG, PNG, WEBP).", unit: "imágenes", unit_one: "imagen",
    extensions: ["jpg", "jpeg", "png", "webp"], accept: "image/jpeg,image/png,image/webp", upload_help: "", preview: "image", windowed: false, tabular: false,
    window_label: "", window_hint: "", resolution_label: "Resolución de entrada",
    resolution_hint: "Más resolución distingue mejor detalles finos (p. ej. tipos de avión), pero entrena más lento y usa más memoria.", test_hint: "",
  },
  {
    id: "audio", label: "Audio", icon: "🎵", description: "Clasifica sonidos (WAV, MP3, OGG, FLAC, M4A…) mediante espectrogramas.", unit: "audios", unit_one: "audio",
    extensions: ["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus", "webm"], accept: ".wav,.mp3,.ogg,.flac,.m4a,.aac,.opus,.webm,audio/*",
    upload_help: "WAV, MP3, OGG, FLAC, M4A, AAC, OPUS o WEBM. Los audios largos se cortan en clips para entrenar.", preview: "audio", windowed: true, tabular: false,
    window_label: "Duración de cada clip (segundos)", window_hint: "Los audios se cortan en clips de esta duración; cada clip es una muestra de entrenamiento. 2–5 s suele funcionar bien.",
    resolution_label: "Resolución del espectrograma", resolution_hint: "Tamaño de la imagen del espectrograma. 224 es suficiente en la mayoría de los casos.",
    test_hint: " Cada audio se analiza por segmentos y se promedia.",
  },
  {
    id: "tabular", label: "Datos tabulares", icon: "📊", description: "Clasifica filas de una tabla (CSV): por ejemplo, predecir una categoría a partir de sus características.",
    unit: "archivos CSV", unit_one: "archivo CSV", extensions: ["csv", "tsv"], accept: ".csv,.tsv,text/csv",
    upload_help: "CSV o TSV con una fila por ejemplo y los nombres de columna en la primera línea. Si subes varios archivos, deben tener las mismas columnas.",
    preview: "table", windowed: false, tabular: true, window_label: "", window_hint: "", resolution_label: "Resolución de entrada", resolution_hint: "",
    test_hint: " Completa los valores de una fila o sube un CSV con varias filas.",
  },
];

const Ctx = createContext<ModalityInfo[]>(FALLBACK);

export function ModalitiesProvider({ children }: { children: ReactNode }) {
  const [list, setList] = useState<ModalityInfo[]>(FALLBACK);
  useEffect(() => {
    api<ModalityInfo[]>("/modalities").then((l) => l.length && setList(l)).catch(() => undefined);
  }, []);
  return <Ctx.Provider value={list}>{children}</Ctx.Provider>;
}

export const useModalities = () => useContext(Ctx);

/** Datos del tipo de dato de un proyecto o modelo (`image` si no se indica). */
export function useModality(id?: string | null): ModalityInfo {
  const list = useModalities();
  return list.find((m) => m.id === (id ?? "image")) ?? list.find((m) => m.id === "image") ?? FALLBACK[0];
}

/** Expresión regular que acepta los nombres de archivo de un tipo de dato. */
export const extensionRegex = (m: ModalityInfo) => new RegExp(`\\.(${m.extensions.join("|")})$`, "i");

export function useExtensionRegex(id?: string | null) {
  const m = useModality(id);
  return useMemo(() => extensionRegex(m), [m]);
}

/** Ícono que distingue el tipo de dato en los listados («🎵 », «📊 »); las imágenes, el tipo original, no llevan. */
export function useModalityBadge() {
  const list = useModalities();
  return (id?: string | null) => (id && id !== "image" ? `${list.find((m) => m.id === id)?.icon ?? ""} ` : "");
}
