import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { ImageItem, Project } from "../../lib/types";

/** Ilustración incluida (sin descargas): un avión con detalles finos para que se note la resolución. */
const SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 384" width="384" height="384">
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5b8fd6"/><stop offset="0.7" stop-color="#b9d4f2"/><stop offset="1" stop-color="#e8eef5"/></linearGradient>
<linearGradient id="body" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#9aa5b1"/><stop offset="1" stop-color="#5d6772"/></linearGradient>
</defs>
<rect width="384" height="384" fill="url(#sky)"/>
<circle cx="318" cy="70" r="26" fill="#fff6d5" opacity="0.9"/>
<path d="M0 300 L60 250 L110 280 L170 230 L240 285 L300 245 L384 290 L384 384 L0 384 Z" fill="#6f8f7a"/>
<path d="M0 330 L80 300 L150 325 L230 295 L310 320 L384 305 L384 384 L0 384 Z" fill="#557463"/>
<rect x="0" y="352" width="384" height="32" fill="#4a4f55"/>
<g fill="#f2f2f2">${Array.from({ length: 12 }, (_, i) => `<rect x="${i * 34 + 6}" y="366" width="18" height="3"/>`).join("")}</g>
<g transform="translate(192 180) rotate(-8)">
<path d="M-150 0 L-40 -14 L60 -12 L130 -4 L150 0 L130 5 L60 12 L-40 14 Z" fill="url(#body)"/>
<path d="M-20 -10 L-70 -78 L-48 -80 L30 -10 Z" fill="#6b7580"/>
<path d="M-20 10 L-70 78 L-48 80 L30 10 Z" fill="#59636d"/>
<path d="M-120 -6 L-150 -58 L-132 -60 L-96 -8 Z" fill="#6b7580"/>
<path d="M-118 6 L-140 36 L-126 37 L-100 7 Z" fill="#59636d"/>
<path d="M72 -10 C90 -18 112 -14 124 -5 L72 -5 Z" fill="#2f4b6e" opacity="0.85"/>
<g stroke="#e0b030" stroke-width="1.2">${Array.from({ length: 7 }, (_, i) => `<line x1="${-60 + i * 7}" y1="-58" x2="${-48 + i * 7}" y2="-46"/>`).join("")}</g>
<text x="-138" y="-28" font-family="monospace" font-size="7" fill="#f4f4f4" transform="rotate(-60 -138 -28)">AF 86-0254</text>
<text x="-10" y="4" font-family="monospace" font-size="6" fill="#20262c">USAF · 512 TFS</text>
<circle cx="20" cy="-1" r="3.2" fill="#fff"/><circle cx="20" cy="-1" r="1.6" fill="#1d3f8a"/>
<g fill="#20262c">${Array.from({ length: 9 }, (_, i) => `<circle cx="${-30 + i * 9}" cy="9" r="0.9"/>`).join("")}</g>
</g>
</svg>`;

export const BUILTIN_SAMPLE = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(SVG)}`;

export interface Sample {
  src: string;
  label: string;
  /** true cuando viene de un proyecto del usuario */
  own: boolean;
}

/** Imagen de muestra para las demos: la primera imagen de un proyecto propio si existe; si no, la ilustración. */
export function useSampleImage(projects: Project[] | null): [Sample, (own: boolean) => void, boolean] {
  const builtin: Sample = { src: BUILTIN_SAMPLE, label: "Ilustración de ejemplo", own: false };
  const [ownSample, setOwnSample] = useState<Sample | null>(null);
  const [useOwn, setUseOwn] = useState(true);
  useEffect(() => {
    const p = projects?.find((x) => (x.modality ?? "image") === "image" && x.image_count > 0);
    if (!p) return;
    let alive = true;
    api<{ items: ImageItem[] }>(`/projects/${p.id}/images?page_size=1`)
      .then((r) => {
        const img = r.items[0];
        if (alive && img) setOwnSample({ src: `/api/images/${img.id}/file`, label: `Imagen de tu proyecto «${p.name}»`, own: true });
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [projects]);
  return [useOwn && ownSample ? ownSample : builtin, setUseOwn, !!ownSample];
}

/** Carga una imagen y la devuelve lista para dibujar en un canvas. */
export function useLoadedImage(src: string): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    let alive = true;
    const el = new Image();
    el.onload = () => alive && setImg(el);
    el.onerror = () => alive && setImg(null);
    el.src = src;
    return () => {
      alive = false;
    };
  }, [src]);
  return img;
}

/** Generador pseudoaleatorio con semilla (las demos se repiten igual hasta que el usuario pide otra corrida). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
