import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "../../lib/api";
import { useAsync } from "../ui";
import MiniChart from "./MiniChart";
import { gaussian, rng, useLoadedImage, type Sample } from "./sample";

// ------------------------------------------------------------------ piezas comunes
export function Slider({ label, value, min, max, step = 1, onChange, display }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; display?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1 flex items-baseline justify-between text-sm font-medium text-slate-700 dark:text-slate-300">
        {label}
        <span className="rounded bg-indigo-50 px-2 py-0.5 font-mono text-sm text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200">{display ?? value}</span>
      </span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full border-0 bg-transparent p-0 shadow-none accent-indigo-600 focus:ring-0"
      />
    </label>
  );
}

/** Explicación estándar de cada ajuste: qué es, qué pasa al subirlo o bajarlo y qué conviene. */
export function Explain({ what, up, down, tip }: { what: ReactNode; up: ReactNode; down: ReactNode; tip: ReactNode }) {
  return (
    <div className="space-y-3 text-sm">
      <p className="text-slate-700 dark:text-slate-300">{what}</p>
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="rounded-lg border border-slate-200 p-3 dark:border-slate-700"><div className="mb-1 font-semibold">⬆ Si lo subes</div><div className="text-slate-600 dark:text-slate-400">{up}</div></div>
        <div className="rounded-lg border border-slate-200 p-3 dark:border-slate-700"><div className="mb-1 font-semibold">⬇ Si lo bajas</div><div className="text-slate-600 dark:text-slate-400">{down}</div></div>
      </div>
      <p className="rounded-lg bg-emerald-50 px-3 py-2 text-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200">✅ {tip}</p>
    </div>
  );
}

export function Verdict({ tone, children }: { tone: "good" | "warn" | "bad"; children: ReactNode }) {
  const cls = {
    good: "bg-emerald-50 text-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200",
    warn: "bg-amber-50 text-amber-900 dark:bg-amber-950/40 dark:text-amber-200",
    bad: "bg-red-50 text-red-900 dark:bg-red-950/40 dark:text-red-200",
  }[tone];
  return <p className={`rounded-lg px-3 py-2 text-sm ${cls}`} aria-live="polite">{children}</p>;
}

export function Panel({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border border-indigo-100 bg-indigo-50/40 p-4 dark:border-indigo-900/50 dark:bg-indigo-950/20">{children}</div>;
}

const fmtNum = (v: number) => (v >= 0.001 ? String(+v.toPrecision(3)) : v.toExponential(0));

// ------------------------------------------------------------------ resolución
const SIZES = [160, 224, 260, 300, 384];
const ZOOM = 0.28; // fracción del lado que se amplía

/** Dibuja la imagen recortada al cuadrado, redimensionada a `size` px como hace el preprocesado. */
function toModelInput(img: HTMLImageElement, size: number) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  const w = img.naturalWidth || 384;
  const h = img.naturalHeight || 384;
  ctx.drawImage(img, 0, 0, w, h, 0, 0, size, size); // el servidor también estira a cuadrado
  return c;
}

export function ResolutionDemo({ sample }: { sample: Sample }) {
  const [idx, setIdx] = useState(1);
  const [focus, setFocus] = useState({ x: 0.5, y: 0.47 }); // texto del fuselaje en la ilustración
  const size = SIZES[idx];
  const img = useLoadedImage(sample.src);
  const fullRef = useRef<HTMLCanvasElement>(null);
  const zoomRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!img || !fullRef.current || !zoomRef.current) return;
    const src = toModelInput(img, size);
    const full = fullRef.current.getContext("2d")!;
    full.imageSmoothingEnabled = false;
    full.clearRect(0, 0, 240, 240);
    full.drawImage(src, 0, 0, size, size, 0, 0, 240, 240);
    const sw = size * ZOOM;
    const sx = Math.min(Math.max(focus.x * size - sw / 2, 0), size - sw);
    const sy = Math.min(Math.max(focus.y * size - sw / 2, 0), size - sw);
    full.strokeStyle = "#f59e0b";
    full.lineWidth = 2;
    full.strokeRect((sx / size) * 240, (sy / size) * 240, ZOOM * 240, ZOOM * 240);
    const zoom = zoomRef.current.getContext("2d")!;
    zoom.imageSmoothingEnabled = false;
    zoom.drawImage(src, sx, sy, sw, sw, 0, 0, 240, 240);
  }, [img, size, focus]);

  const factor = (size / 224) ** 2;
  return (
    <div className="space-y-4">
      <Explain
        what={<>Es el tamaño (en píxeles por lado) al que se reduce cada imagen antes de entrar al modelo. Todas las imágenes se convierten a ese cuadrado, así que <strong>lo que no se ve a esa resolución, el modelo tampoco lo puede aprender</strong>.</>}
        up="Distingue mejor detalles finos (marcas, matrículas, formas pequeñas), pero el tiempo y la memoria crecen con el número de píxeles: 384 px cuesta unas 3 veces lo que 224 px."
        down="Entrena mucho más rápido y usa menos memoria, pero se pierden los detalles pequeños."
        tip="224 px sirve para la mayoría. Usa 300–384 cuando las categorías se distinguen por detalles (p. ej. tipos de avión), con batch size 16 o menos."
      />
      <Panel>
        <Slider label="Resolución de entrada" min={0} max={SIZES.length - 1} value={idx} onChange={setIdx} display={`${size} × ${size} px`} />
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <div className="mb-1 text-xs font-medium text-slate-500">Lo que recibe el modelo (haz clic para elegir dónde ampliar)</div>
            <canvas
              ref={fullRef} width={240} height={240}
              className="aspect-square w-full max-w-[240px] cursor-crosshair rounded-lg bg-slate-200 [image-rendering:pixelated] dark:bg-slate-800"
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setFocus({ x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height });
              }}
            />
          </div>
          <div>
            <div className="mb-1 text-xs font-medium text-slate-500">Zoom al recuadro: {Math.round(size * ZOOM)} px de detalle</div>
            <canvas ref={zoomRef} width={240} height={240} className="aspect-square w-full max-w-[240px] rounded-lg bg-slate-200 [image-rendering:pixelated] dark:bg-slate-800" />
          </div>
        </div>
        <div className="mt-3 grid gap-2 text-sm sm:grid-cols-3">
          <div className="rounded-lg bg-white p-2 dark:bg-slate-900"><div className="text-xs text-slate-500">Píxeles por imagen</div><strong>{(size * size).toLocaleString("es-CL")}</strong></div>
          <div className="rounded-lg bg-white p-2 dark:bg-slate-900"><div className="text-xs text-slate-500">Tiempo de entrenamiento</div><strong>≈ {factor.toFixed(1)}× del estándar</strong></div>
          <div className="rounded-lg bg-white p-2 dark:bg-slate-900"><div className="text-xs text-slate-500">Memoria por imagen</div><strong>≈ {factor.toFixed(1)}×</strong></div>
        </div>
        <p className="mt-2 text-xs text-slate-500">{sample.label}. Compara 160 y 384: en el zoom se ve qué detalles sobreviven.</p>
      </Panel>
    </div>
  );
}

// ------------------------------------------------------------------ augmentation
export function AugmentationDemo({ sample }: { sample: Sample }) {
  const [on, setOn] = useState(true);
  const [seed, setSeed] = useState(7);
  // mismos rangos que el entrenamiento: volteo horizontal, rotación ±0,08 vueltas, zoom ±10 %, contraste ±10 %
  const variants = useMemo(() => {
    const r = rng(seed);
    return Array.from({ length: 6 }, (_, i) => {
      if (!on || i === 0) return { flip: false, rot: 0, zoom: 1, contrast: 1 };
      return { flip: r() < 0.5, rot: (r() * 2 - 1) * 0.08 * 360, zoom: 1 + (r() * 2 - 1) * 0.1, contrast: 1 + (r() * 2 - 1) * 0.1 };
    });
  }, [on, seed]);
  return (
    <div className="space-y-4">
      <Explain
        what="En cada época, cada imagen de entrenamiento se altera al azar (volteo horizontal, rotación de hasta ±29°, zoom ±10 % y contraste ±10 %). El modelo ve «versiones distintas» de la misma foto, así que aprende la forma del objeto y no memoriza la foto exacta."
        up={<>Activado: reduce el <strong>sobreajuste</strong> (cuando el modelo acierta en entrenamiento pero falla con fotos nuevas). Es clave con pocas imágenes por categoría.</>}
        down="Desactivado: entrena un poco más rápido, pero memoriza más. Sólo conviene desactivarlo si voltear o girar cambia el significado (texto, flechas, izquierda/derecha)."
        tip="Déjalo activado casi siempre. Sólo se aplica al entrenar; las fotos de validación y prueba no se alteran."
      />
      <Panel>
        <div className="mb-3 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm font-medium"><input type="checkbox" checked={on} onChange={(e) => setOn(e.target.checked)} /> Data augmentation activado</label>
          <button type="button" className="btn btn-secondary btn-sm" disabled={!on} onClick={() => setSeed((s) => s + 1)}>🎲 Ver otra época</button>
        </div>
        <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
          {variants.map((v, i) => (
            <figure key={i}>
              <div className="aspect-square overflow-hidden rounded-lg bg-slate-300 dark:bg-slate-700">
                <img
                  src={sample.src} alt="" draggable={false}
                  className="h-full w-full object-cover transition-transform duration-500"
                  style={{ transform: `scaleX(${v.flip ? -1 : 1}) rotate(${v.rot}deg) scale(${v.zoom})`, filter: `contrast(${v.contrast})` }}
                />
              </div>
              <figcaption className="mt-1 text-center text-[11px] leading-tight text-slate-500">
                {i === 0 ? "Original" : !on ? "Igual (sin cambios)" : [v.flip && "volteada", `${v.rot > 0 ? "+" : ""}${v.rot.toFixed(0)}°`, `zoom ${Math.round(v.zoom * 100)}%`].filter(Boolean).join(" · ")}
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-500">Así vería el modelo la misma imagen en 6 épocas distintas. {on ? "Cada época es diferente." : "Sin augmentation, ve siempre la misma foto."}</p>
      </Panel>
    </div>
  );
}

// ------------------------------------------------------------------ learning rate
const LRS = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.1];

export function LearningRateDemo() {
  const [idx, setIdx] = useState(4);
  const lr = LRS[idx];
  // ilustración: descenso por gradiente sobre un «valle» f(x) = x²; el paso efectivo es proporcional al learning rate
  const eta = lr * 300;
  const steps = useMemo(() => {
    const xs = [-1.7];
    for (let i = 0; i < 14; i++) xs.push(xs[i] - eta * 2 * xs[i]);
    return xs;
  }, [eta]);
  const loss = steps.map((x) => x * x);
  const verdict: [tone: "good" | "warn" | "bad", text: string] =
    eta < 0.05 ? ["bad", "Muy bajo: casi no avanza. Con paciencia 3 el entrenamiento se cortaría antes de aprender algo útil."]
    : eta < 0.2 ? ["warn", "Bajo: aprende, pero lento. Necesitará bastantes más épocas."]
    : eta <= 0.9 ? ["good", "Bien: baja rápido y se acomoda en el fondo del valle."]
    : eta < 1.95 ? ["warn", "Alto: rebota de un lado a otro del mínimo. Termina llegando, pero la curva es inestable."]
    : eta <= 2.05 ? ["bad", "Muy alto: rebota para siempre sin llegar al mínimo."]
    : ["bad", "Demasiado alto: cada paso empeora el resultado y la pérdida explota (en la práctica aparece «nan»)."];

  const W = 300, H = 190;
  const px = (x: number) => W / 2 + (x / 2.2) * (W / 2 - 10);
  const py = (v: number) => H - 14 - (Math.min(v, 4.8) / 4.8) * (H - 28);
  const parabola = Array.from({ length: 61 }, (_, i) => {
    const x = -2.2 + (i / 60) * 4.4;
    return `${i ? "L" : "M"}${px(x).toFixed(1)} ${py(x * x).toFixed(1)}`;
  }).join(" ");
  const visible = steps.filter((x) => Math.abs(x) <= 2.2);
  return (
    <div className="space-y-4">
      <Explain
        what="En cada paso el modelo mira cuánto se equivocó y corrige sus pesos. El learning rate es el tamaño de esa corrección: como bajar una ladera con niebla dando pasos de cierto largo."
        up="Aprende más rápido, pero los pasos pueden ser tan largos que se pasa del mínimo y rebota, o incluso empeora (pérdida que salta o «nan»)."
        down="Más estable, pero avanza lento y puede necesitar muchas más épocas; si es demasiado bajo, el early stopping corta antes de que aprenda."
        tip="0,001 es el estándar para transfer learning. Bájalo a 0,0003 sólo si la pérdida salta; en el ajuste fino se usa automáticamente lr ÷ 100."
      />
      <Panel>
        <Slider label="Learning rate" min={0} max={LRS.length - 1} value={idx} onChange={setIdx} display={fmtNum(lr)} />
        <div className="mt-4 grid items-start gap-4 md:grid-cols-2">
          <div>
            <div className="mb-1 text-xs font-medium text-slate-500">El «valle» del error: cada punto es un paso de corrección</div>
            <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full rounded-lg bg-white dark:bg-slate-900" role="img" aria-label="Pasos del descenso por gradiente">
              <path d={parabola} fill="none" className="stroke-slate-300 dark:stroke-slate-600" strokeWidth={2} />
              <circle cx={px(0)} cy={py(0)} r={4} className="fill-emerald-500" />
              <polyline
                points={visible.map((x) => `${px(x).toFixed(1)},${py(x * x).toFixed(1)}`).join(" ")}
                fill="none" className="stroke-indigo-400" strokeWidth={1.2} strokeDasharray="3 2"
              />
              {visible.map((x, i) => (
                <circle key={i} cx={px(x)} cy={py(x * x)} r={i === 0 ? 5 : 3.5} className={i === 0 ? "fill-slate-500" : "fill-indigo-600"} opacity={1 - i * 0.045} />
              ))}
              {visible.length < steps.length && <text x={W / 2} y={18} textAnchor="middle" className="fill-red-500 text-[11px] font-semibold">¡Se sale del valle!</text>}
              <text x={px(0)} y={H - 2} textAnchor="middle" className="fill-emerald-600 text-[10px]">mínimo</text>
            </svg>
          </div>
          <div>
            <div className="mb-1 text-xs font-medium text-slate-500">Pérdida (loss) en cada paso</div>
            <MiniChart xLabel="Paso" yMin={0} yMax={3} series={[{ values: loss, className: "stroke-indigo-600", label: "loss de entrenamiento" }]} height={170} />
          </div>
        </div>
        <div className="mt-3"><Verdict tone={verdict[0]}>{verdict[1]}</Verdict></div>
        <p className="mt-2 text-xs text-slate-500">Ilustración simplificada: muestra la forma típica de la curva, no los valores exactos de tu proyecto. Ajuste fino: {fmtNum(lr / 100)}.</p>
      </Panel>
    </div>
  );
}

// ------------------------------------------------------------- épocas y paciencia
function simulateRun(seed: number, n = 100) {
  const r = rng(seed);
  const val: number[] = [];
  const train: number[] = [];
  for (let t = 1; t <= n; t++) {
    val.push(0.3 + 0.95 * Math.exp(-t / 4.5) + 0.0035 * Math.max(0, t - 14) ** 1.15 + 0.03 * gaussian(r));
    train.push(Math.max(0.03, 0.06 + 1.05 * Math.exp(-t / 5.5) + 0.008 * gaussian(r)));
  }
  return { val, train };
}

function earlyStop(val: number[], epochs: number, patience: number) {
  let best = Infinity;
  let bestE = 1;
  for (let e = 1; e <= epochs; e++) {
    if (val[e - 1] < best) {
      best = val[e - 1];
      bestE = e;
    } else if (e - bestE >= patience) {
      return { stop: e, best: bestE, early: e < epochs };
    }
  }
  return { stop: epochs, best: bestE, early: false };
}

export function EpochsPatienceDemo() {
  const [epochs, setEpochs] = useState(15);
  const [patience, setPatience] = useState(3);
  const [seed, setSeed] = useState(3);
  const run = useMemo(() => simulateRun(seed), [seed]);
  const { stop, best, early } = earlyStop(run.val, epochs, patience);
  const val = run.val.slice(0, epochs);
  const train = run.train.slice(0, epochs);
  // la mejor época que había dentro del máximo de épocas: si es claramente mejor, se cortó demasiado pronto
  const ideal = val.indexOf(Math.min(...val)) + 1;
  const missed = early && val[ideal - 1] < val[best - 1] - 0.02;
  const nEpochs = `${patience} época${patience === 1 ? "" : "s"} seguida${patience === 1 ? "" : "s"}`;
  const msg: [tone: "good" | "warn" | "bad", text: ReactNode] = early
    ? missed
      ? ["warn", <>Se detuvo demasiado pronto, en la época <strong>{stop}</strong>: una subida pasajera de la val_loss durante {nEpochs} bastó para cortar. Con más paciencia habría llegado a la época <strong>{ideal}</strong>, con una val_loss más baja ({val[ideal - 1].toFixed(2)} en vez de {val[best - 1].toFixed(2)}).</>]
      : ["good", <>El early stopping detuvo el entrenamiento en la época <strong>{stop}</strong>: la val_loss no mejoró durante {nEpochs}. Se usan los pesos de la época <strong>{best}</strong>, la mejor; no se pierde nada por entrenar de más.</>]
    : best === epochs
      ? ["warn", <>Terminó en la última época (<strong>{epochs}</strong>) y todavía estaba mejorando: sube las épocas máximas.</>]
      : ["good", <>Se usaron todas las épocas ({epochs}). La mejor fue la <strong>{best}</strong> y esos son los pesos que se guardan.</>];
  return (
    <div className="space-y-4">
      <Explain
        what={<>Una <strong>época</strong> es una pasada completa por todas las imágenes de entrenamiento. Tras cada época el modelo se evalúa con imágenes de <strong>validación</strong> que no usa para aprender (val_loss). La <strong>paciencia</strong> es cuántas épocas seguidas sin mejorar la val_loss se toleran antes de detenerse (early stopping).</>}
        up="Más épocas: más oportunidad de aprender (es un máximo; si no mejora, se corta antes). Más paciencia: tolera fluctuaciones y puede encontrar mejoras tardías, a costa de tiempo."
        down="Menos épocas: puede quedar a medio aprender. Menos paciencia: corta antes y ahorra tiempo, pero una fluctuación normal puede detenerlo demasiado pronto."
        tip="15 épocas y paciencia 3–5. Si siempre termina en la última época y sigue mejorando, sube las épocas, no la paciencia."
      />
      <Panel>
        <div className="grid gap-4 sm:grid-cols-2">
          <Slider label="Épocas máximas" min={1} max={100} value={epochs} onChange={setEpochs} />
          <Slider label="Paciencia (early stopping)" min={1} max={20} value={patience} onChange={setPatience} />
        </div>
        <div className="mt-4">
          <MiniChart
            yMin={0} yMax={1.4}
            series={[
              { values: train, className: "stroke-sky-500", label: "loss de entrenamiento", until: stop },
              { values: val, className: "stroke-amber-500", label: "val_loss (validación)", until: stop },
            ]}
            markers={[
              { x: best - 1, label: `mejor época (${best})`, className: "stroke-emerald-500" },
              ...(early ? [{ x: stop - 1, label: `se detiene (${stop})`, className: "stroke-red-500" }] : []),
            ]}
          />
        </div>
        <div className="mt-3 flex flex-wrap items-start gap-3">
          <div className="flex-1"><Verdict tone={msg[0]}>{msg[1]}</Verdict></div>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setSeed((s) => s + 1)}>🎲 Otra corrida</button>
        </div>
        <p className="mt-2 text-xs text-slate-500">La línea tenue después del corte son las épocas que no se ejecutan. Fíjate en que la val_loss vuelve a subir al final: es el sobreajuste que el early stopping evita.</p>
      </Panel>
    </div>
  );
}

// ---------------------------------------------------------------- ajuste fino
const PHASE1 = 12;

export function FineTuneDemo() {
  const [ft, setFt] = useState(5);
  const [similar, setSimilar] = useState(true);
  const { acc, end1, final } = useMemo(() => {
    const r = rng(11);
    const plateau = similar ? 0.76 : 0.92;
    const gain = similar ? 0.15 : 0.03;
    const p1: (number | null)[] = [];
    const p2: (number | null)[] = [];
    for (let t = 1; t <= PHASE1; t++) p1.push(plateau - 0.42 * Math.exp(-t / 2.6) + 0.008 * gaussian(r));
    const e1 = p1[PHASE1 - 1] as number;
    for (let t = 1; t <= ft; t++) p2.push(Math.min(0.995, e1 + gain * (1 - Math.exp(-t / 3.5)) + 0.006 * gaussian(r)));
    return {
      acc: [
        [...p1, ...Array(ft).fill(null)],
        [...Array(PHASE1 - 1).fill(null), e1, ...p2],
      ],
      end1: e1,
      final: ft ? (p2[ft - 1] as number) : e1,
    };
  }, [ft, similar]);
  const blocks = 10;
  const unlocked = ft > 0 ? 3 : 0;
  const pct = (v: number) => `${Math.round(v * 100)} %`;
  return (
    <div className="space-y-4">
      <Explain
        what="El entrenamiento tiene dos fases. En la primera, el modelo base (ya entrenado con millones de fotos de ImageNet) queda congelado y sólo se entrena la «cabeza» que decide la categoría. En el ajuste fino se descongelan las últimas capas del modelo base y se reentrenan con un learning rate 100 veces menor, para que aprendan rasgos propios de tus imágenes."
        up="Suele subir bastante la exactitud, sobre todo cuando las categorías se parecen entre sí. Cuesta más tiempo."
        down="Con 0 se omite la segunda fase: rápido, pero normalmente menos exacto."
        tip="5 épocas para empezar; 10–15 si las categorías son muy parecidas. La paciencia también se aplica en esta fase."
      />
      <Panel>
        <div className="grid items-end gap-4 sm:grid-cols-2">
          <Slider label="Épocas de ajuste fino" min={0} max={30} value={ft} onChange={setFt} />
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={similar} onChange={(e) => setSimilar(e.target.checked)} /> Categorías parecidas entre sí (p. ej. modelos de avión)</label>
        </div>
        <div className="mt-4">
          <div className="mb-1 text-xs font-medium text-slate-500">Capas del modelo</div>
          <div className="flex items-center gap-1" aria-label={`${unlocked} bloques del modelo base se reentrenan`}>
            {Array.from({ length: blocks }, (_, i) => {
              const open = i >= blocks - unlocked;
              return (
                <div key={i} className={`grid h-9 flex-1 place-items-center rounded text-xs transition-colors ${open ? "bg-indigo-500 text-white" : "bg-slate-300 text-slate-600 dark:bg-slate-700 dark:text-slate-300"}`} title={open ? "Se reentrena en el ajuste fino" : "Congelada"}>
                  {open ? "✎" : "🔒"}
                </div>
              );
            })}
            <div className="grid h-9 w-16 place-items-center rounded bg-emerald-500 text-xs font-medium text-white" title="Siempre se entrena">cabeza</div>
          </div>
          <p className="mt-1 text-xs text-slate-500">🔒 congeladas (conservan lo aprendido en ImageNet) · ✎ se reentrenan en el ajuste fino · la cabeza se entrena siempre.</p>
        </div>
        <div className="mt-4">
          <MiniChart
            yMin={0.3} yMax={1} yLabel="exactitud" yFormat={(v) => `${Math.round(v * 100)}%`}
            series={[
              { values: acc[0], className: "stroke-sky-500", label: "fase 1: sólo la cabeza" },
              { values: acc[1], className: "stroke-indigo-600", label: "fase 2: ajuste fino" },
            ]}
            markers={ft ? [{ x: PHASE1 - 1, label: "comienza el ajuste fino", className: "stroke-indigo-400" }] : []}
          />
        </div>
        <Verdict tone={ft === 0 && similar ? "warn" : "good"}>
          {ft === 0
            ? `Sin ajuste fino la exactitud se queda en ~${pct(end1)}.${similar ? " Con categorías parecidas conviene activarlo." : ""}`
            : `Con ${ft} épocas de ajuste fino pasa de ~${pct(end1)} a ~${pct(final)}.${!similar ? " Con categorías muy distintas la mejora es pequeña." : ""}`}
        </Verdict>
      </Panel>
    </div>
  );
}

// ---------------------------------------------------------------- batch size
const BATCHES = [4, 8, 16, 32, 64];

export function BatchDemo({ sample }: { sample: Sample }) {
  const [bi, setBi] = useState(3);
  const [size, setSize] = useState(224);
  const [seed, setSeed] = useState(5);
  const batch = BATCHES[bi];
  const trainImages = 700; // 70 % de 1.000 imágenes
  const steps = Math.ceil(trainImages / batch);
  const mem = (batch / 32) * (size / 224) ** 2;
  const curve = useMemo(() => {
    const r = rng(seed);
    const amp = 0.16 / Math.sqrt(batch / 4);
    return Array.from({ length: 40 }, (_, i) => Math.max(0.02, 0.15 + 0.9 * Math.exp(-i / 9) + amp * gaussian(r)));
  }, [batch, seed]);
  return (
    <div className="space-y-4">
      <Explain
        what="Cuántas imágenes procesa el modelo a la vez antes de corregir sus pesos. Una época se divide en pasos; en cada paso entra un lote (batch) completo a la memoria."
        up="Menos pasos por época (más rápido por época) y una curva de aprendizaje más estable, pero usa más memoria RAM; con resolución alta puede no caber."
        down="Usa menos memoria y cada corrección es más «ruidosa»; cada época tiene más pasos."
        tip="32 a 224 px. Con 300–384 px usa 16 o menos. Si el entrenamiento falla por memoria, baja primero el batch size."
      />
      <Panel>
        <div className="grid gap-4 sm:grid-cols-2">
          <Slider label="Batch size" min={0} max={BATCHES.length - 1} value={bi} onChange={setBi} display={String(batch)} />
          <label className="block">
            <span className="label">Resolución (afecta la memoria)</span>
            <select className="w-full" value={size} onChange={(e) => setSize(Number(e.target.value))}>
              {SIZES.map((s) => <option key={s} value={s}>{s} × {s}</option>)}
            </select>
          </label>
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-1 text-xs font-medium text-slate-500">Un paso = {batch} imágenes en memoria a la vez</div>
            <div className="grid grid-cols-8 gap-1 rounded-lg bg-white p-2 dark:bg-slate-900">
              {Array.from({ length: batch }, (_, i) => (
                <img key={i} src={sample.src} alt="" className="aspect-square w-full rounded-sm object-cover" style={{ filter: `hue-rotate(${(i * 37) % 360}deg)` }} />
              ))}
            </div>
            <div className="mt-3 text-xs text-slate-500">Memoria de entrenamiento comparada con el valor por defecto (32 a 224 px)</div>
            <div className="mt-1 h-3 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
              <div className={`h-full rounded-full transition-all ${mem > 2.5 ? "bg-red-500" : mem > 1.5 ? "bg-amber-500" : "bg-emerald-500"}`} style={{ width: `${Math.min(mem / 4, 1) * 100}%` }} />
            </div>
            <div className="mt-1 text-sm"><strong>≈ {mem.toFixed(1)}×</strong> {mem > 2.5 ? "— riesgo de quedarse sin memoria" : mem > 1.5 ? "— vigila la RAM" : ""}</div>
            <div className="mt-2 text-sm">Pasos por época con 1.000 imágenes (700 de entrenamiento): <strong>{steps}</strong></div>
          </div>
          <div>
            <div className="mb-1 flex items-center justify-between text-xs font-medium text-slate-500">
              Ruido en la curva de aprendizaje
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setSeed((s) => s + 1)}>🎲</button>
            </div>
            <MiniChart xLabel="Paso" yMin={0} yMax={1.4} series={[{ values: curve, className: "stroke-indigo-600", label: "loss por paso" }]} height={170} />
          </div>
        </div>
      </Panel>
    </div>
  );
}

// ---------------------------------------------------------------- modelos base
interface Arch { id: string; label: string; description: string; size_mb: number; speed: string; modality?: string }
const SPEED_RANK: Record<string, number> = { "muy rápido": 1, "rápido": 2, "medio": 3, "lento": 4, "muy lento": 5 };

/** Modelos base de un tipo de dato: redes de visión (imágenes y audio) o modelos de scikit-learn (tablas). */
export function ModelsDemo({ kind = "vision" }: { kind?: "vision" | "tabular" }) {
  const { data } = useAsync(() => api<{ items: Arch[] }>("/architectures"), []);
  const [pick, setPick] = useState<string | null>(null);
  const tabular = kind === "tabular";
  const items = (data?.items ?? []).filter((a) => (a.modality ?? "image") === (tabular ? "tabular" : "image"));
  const maxMb = Math.max(...items.map((a) => a.size_mb), 1);
  const sel = items.find((a) => a.id === pick) ?? items.find((a) => a.id === (tabular ? "random_forest" : "mobilenetv3")) ?? items[0];
  return (
    <div className="space-y-4">
      {tabular ? (
        <Explain
          what="El modelo base es el algoritmo que aprende a clasificar tus filas. Puedes elegir varios: cada uno se entrena con los mismos datos (en segundos) y luego los comparas en «Modelos y pruebas». No hay épocas, resolución ni data augmentation: esos ajustes son de las redes de imágenes."
          up="Los árboles (Random Forest, Gradient Boosting) captan relaciones complejas entre columnas y suelen ganar con datos de unos cientos de filas o más; la red neuronal rinde mejor con muchas filas."
          down="La regresión logística es la más simple y rápida: sirve de referencia y se explica fácilmente, pero sólo capta relaciones lineales."
          tip="Selecciona los cuatro: tardan segundos y así ves cuál funciona mejor con tus datos. Si dos empatan, quédate con el más simple."
        />
      ) : (
        <Explain
          what="El modelo base es la red neuronal preentrenada (con ImageNet) sobre la que se construye tu clasificador. Puedes elegir varios: cada uno se entrena con el mismo dataset y luego los comparas en «Modelos y pruebas». En la «Configuración avanzada por modelo» puedes darle a cada uno parámetros distintos. En audio se usan los mismos modelos, sobre el espectrograma de cada clip."
          up="Un modelo más grande suele ser más exacto, sobre todo con categorías parecidas, pero entrena y responde más lento y pesa más al exportarlo."
          down="Un modelo más pequeño es rápido y liviano (ideal para celulares), con algo menos de exactitud."
          tip="Empieza con MobileNetV3 Large. Si las categorías se parecen mucho, prueba EfficientNetV2 S junto a él y compara."
        />
      )}
      <Panel>
        <div className="space-y-1.5">
          {items.map((a) => (
            <button
              key={a.id} type="button" onClick={() => setPick(a.id)}
              className={`grid w-full grid-cols-[9rem_1fr_5.5rem] items-center gap-2 rounded-lg px-2 py-1 text-left text-sm transition ${sel?.id === a.id ? "bg-white shadow-sm ring-1 ring-indigo-300 dark:bg-slate-900 dark:ring-indigo-700" : "hover:bg-white/60 dark:hover:bg-slate-900/60"}`}
            >
              <span className="truncate font-medium">{a.label}</span>
              <span className="h-2.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                <span className="block h-full rounded-full bg-indigo-500" style={{ width: `${(a.size_mb / maxMb) * 100}%` }} />
              </span>
              <span className="text-right text-xs text-slate-500">{a.size_mb} MB</span>
            </button>
          ))}
        </div>
        {sel && (
          <div className="mt-3 rounded-lg bg-white p-3 text-sm dark:bg-slate-900">
            <div className="font-semibold">{sel.label}</div>
            <div className="mt-1 text-slate-600 dark:text-slate-400">{sel.description}</div>
            <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
              Velocidad en CPU:
              <span className="flex gap-0.5" aria-label={sel.speed}>
                {[1, 2, 3, 4, 5].map((i) => <span key={i} className={`h-2 w-5 rounded-sm ${i <= 6 - (SPEED_RANK[sel.speed] ?? 3) ? "bg-emerald-500" : "bg-slate-200 dark:bg-slate-700"}`} />)}
              </span>
              {sel.speed}
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}
