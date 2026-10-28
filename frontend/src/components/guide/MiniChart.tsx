import type { ReactNode } from "react";

export interface Series {
  /** valores por época (índice 0 = época 1); null = sin dato (no se dibuja) */
  values: (number | null)[];
  className: string; // clase de trazo, p. ej. "stroke-indigo-500"
  label: string;
  dashed?: boolean;
  /** dibujar sólo hasta este índice (excl.); el resto se muestra tenue */
  until?: number;
}

/** Gráfico de líneas mínimo en SVG para las demos de la guía. */
export default function MiniChart({
  series, yMin, yMax, xLabel = "Época", yLabel, markers = [], height = 190, children, yFormat = (v) => v.toFixed(2),
}: {
  series: Series[];
  yMin: number;
  yMax: number;
  xLabel?: string;
  yLabel?: string;
  markers?: { x: number; label: string; className: string }[];
  height?: number;
  children?: ReactNode;
  yFormat?: (v: number) => string;
}) {
  const W = 560;
  const H = height;
  const pad = { l: 42, r: 12, t: 12, b: 28 };
  const n = Math.max(...series.map((s) => s.values.length), 2);
  const x = (i: number) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (Math.min(Math.max(v, yMin), yMax) - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);
  const path = (vals: (number | null)[], from: number, to: number) => {
    let d = "";
    let pen = false;
    for (let i = from; i < to; i++) {
      const v = vals[i];
      if (v == null || !Number.isFinite(v)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)} `;
      pen = true;
    }
    return d;
  };
  const ticks = [yMin, (yMin + yMax) / 2, yMax];
  const xticks = [0, Math.floor((n - 1) / 2), n - 1];
  return (
    <figure className="w-full">
      {yLabel && <div className="text-[11px] text-slate-400">{yLabel}</div>}
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={series.map((s) => s.label).join(", ")}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} className="stroke-slate-200 dark:stroke-slate-700" strokeWidth={1} />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" className="fill-slate-500 text-[12px]">{yFormat(t)}</text>
          </g>
        ))}
        {xticks.map((i) => (
          <text key={i} x={x(i)} y={H - 10} textAnchor="middle" className="fill-slate-500 text-[12px]">{i + 1}</text>
        ))}
        <text x={W - pad.r} y={H - pad.b - 4} textAnchor="end" className="fill-slate-400 text-[12px]">{xLabel}</text>
        {markers.map((m) => (
          <g key={m.label}>
            <line x1={x(m.x)} x2={x(m.x)} y1={pad.t} y2={H - pad.b} className={m.className} strokeWidth={1.5} strokeDasharray="4 3" />
          </g>
        ))}
        {series.map((s) => {
          const until = s.until ?? s.values.length;
          return (
            <g key={s.label} fill="none" strokeLinecap="round" strokeLinejoin="round">
              {until < s.values.length && (
                <path d={path(s.values, Math.max(until - 1, 0), s.values.length)} className={s.className} strokeWidth={2} opacity={0.2} strokeDasharray="3 4" />
              )}
              <path d={path(s.values, 0, until)} className={s.className} strokeWidth={2.4} strokeDasharray={s.dashed ? "6 4" : undefined} />
            </g>
          );
        })}
        {children}
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <svg width="18" height="6" aria-hidden="true"><line x1="0" x2="18" y1="3" y2="3" className={s.className} strokeWidth={2.4} strokeDasharray={s.dashed ? "6 4" : undefined} /></svg>
            {s.label}
          </span>
        ))}
        {markers.map((m) => (
          <span key={m.label} className="inline-flex items-center gap-1.5">
            <svg width="10" height="12" aria-hidden="true"><line x1="5" x2="5" y1="0" y2="12" className={m.className} strokeWidth={1.5} strokeDasharray="3 2" /></svg>
            {m.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
