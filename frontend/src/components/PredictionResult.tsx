import { fmtSeconds, pct } from "../lib/format";
import type { Prediction } from "../lib/types";

/** Barras con la puntuación de las categorías más probables. */
export function ScoreBars({ scores, max = 5 }: { scores: Prediction["scores"]; max?: number }) {
  return (
    <div className="space-y-1">
      {scores.slice(0, max).map((s) => (
        <div key={s.label}>
          <div className="flex justify-between text-xs"><span className="truncate">{s.label}</span><span>{pct(s.score)}</span></div>
          <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800"><div className="h-1.5 rounded-full bg-brand-500" style={{ width: `${s.score * 100}%` }} /></div>
        </div>
      ))}
    </div>
  );
}

/** Resultado por segmento de un audio largo. */
export function Segments({ segments }: { segments?: Prediction["segments"] }) {
  if (!segments || segments.length < 2) return null;
  return (
    <details className="mt-2 text-xs">
      <summary className="cursor-pointer text-slate-500">Resultado por segmento ({segments.length})</summary>
      <ul className="mt-1 max-h-40 space-y-0.5 overflow-auto">
        {segments.map((sg, k) => (
          <li key={k} className="flex justify-between gap-2">
            <span className="shrink-0 tabular-nums text-slate-500">{fmtSeconds(sg.start)}–{fmtSeconds(sg.end)}</span>
            <span className="min-w-0 truncate">{sg.label}</span>
            <span className="shrink-0 tabular-nums">{pct(sg.confidence, 0)}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}
