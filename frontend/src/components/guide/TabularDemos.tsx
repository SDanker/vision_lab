import { useState } from "react";
import { Explain, Panel, Slider, Verdict } from "./Demos";

// ------------------------------------------------------- columnas de entrada
const TABLE_COLS = [
  { name: "N.º", kind: "identificador", values: ["1", "2", "3", "4", "5"] },
  { name: "Nombre", kind: "identificador", values: ["Bulbasaur", "Ivysaur", "Mewtwo", "Pikachu", "Rayquaza"] },
  { name: "Tipo", kind: "texto", values: ["Planta", "Planta", "Psíquico", "Eléctrico", "Dragón"] },
  { name: "Ataque", kind: "número", values: ["49", "62", "110", "55", "150"] },
  { name: "Defensa", kind: "número", values: ["49", "63", "90", "40", "90"] },
  { name: "Velocidad", kind: "número", values: ["45", "60", "130", "90", "95"] },
] as const;
const TARGET_VALUES = ["No", "No", "Sí", "No", "Sí"];

function Bar({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs"><span>{label}</span><span className="font-mono">{(value * 100).toFixed(0)}%</span></div>
      <div className="h-2.5 rounded-full bg-slate-200 dark:bg-slate-800"><div className={`h-2.5 rounded-full ${tone}`} style={{ width: `${value * 100}%` }} /></div>
    </div>
  );
}

/** Qué columnas conviene usar: los identificadores se memorizan y no ayudan con filas nuevas. */
export function TabularColumnsDemo() {
  const [used, setUsed] = useState<Set<string>>(new Set(["Tipo", "Ataque", "Defensa", "Velocidad"]));
  const toggle = (n: string) => setUsed((p) => {
    const next = new Set(p);
    if (next.has(n)) next.delete(n);
    else next.add(n);
    return next;
  });
  const ids = TABLE_COLS.filter((c) => c.kind === "identificador" && used.has(c.name)).length;
  const real = TABLE_COLS.filter((c) => c.kind !== "identificador" && used.has(c.name)).length;
  const baseline = 0.9; // acertar siempre «No»
  const fresh = real + ids === 0 ? baseline : Math.max(0.3, baseline + (0.96 - baseline) * (1 - 0.55 ** real) - 0.05 * ids); // filas nuevas
  const train = real + ids === 0 ? baseline : Math.min(1, fresh + 0.02 + 0.07 * ids); // filas con las que entrenó
  return (
    <div className="space-y-4">
      <Explain
        what={<>Cada fila de tu tabla es un ejemplo. Eliges <strong>una columna a predecir</strong> (aquí «Legendario») y <strong>las columnas de entrada</strong> con las que el modelo decide. Marca y desmarca columnas y mira cómo cambia el resultado simulado.</>}
        up="Más columnas con información real (números y categorías con relación a lo que predices) suelen mejorar la exactitud en filas nuevas."
        down="Quitar columnas útiles empeora el resultado; quitar las que no aportan lo deja igual o mejor, y más simple."
        tip="Deja fuera los identificadores (nombres, números correlativos, códigos únicos): el modelo los memoriza en vez de aprender. La plataforma ya los desmarca sola, pero tú decides."
      />
      <Panel>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                {TABLE_COLS.map((c) => (
                  <th key={c.name} className="px-2 py-1 text-left">
                    <label className="flex cursor-pointer items-center gap-1.5 font-medium">
                      <input type="checkbox" checked={used.has(c.name)} onChange={() => toggle(c.name)} aria-label={`Usar ${c.name}`} />
                      {c.name}
                    </label>
                    <span className={`block text-[10px] font-normal ${c.kind === "identificador" ? "text-amber-700 dark:text-amber-300" : "text-slate-400"}`}>{c.kind === "identificador" ? "⚠ identificador" : c.kind}</span>
                  </th>
                ))}
                <th className="px-2 py-1 text-left font-medium text-indigo-700 dark:text-indigo-300">Legendario<span className="block text-[10px] font-normal text-slate-400">lo que se predice</span></th>
              </tr>
            </thead>
            <tbody>
              {TARGET_VALUES.map((t, r) => (
                <tr key={r} className="border-t border-indigo-100 dark:border-indigo-900/40">
                  {TABLE_COLS.map((c) => <td key={c.name} className={`px-2 py-1 ${used.has(c.name) ? "" : "text-slate-300 dark:text-slate-600"}`}>{c.values[r]}</td>)}
                  <td className="px-2 py-1 font-medium">{t}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <Bar label="Referencia: siempre «No»" value={baseline} tone="bg-slate-400" />
          <Bar label="En las filas con que entrenó" value={train} tone="bg-indigo-500" />
          <Bar label="En filas nuevas (lo que importa)" value={fresh} tone={fresh > baseline + 0.02 ? "bg-emerald-500" : "bg-red-500"} />
        </div>
      </Panel>
      {real + ids === 0 && <Verdict tone="bad">Sin columnas de entrada el modelo no tiene en qué basarse: sólo puede repetir la clase más común.</Verdict>}
      {ids > 0 && <Verdict tone="bad">Incluiste {ids === 1 ? "una columna que es un identificador" : "columnas que son identificadores"}: acierta casi todo lo que ya vio, pero en filas nuevas no mejora (o empeora). Es el sobreajuste más común en tablas.</Verdict>}
      {ids === 0 && real > 0 && real < 3 && <Verdict tone="warn">Funciona, pero con {real === 1 ? "una sola columna" : "sólo 2 columnas"} el modelo tiene poco para decidir. Prueba agregando más columnas con información real.</Verdict>}
      {ids === 0 && real >= 3 && <Verdict tone="good">Buena selección: sólo columnas con información real. En filas nuevas supera con claridad a la referencia.</Verdict>}
      <p className="text-xs text-slate-500">Simulación para entender la idea; los resultados reales dependen de tus datos. Se configura en la pestaña «Datos» del proyecto.</p>
    </div>
  );
}

// -------------------------------------------------------------- leer los resultados
/** Cómo interpretar exactitud, referencia y sobreajuste. Los avisos son los mismos que genera la plataforma. */
export function TabularResultsDemo() {
  const [baseline, setBaseline] = useState(0.9);
  const [train, setTrain] = useState(0.97);
  const [test, setTest] = useState(0.93);
  const gap = train - test;
  const beats = test > baseline + 0.02;
  return (
    <div className="space-y-4">
      <Explain
        what={<>Al terminar, «Modelos y pruebas» muestra la <strong>exactitud en prueba</strong> (filas que el modelo nunca vio) junto a una <strong>referencia</strong>: lo que acertaría si respondiera siempre la categoría más común. Un modelo sólo sirve si la supera.</>}
        up="Si la exactitud en entrenamiento sube mucho más que la de prueba, el modelo está memorizando (sobreajuste)."
        down="Si la exactitud en prueba no supera la referencia, las columnas elegidas no explican la categoría o hay muy pocos datos."
        tip="Con categorías muy desiguales (p. ej. 90 % «No») mira el F1 macro y los resultados por categoría, no sólo la exactitud: un 90 % puede significar no haber aprendido nada."
      />
      <Panel>
        <div className="grid gap-4 sm:grid-cols-3">
          <Slider label="Referencia (siempre la más común)" value={baseline} min={0.3} max={0.99} step={0.01} onChange={setBaseline} display={`${(baseline * 100).toFixed(0)}%`} />
          <Slider label="Exactitud en entrenamiento" value={train} min={0.3} max={1} step={0.01} onChange={setTrain} display={`${(train * 100).toFixed(0)}%`} />
          <Slider label="Exactitud en prueba" value={test} min={0.3} max={1} step={0.01} onChange={setTest} display={`${(test * 100).toFixed(0)}%`} />
        </div>
      </Panel>
      {!beats && <Verdict tone="bad">No supera la referencia ({(baseline * 100).toFixed(0)} %): el modelo no aprendió nada útil. Revisa que las columnas de entrada tengan relación con la que predices y que haya suficientes filas por categoría.</Verdict>}
      {beats && gap > 0.15 && <Verdict tone="warn">Sobreajuste: entrenamiento {(gap * 100).toFixed(0)} puntos por encima de prueba. Prueba con más filas, menos columnas o un modelo más simple (regresión logística).</Verdict>}
      {beats && gap <= 0.15 && <Verdict tone="good">Buen resultado: supera la referencia por {((test - baseline) * 100).toFixed(0)} puntos y entrenamiento y prueba están cerca, así que generaliza.</Verdict>}
    </div>
  );
}
