import { api, errMsg } from "./api";
import { BASE_LABEL } from "./format";
import type { ModelInfo } from "./types";

export const modelLabel = (m: ModelInfo) =>
  `${m.project_name ? `«${m.project_name}» ` : ""}v${m.version} (${BASE_LABEL[m.base_model] ?? m.base_model})`;

/** Pide confirmación y borra los modelos uno a uno. Devuelve `null` si el usuario cancela. */
export async function deleteModels(list: ModelInfo[]): Promise<{ deleted: number; errors: string[] } | null> {
  if (list.length === 0) return null;
  const what = list.length === 1 ? `el modelo ${modelLabel(list[0])}` : `${list.length} modelos:\n- ${list.map(modelLabel).join("\n- ")}`;
  if (!confirm(`¿Borrar ${what}?\n\nSe eliminan sus archivos descargables y no se puede deshacer. El historial de entrenamientos se conserva.`)) return null;
  let deleted = 0;
  const errors: string[] = [];
  for (const m of list) {
    try {
      await api(`/models/${m.id}`, { method: "DELETE" });
      deleted++;
    } catch (e) {
      errors.push(`${modelLabel(m)}: ${errMsg(e)}`);
    }
  }
  return { deleted, errors };
}

export const deleteSummary = (r: { deleted: number; errors: string[] }) =>
  r.errors.length
    ? { tone: "error" as const, text: `${r.deleted} borrado(s). No se pudo borrar: ${r.errors.join(" · ")}` }
    : { tone: "success" as const, text: r.deleted === 1 ? "Modelo borrado." : `${r.deleted} modelos borrados.` };
