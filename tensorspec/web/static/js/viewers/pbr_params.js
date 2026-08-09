/** @param {boolean} shiny
 *  @param {"atom"|"bond"} kind
 *  @returns {{ metalness: number, roughness: number }}
 */
export function pbrMaterialParams(shiny, kind) {
  if (shiny) {
    return { metalness: 0.85, roughness: 0.2 };
  }
  if (kind === "bond") {
    return { metalness: 0.1, roughness: 0.5 };
  }
  return { metalness: 0.1, roughness: 0.45 };
}
