"""Une las etiquetas del anotador Claude y compara la regla del diccionario.

Entradas (en esta carpeta):
- NO_ABRIR_hasta_terminar_anotador_B_piloto.csv      (T0001-T0100, guía 1.0)
- NO_ABRIR_hasta_terminar_anotador_B_calibracion.csv (T0101-T0150, guía 1.1)
- claude/lote_*.txt                                   (T0151-T1130, guía 1.1)
- clave_muestra_NO_ENTREGAR.csv                       (estrato, peso, etiqueta del diccionario)

Salidas:
- etiquetas_claude.csv
- analisis_diccionario_vs_claude.json
"""
import glob
import json

import pandas as pd

cols = ["id_anotacion", "p1_estres_propio", "p2_crisis_electrica", "p3_ansiedad_o_tristeza", "comentario"]
parts = [
    pd.read_csv("NO_ABRIR_hasta_terminar_anotador_B_piloto.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").assign(guia="1.0"),
    pd.read_csv("NO_ABRIR_hasta_terminar_anotador_B_calibracion.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").assign(guia="1.1"),
]
for f in sorted(glob.glob("claude/lote_*.txt")):
    rows = []
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        t = line.split(" ", 4)
        rows.append({"id_anotacion": t[0], "p1_estres_propio": t[1], "p2_crisis_electrica": t[2],
                     "p3_ansiedad_o_tristeza": "" if t[3] == "-" else t[3],
                     "comentario": t[4] if len(t) > 4 else ""})
    parts.append(pd.DataFrame(rows).assign(guia="1.1"))
lab = pd.concat(parts, ignore_index=True)[cols + ["guia"]]

# controles
assert lab.id_anotacion.is_unique, lab[lab.id_anotacion.duplicated()].id_anotacion.tolist()
assert len(lab) == 1130, len(lab)
assert set(lab.p1_estres_propio) <= {"0", "1", "NS"}, set(lab.p1_estres_propio)
assert set(lab.p2_crisis_electrica) <= {"0", "1"}, set(lab.p2_crisis_electrica)
assert set(lab.p3_ansiedad_o_tristeza) <= {"", "A", "D"}, set(lab.p3_ansiedad_o_tristeza)
lab.to_csv("etiquetas_claude.csv", index=False, encoding="utf-8-sig")

k = pd.read_csv("clave_muestra_NO_ENTREGAR.csv", dtype={"id_anotacion": str})
m = lab.merge(k, on="id_anotacion")
assert len(m) == 1130
m = m[m.p1_estres_propio != "NS"].copy()
m["y"] = (m.p1_estres_propio == "1").astype(int)
m["d"] = m.etiqueta_diccionario.astype(int)
m["c"] = (m.p2_crisis_electrica == "1").astype(int)
w = m.peso_muestral


def wmean(x, mask=None):
    mk = mask if mask is not None else pd.Series(True, index=x.index)
    return float((x[mk] * w[mk]).sum() / w[mk].sum())


res = {
    "anotados": 1130,
    "NS_excluidos": int((lab.p1_estres_propio == "NS").sum()),
    "estres_propio_en_muestra": int(m.y.sum()),
    "por_estrato": {e: {"n": int(len(g)), "estres_propio": int(g.y.sum()), "pct": round(100 * g.y.mean(), 1),
                        "crisis_electrica_pct": round(100 * g.c.mean(), 1)}
                    for e, g in m.groupby("estrato")},
    "corpus_ponderado": {
        "prevalencia_estres_propio_pct": round(100 * wmean(m.y), 1),
        "prevalencia_etiqueta_diccionario_pct": round(100 * wmean(m.d), 1),
        "precision_diccionario_pct": round(100 * wmean(m.y, m.d == 1), 1),
        "cobertura_diccionario_pct": round(100 * wmean(m.d, m.y == 1), 1),
        "falsos_negativos_entre_negativos_del_diccionario_pct": round(100 * wmean(m.y, m.d == 0), 1),
        "crisis_electrica_pct": round(100 * wmean(m.c), 1),
        "estres_propio_entre_tuits_de_la_crisis_pct": round(100 * wmean(m.y, m.c == 1), 1),
    },
}
json.dump(res, open("analisis_diccionario_vs_claude.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
