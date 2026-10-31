"""Compara los tres anotadores automáticos (Claude, Qwen3.5-9B y Gemma 4 12B) y la regla del diccionario.

Salidas:
- comparacion_anotadores.json  (acuerdo, kappa por pares, kappa de Fleiss y métricas del diccionario con cada referencia)
- desacuerdos_P1.csv           (tuits donde los tres anotadores no coinciden en P1, con la mayoría)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def leer_llm(p, suf):
    d = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]).drop_duplicates("id_anotacion", keep="last")
    return d[["id_anotacion", "p1", "p2", "p3", "razon"]].rename(columns={c: f"{c}_{suf}" for c in ["p1", "p2", "p3", "razon"]})


c = pd.read_csv("etiquetas_claude.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
k = pd.read_csv("clave_muestra_NO_ENTREGAR.csv", dtype={"id_anotacion": str})
h = pd.read_csv("hoja_anotador_A.csv", dtype=str, encoding="utf-8-sig")[["id_anotacion", "texto"]]

m = (c.rename(columns={"p1_estres_propio": "p1_c", "p2_crisis_electrica": "p2_c", "p3_ansiedad_o_tristeza": "p3_c", "comentario": "razon_c"})
     .merge(leer_llm("anotador_qwen.jsonl", "q"), on="id_anotacion")
     .merge(leer_llm("anotador_gemma.jsonl", "g"), on="id_anotacion")
     .merge(k, on="id_anotacion").merge(h, on="id_anotacion"))
print("tuits con las tres anotaciones:", len(m))
A = {"claude": "c", "qwen": "q", "gemma": "g"}


def kappa(a, b):
    return round(float(cohen_kappa_score(a, b)), 3)


res = {"n": int(len(m))}
for preg in ["p1", "p2"]:
    ok = np.logical_and.reduce([m[f"{preg}_{s}"].isin(["0", "1"]) for s in A.values()])
    r = {"n_sin_NS": int(ok.sum()), "kappa_fleiss": None, "pares": {}}
    for i, (na, sa) in enumerate(A.items()):
        r[f"positivos_{na}"] = int((m[f"{preg}_{sa}"] == "1").sum())
        r[f"NS_{na}"] = int((m[f"{preg}_{sa}"] == "NS").sum())
        for nb, sb in list(A.items())[i + 1:]:
            both = m[m[f"{preg}_{sa}"].isin(["0", "1"]) & m[f"{preg}_{sb}"].isin(["0", "1"])]
            r["pares"][f"{na}-{nb}"] = {"acuerdo_bruto": round(float((both[f"{preg}_{sa}"] == both[f"{preg}_{sb}"]).mean()), 3),
                                       "kappa": kappa(both[f"{preg}_{sa}"], both[f"{preg}_{sb}"])}
    sub = m[ok]
    x = sub[[f"{preg}_{s}" for s in A.values()]].astype(int).values
    unos = x.sum(1)
    pi = (unos * (unos - 1) + (3 - unos) * (2 - unos)) / 6  # kappa de Fleiss, 3 anotadores binarios
    pe = (unos.sum() / x.size) ** 2 + (1 - unos.sum() / x.size) ** 2
    r["kappa_fleiss"] = round(float((pi.mean() - pe) / (1 - pe)), 3)
    r["unanimes_pct"] = round(float(100 * np.isin(unos, [0, 3]).mean()), 1)
    res[preg.upper()] = r


# mayoría en P1: al menos 2 votos iguales entre los que dieron 0/1
def mayoria(r):
    v = [r[f"p1_{s}"] for s in A.values() if r[f"p1_{s}"] in ("0", "1")]
    if len(v) < 2 or v.count("1") == v.count("0"):
        return "NS"
    return "1" if v.count("1") > v.count("0") else "0"


m["p1_mayoria"] = m.apply(mayoria, axis=1)


# métricas de la regla del diccionario con distintas referencias (ponderadas al corpus)
def metricas(col, mask):
    y = (m[col] == "1").astype(int)
    d = m.loc[mask, "etiqueta_diccionario"].astype(int)
    w = m.loc[mask, "peso_muestral"]
    yy = y[mask]
    wm = lambda x, s: float((x[s] * w[s]).sum() / w[s].sum())
    t = pd.Series(True, index=yy.index)
    return {
        "n": int(mask.sum()),
        "positivos": int(yy.sum()),
        "prevalencia_estres_pct": round(100 * wm(yy, t), 1),
        "prevalencia_diccionario_pct": round(100 * wm(d, t), 1),
        "precision_diccionario_pct": round(100 * wm(yy, d == 1), 1),
        "cobertura_diccionario_pct": round(100 * wm(d, yy == 1), 1),
    }


for na, sa in A.items():
    res[f"diccionario_vs_{na}"] = metricas(f"p1_{sa}", m[f"p1_{sa}"].isin(["0", "1"]))
res["diccionario_vs_mayoria"] = metricas("p1_mayoria", m.p1_mayoria.isin(["0", "1"]))
res["diccionario_vs_claude_gemma"] = metricas("p1_c", m.p1_c.isin(["0", "1"]) & (m.p1_c == m.p1_g))
res["diccionario_vs_unanimidad"] = metricas("p1_c", m.p1_c.isin(["0", "1"]) & (m.p1_c == m.p1_q) & (m.p1_c == m.p1_g))

json.dump(res, open("comparacion_anotadores.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))

d = m[~((m.p1_c == m.p1_q) & (m.p1_c == m.p1_g))][
    ["id_anotacion", "texto", "p1_c", "p1_q", "p1_g", "p1_mayoria", "razon_c", "razon_q", "razon_g", "estrato"]]
d.to_csv("desacuerdos_P1.csv", index=False, encoding="utf-8-sig")
print("tuits sin unanimidad en P1:", len(d), "| por estrato:", d.estrato.value_counts().to_dict())
