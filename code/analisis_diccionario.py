"""Análisis del diccionario para el artículo: fechas reales (del ID del tuit), estrés por estrato y precisión por término.

Uso: python analisis_diccionario.py <carpeta del área>. Necesita data/ids_reales_tmp.csv (emparejado con los datos crudos)."""
import sys, json, pandas as pd, numpy as np
B = sys.argv[1]
ids = pd.read_csv(B + r"\data\ids_reales_tmp.csv", dtype=str)
n = pd.to_numeric(ids.id_real, errors="coerce"); print("sin id real:", int(n.isna().sum()))
ids["fecha_real"] = pd.to_datetime(ids.id_real.map(lambda s: (int(s) >> 22) + 1288834974657 if str(s).isdigit() else None), unit="ms").dt.normalize()
print("rango real:", ids.fecha_real.min(), ids.fecha_real.max())
print(ids.fecha_real.dt.to_period("M").value_counts().sort_index().to_string())
ids[["Tweet_ID", "id_real", "fecha_real"]].rename(columns={"Tweet_ID": "tweet_id"}).to_csv(B + r"\data\ids_reales.csv", index=False)

A = B + r"\anotacion"
k = pd.read_csv(A + r"\clave_muestra_NO_ENTREGAR.csv")
c = pd.read_csv(A + r"\etiquetas_claude.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")[["id_anotacion", "p1_estres_propio", "p2_crisis_electrica"]]
def llm(p, s):
    d = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]).drop_duplicates("id_anotacion", keep="last")
    return d[["id_anotacion", "p1", "p2"]].rename(columns={"p1": "p1_" + s, "p2": "p2_" + s})
m = k.merge(c, on="id_anotacion").merge(llm(A + r"\anotador_qwen.jsonl", "q"), on="id_anotacion").merge(llm(A + r"\anotador_gemma.jsonl", "g"), on="id_anotacion")
def may(r):
    v = [x for x in (r.p1_estres_propio, r.p1_q, r.p1_g) if x in ("0", "1")]
    return np.nan if len(v) < 2 or v.count("1") == v.count("0") else float(v.count("1") > v.count("0"))
m["y"] = m.apply(may, axis=1)
v2 = m[["p2_crisis_electrica", "p2_q", "p2_g"]].apply(lambda r: (r == "1").sum() >= 2, axis=1)
m["crisis"] = v2.astype(int)
print("\nPor estrato (mayoría):")
print(m.groupby("estrato").agg(n=("y", "size"), estres_pct=("y", lambda s: round(100 * s.mean(), 1)), crisis_mayoria_pct=("crisis", lambda s: round(100 * s.mean(), 1))).to_string())
print("\nEstrés según si el tuit trata de la crisis (mayoría P2), ponderado:")
w = m.peso_muestral
for cr in (1, 0):
    s = (m.crisis == cr) & m.y.notna()
    print(" crisis=%d: estres %.1f %%  | dicc marca %.1f %%" % (cr, 100 * (w[s] * m.y[s]).sum() / w[s].sum(), 100 * (w[s] * m.etiqueta_diccionario[s]).sum() / w[s].sum()))
print("crisis (P2 mayoría) ponderada en el corpus: %.1f %%" % (100 * (w * m.crisis).sum() / w.sum()))
pos = m[(m.etiqueta_diccionario == 1) & m.y.notna()].copy()
pos["term"] = pos.terminos_activados.str.split("|")
t = pos.explode("term")
tab = t.groupby("term").agg(n=("y", "size"), estres=("y", "sum")).query("n >= 10")
tab["precision_pct"] = (100 * tab.estres / tab.n).round(1)
print("\nPrecisión por término (positivos del diccionario en la muestra, n>=10):")
print(tab.sort_values("n", ascending=False).to_string())
neg = m[(m.etiqueta_diccionario == 0) & (m.y == 1)]
print("\nfalsos negativos del diccionario en la muestra:", len(neg), "de", int(m.y.sum()), "positivos")
