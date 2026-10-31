"""Resume resultados/metricas.jsonl en tablas (promedio y desviación entre semillas).

Uso: uv run python resumir.py [ref]   (ref por defecto: ref_mayoria)
Escribe resultados/tabla_<ref>.md y la imprime.
"""
import json
import sys
from pathlib import Path

import pandas as pd

SAL = Path(__file__).resolve().parent / "resultados"
ref = sys.argv[1] if len(sys.argv) > 1 else "ref_mayoria"
d = pd.DataFrame([json.loads(l) for l in open(SAL / "metricas.jsonl", encoding="utf-8") if l.strip()])
d = d[d.ref == ref]
agg = d.groupby(["modelo", "fuente", "enmascarado"]).agg(
    semillas=("semilla", "nunique"),
    f1=("f1", "mean"), f1_sd=("f1", "std"),
    precision=("precision", "mean"), recall=("recall", "mean"),
    pr_auc=("pr_auc", "mean"), mcc=("mcc", "mean"),
    pos_pred_corpus=("positivos_predichos_corpus_pct", "mean"),
).reset_index()
orden = ["regla_diccionario", "tfidf", "fasttext", "beto_congelado", "beto_ft", "robertuito_ft", "qwen_directo", "gemma_directo"]
agg["o"] = agg.modelo.map({m: i for i, m in enumerate(orden)})
agg = agg.sort_values(["o", "fuente", "enmascarado"]).drop(columns="o")
fmt = agg.copy()
fmt["f1"] = [f"{m:.3f}" + (f" ± {s:.3f}" if pd.notna(s) and n > 1 else "") for m, s, n in zip(agg.f1, agg.f1_sd, agg.semillas)]
fmt = fmt.drop(columns="f1_sd").round(3)
n = int(d.n.iloc[0]) if len(d) else 0
pos = int(d.positivos.iloc[0]) if len(d) else 0
txt = f"Referencia: {ref} (n = {n}, positivos = {pos})\n\n" + fmt.to_markdown(index=False)
(SAL / f"tabla_{ref}.md").write_text(txt, encoding="utf-8")
print(txt)
