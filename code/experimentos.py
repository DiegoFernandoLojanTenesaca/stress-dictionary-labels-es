"""Experimentos del artículo: ¿qué aprenden los modelos entrenados con etiquetas de diccionario?

Entrenamiento: 2.490 tuits (pool sin solaparse con la prueba), con dos fuentes de etiqueta:
    - "diccionario": la regla de palabras clave de la tesis.
    - "qwen": Qwen3.5-9B con la guía 1.1.
Prueba: los 1.130 tuits anotados, con tres referencias:
    - "mayoria": mayoría de Claude, Qwen y Gemma (empates y NS fuera).
    - "claude_gemma": solo tuits donde Claude y Gemma coinciden (no depende de Qwen).
    - "claude": la etiqueta de Claude.
Modelos: regla del diccionario, TF-IDF + regresión logística, fastText del dominio + SVM,
BETO congelado + regresión logística, BETO ajustado y RoBERTuito ajustado.
Variante "enmascarado": se ocultan en entrenamiento y prueba las palabras del diccionario.

Uso:
    uv run python experimentos.py --modelos tfidf fasttext beto_congelado beto_ft robertuito_ft \
        --fuentes diccionario qwen --enmascarar no si --semillas 13 42 77
"""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

AREA = Path(__file__).resolve().parent.parent
ANOT = AREA / "anotacion"
SAL = Path(__file__).resolve().parent / "resultados"
SAL.mkdir(exist_ok=True)

# ---------------------------------------------------------------- datos


def leer_jsonl(p):
    if not Path(p).exists():
        return pd.DataFrame(columns=["id_anotacion", "p1"])
    df = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])
    return df.drop_duplicates("id_anotacion", keep="last")[["id_anotacion", "p1"]]


def cargar():
    hoja = pd.read_csv(ANOT / "hoja_anotador_A.csv", encoding="utf-8-sig")[["id_anotacion", "texto"]]
    clave = pd.read_csv(ANOT / "clave_muestra_NO_ENTREGAR.csv")
    claude = pd.read_csv(ANOT / "etiquetas_claude.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    test = hoja.merge(clave, on="id_anotacion").merge(
        claude[["id_anotacion", "p1_estres_propio"]].rename(columns={"p1_estres_propio": "claude"}), on="id_anotacion")
    test = test.merge(leer_jsonl(ANOT / "anotador_qwen.jsonl").rename(columns={"p1": "qwen"}), on="id_anotacion", how="left")
    test = test.merge(leer_jsonl(ANOT / "anotador_gemma.jsonl").rename(columns={"p1": "gemma"}), on="id_anotacion", how="left")

    def mayoria(r):
        v = [x for x in (r.claude, r.qwen, r.gemma) if x in ("0", "1")]
        unos = v.count("1")
        ceros = v.count("0")
        if len(v) < 2 or unos == ceros:
            return np.nan
        return 1.0 if unos > ceros else 0.0

    test["ref_mayoria"] = test.apply(mayoria, axis=1)
    cg = test.claude.isin(["0", "1"]) & (test.claude == test.gemma)
    test["ref_claude_gemma"] = np.where(cg, test.claude.map({"0": 0.0, "1": 1.0}), np.nan)
    test["ref_claude"] = test.claude.map({"0": 0.0, "1": 1.0})

    pool = pd.read_csv(ANOT / "pool_entrenamiento.csv", encoding="utf-8-sig")
    pool = pool.merge(leer_jsonl(ANOT / "qwen_pool.jsonl").rename(columns={"p1": "qwen"}), on="id_anotacion", how="left")
    pool["y_diccionario"] = pool.etiqueta_diccionario.astype(int)
    pool["y_qwen"] = pool.qwen.map({"0": 0, "1": 1})  # NS o faltante -> NaN
    return test, pool


# ---------------------------------------------------------------- enmascarado de términos del diccionario

RAICES = ["estres", "tension", "tenso", "tensa", "agot", "cansa", "fatig", "exhaust", "insomn", "desvel", "ansi",
          "preocup", "frustr", "irrit", "desesper", "agobi", "concentr", "desenfo", "trabaj", "estudi", "improductiv",
          "retras", "clase", "colaps", "hart", "insoport", "estresor"]
FRASES = ["dormir mal", "no aguanto", "hasta cuando", "el colmo"]


def sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def enmascarar(t):
    """Borra (no reemplaza) los términos del diccionario: un marcador fijo delataría dónde estaban."""
    s = sin_tildes(str(t).lower())
    for f in FRASES:
        s = s.replace(f, " ")
    toks = re.findall(r"\w+|[^\w\s]", s)
    return " ".join(tok for tok in toks if not any(tok.startswith(r) for r in RAICES))


assert enmascarar("Estoy muy estresado, no aguanto el trabajo") == "estoy muy , el"


# ---------------------------------------------------------------- modelos clásicos


def tfidf_lr(xtr, ytr, xte, semilla):
    vec = FeatureUnion([("pal", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                        ("car", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True))])
    a = vec.fit_transform(xtr)
    m = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=semilla).fit(a, ytr)
    return m.predict_proba(vec.transform(xte))[:, 1]


def tokens(t):
    return re.findall(r"\w+", sin_tildes(str(t).lower()))


def fasttext_svm(xtr, ytr, xte, semilla):
    from gensim.models import FastText
    corp = [tokens(t) for t in xtr]
    ft = FastText(sentences=corp, vector_size=128, window=5, min_count=2, sg=1, epochs=20, seed=semilla, workers=1)

    def vec(t):
        tk = tokens(t)
        return np.mean([ft.wv[w] for w in tk], axis=0) if tk else np.zeros(128)

    a, b = np.vstack([vec(t) for t in xtr]), np.vstack([vec(t) for t in xte])
    sc = StandardScaler().fit(a)
    m = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=semilla).fit(sc.transform(a), ytr)
    return m.predict_proba(sc.transform(b))[:, 1]


# ---------------------------------------------------------------- transformers

MODELOS_HF = {"beto": "dccuchile/bert-base-spanish-wwm-uncased", "robertuito": "pysentimiento/robertuito-base-uncased"}


def incrustar(nombre, textos, lote=32):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELOS_HF[nombre])
    mod = AutoModel.from_pretrained(MODELOS_HF[nombre]).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            enc = tok(list(textos[i:i + lote]), padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            h = mod(**enc).last_hidden_state
            msk = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * msk).sum(1) / msk.sum(1)).cpu().numpy())
    del mod
    torch.cuda.empty_cache()
    return np.vstack(out)


def congelado_lr(nombre, xtr, ytr, xte, semilla):
    a, b = incrustar(nombre, xtr), incrustar(nombre, xte)
    sc = StandardScaler().fit(a)
    m = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=semilla).fit(sc.transform(a), ytr)
    return m.predict_proba(sc.transform(b))[:, 1]


def ajuste_fino(nombre, xtr, ytr, xte, semilla, epocas=3, lr=2e-5, lote=16):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    if nombre == "robertuito":  # su preentrenamiento usa @usuario (ponytail: sin demojize de pysentimiento)
        xtr = [re.sub(r"@\w+", "@usuario", t) for t in xtr]
        xte = [re.sub(r"@\w+", "@usuario", t) for t in xte]
    tok = AutoTokenizer.from_pretrained(MODELOS_HF[nombre])
    mod = AutoModelForSequenceClassification.from_pretrained(MODELOS_HF[nombre], num_labels=2).cuda()
    enc = tok(list(xtr), padding=True, truncation=True, max_length=128, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(np.asarray(ytr), dtype=torch.long))
    g = torch.Generator().manual_seed(semilla)
    dl = DataLoader(ds, batch_size=lote, shuffle=True, generator=g)
    pos = float(np.mean(ytr))
    pesos = torch.tensor([1.0 / (2 * (1 - pos)), 1.0 / (2 * pos)], dtype=torch.float).cuda()  # clases balanceadas
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=0.01)
    pasos = epocas * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * pasos), pasos)
    lossf = torch.nn.CrossEntropyLoss(weight=pesos)
    mod.train()
    for _ in range(epocas):
        for ids, am, y in dl:
            out = mod(input_ids=ids.cuda(), attention_mask=am.cuda())
            loss = lossf(out.logits, y.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
    mod.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(xte), 64):
            e = tok(list(xte[i:i + 64]), padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            probs.append(torch.softmax(mod(**e).logits, -1)[:, 1].cpu().numpy())
    del mod
    torch.cuda.empty_cache()
    return np.concatenate(probs)


# ---------------------------------------------------------------- evaluación


def evaluar(test, prob, ref):
    msk = test[ref].notna().values
    y = test.loc[msk, ref].astype(int).values
    s = prob[msk]
    p = (s >= 0.5).astype(int)
    w = test.loc[msk, "peso_muestral"].values
    rng = np.random.default_rng(0)
    f1s = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        f1s.append(f1_score(y[i], p[i], zero_division=0))
    return {
        "n": int(msk.sum()), "positivos": int(y.sum()),
        "precision": round(float(precision_score(y, p, zero_division=0)), 3),
        "recall": round(float(recall_score(y, p, zero_division=0)), 3),
        "f1": round(float(f1_score(y, p, zero_division=0)), 3),
        "f1_ic95": [round(float(np.percentile(f1s, 2.5)), 3), round(float(np.percentile(f1s, 97.5)), 3)],
        "pr_auc": round(float(average_precision_score(y, s)), 3) if len(set(s)) > 2 else None,
        "mcc": round(float(matthews_corrcoef(y, p)), 3),
        "precision_corpus": round(float((w * p * y).sum() / max((w * p).sum(), 1e-9)), 3),
        "recall_corpus": round(float((w * p * y).sum() / max((w * y).sum(), 1e-9)), 3),
        "positivos_predichos_corpus_pct": round(float(100 * (w * p).sum() / w.sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["tfidf", "fasttext"])
    ap.add_argument("--fuentes", nargs="+", default=["diccionario", "qwen"])
    ap.add_argument("--enmascarar", nargs="+", default=["no", "si"])
    ap.add_argument("--semillas", nargs="+", type=int, default=[42])
    a = ap.parse_args()
    test, pool = cargar()
    refs = [r for r in ["ref_mayoria", "ref_claude_gemma", "ref_claude"] if test[r].notna().sum() > 0]
    print("referencias disponibles:", {r: int(test[r].notna().sum()) for r in refs})
    salida = open(SAL / "metricas.jsonl", "a", encoding="utf-8")
    filas = []

    def guardar(r):  # cada fila al disco apenas sale, para no perder nada si algo falla después
        filas.append(r)
        salida.write(json.dumps(r, ensure_ascii=False) + "\n")
        salida.flush()

    # la regla del diccionario y los anotadores LLM como clasificadores directos
    for nombre, prob in [("regla_diccionario", test.etiqueta_diccionario.astype(float).values),
                         ("qwen_directo", test.qwen.map({"1": 1.0, "0": 0.0}).fillna(0).values),
                         ("gemma_directo", test.gemma.map({"1": 1.0, "0": 0.0}).fillna(0).values)]:
        if nombre == "gemma_directo" and test.gemma.isna().all():
            continue
        for ref in refs:
            guardar({"modelo": nombre, "fuente": "-", "enmascarado": "no", "semilla": 0, "ref": ref, **evaluar(test, prob, ref)})
    for mod in a.modelos:
        for fuente in a.fuentes:
            tr = pool[pool[f"y_{fuente}"].notna()]
            ytr = tr[f"y_{fuente}"].astype(int).values
            for enm in a.enmascarar:
                xtr = tr.texto.map(enmascarar).values if enm == "si" else tr.texto.values
                xte = test.texto.map(enmascarar).values if enm == "si" else test.texto.values
                semillas = a.semillas if mod.endswith("_ft") else a.semillas[:1]
                for s in semillas:
                    t0 = time.time()
                    if mod == "tfidf":
                        prob = tfidf_lr(xtr, ytr, xte, s)
                    elif mod == "fasttext":
                        prob = fasttext_svm(xtr, ytr, xte, s)
                    elif mod == "beto_congelado":
                        prob = congelado_lr("beto", xtr, ytr, xte, s)
                    elif mod in ("beto_ft", "robertuito_ft"):
                        prob = ajuste_fino(mod.split("_")[0], xtr, ytr, xte, s)
                    else:
                        raise ValueError(mod)
                    seg = round(time.time() - t0, 1)
                    pd.DataFrame({"id_anotacion": test.id_anotacion, "prob": prob}).to_csv(
                        SAL / f"pred_{mod}_{fuente}_{enm}_{s}.csv", index=False)
                    for ref in refs:
                        r = {"modelo": mod, "fuente": fuente, "enmascarado": enm, "semilla": s, "ref": ref,
                             "n_train": int(len(ytr)), "pos_train": int(ytr.sum()), "segundos": seg, **evaluar(test, prob, ref)}
                        guardar(r)
                    print(f"{mod} | {fuente} | enmascarado={enm} | semilla {s} | {seg}s | "
                          f"F1 mayoría={filas[-len(refs)]['f1']}", flush=True)
    salida.close()
    print("guardado en", salida.name)


if __name__ == "__main__":
    main()
