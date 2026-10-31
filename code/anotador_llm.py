"""Anotador automático genérico con un modelo local en llama-server (guía 1.1, a ciegas).

Misma instrucción, esquema JSON, temperatura 0 y semilla 42 que anotador_qwen.py.

Uso:
    python anotador_llm.py --modelo NOMBRE --entrada CSV --salida JSONL [--puerto 8089]
El CSV de entrada necesita las columnas id_anotacion y texto. Retoma donde quedó.
"""
import argparse
import json
import os
import time
import urllib.request

import pandas as pd

GUIA = open("guia-anotacion-v1.1.md", encoding="utf-8").read()
SISTEMA = (
    "Eres un anotador de un estudio sobre estrés en tuits en español. Sigue al pie de la letra esta guía:\n\n"
    + GUIA
    + "\n\nResponde solo con un objeto JSON con estos campos, en este orden:\n"
    "- razon: una frase corta (máximo 25 palabras) que aplique la prueba de tres pasos.\n"
    '- p1: "1", "0" o "NS".\n'
    '- p2: "1" o "0".\n'
    '- p3: "A", "D" o "" (vacío casi siempre).'
)
ESQUEMA = {
    "type": "object",
    "properties": {
        "razon": {"type": "string"},
        "p1": {"type": "string", "enum": ["1", "0", "NS"]},
        "p2": {"type": "string", "enum": ["1", "0"]},
        "p3": {"type": "string", "enum": ["A", "D", ""]},
    },
    "required": ["razon", "p1", "p2", "p3"],
}


def anotar(url, modelo, texto):
    body = {
        "model": modelo,
        "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": f"Tuit:\n{texto}"}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "anotacion", "schema": ESQUEMA}},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
    return json.loads(out["choices"][0]["message"]["content"]), out.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--puerto", default="8089")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.puerto}/v1/chat/completions"
    df = pd.read_csv(a.entrada, encoding="utf-8-sig")
    hechos = set()
    if os.path.exists(a.salida):
        hechos = {json.loads(l)["id_anotacion"] for l in open(a.salida, encoding="utf-8") if l.strip()}
    pend = df[~df.id_anotacion.isin(hechos)]
    t0 = time.time()
    with open(a.salida, "a", encoding="utf-8") as f:
        for i, r in enumerate(pend.itertuples(), 1):
            try:
                res, uso = anotar(url, a.modelo, r.texto)
            except Exception as e:  # una respuesta rota no detiene el lote; se reintenta al relanzar
                print(f"error en {r.id_anotacion}: {e}", flush=True)
                continue
            f.write(json.dumps({"id_anotacion": r.id_anotacion, **res, "uso": uso}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(pend):
                print(f"{a.modelo}: {i}/{len(pend)} en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
