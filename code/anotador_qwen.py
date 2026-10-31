"""Segundo anotador automático: Qwen3.5-9B (Q4_K_M) en llama.cpp, con la guía 1.1.

Requiere llama-server escuchando en 127.0.0.1:8089 (ver plan-q1.md).
Lee solo el texto de cada tuit (a ciegas: no ve la etiqueta del diccionario ni la de Claude).
Guarda cada respuesta en anotador_qwen.jsonl y retoma donde quedó.

Uso: python anotador_qwen.py [n_maximo]
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd

URL = "http://127.0.0.1:8089/v1/chat/completions"
SALIDA = "anotador_qwen.jsonl"
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


def anotar(texto):
    body = {
        "model": "Qwen3.5-9B-Q4_K_M",
        "messages": [{"role": "system", "content": SISTEMA},
                     {"role": "user", "content": f"Tuit:\n{texto}"}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "anotacion", "schema": ESQUEMA}},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return json.loads(out["choices"][0]["message"]["content"]), out.get("usage", {})


def main():
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
    h = pd.read_csv("hoja_anotador_A.csv", encoding="utf-8-sig")
    hechos = set()
    if os.path.exists(SALIDA):
        hechos = {json.loads(l)["id_anotacion"] for l in open(SALIDA, encoding="utf-8") if l.strip()}
    pendientes = h[~h.id_anotacion.isin(hechos)]
    if limite:
        pendientes = pendientes.head(limite)
    t0 = time.time()
    with open(SALIDA, "a", encoding="utf-8") as f:
        for i, r in enumerate(pendientes.itertuples(), 1):
            res, uso = anotar(r.texto)
            f.write(json.dumps({"id_anotacion": r.id_anotacion, **res, "uso": uso}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(pendientes):
                print(f"{i}/{len(pendientes)} en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
