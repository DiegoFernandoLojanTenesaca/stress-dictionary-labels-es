# What models learn from dictionary labels: data and code

Data and code for the paper "What models learn from dictionary labels: stress in Spanish tweets during Ecuador's electricity crisis" by Diego F. Lojan Tenesaca (ORCID 0009-0003-3882-3889), submitted to *Natural Language Processing*.

The corpus has 3,620 Spanish tweets collected with searches about Ecuador during the 2024 electricity crisis. A 68-term dictionary labels stress; three language models (Claude, Qwen3.5-9B and Gemma 4 12B) annotated a stratified test sample of 1,130 tweets with a written guideline, and Qwen also annotated the 2,490 training tweets.

## Contents

| Path | What it is |
|---|---|
| `data/test_sample.csv` | The 1,130 test tweets, with sampling information, the dictionary label and the labels of the three annotators |
| `data/training_set.csv` | The 2,490 training tweets, with the dictionary label and Qwen's labels |
| `guideline/` | Annotation guideline, in Spanish: version 1.0 (used by Claude on the 100 pilot tweets) and version 1.1 (all other annotations) |
| `code/` | Scripts for the local annotation with llama-server, agreement, experiments and summaries (Python; dependencies in `pyproject.toml` and `uv.lock`) |
| `results/metrics.jsonl` | Every run reported in the paper: model, label source, deletion on or off, seed, reference and metrics |
| `results/dictionary_analysis.txt` | Precision of the dictionary by term and by stratum |

## Columns

Labels use `1` for yes, `0` for no and `NS` for "cannot decide". Question 1 (`*_q1`) is whether the author expresses their own stress, question 2 (`*_q2`) whether the tweet is about an electricity crisis, and question 3 (`*_q3`) whether anxiety (`A`) or sadness (`D`) shows instead of stress (empty when neither).

`test_sample.csv`: `sample_id`, `tweet_id`, `date` (taken from the tweet ID), `stratum` (dictionary label crossed with mention of the crisis; `pos_solo_genericos` are positives triggered only by general words), `sample_weight` (stratum size divided by tweets sampled from it), `pilot` (1 for the 100 pilot tweets), `dictionary_label`, `dictionary_terms` (dictionary terms found in the tweet), `crisis_keyword`, `claude_q1` to `claude_q3`, `claude_guideline` (1.0 or 1.1), `claude_comment`, `qwen_q1` to `qwen_q3`, `qwen_reason`, `gemma_q1` to `gemma_q3`, `gemma_reason`.

`training_set.csv`: `sample_id`, `tweet_id`, `date`, `dictionary_label`, `dictionary_terms`, `crisis_keyword`, `qwen_q1` to `qwen_q3`, `qwen_reason`.

The references in the paper are built from these columns. R1 is the set of tweets where `claude_q1` and `gemma_q1` agree, R2 is the majority of the three annotators, and R3 is Claude alone; `NS` labels are left out.

## What is not included

- **Tweet text:** following X's terms, only tweet IDs are shared. The texts can be retrieved from the IDs. The scripts in `code/` read the original files with text and use the author's folder layout, so they need those files to run.
- **Quotes inside Claude's comments:** twenty comments quoted words from the tweet; those quotes were replaced with `[quote removed]`.

## Known issues

- One training tweet (`E2490`) has no recoverable tweet ID.
- Four tweets appear twice in the corpus with slightly different text, for example with a user mention added. One of them (ID `1776683072613765332`) is in both the test sample and the training set.

## License

Code: MIT (see `LICENSE`). Data and guideline: Creative Commons Attribution 4.0 (CC BY 4.0).

## Related work

- Lojan Tenesaca, D. F. (2025). *Ajuste de hiperparámetros en SVM con Optuna (TPE) para detectar estrés durante la crisis energética en Ecuador, 2024*. Undergraduate thesis, Universidad Nacional de Loja. https://dspace.unl.edu.ec/handle/123456789/35029