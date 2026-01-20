# evaluation_metric.py

import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import utils

# --------------------------------------------------------------------------------------------------------------------
# Data loading / cleaning
# --------------------------------------------------------------------------------------------------------------------
def load_predictions(
    csv_path: str,
    ref_col: str = "story",
    hyp_col: str = "generated_story"
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    if ref_col not in df.columns or hyp_col not in df.columns:
        raise ValueError(
            f"Input CSV must contain '{ref_col}' and '{hyp_col}' columns."
        )

    df[ref_col] = df[ref_col].astype(str).apply(utils.normalize_whitespace)
    df[hyp_col] = df[hyp_col].astype(str).apply(utils.normalize_whitespace)

    return df

# --------------------------------------------------------------------------------------------------------------------
# BLEU Score
# --------------------------------------------------------------------------------------------------------------------
def compute_bleu_scores(
    df: pd.DataFrame,
    ref_col: str = "story",
    hyp_col: str = "generated_story",
    max_n: int = 4,
    add_column: bool = True
) -> Tuple[float, List[float]]:

    smoothie = SmoothingFunction().method1
    refs_raw = df[ref_col].astype(str).tolist()
    hyps_raw = df[hyp_col].astype(str).tolist()
    references = [[r.split()] for r in refs_raw]
    hypotheses = [h.split() for h in hyps_raw]

    # Corpus BLEU with uniform weights up to max_n
    weights = tuple(1.0 / max_n for _ in range(max_n))
    corpus_bleu_score = corpus_bleu(
        references,
        hypotheses,
        weights=weights,
        smoothing_function=smoothie,
    )

    # Sentence BLEU
    sentence_bleu_scores: List[float] = []
    for ref_tokens, hyp_tokens in zip(references, hypotheses):
        s_bleu = sentence_bleu(
            ref_tokens,
            hyp_tokens,
            weights=weights,
            smoothing_function=smoothie,
        )
        sentence_bleu_scores.append(float(s_bleu))

    if add_column:
        df["bleu"] = sentence_bleu_scores

    return float(corpus_bleu_score), sentence_bleu_scores

# --------------------------------------------------------------------------------------------------------------------
# Perplexity (using a pretrained causal language model)
# --------------------------------------------------------------------------------------------------------------------
def compute_perplexity(
    texts: List[str],
    model_name: str = "gpt2",
    max_length: int = 512,
    batch_size: int = 4,
    device: Optional[str] = None
) -> Tuple[List[float], float]:
    if len(texts) == 0:
        return [], float("nan")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [utils.normalize_whitespace(t) for t in texts]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    per_text_ppl: List[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encodings = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)

            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            ppl = torch.exp(loss).detach().cpu().item()

            # Assign same batch-level perplexity to each text (approximation)
            per_text_ppl.extend([float(ppl)] * len(batch))

    mean_ppl = float(np.mean(per_text_ppl))
    return per_text_ppl, mean_ppl

# --------------------------------------------------------------------------------------------------------------------
# BERTScore
# --------------------------------------------------------------------------------------------------------------------
def compute_bert_scores(
    df: pd.DataFrame,
    ref_col: str = "story",
    hyp_col: str = "generated_story",
    model_type: str = "bert-base-uncased",
    lang: str = "en",
    add_columns: bool = True
) -> Tuple[float, float, float]:
    refs = df[ref_col].astype(str).tolist()
    hyps = df[hyp_col].astype(str).tolist()

    P, R, F1 = bert_score(
        cands=hyps,
        refs=refs,
        lang=lang,
        model_type=model_type,
        verbose=False,
    )

    if add_columns:
        df["bert_precision"] = P.numpy()
        df["bert_recall"] = R.numpy()
        df["bert_f1"] = F1.numpy()

    return float(P.mean()), float(R.mean()), float(F1.mean())

# --------------------------------------------------------------------------------------------------------------------
# ROUGE Score
# --------------------------------------------------------------------------------------------------------------------
def compute_rouge_scores(
    df: pd.DataFrame,
    ref_col: str = "story",
    hyp_col: str = "generated_story",
    add_columns: bool = True
) -> Tuple[float, float, float]:

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_list, r2_list, rl_list = [], [], []

    for ref, hyp in zip(df[ref_col], df[hyp_col]):
        scores = scorer.score(ref, hyp)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)

    if add_columns:
        df["rouge1"] = r1_list
        df["rouge2"] = r2_list
        df["rougeL"] = rl_list

    return (
        float(np.mean(r1_list)),
        float(np.mean(r2_list)),
        float(np.mean(rl_list)),
    )

# --------------------------------------------------------------------------------------------------------------------
# Keyword-based Metrics (per-row keywords from df["words"])
# --------------------------------------------------------------------------------------------------------------------
def compute_keyword_metrics(
    df: pd.DataFrame,
    hyp_col: str = "generated_story",
    words_col: str = "words",
    add_columns: bool = True,
    case_sensitive: bool = False,
) -> Tuple[float, float]:
    generated_texts = df[hyp_col].astype(str).tolist()
    keywords_lists = df[words_col].tolist()

    strict_matches: List[float] = []
    coverage_pct: List[float] = []

    for text, kw_list in zip(generated_texts, keywords_lists):
        # If no keywords for this example -> NaN
        if not isinstance(kw_list, list) or len(kw_list) == 0:
            strict_matches.append(float("nan"))
            coverage_pct.append(float("nan"))
            continue

        if case_sensitive:
            t = text
            kws = [str(kw) for kw in kw_list]
        else:
            t = text.lower()
            kws = [str(kw).lower() for kw in kw_list]

        present_count = 0
        for kw in kws:
            if kw in t:
                present_count += 1

        total_kw = len(kws)
        strict = 1.0 if present_count == total_kw else 0.0
        pct = (present_count / total_kw) * 100.0

        strict_matches.append(strict)
        coverage_pct.append(pct)

    strict_arr = np.array(strict_matches, dtype=float)
    pct_arr = np.array(coverage_pct, dtype=float)

    if np.all(np.isnan(strict_arr)):
        keyword_strict_accuracy = float("nan")
    else:
        keyword_strict_accuracy = float(np.nanmean(strict_arr))

    if np.all(np.isnan(pct_arr)):
        keyword_avg_percentage = float("nan")
    else:
        keyword_avg_percentage = float(np.nanmean(pct_arr))

    if add_columns:
        df["keyword_strict_match"] = strict_matches
        df["keyword_coverage_pct"] = coverage_pct

    return keyword_strict_accuracy, keyword_avg_percentage

# --------------------------------------------------------------------------------------------------------------------
# High-level helper to run all metrics
# --------------------------------------------------------------------------------------------------------------------
def evaluate_text_generation(
    df: pd.DataFrame,
    ref_col: str = "story",
    hyp_col: str = "generated_story",
    bleu_max_n: int = 4,
    ppl_model_name: str = "gpt2",
    ppl_max_length: int = 512,
    ppl_batch_size: int = 4,
    bert_model_type: str = "bert-base-uncased",
    bert_lang: str = "en",
    id_col: str = "id",
    words_col: str = "words",
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    # BLEU
    corpus_bleu_score, _ = compute_bleu_scores(
        df,
        ref_col=ref_col,
        hyp_col=hyp_col,
        max_n=bleu_max_n,
        add_column=True,
    )

    # Perplexity
    per_text_ppl, mean_ppl = compute_perplexity(
        df[hyp_col].astype(str).tolist(),
        model_name=ppl_model_name,
        max_length=ppl_max_length,
        batch_size=ppl_batch_size,
    )
    df["perplexity"] = per_text_ppl

    # BERTScore
    P_mean, R_mean, F1_mean = compute_bert_scores(
        df,
        ref_col=ref_col,
        hyp_col=hyp_col,
        model_type=bert_model_type,
        lang=bert_lang,
        add_columns=True,
    )

    # ROUGE
    rouge1_mean, rouge2_mean, rougel_mean = compute_rouge_scores(
        df,
        ref_col=ref_col,
        hyp_col=hyp_col,
        add_columns=True,
    )

    # Keyword metrics (per-row keywords in df[words_col])
    keyword_strict_acc, keyword_avg_pct = compute_keyword_metrics(
        df,
        hyp_col=hyp_col,
        words_col=words_col,
        add_columns=True,
    )

    summary: Dict[str, float] = {
        "corpus_bleu": corpus_bleu_score,
        "avg_perplexity": mean_ppl,
        "bert_precision": P_mean,
        "bert_recall": R_mean,
        "bert_f1": F1_mean,
        "rouge1": rouge1_mean,
        "rouge2": rouge2_mean,
        "rougeL": rougel_mean,
        "keyword_strict_accuracy": keyword_strict_acc,
        "keyword_avg_percentage": keyword_avg_pct,
    }

    return df, summary
