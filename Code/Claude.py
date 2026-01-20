#!/usr/bin/env python3

import os
import ast
import json
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from tqdm import tqdm
from dotenv import load_dotenv

import boto3
from botocore.config import Config
import logging

# ---------------------------
# Basic logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------
# CONFIG 
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

BASE_MODEL = "google/flan-t5-small"
MODEL_DIR = os.path.join(BASE_DIR, "story-flan-t5-lora", "epoch_3")  # 
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")

OUTPUT_PRED_CSV = os.path.join(BASE_DIR, "val_predictions_consistent.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "evaluation_results_consistent.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_INPUT_LENGTH = 128

GEN_PARAMS = dict(
    max_new_tokens=150,
    num_beams=3,
    temperature=0.7,
    repetition_penalty=1.2,
    early_stopping=True
)

# ---------------------------
# Load dotenv (for AWS keys)
# ---------------------------
load_dotenv()  #

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# ---------------------------
# Utilities
# ---------------------------
def normalize_whitespace(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def extract_keywords(words_raw):
    if isinstance(words_raw, str) and words_raw.startswith("["):
        try:
            return [w.lower().strip() for w in ast.literal_eval(words_raw)]
        except Exception:
            return []
    return []

def keyword_metric(keywords: List[str], generated: str, reference: str):
    gen = generated.lower()
    ref = reference.lower()
    found_gen = sum(1 for w in keywords if w in gen)
    found_ref = sum(1 for w in keywords if w in ref)
    precision = found_gen / len(keywords) if keywords else 0
    recall = found_gen / found_ref if found_ref > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, f1

# ---------------------------
# Dataset + Collate
# ---------------------------
class InferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_input_length=128):
        df = pd.read_csv(csv_path).dropna(subset=["story_beginning_prompt", "story"]).reset_index(drop=True)
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        words_raw = row.get("words", "")
        keywords_list = extract_keywords(words_raw)
        keyword_str = ", ".join(keywords_list) if keywords_list else "happy, friend, sun"

        prompt = (
            f"You are a kind children's story writer.\n"
            f"Use these words: {keyword_str}\n"
            f"Start with: {row['story_beginning_prompt']}\n"
            f"Write a complete short story:"
        )

        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
            return_tensors=None
        )
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "original_prompt": prompt,
            "ground_truth": row["story"],
            "words_raw": words_raw
        }

def collate_batch(samples):
    ids_list = [s["input_ids"] for s in samples]
    padded = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (padded != tokenizer.pad_token_id).long()
    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "prompts": [s["original_prompt"] for s in samples],
        "gts": [s["ground_truth"] for s in samples],
        "words_raw": [s["words_raw"] for s in samples]
    }

# ---------------------------
# Load tokenizer + model + LoRA
# ---------------------------
logging.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

logging.info("Loading base pretrained model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=device_dtype,
    device_map="auto" if DEVICE == "cuda" else None
)

if not os.path.exists(MODEL_DIR):
    logging.error("MODEL_DIR does not exist: %s", MODEL_DIR)
    raise FileNotFoundError(MODEL_DIR)

logging.info("Applying LoRA adapter from: %s", MODEL_DIR)

model = PeftModel.from_pretrained(base_model, MODEL_DIR, adapter_name="default")
model.to(DEVICE)
model.eval()
logging.info("Model loaded and set to eval.")

# ---------------------------
# Claude via AWS Bedrock
# ---------------------------
def call_claude_refine(text: str, max_tokens: int = 512) -> str:
    """
    Send text to Claude via AWS Bedrock to 'refine' it.
    Requires AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY env vars or .env.
    If Bedrock fails, returns original text.
    """
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        logging.warning("AWS credentials not found in environment — skipping Claude refinement.")
        return text

    try:
        session_kwargs = {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "region_name": AWS_REGION
        }
        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

        session = boto3.Session(**session_kwargs)
        bedrock = session.client("bedrock-runtime", config=Config(region_name=AWS_REGION))

        # Prompt/format for Claude
        prompt = (
            "You are an assistant that polishes and refines short children's stories. "
            "Preserve the core events, characters and keywords. Improve clarity, flow, "
            "readability for ages 4-8, and fix grammar. Do not hallucinate new facts.\n\n"
            f"Story:\n{text}\n\nRefined story:"
        )

       
        body = {
            "input": prompt,
            "max_tokens_to_sample": max_tokens,
            "temperature": 0.7
        }

       
        model_id = "anthropic.claude-3"  

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body).encode("utf-8"),
        )

        raw = response.get("body")
        
        if raw is None:
            logging.error("No 'body' in Bedrock response; returning original.")
            return text

        try:
            payload = raw.read().decode("utf-8")
            parsed = json.loads(payload)
           
            if isinstance(parsed, dict):
                # heuristic extraction
                if "output" in parsed and isinstance(parsed["output"], str):
                    refined = parsed["output"]
                elif "generations" in parsed:
                    gens = parsed["generations"]
                    if isinstance(gens, list) and len(gens) > 0:
                        
                        first = gens[0]
                        if isinstance(first, dict) and "text" in first:
                            refined = first["text"]
                        else:
                            refined = str(first)
                    else:
                        refined = text
                elif "choices" in parsed and isinstance(parsed["choices"], list) and len(parsed["choices"])>0:
           
                    refined = parsed["choices"][0].get("text", text)
                else:
                  
                    refined = json.dumps(parsed)
            else:
                refined = str(parsed)
        except Exception as e:
            logging.exception("Failed to parse Bedrock response body: %s", e)
            return text

       
        refined = normalize_whitespace(refined)
        return refined

    except Exception as e:
        logging.exception("Claude Bedrock invocation failed: %s", e)
        return text

# ---------------------------
# Evaluation functions 
# ---------------------------

def load_predictions(csv_path: str, ref_col: str = "story", hyp_col: str = "generated_story") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if ref_col not in df.columns or hyp_col not in df.columns:
        raise ValueError(f"Input CSV must contain '{ref_col}' and '{hyp_col}' columns.")
    df[ref_col] = df[ref_col].astype(str).apply(normalize_whitespace)
    df[hyp_col] = df[hyp_col].astype(str).apply(normalize_whitespace)
    return df

def compute_bleu_scores(df: pd.DataFrame, ref_col: str = "story", hyp_col: str = "generated_story", max_n: int = 4, add_column: bool = True) -> Tuple[float, List[float]]:
    smoothie = SmoothingFunction().method1
    refs_raw = df[ref_col].astype(str).tolist()
    hyps_raw = df[hyp_col].astype(str).tolist()
    references = [[r.split()] for r in refs_raw]
    hypotheses = [h.split() for h in hyps_raw]
    weights = tuple(1.0 / max_n for _ in range(max_n))
    corpus_bleu_score = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoothie)
    sentence_bleu_scores: List[float] = []
    for ref_tokens, hyp_tokens in zip(references, hypotheses):
        s_bleu = sentence_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=smoothie)
        sentence_bleu_scores.append(float(s_bleu))
    if add_column:
        df["bleu"] = sentence_bleu_scores
    return float(corpus_bleu_score), sentence_bleu_scores

def compute_perplexity(texts: List[str], model_name: str = "gpt2", max_length: int = 512, batch_size: int = 4, device: Optional[str] = None) -> Tuple[List[float], float]:
    if len(texts) == 0:
        return [], float("nan")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [normalize_whitespace(t) for t in texts]
    tokenizer_ppl = AutoTokenizer.from_pretrained(model_name)
    if tokenizer_ppl.pad_token_id is None and getattr(tokenizer_ppl, "eos_token_id", None) is not None:
        tokenizer_ppl.pad_token = tokenizer_ppl.eos_token
    model_ppl = AutoModelForCausalLM.from_pretrained(model_name)
    model_ppl.to(device)
    model_ppl.eval()
    per_text_ppl: List[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encodings = tokenizer_ppl(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            outputs = model_ppl(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            ppl = torch.exp(loss).detach().cpu().item()
            per_text_ppl.extend([float(ppl)] * len(batch))
    mean_ppl = float(np.mean(per_text_ppl)) if per_text_ppl else float("nan")
    return per_text_ppl, mean_ppl

def compute_bert_scores(df: pd.DataFrame, ref_col: str = "story", hyp_col: str = "generated_story", model_type: str = "bert-base-uncased", lang: str = "en", add_columns: bool = True) -> Tuple[float, float, float]:
    refs = df[ref_col].astype(str).tolist()
    hyps = df[hyp_col].astype(str).tolist()
    P, R, F1 = bert_score(cands=hyps, refs=refs, lang=lang, model_type=model_type, verbose=False)
    if add_columns:
        df["bert_precision"] = P.numpy()
        df["bert_recall"] = R.numpy()
        df["bert_f1"] = F1.numpy()
    return float(P.mean()), float(R.mean()), float(F1.mean())

def compute_rouge_scores(df: pd.DataFrame, ref_col: str = "story", hyp_col: str = "generated_story", add_columns: bool = True) -> Tuple[float, float, float]:
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
    return float(np.mean(r1_list)), float(np.mean(r2_list)), float(np.mean(rl_list))

def compute_keyword_metrics(df: pd.DataFrame, hyp_col: str = "generated_story", keywords: Optional[List[str]] = None, add_columns: bool = True, case_sensitive: bool = False) -> Tuple[float, float]:
    if not keywords:
        return float("nan"), float("nan")
    keywords_norm = [kw.lower() for kw in keywords] if not case_sensitive else keywords
    strict_matches: List[int] = []
    coverage_pct: List[float] = []
    for text in df[hyp_col].astype(str):
        t = text if case_sensitive else text.lower()
        present_count = sum(1 for kw in keywords_norm if kw in t)
        total_kw = len(keywords_norm)
        strict = 1 if present_count == total_kw else 0
        pct = (present_count / total_kw) * 100.0 if total_kw > 0 else 0.0
        strict_matches.append(strict)
        coverage_pct.append(pct)
    keyword_strict_accuracy = float(np.mean(strict_matches))
    keyword_avg_percentage = float(np.mean(coverage_pct))
    if add_columns:
        df["keyword_strict_match"] = strict_matches
        df["keyword_coverage_pct"] = coverage_pct
    return keyword_strict_accuracy, keyword_avg_percentage



def evaluate_text_generation(df: pd.DataFrame, ref_col: str = "story", hyp_col: str = "generated_story", bleu_max_n: int = 4, ppl_model_name: str = "gpt2", ppl_max_length: int = 512, ppl_batch_size: int = 4, bert_model_type: str = "bert-base-uncased", bert_lang: str = "en", id_col: str = "id", keywords: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    corpus_bleu_score, _ = compute_bleu_scores(df, ref_col=ref_col, hyp_col=hyp_col, max_n=bleu_max_n, add_column=True)
    per_text_ppl, mean_ppl = compute_perplexity(df[hyp_col].astype(str).tolist(), model_name=ppl_model_name, max_length=ppl_max_length, batch_size=ppl_batch_size)
    df["perplexity"] = per_text_ppl
    P_mean, R_mean, F1_mean = compute_bert_scores(df, ref_col=ref_col, hyp_col=hyp_col, model_type=bert_model_type, lang=bert_lang, add_columns=True)
    rouge1_mean, rouge2_mean, rougel_mean = compute_rouge_scores(df, ref_col=ref_col, hyp_col=hyp_col, add_columns=True)

    keyword_strict_acc, keyword_avg_pct = compute_keyword_metrics(df, hyp_col=hyp_col, keywords=keywords, add_columns=True) if keywords else (float("nan"), float("nan"))
    # keyword_strict_acc, keyword_avg_pct = compute_keyword_metrics(df, hyp_col="generated_story", ref_col="story",words_col="words")

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

# ---------------------------
# Run inference, optionally refine with Claude
# ---------------------------
def run_pipeline(use_claude: bool = False, keywords_for_eval: Optional[List[str]] = None, ppl_model: str = "gpt2"):
    ds = InferenceDataset(VAL_CSV, tokenizer, MAX_INPUT_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    generated, references, keywords_col = [], [], []

    logging.info("Running inference on %d samples", len(ds))
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **GEN_PARAMS)

        texts = tokenizer.batch_decode(out, skip_special_tokens=True)

        for text, gt, kw_raw in zip(texts, batch["gts"], batch["words_raw"]):
            refined_text = text
            if use_claude:
                refined_text = call_claude_refine(text)
            generated.append(refined_text)
            references.append(gt)
            keywords_col.append(kw_raw)

    # Save predictions
    out_df = pd.DataFrame({
        "generated_story": generated,
        "story": references,
        "words": keywords_col
    })
    out_df.to_csv(OUTPUT_PRED_CSV, index=False)
    logging.info("Saved predictions to %s", OUTPUT_PRED_CSV)

    # Run evaluation
    eval_df, summary = evaluate_text_generation(
        out_df,
        ref_col="story",
        hyp_col="generated_story",
        ppl_model_name=ppl_model,
        keywords=keywords_for_eval
    )

    # Save evaluation results (json + csv with per-example metrics)
    eval_df.to_csv(OUTPUT_PRED_CSV.replace(".csv", "_with_metrics.csv"), index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=4)
    logging.info("Saved evaluation summary to %s", OUTPUT_JSON)

    # Print summary
    logging.info("EVALUATION SUMMARY:")
    for k, v in summary.items():
        logging.info("%s: %s", k, v)

    return summary

# ---------------------------
# CLI-ish quick-run check
# ---------------------------
if __name__ == "__main__":
    
    use_claude_flag = True
   
    keywords_eval_list = None
    if os.getenv("KEYWORDS_EVAL"):
        keywords_eval_list = [k.strip() for k in os.getenv("KEYWORDS_EVAL").split(",") if k.strip()]

    ppl_model_name = os.getenv("PPL_MODEL", "gpt2")

    run_pipeline(use_claude=use_claude_flag, keywords_for_eval=keywords_eval_list, ppl_model=ppl_model_name)
    
