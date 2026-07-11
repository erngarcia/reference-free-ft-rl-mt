import argparse
import os
import csv
from datetime import datetime

import yaml
import torch
from torch.optim import AdamW
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sacrebleu.metrics import CHRF, BLEU
from comet import download_model, load_from_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

chrf_metric = CHRF()
bleu_metric = BLEU(effective_order=True)


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config_and_args(config, args):
    """
    CLI args take priority over config values.
    Only override config if CLI arg was explicitly set.
    """
    optional_overrides = [
        "languages", "model_name", "lr", "n_bootstrap",
        "train_data_dir", "eval_data_dir", "vanilla_dir", "grpo_dir",
        "output_dir", "tag",
    ]
    for field in optional_overrides:
        cli_val = getattr(args, field, None)
        if cli_val is not None:
            config[field] = cli_val

    return config


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SFT ablation runner (comparison baseline for GRPO)")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. sft_ablation.yaml)")

    # All args are optional — they override config values if provided
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=None)
    parser.add_argument("--train_data_dir", default=None)
    parser.add_argument("--eval_data_dir", default=None)
    parser.add_argument("--vanilla_dir", default=None)
    parser.add_argument("--grpo_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


# ============================================================
# LOGGER
# ============================================================

class TrainingLogger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        write_header = (
            not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        )
        self._file   = open(filepath, "a", newline="")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(["lang", "step", "loss"])

    def log(self, lang, step, loss):
        self._writer.writerow([lang, step, round(loss, 6)])
        self._file.flush()

    def close(self):
        self._file.close()


# ============================================================
# MODEL LOADING
# ============================================================

def make_bnb_config(cfg):
    return BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
    )


def make_lora_config(cfg):
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        task_type="SEQ_2_SEQ_LM",
    )


def load_model(cfg):
    bnb  = make_bnb_config(cfg)
    lora = make_lora_config(cfg)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
    )
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=False
    )
    model = get_peft_model(base_model, lora)
    model.enable_input_require_grads()
    return model


# ============================================================
# SFT TRAIN STEP
# ============================================================

def sft_train_step(model, optimizer, tokenizer, src_text, tgt_text, tgt_lang_key, cfg):
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = tgt_lang_key

    inputs = tokenizer(
        src_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg["max_length"],
    ).to(DEVICE)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["max_length"],
        ).input_ids.to(DEVICE)

    labels[labels == tokenizer.pad_token_id] = -100

    model.train()
    optimizer.zero_grad()

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    loss_val = loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    del inputs, labels, outputs, loss
    torch.cuda.empty_cache()

    return loss_val


# ============================================================
# TRANSLATION
# ============================================================

def translate_corpus(model, tokenizer, df, lang, cfg):
    model.eval()
    tokenizer.src_lang = "eng_Latn"
    tgt_token_id = tokenizer.convert_tokens_to_ids(lang)
    translations = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Translating"):
        inputs = tokenizer(row["src"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_new_tokens=cfg["max_new_tokens_eval"],
                num_beams=cfg["num_beams"],
                length_penalty=1.0,
            )
        translations.append({
            "src": row["src"],
            "mt": tokenizer.decode(gen_ids[0], skip_special_tokens=True),
            "ref": row["ref"],
        })
        del inputs, gen_ids

    torch.cuda.empty_cache()
    return translations


# ============================================================
# SCORING
# ============================================================

def score_translations(translations, comet_model, cfg):
    hypotheses = [t["mt"] for t in translations]
    references = [t["ref"] for t in translations]

    chrf_score = chrf_metric.corpus_score(hypotheses, [references]).score
    bleu_score = bleu_metric.corpus_score(hypotheses, [references]).score

    comet_input = [{"src": t["src"], "mt": t["mt"]} for t in translations]
    with torch.no_grad():
        comet_output = comet_model.predict(
            comet_input,
            batch_size=cfg["comet_batch_size"],
            gpus=1 if DEVICE == "cuda" else 0,
        )
    comet_scores = comet_output.scores
    del comet_output
    torch.cuda.empty_cache()

    for i, score in enumerate(comet_scores):
        translations[i]["comet_score"] = score

    return translations, float(np.mean(comet_scores)), chrf_score, bleu_score


# ============================================================
# BOOTSTRAP SIGNIFICANCE
# ============================================================

def chrf_fn(hyps, refs):
    return chrf_metric.corpus_score(hyps, [refs]).score


def bleu_fn(hyps, refs):
    return bleu_metric.corpus_score(hyps, [refs]).score


def paired_bootstrap_delta(hyps_a, hyps_b, references, metric_fn, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(references)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta = (
            metric_fn([hyps_b[i] for i in idx], [references[i] for i in idx]) -
            metric_fn([hyps_a[i] for i in idx], [references[i] for i in idx])
        )
        deltas.append(delta)
    p_value = np.mean(np.array(deltas) <= 0)
    observed = metric_fn(hyps_b, references) - metric_fn(hyps_a, references)
    return observed, p_value


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "~"
    return "ns"


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # Load config and merge CLI overrides
    cfg = load_config(args.config)
    cfg = merge_config_and_args(cfg, args)

    # Set derived defaults
    if cfg.get("tag") is None:
        cfg["tag"] = f"sft_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if cfg.get("output_dir") is None:
        cfg["output_dir"] = os.path.join("results", cfg["tag"])

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Save resolved config alongside results for reproducibility
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"  Config:     {args.config}")
    print(f"  Tag:        {cfg['tag']}")
    print(f"  Model:      {cfg['model_name']}")
    print(f"  Languages:  {cfg['languages']}")
    print(f"  Train dir:  {cfg['train_data_dir']}")
    print(f"  Eval dir:   {cfg['eval_data_dir']}")
    print(f"{'='*60}\n")

    logger    = TrainingLogger(os.path.join(out_dir, "training_metrics.csv"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    print("Loading COMET...")
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
    comet_model.eval()

    # Load vanilla/GRPO translations once per language for 3-way comparison
    summary = []

    try:
        for lang in cfg["languages"]:
            print(f"\n{'='*20} {lang} {'='*20}")

            train_path   = os.path.join(cfg["train_data_dir"], lang, "dev.parquet")
            eval_path    = os.path.join(cfg["eval_data_dir"], lang, "devtest.parquet")
            grpo_path    = os.path.join(cfg["grpo_dir"], f"results_{lang}_grpo.csv")
            vanilla_path = os.path.join(cfg["vanilla_dir"], f"results_{lang}_vanilla.csv")

            if not os.path.exists(train_path):
                print(f"  [SKIP] No training data for {lang}")
                continue

            df_train = pd.read_parquet(train_path)
            df_eval  = pd.read_parquet(eval_path)

            # --- TRAIN SFT MODEL ---
            print(f"  Training SFT model ({len(df_train)} steps)...")
            model     = load_model(cfg)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["lr"],
            )

            pbar = tqdm(df_train.iterrows(), total=len(df_train), desc=f"  SFT {lang}")
            for i, (_, row) in enumerate(pbar):
                loss_val = sft_train_step(
                    model, optimizer, tokenizer, row["src"], row["ref"], lang, cfg
                )
                logger.log(lang, i, loss_val)
                if i % 10 == 0:
                    pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            # Save SFT adapter
            adapter_path = os.path.join(out_dir, "adapters", lang)
            os.makedirs(adapter_path, exist_ok=True)
            model.save_pretrained(adapter_path)
            print(f"  Saved SFT adapter → {adapter_path}")

            # --- EVALUATE SFT ---
            print("  Evaluating SFT...")
            sft_translations = translate_corpus(model, tokenizer, df_eval, lang, cfg)
            sft_translations, sft_comet, sft_chrf, sft_bleu = score_translations(
                sft_translations, comet_model, cfg
            )
            res_dir = os.path.join(out_dir, "results")
            os.makedirs(res_dir, exist_ok=True)
            pd.DataFrame(sft_translations).to_csv(
                os.path.join(res_dir, f"{lang}_sft.csv"), index=False
            )

            del model
            optimizer.state.clear()
            torch.cuda.empty_cache()

            # --- LOAD VANILLA AND GRPO RESULTS ---
            vanilla_df = pd.read_csv(vanilla_path) if os.path.exists(vanilla_path) else None
            grpo_df    = pd.read_csv(grpo_path)    if os.path.exists(grpo_path)    else None

            # --- SIGNIFICANCE: SFT vs VANILLA ---
            sft_hyps = [t["mt"] for t in sft_translations]
            refs     = [t["ref"] for t in sft_translations]

            sft_vs_van_chrf_delta, sft_vs_van_chrf_p = "n/a", "n/a"
            sft_vs_van_bleu_delta, sft_vs_van_bleu_p = "n/a", "n/a"
            sft_vs_grpo_chrf_delta, sft_vs_grpo_chrf_p = "n/a", "n/a"
            sft_vs_grpo_bleu_delta, sft_vs_grpo_bleu_p = "n/a", "n/a"
            van_chrf, van_bleu = None, None
            grpo_chrf, grpo_bleu = None, None

            if vanilla_df is not None:
                merged_van = pd.DataFrame(sft_translations)[["src"]].merge(
                    vanilla_df[["src", "mt", "comet_score"]], on="src"
                )
                van_hyps = merged_van["mt"].tolist()
                van_chrf = chrf_fn(van_hyps, refs)
                van_bleu = bleu_fn(van_hyps, refs)

                sft_vs_van_chrf_delta, sft_vs_van_chrf_p = paired_bootstrap_delta(
                    van_hyps, sft_hyps, refs, chrf_fn, cfg["n_bootstrap"], cfg["bootstrap_seed"]
                )
                sft_vs_van_bleu_delta, sft_vs_van_bleu_p = paired_bootstrap_delta(
                    van_hyps, sft_hyps, refs, bleu_fn, cfg["n_bootstrap"], cfg["bootstrap_seed"]
                )

            # --- SIGNIFICANCE: GRPO vs SFT ---
            if grpo_df is not None:
                merged_grpo = pd.DataFrame(sft_translations)[["src"]].merge(
                    grpo_df[["src", "mt", "comet_score"]], on="src"
                )
                grpo_hyps = merged_grpo["mt"].tolist()
                grpo_chrf = chrf_fn(grpo_hyps, refs)
                grpo_bleu = bleu_fn(grpo_hyps, refs)

                sft_vs_grpo_chrf_delta, sft_vs_grpo_chrf_p = paired_bootstrap_delta(
                    sft_hyps, grpo_hyps, refs, chrf_fn, cfg["n_bootstrap"], cfg["bootstrap_seed"]
                )
                sft_vs_grpo_bleu_delta, sft_vs_grpo_bleu_p = paired_bootstrap_delta(
                    sft_hyps, grpo_hyps, refs, bleu_fn, cfg["n_bootstrap"], cfg["bootstrap_seed"]
                )

            # --- PRINT LANGUAGE SUMMARY ---
            print(f"\n  {'System':<10} | {'chrF':>7} | {'BLEU':>7} | {'COMET':>7}")
            print(f"  {'-'*40}")
            if van_chrf:
                print(f"  {'Vanilla':<10} | {van_chrf:>7.4f} | {van_bleu:>7.4f} | {'see eval':>7}")
            print(f"  {'SFT':<10} | {sft_chrf:>7.4f} | {sft_bleu:>7.4f} | {sft_comet:>7.4f}")
            if grpo_chrf:
                print(f"  {'GRPO':<10} | {grpo_chrf:>7.4f} | {grpo_bleu:>7.4f} | {'see eval':>7}")

            if isinstance(sft_vs_van_chrf_delta, float):
                print(f"\n  SFT vs Vanilla | ΔchrF: {sft_vs_van_chrf_delta:+.4f} {sig_label(sft_vs_van_chrf_p)} | ΔBLEU: {sft_vs_van_bleu_delta:+.4f} {sig_label(sft_vs_van_bleu_p)}")
            if isinstance(sft_vs_grpo_chrf_delta, float):
                print(f"  GRPO vs SFT    | ΔchrF: {sft_vs_grpo_chrf_delta:+.4f} {sig_label(sft_vs_grpo_chrf_p)} | ΔBLEU: {sft_vs_grpo_bleu_delta:+.4f} {sig_label(sft_vs_grpo_bleu_p)}")

            summary.append({
                "lang":                   lang,
                "van_chrf":               round(van_chrf, 4) if van_chrf else "n/a",
                "sft_chrf":               round(sft_chrf, 4),
                "grpo_chrf":              round(grpo_chrf, 4) if grpo_chrf else "n/a",
                "sft_vs_van_chrf_delta":  round(sft_vs_van_chrf_delta, 4) if isinstance(sft_vs_van_chrf_delta, float) else "n/a",
                "sft_vs_van_chrf_p":      round(sft_vs_van_chrf_p, 4) if isinstance(sft_vs_van_chrf_p, float) else "n/a",
                "sft_vs_van_chrf_sig":    sig_label(sft_vs_van_chrf_p) if isinstance(sft_vs_van_chrf_p, float) else "n/a",
                "grpo_vs_sft_chrf_delta": round(sft_vs_grpo_chrf_delta, 4) if isinstance(sft_vs_grpo_chrf_delta, float) else "n/a",
                "grpo_vs_sft_chrf_p":     round(sft_vs_grpo_chrf_p, 4) if isinstance(sft_vs_grpo_chrf_p, float) else "n/a",
                "grpo_vs_sft_chrf_sig":   sig_label(sft_vs_grpo_chrf_p) if isinstance(sft_vs_grpo_chrf_p, float) else "n/a",
                "van_bleu":               round(van_bleu, 4) if van_bleu else "n/a",
                "sft_bleu":               round(sft_bleu, 4),
                "grpo_bleu":              round(grpo_bleu, 4) if grpo_bleu else "n/a",
                "sft_vs_van_bleu_delta":  round(sft_vs_van_bleu_delta, 4) if isinstance(sft_vs_van_bleu_delta, float) else "n/a",
                "sft_vs_van_bleu_sig":    sig_label(sft_vs_van_bleu_p) if isinstance(sft_vs_van_bleu_p, float) else "n/a",
                "grpo_vs_sft_bleu_delta": round(sft_vs_grpo_bleu_delta, 4) if isinstance(sft_vs_grpo_bleu_delta, float) else "n/a",
                "grpo_vs_sft_bleu_sig":   sig_label(sft_vs_grpo_bleu_p) if isinstance(sft_vs_grpo_bleu_p, float) else "n/a",
                "sft_comet":              round(sft_comet, 4),
            })
    finally:
        logger.close()

    # --- FINAL SUMMARY TABLE ---
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "sft_ablation_summary.csv"), index=False)

    print(f"\n\n{'='*85}")
    print("THREE-WAY COMPARISON: Vanilla vs SFT vs GRPO")
    print(f"{'='*85}")
    print(f"{'Lang':<12} | {'Van chrF':>8} | {'SFT chrF':>8} | {'GRPO chrF':>9} | {'SFT>Van':>8} | {'GRPO>SFT':>9}")
    print(f"{'-'*85}")
    for r in summary:
        print(
            f"{r['lang']:<12} | "
            f"{str(r['van_chrf']):>8} | "
            f"{str(r['sft_chrf']):>8} | "
            f"{str(r['grpo_chrf']):>9} | "
            f"{str(r['sft_vs_van_chrf_delta']):>5} {str(r['sft_vs_van_chrf_sig']):>3} | "
            f"{str(r['grpo_vs_sft_chrf_delta']):>6} {str(r['grpo_vs_sft_chrf_sig']):>3}"
        )
    print(f"{'='*85}")
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ~ p<0.10  ns = not significant")
    print(f"\nOutputs in: {out_dir}/")
    print("  config.yaml            — resolved config for reproducibility")
    print("  adapters/{lang}/       — SFT LoRA adapter per language")
    print("  results/{lang}_sft.csv — per-sentence SFT translations")
    print("  sft_ablation_summary.csv — three-way comparison table")


if __name__ == "__main__":
    main()
