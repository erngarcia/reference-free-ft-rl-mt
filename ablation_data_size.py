import argparse
import os
import gc
from datetime import datetime

import yaml
import torch
import pandas as pd
from torch.optim import AdamW

import train_grpo as grpo
import sft_ablation as sft

DEVICE = grpo.DEVICE


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
        "languages", "train_sizes", "model_name", "reward", "reward_mode",
        "train_file", "eval_data_dir", "final_eval_data_dir",
        "output_dir", "tag", "n_bootstrap",
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
    parser = argparse.ArgumentParser(
        description="Training data size ablation: vanilla vs SFT vs GRPO across N"
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. ablation_data_size.yaml)")

    # All args are optional — they override config values if provided
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--train_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--reward", choices=["hybrid", "semantic", "comet"], default=None)
    parser.add_argument("--reward_mode",
                        choices=["standard", "soft_rank", "contrastive",
                                 "soft_rank_contrastive"],
                        default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--eval_data_dir", default=None)
    parser.add_argument("--final_eval_data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--n_bootstrap", type=int, default=None)
    return parser.parse_args()


# ============================================================
# SIGNIFICANCE HELPER (reuses bootstrap machinery from sft_ablation)
# ============================================================

def compare(hyps_a, hyps_b, refs, cfg):
    """Paired bootstrap test: does system b beat system a?"""
    chrf_delta, chrf_p = sft.paired_bootstrap_delta(
        hyps_a, hyps_b, refs, sft.chrf_fn, cfg["n_bootstrap"], cfg["bootstrap_seed"]
    )
    bleu_delta, bleu_p = sft.paired_bootstrap_delta(
        hyps_a, hyps_b, refs, sft.bleu_fn, cfg["n_bootstrap"], cfg["bootstrap_seed"]
    )
    return {
        "chrf_delta": round(chrf_delta, 4), "chrf_p": round(chrf_p, 4),
        "chrf_sig": sft.sig_label(chrf_p),
        "bleu_delta": round(bleu_delta, 4), "bleu_p": round(bleu_p, 4),
        "bleu_sig": sft.sig_label(bleu_p),
    }


# ============================================================
# PER-(LANG, N) TRAINING RUNS
# ============================================================

def run_grpo(cfg, res, df_train, df_eval, lang, out_dir):
    rewarder = grpo.GRPORewarder(res, cfg)
    logger   = grpo.TrainingLogger(os.path.join(out_dir, "grpo_training_metrics.csv"))
    actor    = grpo.load_actor(cfg)
    try:
        grpo.run_training(
            actor, res.tokenizer, rewarder, logger,
            df_train, df_eval, lang, out_dir, cfg,
        )
    finally:
        logger.close()
    return actor


def run_sft(cfg, tokenizer, df_train, lang):
    model     = sft.load_model(cfg)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["sft_lr"]
    )
    for _ in range(cfg.get("sft_epochs", 1)):
        for row in df_train.itertuples():
            sft.sft_train_step(model, optimizer, tokenizer, row.src, row.ref, lang, cfg)
    optimizer.state.clear()
    del optimizer
    torch.cuda.empty_cache()
    return model


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = merge_config_and_args(cfg, args)

    if cfg.get("tag") is None:
        cfg["tag"] = f"data_size_ablation_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if cfg.get("output_dir") is None:
        cfg["output_dir"] = os.path.join("results", cfg["tag"])
    if cfg.get("final_eval_data_dir") is None:
        cfg["final_eval_data_dir"] = cfg["eval_data_dir"]

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"  Config:      {args.config}")
    print(f"  Tag:         {cfg['tag']}")
    print(f"  Model:       {cfg['model_name']}")
    print(f"  Languages:   {cfg['languages']}")
    print(f"  Train sizes: {cfg['train_sizes']}")
    print(f"{'='*60}\n")

    res     = grpo.SharedResources(cfg)
    summary = []
    summary_path = os.path.join(out_dir, "data_size_ablation_summary.csv")

    for lang in cfg["languages"]:
        print(f"\n{'='*20} {lang} {'='*20}")

        final_path = os.path.join(cfg["final_eval_data_dir"], lang, "devtest.parquet")
        df_final   = pd.read_parquet(final_path)

        # Vanilla baseline is independent of N — computed once per language
        print("  Evaluating vanilla baseline...")
        vanilla_model = grpo.load_vanilla_model(cfg)
        vt = grpo.translate_corpus(vanilla_model, res.tokenizer, df_final, lang, cfg)
        vt, v_com, v_chrf, v_bleu, v_bs = grpo.score_corpus(vt, res.comet_da, lang, cfg)
        van_hyps = [t["mt"] for t in vt]
        refs     = [t["ref"] for t in vt]
        print(f"  Vanilla | chrF: {v_chrf:.4f} | COMET-22: {v_com:.4f}")
        del vanilla_model
        gc.collect()
        torch.cuda.empty_cache()

        for n in cfg["train_sizes"]:
            print(f"\n  --- N = {n} ---")
            n_dir = os.path.join(out_dir, lang, f"n{n}")
            os.makedirs(n_dir, exist_ok=True)

            train_path = os.path.join(cfg["train_file"], lang, "dev.parquet")
            eval_path  = os.path.join(cfg["eval_data_dir"], lang, "dev.parquet")
            df_train   = pd.read_parquet(train_path).iloc[:n]
            df_eval    = pd.read_parquet(eval_path)

            # --- GRPO ---
            print(f"  Training GRPO ({n} sentences)...")
            actor = run_grpo(cfg, res, df_train, df_eval, lang, n_dir)
            actor.save_pretrained(os.path.join(n_dir, "adapters_grpo"))
            gt = grpo.translate_corpus(actor, res.tokenizer, df_final, lang, cfg)
            gt, g_com, g_chrf, g_bleu, g_bs = grpo.score_corpus(gt, res.comet_da, lang, cfg)
            pd.DataFrame(gt).to_csv(os.path.join(n_dir, f"{lang}_n{n}_grpo.csv"), index=False)
            grpo_hyps = [t["mt"] for t in gt]
            del actor
            gc.collect()
            torch.cuda.empty_cache()

            # --- SFT ---
            print(f"  Training SFT ({n} sentences)...")
            sft_model = run_sft(cfg, res.tokenizer, df_train, lang)
            sft_model.save_pretrained(os.path.join(n_dir, "adapters_sft"))
            st = grpo.translate_corpus(sft_model, res.tokenizer, df_final, lang, cfg)
            st, s_com, s_chrf, s_bleu, s_bs = grpo.score_corpus(st, res.comet_da, lang, cfg)
            pd.DataFrame(st).to_csv(os.path.join(n_dir, f"{lang}_n{n}_sft.csv"), index=False)
            sft_hyps = [t["mt"] for t in st]
            del sft_model
            gc.collect()
            torch.cuda.empty_cache()

            # --- Significance ---
            sft_vs_van  = compare(van_hyps, sft_hyps, refs, cfg)
            grpo_vs_van = compare(van_hyps, grpo_hyps, refs, cfg)
            grpo_vs_sft = compare(sft_hyps, grpo_hyps, refs, cfg)

            print(
                f"  N={n:<5} | Van chrF: {v_chrf:.4f} | "
                f"SFT chrF: {s_chrf:.4f} ({sft_vs_van['chrf_sig']} vs van) | "
                f"GRPO chrF: {g_chrf:.4f} ({grpo_vs_van['chrf_sig']} vs van, "
                f"{grpo_vs_sft['chrf_sig']} vs sft, p={grpo_vs_sft['chrf_p']})"
            )

            summary.append({
                "lang": lang, "n": n,
                "vanilla_chrf": round(v_chrf, 4), "vanilla_comet22": round(v_com, 4),
                "sft_chrf":     round(s_chrf, 4), "sft_comet22":     round(s_com, 4),
                "grpo_chrf":    round(g_chrf, 4), "grpo_comet22":    round(g_com, 4),
                "sft_vs_van_chrf_delta":  sft_vs_van["chrf_delta"],
                "sft_vs_van_chrf_p":      sft_vs_van["chrf_p"],
                "sft_vs_van_chrf_sig":    sft_vs_van["chrf_sig"],
                "grpo_vs_van_chrf_delta": grpo_vs_van["chrf_delta"],
                "grpo_vs_van_chrf_p":     grpo_vs_van["chrf_p"],
                "grpo_vs_van_chrf_sig":   grpo_vs_van["chrf_sig"],
                "grpo_vs_sft_chrf_delta": grpo_vs_sft["chrf_delta"],
                "grpo_vs_sft_chrf_p":     grpo_vs_sft["chrf_p"],
                "grpo_vs_sft_chrf_sig":   grpo_vs_sft["chrf_sig"],
                "sft_vs_van_bleu_sig":    sft_vs_van["bleu_sig"],
                "grpo_vs_van_bleu_sig":   grpo_vs_van["bleu_sig"],
                "grpo_vs_sft_bleu_sig":   grpo_vs_sft["bleu_sig"],
            })

            # Write incrementally so a crash doesn't lose completed runs
            pd.DataFrame(summary).to_csv(summary_path, index=False)

    # --- FINAL TABLE ---
    print(f"\n\n{'='*100}")
    print("TRAINING DATA SIZE ABLATION")
    print(f"{'='*100}")
    print(
        f"{'Lang':<10} | {'N':>5} | {'Van':>7} | {'SFT':>7} | {'GRPO':>7} | "
        f"{'SFT>Van':>8} | {'GRPO>Van':>9} | {'GRPO>SFT (p)':>16}"
    )
    print(f"{'-'*100}")
    for r in summary:
        print(
            f"{r['lang']:<10} | {r['n']:>5} | {r['vanilla_chrf']:>7.4f} | "
            f"{r['sft_chrf']:>7.4f} | {r['grpo_chrf']:>7.4f} | "
            f"{r['sft_vs_van_chrf_sig']:>8} | {r['grpo_vs_van_chrf_sig']:>9} | "
            f"{r['grpo_vs_sft_chrf_sig']:>3} (p={r['grpo_vs_sft_chrf_p']}){'':>1}"
        )
    print(f"{'='*100}")
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ~ p<0.10  ns = not significant")
    print(f"\nOutputs in: {out_dir}/")
    print("  config.yaml                     — resolved config for reproducibility")
    print("  {lang}/n{N}/                     — per-run GRPO/SFT adapters + training logs")
    print("  {lang}/n{N}/{lang}_n{N}_*.csv    — per-sentence translations for GRPO/SFT")
    print("  data_size_ablation_summary.csv  — full comparison table")


if __name__ == "__main__":
    main()
