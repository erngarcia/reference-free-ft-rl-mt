import argparse
import os
import gc
from datetime import datetime
from itertools import combinations

import yaml
import torch
import pandas as pd

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
        "languages", "reward_components", "model_name", "reward_mode",
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
        description="Reward component ablation: LaBSE-only vs COMET-Kiwi-only vs hybrid"
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. ablations_configs/ablation_reward_components.yaml)")

    # All args are optional — they override config values if provided
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--reward_components", nargs="+",
                        choices=["hybrid", "semantic", "comet"], default=None)
    parser.add_argument("--model_name", default=None)
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
# PER-(LANG, REWARD COMPONENT) TRAINING RUN
# ============================================================

def run_variant(cfg, res, reward_component, df_train, df_eval, lang, out_dir):
    cfg_run = dict(cfg)
    cfg_run["reward"] = reward_component

    rewarder = grpo.GRPORewarder(res, cfg_run)
    logger   = grpo.TrainingLogger(os.path.join(out_dir, "training_metrics.csv"))
    actor    = grpo.load_actor(cfg_run)
    try:
        grpo.run_training(
            actor, res.tokenizer, rewarder, logger,
            df_train, df_eval, lang, out_dir, cfg_run,
        )
    finally:
        logger.close()
    return actor


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = merge_config_and_args(cfg, args)

    if cfg.get("tag") is None:
        cfg["tag"] = f"reward_components_ablation_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if cfg.get("output_dir") is None:
        cfg["output_dir"] = os.path.join("results", cfg["tag"])
    if cfg.get("final_eval_data_dir") is None:
        cfg["final_eval_data_dir"] = cfg["eval_data_dir"]

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"  Config:            {args.config}")
    print(f"  Tag:               {cfg['tag']}")
    print(f"  Model:             {cfg['model_name']}")
    print(f"  Languages:         {cfg['languages']}")
    print(f"  Reward components: {cfg['reward_components']}")
    print(f"{'='*60}\n")

    # Force LaBSE + COMET-Kiwi to load once, up front, regardless of which
    # reward component is being ablated in a given run.
    res_cfg = dict(cfg)
    res_cfg["reward"] = "hybrid"
    res = grpo.SharedResources(res_cfg)

    scores_summary = []
    sig_summary    = []
    scores_path    = os.path.join(out_dir, "reward_components_scores.csv")
    sig_path       = os.path.join(out_dir, "reward_components_significance.csv")

    for lang in cfg["languages"]:
        print(f"\n{'='*20} {lang} {'='*20}")

        train_path = os.path.join(cfg["train_file"], lang, "dev.parquet")
        eval_path  = os.path.join(cfg["eval_data_dir"], lang, "dev.parquet")
        final_path = os.path.join(cfg["final_eval_data_dir"], lang, "devtest.parquet")

        df_final = pd.read_parquet(final_path)

        # Vanilla baseline — independent of reward component
        print("  Evaluating vanilla baseline...")
        vanilla_model = grpo.load_vanilla_model(cfg)
        vt = grpo.translate_corpus(vanilla_model, res.tokenizer, df_final, lang, cfg)
        vt, v_com, v_chrf, v_bleu, v_bs = grpo.score_corpus(vt, res.comet_da, lang, cfg)
        refs = [t["ref"] for t in vt]
        systems = {"vanilla": [t["mt"] for t in vt]}
        scores_summary.append({
            "lang": lang, "system": "vanilla",
            "chrf": round(v_chrf, 4), "comet22": round(v_com, 4),
            "bleu": round(v_bleu, 4), "bertscore": round(v_bs, 4),
        })
        print(f"  Vanilla | chrF: {v_chrf:.4f} | COMET-22: {v_com:.4f}")
        del vanilla_model
        gc.collect()
        torch.cuda.empty_cache()

        # One GRPO run per reward component
        for component in cfg["reward_components"]:
            print(f"\n  --- Reward: {component} ---")
            run_dir = os.path.join(out_dir, lang, component)
            os.makedirs(run_dir, exist_ok=True)

            df_train = pd.read_parquet(train_path)
            df_eval  = pd.read_parquet(eval_path)

            actor = run_variant(cfg, res, component, df_train, df_eval, lang, run_dir)
            actor.save_pretrained(os.path.join(run_dir, "adapters"))

            gt = grpo.translate_corpus(actor, res.tokenizer, df_final, lang, cfg)
            gt, g_com, g_chrf, g_bleu, g_bs = grpo.score_corpus(gt, res.comet_da, lang, cfg)
            pd.DataFrame(gt).to_csv(os.path.join(run_dir, f"{lang}_{component}.csv"), index=False)

            systems[component] = [t["mt"] for t in gt]
            scores_summary.append({
                "lang": lang, "system": component,
                "chrf": round(g_chrf, 4), "comet22": round(g_com, 4),
                "bleu": round(g_bleu, 4), "bertscore": round(g_bs, 4),
            })
            print(f"  {component:<10} | chrF: {g_chrf:.4f} | COMET-22: {g_com:.4f}")

            del actor
            gc.collect()
            torch.cuda.empty_cache()

            pd.DataFrame(scores_summary).to_csv(scores_path, index=False)

        # Pairwise significance across all systems present for this language
        print("\n  Pairwise significance:")
        for sys_a, sys_b in combinations(systems.keys(), 2):
            res_cmp = compare(systems[sys_a], systems[sys_b], refs, cfg)
            print(
                f"    {sys_b} vs {sys_a} | ΔchrF: {res_cmp['chrf_delta']:+.4f} "
                f"{res_cmp['chrf_sig']} (p={res_cmp['chrf_p']}) | "
                f"ΔBLEU: {res_cmp['bleu_delta']:+.4f} {res_cmp['bleu_sig']}"
            )
            sig_summary.append({
                "lang": lang, "system_a": sys_a, "system_b": sys_b,
                "chrf_delta": res_cmp["chrf_delta"], "chrf_p": res_cmp["chrf_p"],
                "chrf_sig":   res_cmp["chrf_sig"],
                "bleu_delta": res_cmp["bleu_delta"], "bleu_p": res_cmp["bleu_p"],
                "bleu_sig":   res_cmp["bleu_sig"],
            })
        pd.DataFrame(sig_summary).to_csv(sig_path, index=False)

    # --- FINAL TABLES ---
    scores_df = pd.DataFrame(scores_summary)
    sig_df    = pd.DataFrame(sig_summary)

    print(f"\n\n{'='*80}")
    print("REWARD COMPONENT ABLATION — SCORES")
    print(f"{'='*80}")
    print(scores_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("REWARD COMPONENT ABLATION — PAIRWISE SIGNIFICANCE (chrF)")
    print(f"{'='*80}")
    print(sig_df[["lang", "system_a", "system_b", "chrf_delta", "chrf_p", "chrf_sig"]].to_string(index=False))
    print(f"{'='*80}")
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ~ p<0.10  ns = not significant")

    print(f"\nOutputs in: {out_dir}/")
    print("  config.yaml                        — resolved config for reproducibility")
    print("  {lang}/{component}/                — per-run GRPO adapter + training log")
    print("  {lang}/{component}/{lang}_{component}.csv — per-sentence translations")
    print("  reward_components_scores.csv       — chrF/COMET-22/BLEU/BERTScore per system")
    print("  reward_components_significance.csv — pairwise bootstrap significance")


if __name__ == "__main__":
    main()
