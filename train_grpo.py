import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import gc
import time
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import CHRF, BLEU
from bert_score import score as bert_score_fn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

chrf_metric = CHRF(word_order=2)
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
    # Fields where CLI None means "not set" (use config default)
    optional_overrides = [
        "languages", "reward", "reward_mode", "soft_rank_temp",
        "train_file", "eval_data_dir", "final_eval_data_dir",
        "vanilla_dir", "output_dir", "tag", "n_segments",
        "model_name", "n_epochs", "eval_every", "eval_batch",
        "checkpoint_step",
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
    parser = argparse.ArgumentParser(description="GRPO MT experiment runner")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. configs/exp_a_600m.yaml)")

    # All args are optional — they override config values if provided
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--reward", choices=["hybrid", "semantic", "comet"], default=None)
    parser.add_argument("--reward_mode",
                        choices=["standard", "soft_rank", "contrastive",
                                 "soft_rank_contrastive"],
                        default=None)
    parser.add_argument("--soft_rank_temp", type=float, default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--eval_data_dir", default=None)
    parser.add_argument("--final_eval_data_dir", default=None)
    parser.add_argument("--vanilla_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--n_segments", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_batch", type=int, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    return parser.parse_args()


# ============================================================
# SHARED RESOURCES
# ============================================================

class SharedResources:
    def __init__(self, cfg):
        print(f"Loading tokenizer ({cfg['model_name']})...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

        print("Loading COMET-Kiwi (reward model, reference-free)...")
        self.comet_kiwi = load_from_checkpoint(
            download_model("Unbabel/wmt22-cometkiwi-da")
        )
        self.comet_kiwi.eval()

        print("Loading COMET-22 (eval model, reference-based)...")
        self.comet_da = load_from_checkpoint(
            download_model("Unbabel/wmt22-comet-da")
        )
        self.comet_da.eval()

        self.labse = None
        if cfg["reward"] in ("hybrid", "semantic"):
            print("Loading LaBSE...")
            self.labse = SentenceTransformer(
                "sentence-transformers/LaBSE"
            ).to(DEVICE)

        print(f"Resources ready. Reward: [{cfg['reward']}]")


# ============================================================
# BERTSCORE HELPER
# ============================================================

def compute_bertscore(hypotheses, references, lang, bertscore_lang_map):
    bs_lang = bertscore_lang_map.get(lang)
    if bs_lang and bs_lang != "en":
        kwargs = dict(lang=bs_lang, rescale_with_baseline=True)
    else:
        kwargs = dict(
            model_type="bert-base-multilingual-cased",
            rescale_with_baseline=False
        )
    _, _, F1 = bert_score_fn(
        hypotheses, references, device=DEVICE, verbose=False, **kwargs
    )
    torch.cuda.empty_cache()
    return float(F1.mean().item()), F1.tolist()


# ============================================================
# REWARDER
# ============================================================

class GRPORewarder:
    def __init__(self, resources, cfg):
        self.res          = resources
        self.mode         = cfg["reward"]
        self.shaping      = cfg["reward_mode"]
        self.sr_temp      = cfg["soft_rank_temp"]
        self.n_inversions = 0
        self.n_steps      = 0

    def get_reward(self, src_text, hyps, baseline_hyp=None):
        raw = self._compute_raw(src_text, hyps)
        self.n_steps += 1

        if self.shaping == "standard":
            return raw.copy(), raw.copy(), None
        elif self.shaping == "soft_rank":
            return self._soft_rank(raw), raw.copy(), None
        elif self.shaping == "contrastive":
            assert baseline_hyp is not None
            baseline_score = float(self._compute_raw(src_text, [baseline_hyp])[0])
            contrastive    = raw - baseline_score
            if contrastive.max() < 0:
                self.n_inversions += 1
            return contrastive, raw.copy(), baseline_score
        elif self.shaping == "soft_rank_contrastive":
            assert baseline_hyp is not None
            baseline_score = float(self._compute_raw(src_text, [baseline_hyp])[0])
            contrastive    = raw - baseline_score
            if contrastive.max() < 0:
                self.n_inversions += 1
            return self._soft_rank(contrastive), raw.copy(), baseline_score
        else:
            raise ValueError(f"Unknown reward shaping: {self.shaping}")

    def _soft_rank(self, scores):
        scores         = np.array(scores, dtype=np.float64)
        scores_shifted = scores - scores.mean()
        exp_scores     = np.exp(scores_shifted / self.sr_temp)
        return (exp_scores / (exp_scores.sum() + 1e-8)).astype(np.float32)

    def _compute_raw(self, src_text, hyps):
        if self.mode == "hybrid":
            return (0.5 * self._labse(src_text, hyps)
                    + 0.5 * self._comet_kiwi(src_text, hyps))
        elif self.mode == "semantic":
            return self._labse(src_text, hyps)
        elif self.mode == "comet":
            return self._comet_kiwi(src_text, hyps)

    def _labse(self, src_text, hyps):
        with torch.no_grad():
            embs   = self.res.labse.encode(
                [src_text] + hyps,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            scores = (embs[0] @ embs[1:].T).cpu().numpy()
        del embs
        torch.cuda.empty_cache()
        return scores

    def _comet_kiwi(self, src_text, hyps):
        data = [{"src": src_text, "mt": h} for h in hyps]
        with torch.no_grad():
            out    = self.res.comet_kiwi.predict(
                data,
                batch_size=len(hyps),
                gpus=1 if DEVICE == "cuda" else 0
            )
            scores = np.array(out.scores)
        del out
        torch.cuda.empty_cache()
        return scores

    def inversion_rate(self):
        return 0.0 if self.n_steps == 0 else self.n_inversions / self.n_steps


# ============================================================
# TEMPERATURE CONTROLLER
# ============================================================

class TemperatureController:
    def __init__(self, cfg):
        self.temp                 = cfg["temp_initial"]
        self.max_temp             = cfg["temp_max"]
        self.step                 = cfg["temp_step"]
        self.patience             = cfg["temp_patience"]
        self.cooldown             = cfg["temp_cooldown"]
        self.history              = []
        self.steps_since_increase = cfg["temp_cooldown"]
        self._history_cap         = cfg["temp_patience"] * 3

    def update(self, reward):
        self.history.append(float(reward))
        if len(self.history) > self._history_cap:
            self.history = self.history[-self.patience * 2:]
        self.steps_since_increase += 1
        if (len(self.history) >= self.patience
                and self.steps_since_increase >= self.cooldown
                and self.temp < self.max_temp):
            recent = np.mean(self.history[-self.patience // 2:])
            older  = np.mean(
                self.history[-self.patience: -self.patience // 2]
            )
            if recent - older < 0.002:
                self.temp = min(self.temp + self.step, self.max_temp)
                self.steps_since_increase = 0
                print(
                    f"\n  [TempCtrl] Plateau → temperature: {self.temp:.2f}"
                )
        return self.temp


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


def load_actor(cfg):
    bnb  = make_bnb_config(cfg)
    lora = make_lora_config(cfg)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
    )
    base  = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=False
    )
    actor = get_peft_model(base, lora)
    actor.enable_input_require_grads()
    return actor


def load_vanilla_model(cfg):
    bnb   = make_bnb_config(cfg)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
    )
    model.eval()
    return model


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
            self._writer.writerow([
                "lang", "step", "loss", "reward", "reward_raw",
                "baseline_score", "kl", "chrf", "temperature",
            ])

    def log(self, lang, step, loss, reward, reward_raw,
            baseline_score, kl, chrf=None, temp=None):
        self._writer.writerow([
            lang, step,
            round(loss,           6),
            round(reward,         6),
            round(reward_raw,     6),
            round(baseline_score, 6) if baseline_score is not None else "",
            round(kl,             6),
            round(chrf,           4) if chrf is not None else "",
            round(temp,           4) if temp is not None else "",
        ])
        self._file.flush()

    def close(self):
        self._file.close()


# ============================================================
# CORE TRAINING FUNCTIONS
# ============================================================

def get_log_probs(model, tokenizer, input_ids, attention_mask, gen_ids):
    outputs         = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=gen_ids[:, :-1],
        return_dict=True,
    )
    log_probs       = F.log_softmax(outputs.logits, dim=-1)
    actual_ids      = gen_ids[:, 1:].unsqueeze(-1)
    token_log_probs = torch.gather(log_probs, -1, actual_ids).squeeze(-1)
    mask            = gen_ids[:, 1:] != tokenizer.pad_token_id
    return (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1)


def decode_greedy_baseline(actor, tokenizer, inputs, tgt_token_id, cfg):
    with actor.disable_adapter(), torch.no_grad():
        baseline_ids = actor.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            do_sample=False,
            num_beams=1,
            max_new_tokens=cfg["max_new_tokens_train"],
        )
    return tokenizer.decode(baseline_ids[0], skip_special_tokens=True)


def train_step(actor, tokenizer, rewarder, optimizer,
               src_text, lang, temp_controller, cfg):
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = lang
    tgt_token_id       = tokenizer.convert_tokens_to_ids(lang)
    inputs             = tokenizer(src_text, return_tensors="pt").to(DEVICE)
    current_temp       = temp_controller.temp

    baseline_hyp = None
    if rewarder.shaping in ("contrastive", "soft_rank_contrastive"):
        baseline_hyp = decode_greedy_baseline(
            actor, tokenizer, inputs, tgt_token_id, cfg
        )

    actor.eval()
    with torch.no_grad():
        gen_ids = actor.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            do_sample=True,
            num_return_sequences=cfg["k"],
            max_new_tokens=cfg["max_new_tokens_train"],
            temperature=current_temp,
        )

    hyps        = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    gen_ids_ref = gen_ids
    del gen_ids
    torch.cuda.empty_cache()

    shaped_rewards, raw_rewards, baseline_score = rewarder.get_reward(
        src_text, hyps, baseline_hyp=baseline_hyp
    )
    avg_reward_shaped = float(shaped_rewards.mean())
    avg_reward_raw    = float(raw_rewards.mean())

    advantages = torch.tensor(
        (shaped_rewards - shaped_rewards.mean())
        / (shaped_rewards.std() + 1e-4),
        dtype=torch.float32,
    ).to(DEVICE)

    K          = cfg["k"]
    exp_inputs = {k: v.expand(K, -1) for k, v in inputs.items()}

    with actor.disable_adapter(), torch.no_grad():
        ref_chunks = []
        for ci in range(0, K, 8):
            chunk_in = {k: v[ci:ci+8] for k, v in exp_inputs.items()}
            ref_chunks.append(
                get_log_probs(
                    actor, tokenizer,
                    chunk_in["input_ids"],
                    chunk_in["attention_mask"],
                    gen_ids_ref[ci:ci+8],
                )
            )
        ref_log_probs = torch.cat(ref_chunks).detach()
    del ref_chunks

    actor.train()
    optimizer.zero_grad()
    curr_log_probs = get_log_probs(
        actor, tokenizer,
        exp_inputs["input_ids"],
        exp_inputs["attention_mask"],
        gen_ids_ref,
    )
    ratio       = torch.exp(curr_log_probs - ref_log_probs)
    clipped     = torch.clamp(ratio, 0.8, 1.2)
    policy_loss = -torch.min(
        ratio * advantages, clipped * advantages
    ).mean()
    kl          = (
        torch.exp(curr_log_probs - ref_log_probs)
        - (curr_log_probs - ref_log_probs) - 1
    ).mean()
    total_loss  = policy_loss + cfg["kl_beta"] * kl

    loss_val = total_loss.item()
    kl_val   = kl.item()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer.step()

    del (gen_ids_ref, hyps, shaped_rewards, raw_rewards, advantages,
         curr_log_probs, ref_log_probs, exp_inputs, ratio, clipped,
         policy_loss, kl, total_loss)
    torch.cuda.empty_cache()

    return loss_val, avg_reward_shaped, avg_reward_raw, baseline_score, kl_val, current_temp


def intermediate_eval(actor, tokenizer, df_eval, lang, cfg):
    actor.eval()
    tokenizer.src_lang = "eng_Latn"
    tgt_token_id       = tokenizer.convert_tokens_to_ids(lang)
    samples            = df_eval.sample(
        n=min(cfg["eval_n_samples"], len(df_eval)), random_state=42
    )
    srcs, refs, hyps   = samples["src"].tolist(), samples["ref"].tolist(), []

    for i in range(0, len(srcs), cfg["eval_batch_size"]):
        inputs = tokenizer(
            srcs[i:i+cfg["eval_batch_size"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            gen_ids = actor.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                do_sample=False,
                num_beams=4,
                max_new_tokens=cfg["max_new_tokens_eval"],
            )
        hyps.extend(
            tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        )
        del inputs, gen_ids

    torch.cuda.empty_cache()
    return chrf_metric.corpus_score(hyps, [refs]).score


def run_training(actor, tokenizer, rewarder, logger,
                 df_train, df_eval, lang, out_dir, cfg):
    temp_controller   = TemperatureController(cfg)
    optimizer         = AdamW(
        filter(lambda p: p.requires_grad, actor.parameters()),
        lr=cfg["lr"],
    )
    best_chrf         = -float("inf")
    best_chrf_step    = -1
    best_adapter_path = os.path.join(out_dir, "adapters_best", lang)
    checkpoint_step   = cfg.get("checkpoint_step", 0)
    ckpt_adapter_path = (
        os.path.join(out_dir, f"adapters_step{checkpoint_step}", lang)
        if checkpoint_step > 0 else None
    )
    ckpt_saved  = False
    global_step = 0

    for epoch in range(cfg["n_epochs"]):
        if cfg["n_epochs"] > 1:
            print(f"\n  Epoch {epoch + 1}/{cfg['n_epochs']}")
        pbar = tqdm(df_train.itertuples(), total=len(df_train), desc=lang)

        for row in pbar:
            (loss_val, avg_reward_shaped, avg_reward_raw,
             baseline_score, kl_val, current_temp) = train_step(
                actor, tokenizer, rewarder, optimizer,
                row.src, lang, temp_controller, cfg,
            )

            if (ckpt_adapter_path and not ckpt_saved
                    and global_step == checkpoint_step):
                os.makedirs(ckpt_adapter_path, exist_ok=True)
                actor.save_pretrained(ckpt_adapter_path)
                ckpt_saved = True
                print(
                    f"\n  Step {global_step} | "
                    f"★ Checkpoint saved → {ckpt_adapter_path}"
                )

            chrf_val = None
            if global_step % cfg["eval_every"] == 0 and global_step > 0:
                chrf_val = intermediate_eval(
                    actor, tokenizer, df_eval, lang, cfg
                )
                if chrf_val > best_chrf:
                    best_chrf      = chrf_val
                    best_chrf_step = global_step
                    os.makedirs(best_adapter_path, exist_ok=True)
                    actor.save_pretrained(best_adapter_path)
                    print(
                        f"\n  Step {global_step} (ep{epoch+1}) | "
                        f"chrF: {chrf_val:.4f} | temp: {current_temp:.2f} | "
                        f"★ New best — saved"
                    )
                else:
                    print(
                        f"\n  Step {global_step} (ep{epoch+1}) | "
                        f"chrF: {chrf_val:.4f} | temp: {current_temp:.2f} | "
                        f"best: {best_chrf:.4f} @ step {best_chrf_step}"
                    )
                actor.train()

            logger.log(
                lang, global_step, loss_val, avg_reward_shaped,
                avg_reward_raw, baseline_score, kl_val, chrf_val, current_temp,
            )

            if global_step % 10 == 0:
                pbar.set_postfix({
                    "ep":    f"{epoch+1}/{cfg['n_epochs']}",
                    "r_raw": f"{avg_reward_raw:.4f}",
                    "kl":    f"{kl_val:.4f}",
                    "temp":  f"{current_temp:.2f}",
                    "best":  f"{best_chrf:.2f}",
                })

            global_step += 1
            torch.cuda.empty_cache()

    if rewarder.shaping in ("contrastive", "soft_rank_contrastive"):
        print(
            f"\n  Inversion rate: {rewarder.inversion_rate():.3f} "
            f"({rewarder.n_inversions}/{rewarder.n_steps} steps)"
        )

    if ckpt_adapter_path and not ckpt_saved:
        print(
            f"\n  [WARN] checkpoint_step={checkpoint_step} never reached "
            f"(total steps: {global_step}). No step checkpoint saved for {lang}."
        )

    print(f"\n  Training complete. Total steps: {global_step}")
    print(f"  Best chrF: {best_chrf:.4f} at step {best_chrf_step}")

    optimizer.state.clear()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return best_chrf, best_chrf_step


# ============================================================
# EVALUATION
# ============================================================

def translate_corpus(model, tokenizer, df, lang, cfg):
    model.eval()
    tokenizer.src_lang = "eng_Latn"
    tgt_token_id       = tokenizer.convert_tokens_to_ids(lang)
    translations       = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  Translating", leave=False):
        inputs = tokenizer(row["src"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_new_tokens=cfg["max_new_tokens_eval"],
                num_beams=5,
                length_penalty=1.0,
            )
        translations.append({
            "src": row["src"],
            "mt":  tokenizer.decode(gen_ids[0], skip_special_tokens=True),
            "ref": row["ref"],
        })
        del inputs, gen_ids

    torch.cuda.empty_cache()
    return translations


def score_corpus(translations, comet_da_model, lang, cfg):
    hyps = [t["mt"]  for t in translations]
    refs = [t["ref"] for t in translations]
    srcs = [t["src"] for t in translations]

    chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
    bleu_score = bleu_metric.corpus_score(hyps, [refs]).score

    comet_input = [
        {"src": s, "mt": m, "ref": r}
        for s, m, r in zip(srcs, hyps, refs)
    ]
    with torch.no_grad():
        out = comet_da_model.predict(
            comet_input,
            batch_size=16,
            gpus=1 if DEVICE == "cuda" else 0,
        )
    comet_scores = out.scores
    del out
    torch.cuda.empty_cache()

    bertscore_mean, bs_per_sent = compute_bertscore(
        hyps, refs, lang, cfg["bertscore_lang_map"]
    )
    torch.cuda.empty_cache()

    for i, (cs, bs) in enumerate(zip(comet_scores, bs_per_sent)):
        translations[i]["comet22_score"] = cs
        translations[i]["bertscore_f1"]  = bs

    return (
        translations,
        float(np.mean(comet_scores)),
        chrf_score,
        bleu_score,
        bertscore_mean,
    )


def load_vanilla_from_archive(lang, vanilla_dir, comet_da_model, cfg):
    path = os.path.join(vanilla_dir, f"results_{lang}_vanilla.csv")
    if not os.path.exists(path):
        return None

    df         = pd.read_csv(path)
    hyps       = df["mt"].tolist()
    refs       = df["ref"].tolist()
    srcs       = df["src"].tolist()
    chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
    bleu_score = bleu_metric.corpus_score(hyps, [refs]).score

    comet_input = [
        {"src": s, "mt": m, "ref": r}
        for s, m, r in zip(srcs, hyps, refs)
    ]
    with torch.no_grad():
        out = comet_da_model.predict(
            comet_input,
            batch_size=16,
            gpus=1 if DEVICE == "cuda" else 0,
        )
    comet_scores = out.scores
    del out
    torch.cuda.empty_cache()
    comet_mean = float(np.mean(comet_scores))

    bs_mean, _ = compute_bertscore(
        hyps, refs, lang, cfg["bertscore_lang_map"]
    )
    torch.cuda.empty_cache()

    print(
        f"  Vanilla  | chrF: {chrf_score:.4f} | COMET-22: {comet_mean:.4f} | "
        f"BERTScore: {bs_mean:.4f}  [archive, rescored]"
    )
    return comet_mean, chrf_score, bleu_score, bs_mean


def run_evaluation(model, tokenizer, comet_da_model, df_eval,
                   lang, out_dir, tag, cfg, label="grpo"):
    res_dir = os.path.join(out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    # Vanilla baseline
    van_scores = None
    if cfg.get("vanilla_dir"):
        van_scores = load_vanilla_from_archive(
            lang, cfg["vanilla_dir"], comet_da_model, cfg
        )

    if van_scores is None:
        print("  Evaluating vanilla baseline (fresh)...")
        vanilla_model = load_vanilla_model(cfg)
        vt            = translate_corpus(vanilla_model, tokenizer, df_eval, lang, cfg)
        vt, v_com, v_chrf, v_bleu, v_bs = score_corpus(
            vt, comet_da_model, lang, cfg
        )
        pd.DataFrame(vt).to_csv(
            os.path.join(res_dir, f"{lang}_vanilla.csv"), index=False
        )
        print(
            f"  Vanilla  | chrF: {v_chrf:.4f} | COMET-22: {v_com:.4f} | "
            f"BERTScore: {v_bs:.4f}"
        )
        van_scores = (v_com, v_chrf, v_bleu, v_bs)
        del vanilla_model
        gc.collect()
        torch.cuda.empty_cache()

    v_com, v_chrf, v_bleu, v_bs = van_scores

    # GRPO / best model
    print(f"  Evaluating [{label}]...")
    gt = translate_corpus(model, tokenizer, df_eval, lang, cfg)
    gt, g_com, g_chrf, g_bleu, g_bs = score_corpus(gt, comet_da_model, lang, cfg)
    pd.DataFrame(gt).to_csv(
        os.path.join(res_dir, f"{lang}_{label}_{tag}.csv"), index=False
    )

    print(
        f"  {label.upper():<12} | chrF: {g_chrf:.4f} | COMET-22: {g_com:.4f} | "
        f"BERTScore: {g_bs:.4f}"
    )
    print(
        f"  Delta        | chrF: {g_chrf-v_chrf:+.4f} | COMET-22: {g_com-v_com:+.4f} | "
        f"BERTScore: {g_bs-v_bs:+.4f}"
    )

    return {
        "lang":              lang,
        "label":             label,
        "vanilla_chrf":      round(v_chrf, 4),
        "grpo_chrf":         round(g_chrf, 4),
        "chrf_delta":        round(g_chrf - v_chrf, 4),
        "vanilla_comet22":   round(v_com,  4),
        "grpo_comet22":      round(g_com,  4),
        "comet22_delta":     round(g_com  - v_com,  4),
        "vanilla_bleu":      round(v_bleu, 4),
        "grpo_bleu":         round(g_bleu, 4),
        "bleu_delta":        round(g_bleu - v_bleu, 4),
        "vanilla_bertscore": round(v_bs,   4),
        "grpo_bertscore":    round(g_bs,   4),
        "bertscore_delta":   round(g_bs   - v_bs,   4),
    }


# ============================================================
# SUMMARY WRITER
# ============================================================

def write_summary(summary, out_dir, cfg):
    if not summary:
        return

    tag        = cfg["tag"]
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        os.path.join(out_dir, "evaluation_summary.csv"), index=False
    )

    txt_path = os.path.join(out_dir, "evaluation_summary.txt")
    sep      = "=" * 120
    with open(txt_path, "w") as f:
        f.write(
            f"{sep}\n"
            f"RESULTS — {tag}\n"
            f"  Reward mode: {cfg['reward_mode']} | Epochs: {cfg['n_epochs']} | "
            f"Eval every: {cfg['eval_every']} | "
            f"Checkpoint step: {cfg['checkpoint_step']}\n"
            f"  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{sep}\n"
            f"{'Lang':<12} | {'Label':<12} | {'Van chrF':>8} | {'GRPO chrF':>9} | "
            f"{'ΔchrF':>7} | {'Van C22':>8} | {'GRPO C22':>9} | {'ΔC22':>6} | "
            f"{'Van BS':>7} | {'GRPO BS':>8} | {'ΔBS':>5}\n"
            f"{'-'*120}\n"
        )
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['lang']:<12} | {row['label']:<12} | "
                f"{row['vanilla_chrf']:>8.4f} | {row['grpo_chrf']:>9.4f} | "
                f"{row['chrf_delta']:>+7.4f} | "
                f"{row['vanilla_comet22']:>8.4f} | {row['grpo_comet22']:>9.4f} | "
                f"{row['comet22_delta']:>+6.4f} | "
                f"{row['vanilla_bertscore']:>7.4f} | {row['grpo_bertscore']:>8.4f} | "
                f"{row['bertscore_delta']:>+5.4f}\n"
            )
        f.write(f"{sep}\n")

    print(f"\n  Summary written → {txt_path}")


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
        cfg["tag"] = (
            f"{cfg['reward']}_{cfg['reward_mode']}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
    if cfg.get("output_dir") is None:
        cfg["output_dir"] = os.path.join("results", cfg["tag"])
    if cfg.get("final_eval_data_dir") is None:
        cfg["final_eval_data_dir"] = cfg["eval_data_dir"]

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Save resolved config alongside results for reproducibility
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"  Config:          {args.config}")
    print(f"  Tag:             {cfg['tag']}")
    print(f"  Model:           {cfg['model_name']}")
    print(f"  Languages:       {cfg['languages']}")
    print(f"  Reward:          {cfg['reward']}")
    print(f"  Reward mode:     {cfg['reward_mode']}")
    if "soft_rank" in cfg["reward_mode"]:
        print(f"  SR temp:         {cfg['soft_rank_temp']}")
    print(f"  Epochs:          {cfg['n_epochs']}")
    print(f"  Eval every:      {cfg['eval_every']} steps")
    print(f"  Checkpoint step: {cfg['checkpoint_step'] if cfg['checkpoint_step'] > 0 else 'disabled'}")
    print(f"  Train file:      {cfg['train_file']}")
    print(f"  Eval dir:        {cfg['eval_data_dir']}")
    print(f"{'='*60}\n")

    res      = SharedResources(cfg)
    rewarder = GRPORewarder(res, cfg)
    logger   = TrainingLogger(os.path.join(out_dir, "training_metrics.csv"))
    actor    = load_actor(cfg)
    summary  = []

    try:
        for lang in cfg["languages"]:
            print(f"\n{'='*20} {lang} {'='*20}")

            train_path = (
                cfg["train_file"] if os.path.isfile(cfg["train_file"])
                else os.path.join(cfg["train_file"], lang, "dev.parquet")
            )
            if not os.path.exists(train_path):
                print(f"  [SKIP] Train file not found: {train_path}")
                continue

            eval_path = os.path.join(
                cfg["eval_data_dir"], lang, "dev.parquet"
            )
            if not os.path.exists(eval_path):
                print(f"  [SKIP] No eval data at {eval_path}")
                continue

            final_eval_path = os.path.join(
                cfg["final_eval_data_dir"], lang, "devtest.parquet"
            )
            if not os.path.exists(final_eval_path):
                print(f"  [WARN] No devtest at {final_eval_path} — using dev")
                final_eval_path = eval_path

            df_train = pd.read_parquet(train_path)
            df_eval  = pd.read_parquet(eval_path)
            df_final = pd.read_parquet(final_eval_path)

            if cfg.get("n_segments"):
                df_train = df_train.iloc[:cfg["n_segments"]]

            print(
                f"  Train: {len(df_train)} | Eval: {len(df_eval)} | "
                f"Final: {len(df_final)}"
            )

            # 1. Train
            print(
                f"\n[1/3] Training ({len(df_train)} steps × {cfg['n_epochs']} epochs) "
                f"— reward_mode: {cfg['reward_mode']}, "
                f"eval_every: {cfg['eval_every']}..."
            )
            best_chrf, best_step = run_training(
                actor, res.tokenizer, rewarder, logger,
                df_train, df_eval, lang, out_dir, cfg,
            )

            # 2. Save final adapter
            final_adapter_path = os.path.join(out_dir, "adapters", lang)
            os.makedirs(final_adapter_path, exist_ok=True)
            actor.save_pretrained(final_adapter_path)
            print(f"  Final adapter → {final_adapter_path}")

            # 3. Evaluate final adapter
            print(f"\n[2/3] Evaluating final adapter...")
            result_final = run_evaluation(
                actor, res.tokenizer, res.comet_da, df_final,
                lang, out_dir, cfg["tag"], cfg, label="grpo_final",
            )
            summary.append(result_final)

            # 4. Evaluate best adapter
            best_adapter_path = os.path.join(out_dir, "adapters_best", lang)
            if os.path.exists(best_adapter_path) and best_step != -1:
                print(
                    f"\n[3/3] Evaluating best adapter "
                    f"(step {best_step}, intermediate chrF {best_chrf:.4f})..."
                )
                base_for_best = load_vanilla_model(cfg)
                best_model    = PeftModel.from_pretrained(
                    base_for_best, best_adapter_path
                )
                best_model.eval()

                result_best = run_evaluation(
                    best_model, res.tokenizer, res.comet_da, df_final,
                    lang, out_dir, cfg["tag"], cfg, label="grpo_best",
                )
                result_best["best_step"] = best_step
                summary.append(result_best)

                del best_model, base_for_best
                gc.collect()
                torch.cuda.empty_cache()

            write_summary(summary, out_dir, cfg)

            del actor
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

            if lang != cfg["languages"][-1]:
                print("\nResetting actor...")
                actor = load_actor(cfg)

    finally:
        logger.close()

    write_summary(summary, out_dir, cfg)

    if summary:
        summary_df = pd.DataFrame(summary)
        print(f"\n\n{'='*120}")
        print(f"FINAL RESULTS — {cfg['tag']}")
        print(f"  Reward mode: {cfg['reward_mode']} | Epochs: {cfg['n_epochs']}")
        print(f"{'='*120}")
        print(
            f"{'Lang':<12} | {'Label':<12} | {'Van chrF':>8} | {'GRPO chrF':>9} | "
            f"{'ΔchrF':>7} | {'Van C22':>8} | {'GRPO C22':>9} | {'ΔC22':>6} | "
            f"{'Van BS':>7} | {'GRPO BS':>8} | {'ΔBS':>5}"
        )
        print("-" * 120)
        for _, row in summary_df.iterrows():
            print(
                f"{row['lang']:<12} | {row['label']:<12} | "
                f"{row['vanilla_chrf']:>8.4f} | {row['grpo_chrf']:>9.4f} | "
                f"{row['chrf_delta']:>+7.4f} | "
                f"{row['vanilla_comet22']:>8.4f} | {row['grpo_comet22']:>9.4f} | "
                f"{row['comet22_delta']:>+6.4f} | "
                f"{row['vanilla_bertscore']:>7.4f} | {row['grpo_bertscore']:>8.4f} | "
                f"{row['bertscore_delta']:>+5.4f}"
            )
        print(f"{'='*120}")

    print(f"\nOutputs in: {out_dir}/")
    print(f"  config.yaml            — resolved config for reproducibility")
    print(f"  adapters/              — final adapter (last step)")
    print(f"  adapters_best/         — best adapter (highest intermediate chrF)")
    if cfg["checkpoint_step"] > 0:
        print(
            f"  adapters_step{cfg['checkpoint_step']}/   — fixed-step snapshot"
        )


if __name__ == "__main__":
    main()
