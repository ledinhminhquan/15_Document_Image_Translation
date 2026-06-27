"""Fine-tune the MT core (the only trainable stage) - HF Seq2SeqTrainer, metric=chrF.

Trains a seq2seq translator (``facebook/m2m100_418M`` default; NLLB / Marian supported)
on flat OPUS-100 sentence pairs so it produces fluent translations of the short text
blocks that the OCR front-end extracts from document images. Resume-safe
(``get_last_checkpoint``); bf16/tf32 on H100/A100, fp16 on T4; early stopping on chrF.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig
from ..data.dataset import load_pairs, seed_split
from ..logging_utils import get_logger
from ..models import model_registry as reg
from . import metrics as M

logger = get_logger(__name__)

_NLLB_CODES = {"en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "vi": "vie_Latn"}


def train_mt(cfg: AppConfig, limit: Optional[int] = None, resume: bool = True,
             base_model: Optional[str] = None) -> Dict:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    mc = cfg.mt
    model_id = base_model or mc.base_model
    torch.backends.cuda.matmul.allow_tf32 = bool(mc.tf32)
    cap = limit or cfg.data.max_train_samples

    train_pairs = load_pairs(cfg, split="train", limit=cap)
    eval_pairs = load_pairs(cfg, split="validation", limit=cfg.data.max_eval_samples)
    if len(eval_pairs) <= 2:
        _, eval_pairs = seed_split(cfg.data.seed)
    train_ex = [{"source": p.src, "target": p.tgt} for p in train_pairs]
    eval_ex = [{"source": p.src, "target": p.tgt} for p in eval_pairs]
    logger.info("Training %s on %d pairs (eval %d), %s->%s",
                model_id, len(train_ex), len(eval_ex), mc.src_lang, mc.tgt_lang)

    tok = AutoTokenizer.from_pretrained(model_id)
    is_nllb = "nllb" in model_id.lower()
    if hasattr(tok, "src_lang"):
        tok.src_lang = _NLLB_CODES.get(mc.src_lang, mc.src_lang) if is_nllb else mc.src_lang
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if bool(mc.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    tgt_code = _NLLB_CODES.get(mc.tgt_lang, mc.tgt_lang) if is_nllb else mc.tgt_lang
    forced_bos = None
    if hasattr(tok, "get_lang_id"):
        forced_bos = tok.get_lang_id(mc.tgt_lang)
    elif hasattr(tok, "convert_tokens_to_ids"):
        try:
            forced_bos = tok.convert_tokens_to_ids(tgt_code)
        except Exception:
            forced_bos = None
    if forced_bos is not None:
        model.config.forced_bos_token_id = forced_bos

    def to_ds(ex):
        return Dataset.from_dict({"source": [e["source"] for e in ex], "target": [e["target"] for e in ex]})

    def preprocess(batch):
        mi = tok(batch["source"], max_length=mc.max_source_length, truncation=True)
        labels = tok(text_target=batch["target"], max_length=mc.max_target_length, truncation=True)
        mi["labels"] = labels["input_ids"]
        return mi

    train_ds = to_ds(train_ex).map(preprocess, batched=True, remove_columns=["source", "target"])
    eval_ds = to_ds(eval_ex).map(preprocess, batched=True, remove_columns=["source", "target"])
    collator = DataCollatorForSeq2Seq(tok, model=model)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dpred = tok.batch_decode(preds, skip_special_tokens=True)
        dref = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf": M.chrf(dpred, dref), "bleu": M.bleu(dpred, dref)}

    out_dir = mc.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=mc.num_train_epochs, learning_rate=mc.learning_rate,
        per_device_train_batch_size=mc.per_device_train_batch_size,
        per_device_eval_batch_size=mc.per_device_eval_batch_size,
        gradient_accumulation_steps=mc.gradient_accumulation_steps,
        weight_decay=mc.weight_decay, warmup_ratio=mc.warmup_ratio, label_smoothing_factor=mc.label_smoothing,
        max_grad_norm=mc.max_grad_norm, bf16=bool(mc.bf16), fp16=bool(mc.fp16),
        predict_with_generate=True, generation_max_length=mc.max_target_length, generation_num_beams=mc.num_beams,
        eval_strategy="steps", save_strategy="steps", eval_steps=mc.eval_steps, save_steps=mc.save_steps,
        save_total_limit=2, logging_steps=mc.logging_steps, seed=mc.seed, report_to=[], group_by_length=True,
        load_best_model_at_end=True, metric_for_best_model="chrf", greater_is_better=True)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=mc.early_stopping_patience)] \
        if mc.early_stopping_patience > 0 else []
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                             data_collator=collator, tokenizer=tok, compute_metrics=compute_metrics,
                             callbacks=callbacks)
    last = get_last_checkpoint(str(out_dir)) if resume and out_dir.exists() else None
    if last:
        logger.info("Resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    metrics = {}
    try:
        metrics = {k: float(v) for k, v in trainer.evaluate().items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.info("final eval failed (%s)", exc)

    version = reg.make_version(model_id)
    final_dir = out_dir / version
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    reg.write_metadata(final_dir, version=version, base_model=model_id,
                       dataset_signature={"train": len(train_ex), "dataset": cfg.data.mt_dataset,
                                          "config": cfg.data.mt_config, "seed": mc.seed},
                       metrics=metrics, extra={"src_lang": mc.src_lang, "tgt_lang": mc.tgt_lang})
    reg.update_latest_pointer(out_dir, final_dir)
    (out_dir / "last_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("MT training done -> %s", final_dir)
    return {"version": version, "model_dir": str(final_dir), "base_model": model_id,
            "n_train": len(train_ex), "metrics": metrics}


__all__ = ["train_mt"]
