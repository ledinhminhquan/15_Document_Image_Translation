"""Command-line interface - the single entrypoint for the imgtrans system.

    imgtrans <command> [options]

Commands: data, gen-synthetic, train-mt, train-baseline, tune, evaluate, translate-image,
translate-text, demo-agent, serve, benchmark, error-analysis, layout-fidelity, monitor-log,
generate-report, generate-slides, autopilot, grade.

All console output is ASCII-only (Windows cp1252 safe); stdout stays pipeable JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Document-Image Machine Translation System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all
    print(json.dumps(download_all(_load(args), render_synthetic=args.render), indent=2, ensure_ascii=False))


def cmd_gen_synthetic(args):
    from .data.dataset import build_synthetic_eval
    print(json.dumps(build_synthetic_eval(_load(args), n_pages=args.n_pages, split=args.split),
                     indent=2, ensure_ascii=False))


def cmd_train_mt(args):
    from .training.train_mt import train_mt
    print(json.dumps(train_mt(_load(args), limit=args.limit, base_model=args.base_model), indent=2))


def cmd_train_baseline(args):
    from .training.train_baseline import build_baseline
    print(json.dumps(build_baseline(_load(args), limit=args.limit), indent=2, ensure_ascii=False))


def cmd_tune(args):
    from .training.tune import tune
    print(json.dumps(tune(_load(args), load_model=not args.fast), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    rep = evaluate(_load(args), limit=args.limit, ocr_noise=args.ocr_noise, load_model=not args.fast)
    print(json.dumps(rep.get("headline", rep), indent=2, ensure_ascii=False))


def cmd_translate_image(args):
    from .agent.imgtrans_agent import ImgTransAgent
    cfg = _load(args)
    agent = ImgTransAgent(cfg, load_model=not args.fast)
    p = Path(args.image)
    kwargs = {"mode": args.mode or cfg.render.mode, "render_path": args.out or "", "save": not args.no_save}
    if p.suffix.lower() == ".pdf":
        job = agent.run(pdf_path=str(p), filename=p.name, **kwargs)
    else:
        from PIL import Image
        job = agent.run(image=Image.open(p), filename=p.name, **kwargs)
    sd = job.to_dict()
    sd.pop("trace", None)
    print(json.dumps(sd, indent=2, ensure_ascii=False))


def cmd_translate_text(args):
    from .agent.imgtrans_agent import ImgTransAgent
    agent = ImgTransAgent(_load(args), load_model=not args.fast)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    out = agent.translate_text(text, mode="text_only")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.imgtrans_agent import ImgTransAgent
    from .data import samples
    agent = ImgTransAgent(_load(args), load_model=not args.fast)
    for i, spec in enumerate(samples.seed_pages()):
        job = agent.run(spec=spec, mode="text_only", save=False)
        sd = job.to_dict()
        print(f"\n[page{i}] kind={sd['input_kind']} status={sd['status']} "
              f"blocks={sd['n_blocks']} translatable={sd['n_translatable']} "
              f"fit_rate={sd['fit_rate']} decisions={[(x['id'], x['branch']) for x in sd['decisions']]}")
        print("  TGT:", sd["translated_text"].replace("\n\n", " | ")[:140])


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["IMGTRANS_INFER_CONFIG"] = str(args.config)
    target = "imgtrans.api.app_combined:app" if args.ui else "imgtrans.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args)), indent=2, ensure_ascii=False))


def cmd_layout_fidelity(args):
    from .analysis.layout_fidelity import layout_fidelity_report
    print(json.dumps(layout_fidelity_report(_load(args)), indent=2, ensure_ascii=False))


def cmd_monitor_log(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imgtrans", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="prefetch/sanity-check the datasets (streaming probes)")
    sp.add_argument("--render", action="store_true", help="also render synthetic eval pages")
    sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("gen-synthetic", help="render synthetic document-image eval pages")
    sp.add_argument("--n-pages", type=int, default=None); sp.add_argument("--split", default="eval")
    sp.set_defaults(func=cmd_gen_synthetic)
    sp = sub.add_parser("train-mt", help="fine-tune the MT core (Seq2SeqTrainer, chrF)")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--base-model", default=None)
    sp.set_defaults(func=cmd_train_mt)
    sp = sub.add_parser("train-baseline", help="persist the dictionary MT baseline (no GPU)")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_baseline)
    sp = sub.add_parser("tune", help="MT beam-width sweep on the seed pages")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="MT chrF/BLEU + OCR CER/WER + end-to-end + fidelity")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--ocr-noise", type=float, default=0.0)
    sp.add_argument("--fast", action="store_true", help="dictionary baseline (no model download)")
    sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("translate-image", help="translate a document image / PDF (overlay or text)")
    sp.add_argument("--image", required=True); sp.add_argument("--mode", default="")
    sp.add_argument("--out", default=""); sp.add_argument("--no-save", action="store_true")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_translate_image)
    sp = sub.add_parser("translate-text", help="translate raw text blocks (no OCR; demo/offline)")
    sp.add_argument("--text", default=""); sp.add_argument("--file", default="")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_translate_text)
    sp = sub.add_parser("demo-agent", help="run the agent on the synthetic seed pages")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server (+ --ui for the Gradio demo)")
    sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--ui", action="store_true"); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark of the agent")
    sp.add_argument("--n", type=int, default=10); sp.add_argument("--warmup", type=int, default=2)
    sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="per-page OCR / translation / fit error analysis")
    sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("layout-fidelity", help="overlay fit-rate / shrink / overflow report")
    sp.set_defaults(func=cmd_layout_fidelity)
    sp = sub.add_parser("monitor-log", help="production monitoring report from the job log")
    sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor_log)
    sp = sub.add_parser("generate-report", help="generate the PDF report")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check")
    sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
