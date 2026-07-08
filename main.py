from training.training_config import make_125M_hofa, LanguageModelingExperimentConfig
import sys
import types


import argparse


def main():
    config = LanguageModelingExperimentConfig() # make_125M_hofa()

    parser = argparse.ArgumentParser(description="HOFA Project Main Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("download-data", help="Download and cache the dataset")
    
    parser_train = subparsers.add_parser("train", help="Run training, plotting, or evaluation")
    parser_train.add_argument("--plot", action="store_true", help="Plot training metrics")
    parser_train.add_argument("--eval", action="store_true", help="Evaluate on HellaSwag using lm-eval")

    parser_infer = subparsers.add_parser("infer", help="Run interactive generation or validation")
    parser_infer.add_argument("--debug", action="store_true", help="Run debug inference (no cache, training model)")
    parser_infer.add_argument("--validate", action="store_true", help="Validate custom inference kernel against reference")

    parser_profile = subparsers.add_parser("profile", help="Run profiling benchmarks")
    parser_profile.add_argument("--prefill", action="store_true", help="Profile prefill (MHA vs HOFA)")
    parser_profile.add_argument("--decode", action="store_true", help="Profile decode throughput and memory")
    parser_profile.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save plotting results (default: True)")
    parser_profile.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True, help="Use cached results if available (default: True)")

    args = parser.parse_args()

    if args.command == "download-data":
        from training.download_fineweb import download_and_tokenize
        download_and_tokenize(config)
    elif args.command == "train":
        if args.plot:
            from training.plot_training import plot_training_metrics
            plot_training_metrics(config)
        elif args.eval:
            from benchmarks.benchmark_swag import evaluate_hellaswag
            from benchmarks.benchmarks_configs import EvalExperimentConfig
            evaluate_hellaswag(config, EvalExperimentConfig())
        else:
            from training.train import train
            train(config)
    elif args.command == "infer":
        if args.validate:
            from src.modules.validate_inference import validate_inference
            validate_inference(config)
        elif args.debug:
            from training.inference_debug import do_debug_inference
            do_debug_inference(config)
        else:
            from training.inference import do_inference
            do_inference(config)
    elif args.command == "profile":
        force_rerun = not args.use_cache
        if args.prefill:
            from benchmarks.profile_prefill import run_profiling_experiment
            from benchmarks.benchmarks_configs import PrefillExperimentConfig
            run_profiling_experiment(config=PrefillExperimentConfig(), force_rerun=force_rerun, save_results=args.save)
        elif args.decode:
            from benchmarks.profile_decode import run_decode_profiling
            from benchmarks.benchmarks_configs import DecodeExperimentConfig
            run_decode_profiling(config=DecodeExperimentConfig(), force_rerun=force_rerun, save_results=args.save)
        else:
            print("Please specify either --prefill or --decode to profile.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
