from training.training_config import make_13M_mha, make_13M_hofa, make_125M_hofa, make_125M_mha, LanguageModelingExperimentConfig


import argparse


def main():
    config = make_125M_hofa() # make_125M_mha()

    parser = argparse.ArgumentParser(description="HOFA Project Main Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("download-data", help="Download and cache the dataset")
    
    parser_train = subparsers.add_parser("train", help="Run training, plotting, or evaluation")
    parser_train.add_argument("--plot", action="store_true", help="Plot training metrics")
    parser_train.add_argument("--shared", action="store_true", help="Plot shared training metrics for HOFA and MHA")
    parser_train.add_argument("--eval", action="store_true", help="Evaluate the model on zero-shot reasoning benchmarks")
    parser_train.add_argument("--simple", action="store_true", help="Only run the HellaSwag benchmark instead of the full suite")

    parser_infer = subparsers.add_parser("infer", help="Run interactive generation or validation")
    parser_infer.add_argument("--debug", action="store_true", help="Print debug information (gate bias, etc.)")
    parser_infer.add_argument("--train", action="store_true", help="Run inference using the training class instead of the inference class")
    parser_infer.add_argument("--latest_ckp", action="store_true", help="Load the latest checkpoint instead of the best validation one")
    parser_infer.add_argument("--validate", action="store_true", help="Validate custom inference kernel against reference")
    parser_infer.add_argument("--verbose", action="store_true", help="Run deep dive into RMSNorm and intermediate tensors")

    parser_profile = subparsers.add_parser("profile", help="Run profiling benchmarks")
    parser_profile.add_argument("--prefill", action="store_true", help="Profile prefill (MHA vs HOFA)")
    parser_profile.add_argument("--decode", action="store_true", help="Profile decode throughput and memory")
    parser_profile.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save plotting results (default: True)")
    parser_profile.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True, help="Use cached results if available (default: True)")

    parser_bench = subparsers.add_parser("benchmark", help="Run synthetic capability benchmarks")
    parser_bench.add_argument("--induction", action="store_true", help="Run the Induction Head capability benchmark")
    parser_bench.add_argument("--copy", action="store_true", help="Run the Sequential Copying capability benchmark")

    args = parser.parse_args()

    if args.command == "download-data":
        from training.download_fineweb import download_and_tokenize
        download_and_tokenize(config)
        import os
        print("\n[INFO] Data download completed successfully! Exiting immediately to prevent HuggingFace teardown bugs.")
        os._exit(0)
    elif args.command == "train":
        if args.plot:
            from training.plot_training import plot_training_metrics
            if args.shared:
                plot_training_metrics([make_125M_hofa(), make_125M_mha()])
            else:
                plot_training_metrics(config)
        elif args.eval:
            from benchmarks.benchmark_zeroshot import evaluate_zeroshot
            from benchmarks.benchmarks_configs import EvalExperimentConfig
            evaluate_zeroshot(config, EvalExperimentConfig(), is_simple=args.simple)
        else:
            from training.train import train
            train(config)
    elif args.command == "infer":
        from src.config import InferenceConfig
        import src.inference
        
        src.inference.DEBUG_MODE = args.debug
        inference_cfg = InferenceConfig()
        
        if args.validate:
            from benchmarks.validate_kernels import validate_inference
            import copy
            
            original_r = config.model_config.r if isinstance(config.model_config.r, int) else config.model_config.r[0]
            print(f"\n{'='*70}\nVALIDATING FOR r = {original_r}\n{'='*70}")
            validate_inference(config, verbose=args.verbose)
            
            if original_r != 0:
                print(f"\n{'='*70}\nVALIDATING FOR r = 0\n{'='*70}")
                config_r0 = copy.deepcopy(config)
                config_r0.model_config.r = 0
                validate_inference(config_r0, verbose=args.verbose)
        else:
            from training.inference import do_inference
            do_inference(config, inference_cfg, use_debug=args.train, latest=args.latest_ckp)
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
    elif args.command == "benchmark":
        if args.induction:
            from benchmarks.benchmark_induction import run_induction_experiment
            run_induction_experiment()
        elif args.copy:
            from benchmarks.benchmark_copying import run_copying_experiment
            run_copying_experiment()
        else:
            print("Please specify either --induction or --copy to run a benchmark.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
