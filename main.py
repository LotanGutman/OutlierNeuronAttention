from training.training_config import (
    make_70M_MHA, make_70M_HOFA, make_70M_HOFA_flat32, 
    make_70M_HOFA_depth_axis, make_70M_HOFA_width_axis, make_70M_HOFA_fixed_blend, 
    make_125M_HOFA, make_125M_MHA, make_350M_HOFA, make_350M_MHA
)


import argparse


def main():
    config = make_350M_HOFA()

    parser = argparse.ArgumentParser(description="HOFA Project Main Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_train = subparsers.add_parser("train", help="Run training, plotting, dataset downloading, or zero-shot evaluation")
    parser_train.add_argument("--download-data", action="store_true", help="Download and cache the dataset for training")
    parser_train.add_argument("--plot", action="store_true", help="Plot training metrics")
    parser_train.add_argument("--shared", action="store_true", help="Plot shared training metrics for HOFA and MHA")
    parser_train.add_argument("--70m", dest="plot_70m", action="store_true", help="Plot shared training metrics for all 70M ablations")
    parser_train.add_argument("--eval", action="store_true", help="Run zero-shot reasoning evaluation on the active trained model checkpoint")

    parser_eval = subparsers.add_parser("eval", help="Run zero-shot reasoning benchmark evaluations")
    parser_eval.add_argument("--full", action="store_true", help="Evaluate pretrained baseline models from Hugging Face")
    parser_eval.add_argument("--shared", action="store_true", help="Evaluate both HOFA and MHA variants together (e.g. 125M_HOFA & 125M_MHA)")
    parser_eval.add_argument("--scale", type=str, choices=["70M", "125M", "350M"], default="125M", help="Model scale to evaluate (default: 125M)")
    parser_eval.add_argument("--simple", action="store_true", help="Only run the HellaSwag benchmark instead of the full suite")
    parser_eval.add_argument("--limit", type=int, default=None, help="Sample evaluation limit per task (default: None for full evaluation, specify e.g. 1000 for quick testing)")
    parser_eval.add_argument("--force", action="store_true", help="Force rerun evaluation skipping cache")
    
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
    parser_profile.add_argument("--plot-shared", action="store_true", help="Plot a shared 2x3 figure of both prefill and decode results")

    parser_bench = subparsers.add_parser("benchmark", help="Run synthetic capability benchmarks")
    parser_bench.add_argument("--induction", action="store_true", help="Run the Induction Head capability benchmark")
    parser_bench.add_argument("--induction-degradation", action="store_true", help="Run the r=10 extended context degradation benchmark")
    parser_bench.add_argument("--copy", action="store_true", help="Run the Sequential Copying capability benchmark")
    parser_bench.add_argument("--alpha", action="store_true", help="Run the Alpha (gate) distribution analysis")
    parser_bench.add_argument("--keff", action="store_true", help="Run the K_eff Probability Mass benchmark")
    parser_bench.add_argument("--distance", action="store_true", help="Run the Effective Attention Distance benchmark")
    parser_bench.add_argument("--plot", action="store_true", help="Plot benchmark results")

    args = parser.parse_args()

    if args.command == "train":
        if args.download_data:
            from training.download_fineweb import download_and_tokenize
            download_and_tokenize(config)
            import os
            print("\n[INFO] Data download completed successfully! Exiting immediately to prevent HuggingFace teardown bugs.")
            os._exit(0)
        elif args.eval:
            from benchmarks.benchmarks_configs import EvalExperimentConfig
            from benchmarks.benchmark_zeroshot import evaluate_zeroshot
            eval_cfg = EvalExperimentConfig()
            if args.shared:
                eval_configs = [make_125M_HOFA(), make_125M_MHA()]
                evaluate_zeroshot(eval_configs, eval_cfg)
            else:
                evaluate_zeroshot(config, eval_cfg, True)
        elif args.plot:
            from training.plot_training import plot_training_metrics
            if args.plot_70m:
                plot_training_metrics([
                    make_70M_MHA(), make_70M_HOFA(), make_70M_HOFA_flat32(),
                    make_70M_HOFA_depth_axis(), make_70M_HOFA_width_axis(), make_70M_HOFA_fixed_blend()
                ], subdirectory="70M")
            elif args.shared:
                plot_training_metrics([make_125M_HOFA(), make_125M_MHA()], subdirectory="125M")
            else:
                plot_training_metrics(config)
        else:
            from training.train import train
            train(config)
    elif args.command == "eval":
        from benchmarks.benchmarks_configs import EvalExperimentConfig
        sample_limit = None if args.limit is None or args.limit <= 0 else args.limit
        eval_cfg = EvalExperimentConfig(limit=sample_limit, force_rerun=args.force)

        if args.full:
            from benchmarks.benchmark_zeroshot import evaluate_pretrained_zeroshot
            evaluate_pretrained_zeroshot(eval_cfg, is_simple=args.simple, scale=args.scale)
        elif args.shared:
            from benchmarks.benchmark_zeroshot import evaluate_zeroshot
            if args.scale == "70M":
                eval_configs = [make_70M_HOFA(), make_70M_MHA()]
            elif args.scale == "350M":
                eval_configs = [make_350M_HOFA(), make_350M_MHA()]
            else:
                eval_configs = [make_125M_HOFA(), make_125M_MHA()]
            evaluate_zeroshot(eval_configs, eval_cfg, is_simple=args.simple)
        else:
            from benchmarks.benchmark_zeroshot import evaluate_zeroshot
            eval_config = config
            if args.scale == "70M":
                eval_config = make_70M_HOFA()
            elif args.scale == "125M":
                eval_config = make_125M_HOFA()
            elif args.scale == "350M":
                eval_config = make_350M_HOFA()
            evaluate_zeroshot(eval_config, eval_cfg, is_simple=args.simple)
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
        if args.plot_shared:
            from benchmarks.plotting.plot_shared_profiling import plot_shared_profiling
            plot_shared_profiling()
        elif args.prefill:
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
            if args.plot:
                from benchmarks.plotting.plot_induction import plot_unified_trendline, plot_feature_norm_disparity
                plot_unified_trendline()
                plot_feature_norm_disparity()
            else:
                from benchmarks.benchmark_induction import run_induction_experiment
                run_induction_experiment()
        elif args.induction_degradation:
            if args.plot:
                from benchmarks.benchmark_induction_degradation import plot_induction_degradation
                plot_induction_degradation()
            else:
                from benchmarks.benchmark_induction_degradation import run_induction_degradation_experiment
                run_induction_degradation_experiment()
        elif args.copy:
            if args.plot:
                from benchmarks.benchmark_copying import plot_copying_experiment
                plot_copying_experiment()
            else:
                from benchmarks.benchmark_copying import run_copying_experiment
                run_copying_experiment()
        elif args.alpha:
            from benchmarks.analyze_alpha import run_alpha_analysis
            run_alpha_analysis(config)
        elif args.keff:
            if args.plot:
                from benchmarks.plotting.plot_K_eff import run_plot_k_eff
                run_plot_k_eff()
            else:
                from benchmarks.benchmark_K_eff import run_k_eff_experiment
                run_k_eff_experiment()
        elif args.distance:
            if args.plot:
                from benchmarks.plotting.plot_distance import run_plot_distance
                run_plot_distance()
            else:
                from benchmarks.benchmark_distance import run_distance_experiment
                run_distance_experiment()
        else:
            print("Please specify either --induction, --copy, --alpha, --keff, or --distance to run a benchmark.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
