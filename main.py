from src.utils.platform_utils import configure_pytorch_allocator
configure_pytorch_allocator()
import json
import argparse
import os

from training.training_config import (
    DatasetType,
    YaRNConfig,
    get_model_config,
    make_70M_MHA, make_70M_HOFA, make_70M_HOFA_flat32, 
    make_70M_HOFA_depth_axis, make_70M_HOFA_width_axis, make_70M_HOFA_fixed_blend,
    make_70M_MHA_low_rank_k, make_70M_HOFA_increasing
)



def main():
    parser = argparse.ArgumentParser(description="HOFA Project Main Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_train = subparsers.add_parser("train", help="Run training, plotting, dataset downloading, or zero-shot evaluation")
    parser_train.add_argument("--scale", type=str, choices=["13M", "30M", "70M", "125M", "350M"], default="350M", help="Model scale to train or evaluate (default: 350M)")
    parser_train.add_argument("--cpt", action="store_true", help="Continual pretraining mode at 16k context on PG19")
    parser_train.add_argument("--model", type=str, choices=["hofa", "mha", "hofa_alpha", "mha_low_rank_k", "hofa_increasing"], default="hofa", help="Model variant to train (default: hofa)")
    parser_train.add_argument("--alpha", action="store_true", help="Enable identity-initialized learnable channel-wise decay (log_alpha)")
    parser_train.add_argument("--dataset", type=str, choices=[d.value for d in DatasetType], default=None, help="Dataset to download/tokenize (default: fineweb, or pg19 if --cpt is set)")
    parser_train.add_argument("--download-data", action="store_true", help="Download and cache the dataset for training")
    parser_train.add_argument("--val-only", action="store_true", help="Download and cache only the validation dataset")
    parser_train.add_argument("--plot", action="store_true", help="Plot training metrics")
    parser_train.add_argument("--shared", action="store_true", help="Plot shared training metrics for HOFA and MHA")
    parser_train.add_argument("--70m", dest="plot_70m", action="store_true", help="Plot shared training metrics for all 70M ablations")
    parser_train.add_argument("--eval", action="store_true", help="Run zero-shot reasoning evaluation on the active trained model checkpoint")

    parser_eval = subparsers.add_parser("eval", help="Run zero-shot reasoning benchmark evaluations")
    parser_eval.add_argument("--model", type=str, choices=["hofa", "mha", "hofa_alpha", "mha_low_rank_k", "hofa_increasing"], default="hofa", help="Model variant to evaluate (default: hofa)")
    parser_eval.add_argument("--full", action="store_true", help="Evaluate pretrained baseline models from Hugging Face")
    parser_eval.add_argument("--shared", action="store_true", help="Evaluate both HOFA and MHA variants together (e.g. 125M_HOFA & 125M_MHA)")
    parser_eval.add_argument("--scale", type=str, choices=["70M", "125M", "350M"], default="125M", help="Model scale to evaluate (default: 125M)")
    parser_eval.add_argument("--simple", action="store_true", help="Only run the HellaSwag benchmark instead of the full suite")
    parser_eval.add_argument("--limit", type=int, default=None, help="Sample evaluation limit per task (default: None for full evaluation, specify e.g. 1000 for quick testing)")
    parser_eval.add_argument("--force", action="store_true", help="Force rerun evaluation skipping cache")
    
    parser_infer = subparsers.add_parser("infer", help="Run interactive generation or validation")
    parser_infer.add_argument("--scale", type=str, choices=["70M", "125M", "350M"], default="350M", help="Model scale for inference or validation (default: 350M)")
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

    parser_bench = subparsers.add_parser("benchmark", aliases=["bench"], help="Run synthetic capability benchmarks")
    parser_bench.add_argument("--induction", action="store_true", help="Run the Induction Head capability benchmark")
    parser_bench.add_argument("--induction-degradation", action="store_true", help="Run the r=10 extended context degradation benchmark")
    parser_bench.add_argument("--copy", action="store_true", help="Run the Sequential Copying capability benchmark")
    parser_bench.add_argument("--alpha", action="store_true", help="Run the Alpha (mixing gate) distribution analysis, or include learnable-alpha HOFA with --extrapolation")
    parser_bench.add_argument("--keff", action="store_true", help="Run the K_eff Probability Mass benchmark")
    parser_bench.add_argument("--models", type=str, default=None, help="Comma-separated list of HuggingFace models for keff benchmark")
    parser_bench.add_argument("--distance", action="store_true", help="Run the Effective Attention Distance benchmark")
    parser_bench.add_argument("--extrapolation", dest="extrapolation", action="store_true", help="Run the context length extrapolation benchmark")
    parser_bench.add_argument("--temp-yarn", dest="temp_yarn", action="store_true", help="Run the position-segment pre-softmax temperature scaling benchmark (YaRN Fig. 5)")
    parser_bench.add_argument("--model", type=str, choices=["both", "hofa", "mha", "hofa_alpha", "all"], default="both", help="Model variant to evaluate (default: both)")
    parser_bench.add_argument("--scale", type=str, choices=["70M", "125M", "350M", "all"], default="350M", help="Model scale for benchmark evaluation (default: 350M, or 'all' for cross-scale)")
    parser_bench.add_argument("--max-books", type=int, default=None, help="Maximum number of eligible PG-19 test books to evaluate (default: None for all eligible books)")
    parser_bench.add_argument("--force", action="store_true", help="Force re-run and bypass cached results")
    parser_bench.add_argument("--cpt", action=argparse.BooleanOptionalAction, default=True, help="Evaluate CPT checkpoints (default: True)")
    parser_bench.add_argument("--base", action="store_true", help="Evaluate base (non-CPT) checkpoints (equivalent to --no-cpt)")
    parser_bench.add_argument("--plot", action="store_true", help="Plot benchmark results")
    parser_bench.add_argument("--combined", action="store_true", help="With --extrapolation: plot base vs. extended-pretraining (CPT) side by side from caches")

    args = parser.parse_args()
    model_arg = getattr(args, 'model', 'hofa')
    if (args.command == "train" and getattr(args, 'alpha', False)) or (args.command in ("benchmark", "bench") and getattr(args, 'extrapolation', False) and getattr(args, 'alpha', False)) or model_arg == "hofa_alpha":
        model_arg = "hofa_alpha"
    elif model_arg not in ["hofa", "mha", "both", "all", "mha_low_rank_k", "hofa_increasing"]:
        model_arg = "hofa"

    scale_arg = getattr(args, 'scale', '350M')
    config = None
    if scale_arg != "all":
        try:
            config = get_model_config(
                scale=scale_arg,
                model=model_arg,
                is_cpt=getattr(args, 'cpt', False) and not getattr(args, 'base', False)
            )
        except Exception:
            config = None

    if args.command == "train":
        if args.download_data:
            chosen_dataset = DatasetType(args.dataset) if args.dataset else (DatasetType.PG19 if getattr(args, 'cpt', False) else DatasetType.FINEWEB)
            if chosen_dataset == DatasetType.PG19:
                from training.download_pg19 import download_and_tokenize_pg19
                download_and_tokenize_pg19(config)
            else:
                from training.download_fineweb import download_and_tokenize
                download_and_tokenize(config, val_only=args.val_only)
            os._exit(0)
        elif args.eval:
            from benchmarks.benchmarks_configs import EvalExperimentConfig
            from benchmarks.benchmark_zeroshot import evaluate_zeroshot
            eval_cfg = EvalExperimentConfig()
            if args.shared:
                scale = getattr(args, 'scale', '125M')
                eval_configs = [get_model_config(scale, "hofa"), get_model_config(scale, "mha")]
                evaluate_zeroshot(eval_configs, eval_cfg)
            else:
                evaluate_zeroshot(config, eval_cfg, True)
        elif args.plot:
            from training.plot_training import plot_training_metrics
            if args.plot_70m:
                plot_training_metrics([
                    make_70M_MHA(), make_70M_HOFA(), make_70M_HOFA_flat32(),
                    make_70M_HOFA_depth_axis(), make_70M_HOFA_width_axis(), make_70M_HOFA_fixed_blend(),
                    make_70M_MHA_low_rank_k(), make_70M_HOFA_increasing()
                ], subdirectory="70M")
            elif args.shared:
                scale = getattr(args, 'scale', '350M')
                is_cpt = getattr(args, 'cpt', False)
                h_name = "hofa_alpha" if getattr(args, 'alpha', False) else "hofa"
                h_cfg = get_model_config(scale=scale, model=h_name, is_cpt=is_cpt)
                m_cfg = get_model_config(scale=scale, model="mha", is_cpt=is_cpt)
                subdir = f"{scale}_CPT" if is_cpt else scale
                plot_training_metrics([h_cfg, m_cfg], subdirectory=subdir)
            else:
                plot_training_metrics(config)
        else:
            if getattr(args, 'cpt', False):
                from training.train_cpt import train_cpt
                train_cpt(config)
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
            scale = getattr(args, 'scale', '125M')
            eval_configs = [get_model_config(scale, "hofa"), get_model_config(scale, "mha")]
            evaluate_zeroshot(eval_configs, eval_cfg, is_simple=args.simple)
        else:
            from benchmarks.benchmark_zeroshot import evaluate_zeroshot
            scale = getattr(args, 'scale', '350M')
            eval_config = get_model_config(scale, getattr(args, 'model', 'hofa'))
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
    elif args.command in ("benchmark", "bench"):
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
        elif args.alpha and not args.extrapolation:
            from benchmarks.analyze_alpha import run_alpha_analysis
            run_alpha_analysis(config)
        elif args.keff:
            from benchmarks.benchmarks_configs import KEffExperimentConfig
            from benchmarks.plotting.plot_K_eff import run_plot_k_eff
            
            keff_cfg = KEffExperimentConfig()
            if getattr(args, 'models', None):
                keff_cfg.models = [m.strip() for m in args.models.split(",") if m.strip()]
            
            if args.plot:
                run_plot_k_eff(keff_cfg)
            else:
                from benchmarks.benchmark_K_eff import run_k_eff_experiment
                run_k_eff_experiment(keff_cfg)
        elif args.distance:
            if args.plot:
                from benchmarks.plotting.plot_distance import run_plot_distance
                run_plot_distance()
            else:
                from benchmarks.benchmark_distance import run_distance_experiment
                run_distance_experiment()
        elif args.extrapolation and getattr(args, 'combined', False):
            from benchmarks.plotting.plot_extensions import plot_extensions
            plot_extensions(scale=getattr(args, 'scale', '125M'))
        elif args.extrapolation:
            from benchmarks.benchmark_long_context import run_long_context_experiment, plot_long_context
            from benchmarks.benchmarks_configs import LongContextExtrapolationConfig, make_cross_scale_base_extrapolation_config
            scale = getattr(args, 'scale', '350M')
            model_flag = getattr(args, 'model', 'both').lower()
            is_cpt = getattr(args, 'cpt', True) and not getattr(args, 'base', False)
            max_books = getattr(args, 'max_books', None)
            force_rerun = getattr(args, 'force', False)

            if scale == "all":
                extrap_cfg = make_cross_scale_base_extrapolation_config(
                    scales=("70M", "125M", "350M"),
                    max_books=max_books,
                    force_rerun=force_rerun,
                )
            else:
                scale_models = []
                use_alpha = getattr(args, 'alpha', False) or model_flag == "hofa_alpha"
                h_name = "hofa_alpha" if use_alpha else "hofa"

                if model_flag in ["hofa", "hofa_alpha", "both", "all"]:
                    h_cfg = get_model_config(scale, h_name, is_cpt=is_cpt)
                    scale_models.append(h_cfg if isinstance(h_cfg, YaRNConfig) else YaRNConfig(model=h_cfg, original_max_seq_len=getattr(h_cfg, "seq_len", 1024)))
                if model_flag in ["mha", "both", "all"]:
                    m_cfg = get_model_config(scale, "mha", is_cpt=is_cpt)
                    scale_models.append(m_cfg if isinstance(m_cfg, YaRNConfig) else YaRNConfig(model=m_cfg, original_max_seq_len=getattr(m_cfg, "seq_len", 1024)))
                if model_flag == "all":
                    other_h = "hofa" if use_alpha else "hofa_alpha"
                    ha_cfg = get_model_config(scale, other_h, is_cpt=is_cpt)
                    scale_models.append(ha_cfg if isinstance(ha_cfg, YaRNConfig) else YaRNConfig(model=ha_cfg, original_max_seq_len=getattr(ha_cfg, "seq_len", 1024)))

                cache_name = f"{scale}_long_context_extrapolation_results.json" if is_cpt else f"{scale}_base_extrapolation_64k.json"
                plot_name = f"{scale}_long_context_extrapolation_results" if is_cpt else f"{scale}_base_extrapolation_64k"
                extrap_cfg = LongContextExtrapolationConfig(
                    models=scale_models,
                    max_books=max_books,
                    cache_file_name=cache_name,
                    plot_file_name=plot_name,
                    use_fixed_yarn=is_cpt,
                    force_rerun=force_rerun,
                )

            if args.plot:
                plot_long_context(extrap_cfg)
            else:
                run_long_context_experiment(extrap_cfg)
        elif getattr(args, 'temp_yarn', False):
            from benchmarks.benchmark_temperature_YaRN import run_temperature_yarn_experiment
            from benchmarks.plotting.plot_temperature_YaRN import plot_temperature_yarn
            from benchmarks.benchmarks_configs import TemperatureYaRNConfig
            scale = getattr(args, 'scale', '350M')
            model_flag = getattr(args, 'model', 'both').lower()
            is_cpt = getattr(args, 'cpt', True) and not getattr(args, 'base', False)
            use_alpha = getattr(args, 'alpha', False) or model_flag == "hofa_alpha"

            eval_models = []
            h_name = "hofa_alpha" if use_alpha else "hofa"
            if model_flag in ["hofa", "hofa_alpha", "both", "all"]:
                h_cfg = get_model_config(scale, h_name, is_cpt=is_cpt)
                eval_models.append(h_cfg if isinstance(h_cfg, YaRNConfig) else YaRNConfig(model=h_cfg))
            if model_flag in ["mha", "both", "all"]:
                m_cfg = get_model_config(scale, "mha", is_cpt=is_cpt)
                eval_models.append(m_cfg if isinstance(m_cfg, YaRNConfig) else YaRNConfig(model=m_cfg))
            if model_flag == "all":
                other_h = "hofa" if use_alpha else "hofa_alpha"
                ha_cfg = get_model_config(scale, other_h, is_cpt=is_cpt)
                eval_models.append(ha_cfg if isinstance(ha_cfg, YaRNConfig) else YaRNConfig(model=ha_cfg))

            cache_name = f"{scale}_temperature_yarn_results.json"
            plot_name = f"{scale}_temperature_scaling"
            temp_cfg = TemperatureYaRNConfig(
                models=eval_models,
                cache_file_name=cache_name,
                plot_file_name=plot_name,
                force_rerun=getattr(args, 'force', False),
            )
            if args.plot:
                plot_temperature_yarn(temp_cfg)
            else:
                run_temperature_yarn_experiment(temp_cfg)
                plot_temperature_yarn(temp_cfg)
        else:
            print("Please specify a valid benchmark flag (e.g. --induction, --copy, --alpha, --keff, --distance, --extrapolation, or --temp-yarn).")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
