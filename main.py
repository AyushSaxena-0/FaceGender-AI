"""
Unified CLI Entrypoint for Face Presentation Classifier.
Usage:
    python main.py web             # Launch Gradio Web UI
    python main.py api             # Launch FastAPI REST API
    python main.py train           # Train Model on RTX 4050 GPU
    python main.py profile         # Profile RTX 4050 Hardware & VRAM
    python main.py benchmark       # Run Inference Latency & FPS Benchmark
    python main.py debug           # Run Complete 11-Step ML Pipeline Audit
    python main.py generate-data   # Generate Synthetic Dataset
    python main.py export          # Export Model to ONNX & TorchScript
    python main.py compare         # Benchmark Model Backbones
    python main.py error-analysis  # Perform Error Analysis
"""

import sys
import argparse
import uvicorn
from utils.logger import setup_logger

logger = setup_logger("MainCLI")


def launch_web(host: str = "0.0.0.0", port: int = 7860):
    from gradio_app.app import build_gradio_ui
    logger.info(f"Launching Gradio Web App on http://{host}:{port} ...")
    app = build_gradio_ui()
    app.queue().launch(server_name=host, server_port=port, share=False)


def launch_api(host: str = "0.0.0.0", port: int = 8000):
    logger.info(f"Launching FastAPI Server on http://{host}:{port} ...")
    uvicorn.run("api.app:app", host=host, port=port, reload=False)


def run_train_cmd(config_path: str = "configs/config.yaml"):
    from training.train import run_rtx4050_training
    logger.info("Starting RTX 4050 GPU model training pipeline...")
    run_rtx4050_training(config_path)


def run_profile_cmd():
    from utils.hardware_info import profile_hardware
    logger.info("Profiling RTX 4050 Hardware & VRAM Memory...")
    profile_hardware()


def run_benchmark_cmd(config_path: str = "configs/config.yaml"):
    import yaml
    from inference.pipeline import FacePresentationPipeline
    from validation.benchmark import benchmark_inference

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = FacePresentationPipeline(config_path=config_path)
    input_size = tuple(config.get("dataset", {}).get("input_size", [224, 224]))
    benchmark_inference(pipeline.model, input_size, str(pipeline.device))


def run_debug_cmd(config_path: str = "configs/config.yaml"):
    from utils.audit_pipeline import run_pipeline_audit
    logger.info("Executing 11-Step ML Pipeline Debugging Audit...")
    run_pipeline_audit(config_path)


def run_data_gen(dataset_path: str = "dataset"):
    from data.generate_sample_data import generate_dataset_structure
    logger.info(f"Generating synthetic dataset inside '{dataset_path}'...")
    generate_dataset_structure(dataset_path)


def run_export_cmd(config_path: str = "configs/config.yaml"):
    import yaml
    from inference.pipeline import FacePresentationPipeline
    from models.exporter import ModelExporter

    logger.info("Exporting trained model...")
    pipeline = FacePresentationPipeline(config_path=config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    ts_path = paths.get("torchscript_model_path", "weights/model.pt")
    onnx_path = paths.get("onnx_model_path", "weights/model.onnx")

    exporter = ModelExporter(
        model=pipeline.model,
        input_size=tuple(config.get("dataset", {}).get("input_size", [224, 224])),
        device=str(pipeline.device)
    )

    exporter.export_torchscript(ts_path)
    exporter.export_onnx(onnx_path)
    logger.info("Export complete.")


def run_compare_cmd(config_path: str = "configs/config.yaml"):
    from utils.model_comparison import compare_models
    logger.info("Running model comparison benchmark across backbones...")
    compare_models(config_path)


def run_error_analysis_cmd(config_path: str = "configs/config.yaml"):
    import yaml
    from datasets.dataset_merger import merge_and_split_datasets
    from datasets.multi_dataset import RTX4050FaceDataset
    from torch.utils.data import DataLoader
    from training.rtx4050_trainer import RTX4050Trainer
    from validation.error_logger import log_misclassified_images

    logger.info("Running error analysis...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _, val_samples, _ = merge_and_split_datasets()
    val_dataset = RTX4050FaceDataset(val_samples, tuple(config.get("dataset", {}).get("input_size", [224, 224])), "val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    trainer = RTX4050Trainer(config)
    log_misclassified_images(trainer.model, val_loader, trainer.device)


def main():
    parser = argparse.ArgumentParser(description="Face Presentation Classifier Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    subparsers.add_parser("web", help="Launch Gradio Web Interface")
    subparsers.add_parser("api", help="Launch FastAPI Server")

    train_p = subparsers.add_parser("train", help="Train Model on RTX 4050 GPU")
    train_p.add_argument("--config", type=str, default="configs/config.yaml")

    subparsers.add_parser("profile", help="Profile RTX 4050 Hardware & VRAM")
    subparsers.add_parser("benchmark", help="Run Inference Benchmark (Latency & FPS)")

    debug_p = subparsers.add_parser("debug", help="Run 11-Step ML Pipeline Audit")
    debug_p.add_argument("--config", type=str, default="configs/config.yaml")

    data_p = subparsers.add_parser("generate-data", help="Generate Synthetic Dataset")
    data_p.add_argument("--dir", type=str, default="dataset")

    export_p = subparsers.add_parser("export", help="Export Model to TorchScript / ONNX")
    export_p.add_argument("--config", type=str, default="configs/config.yaml")

    subparsers.add_parser("audit", help="Run Dataset Audit & Generate Report")
    subparsers.add_parser("compare", help="Benchmark Model Backbones")
    subparsers.add_parser("error-analysis", help="Perform Error Analysis")

    args = parser.parse_args()

    if args.command == "web":
        launch_web()
    elif args.command == "api":
        launch_api()
    elif args.command == "train":
        run_train_cmd(args.config)
    elif args.command == "profile":
        run_profile_cmd()
    elif args.command == "benchmark":
        run_benchmark_cmd(args.config if hasattr(args, "config") else "configs/config.yaml")
    elif args.command == "debug" or args.command == "audit":
        run_debug_cmd(args.config if hasattr(args, "config") else "configs/config.yaml")
    elif args.command == "generate-data":
        run_data_gen(args.dir)
    elif args.command == "export":
        run_export_cmd(args.config)
    elif args.command == "compare":
        run_compare_cmd()
    elif args.command == "error-analysis":
        run_error_analysis_cmd()
    else:
        launch_web()


if __name__ == "__main__":
    main()
