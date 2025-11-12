import os
import argparse
from glob import glob
import prettytable as pt
from concurrent.futures import ThreadPoolExecutor
from metrics import evaluator
from config import load_config
import threading


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ESCNet Evaluation Script")
    parser.add_argument(
        "--pred_root", type=str, help="Prediction root", default="preds"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Save directory for evaluation results",
        default="results",
    )
    parser.add_argument(
        "--check_integrity", type=bool, help="Check file integrity", default=True
    )
    parser.add_argument(
        "--model_lst",
        type=str,
        help="Comma-separated list of models to evaluate",
        default=None,
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        help="Number of threads for parallel evaluation",
        default=4,
    )
    parser.add_argument(
    "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()

    args.metrics = "+".join(["S", "MAE", "E", "F", "WF", "MSE"])
    args.model_folder = args.pred_root
    args.model_lst = [
        d
        for d in os.listdir(args.model_folder)
        if os.path.isdir(os.path.join(args.model_folder, d))
    ]
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def check_file_integrity(args, config):
    """Check the integrity of ground-truth and prediction files."""
    if args.check_integrity:
        print("Checking file integrity...")

        for model_name in args.model_lst:
            gt_path = os.path.join(config.test_dir, "GT_Object")
            pred_path = os.path.join(args.pred_root, model_name)

            gt_files = sorted(os.listdir(gt_path))
            pred_files = sorted(os.listdir(pred_path))

            if len(gt_files) != len(pred_files):
                print(
                    f"Ground-truth files: {len(gt_files)}, Prediction files: {len(pred_files)}"
                )
                print("File count mismatch between ground-truth and predictions!")
            else:
                print(f"Model {model_name}: File integrity check passed.")
    else:
        print("Skipping file integrity check.")


def evaluate_model(model_name, gt_paths, args, tb):
    print(f"Evaluating model: {model_name}")
    pred_paths = [
        p.replace(
            os.path.join(config.test_dir),
            # os.path.join(config.test_dir, args.gt_root),
            os.path.join(args.pred_root, model_name),
        ).replace("/GT_Object/", "/")
        for p in gt_paths
    ]

    em, sm, fm, mae, wfm, mba, biou = evaluator(
        gt_paths=gt_paths,
        pred_paths=pred_paths,
        metrics=args.metrics.split("+"),
    )

    scores = [
        sm.round(3),
        wfm.round(3),
        fm["curve"].mean().round(3),
        em["curve"].mean().round(3),
        mae.round(3),
    ]
    scores = [
        f".{score:.3f}".split(".")[-1] if score <= 1 else f"{score:<4}"
        for score in scores
    ]
    tb.add_row([model_name] + scores)
    return tb


def evaluate_models_parallel(args, config):
    lock = threading.Lock()
    print("Starting evaluation...")
    gt_src = os.path.join(config.test_dir)
    print(f"Ground-truth source: {gt_src}")
    gt_paths = sorted(glob(os.path.join(gt_src, "GT_Object", "*")))

    tb = pt.PrettyTable()
    tb.vertical_char = "&"
    tb.field_names = [
        "Method",
        "Smeasure",
        "wFmeasure",
        "meanFm",
        "meanEm",
        "MAE",
    ]

    n_threads = args.n_threads
    model_chunks = [args.model_lst[i::n_threads] for i in range(n_threads)]

    def worker(model_chunk):
        local_tb = pt.PrettyTable()
        local_tb.field_names = tb.field_names
        for model_name in model_chunk:
            local_tb = evaluate_model(model_name, gt_paths, args, local_tb)
        return local_tb

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = executor.map(worker, model_chunks)

    all_rows = []
    for result in results:
        all_rows.extend(result._rows)

    all_rows.sort(key=lambda row: row[0])

    tb.add_rows(all_rows)
    print(tb)
    results_file = os.path.join(args.save_dir, "result.txt")
    with lock:
        with open(results_file, "a+") as file:
            file.write(str(tb) + "\n")
    


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    check_file_integrity(args, config)
    evaluate_models_parallel(args, config)
