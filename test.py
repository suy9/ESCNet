import os
import argparse
from glob import glob
import pdb
from tqdm import tqdm
from models.ESCNet import ESCNet
import cv2
import torch
import multiprocessing as mp

from dataset import MyData
from utils import save_tensor_img, check_state_dict
from config import load_config



def inference(model, data_loader_test, pred_root, method, device):
    model.eval()
    for batch in (
        tqdm(data_loader_test, total=len(data_loader_test))
    ):
        inputs = batch[0].to(device)
        gts = batch[1].to(device)
        label_paths = batch[-1]
        os.makedirs(os.path.join(pred_root, method), exist_ok=True)
        with torch.no_grad():
            out_edge, scaled_preds = model(inputs)
        pred_lvl = (scaled_preds[-1].sigmoid() >= 0.5).float()

        for idx_sample in range(pred_lvl.shape[0]):
            res = torch.nn.functional.interpolate(
                pred_lvl[idx_sample].unsqueeze(0),
                size=cv2.imread(
                    label_paths[idx_sample], cv2.IMREAD_GRAYSCALE
                ).shape[:2],
                mode="bilinear",
                align_corners=True,
            )
            save_tensor_img(
                res,
                os.path.join(
                    os.path.join(pred_root, method),
                    label_paths[idx_sample].replace("\\", "/").split("/")[-1],
                ),
            ) 
    return None


def load_and_infer_on_gpu(model, weights, data_loader_test, pred_root, testset, device):
    # Load model weights
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(
        state_dict
    )  
    
    inference(
        model,
        data_loader_test=data_loader_test,
        pred_root=pred_root,
        method=weights.split(os.sep)[-1].rstrip(".pth"),
        device=device,
    )

    del state_dict
    torch.cuda.empty_cache()


def run_inference_on_gpu(config, device_id, weights_lst, data_loader_test, pred_root, testset):
    # Assign the current GPU
    current_device = f"cuda:{device_id}"
    model = ESCNet(config ,pretrained=False).to(current_device)

    # Load and infer each model assigned to this GPU
    for weights in weights_lst:
        print(f"Running inference on GPU {current_device} with model {weights}")
        load_and_infer_on_gpu(
            model, weights, data_loader_test, pred_root, testset, current_device
        )


def main(args):
    config = load_config(args.config)
    
    device_ids = list(config.device_ids) 
    
    num_gpus = config.device_ids.__len__()
    
    ckpt_folder = os.path.join(config.save_model_dir, config.name)
    
    if config.precisionHigh:
        torch.set_float32_matmul_precision("high")

    if args.ckpt:
        print("Testing with model {}".format(args.ckpt))
    else:
        print("Testing with models in {}".format(ckpt_folder))

    weights_lst = sorted(
        ([args.ckpt] if args.ckpt else glob(os.path.join(ckpt_folder, "*.pth"))),
        key=lambda x: int(x.split("epoch_")[-1].split(".pth")[0]),
        reverse=True,
    )

    test_set = config.test_dir
    print(f">>>> Test_set: {test_set}...")

    data_loader_test = torch.utils.data.DataLoader(
        dataset=MyData(config, test_set, image_size=config.img_size, is_train=False),
        batch_size=config.batch_size_valid,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    models_per_gpu = len(weights_lst) // num_gpus
    extra_models = len(weights_lst) % num_gpus

    gpu_model_assignments = []
    for i, device_id in enumerate(device_ids):
        models_for_gpu = weights_lst[i * models_per_gpu : (i + 1) * models_per_gpu]
        if i < extra_models:
            models_for_gpu.append(weights_lst[num_gpus * models_per_gpu + i])
        gpu_model_assignments.append((device_id, models_for_gpu))

    processes = []
    for device_id, models_for_gpu in gpu_model_assignments:
        p = mp.Process(
            target=run_inference_on_gpu,
            args=(
                config,
                device_id,
                models_for_gpu,
                data_loader_test,
                args.pred_root,
                test_set,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="ESCNet Test Script")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a specific checkpoint.")
    parser.add_argument("--pred_root", default="preds", type=str, help="Output folder")
    args = parser.parse_args()
    main(args)
