# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Config, load_config
from dataset import MyData
from loss import EdgeDiceLoss, StructureLoss
from models.ESCNet import ESCNet
from utils import Logger, AverageMeter, set_seed


def setup_ddp(rank: int, world_size: int):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    init_process_group(backend="nccl", init_method="env://")


def cleanup_ddp():
    destroy_process_group()


class Trainer:
    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = self.config.device_ids[self.rank]

        torch.cuda.set_device(self.device)
        if self.config.precisionHigh:
            torch.set_float32_matmul_precision("high")
        
        os.makedirs(os.path.join(self.config.save_model_dir, self.config.name), exist_ok=True)

        self.logger = (
            Logger(config, os.path.join(self.config.save_model_dir, self.config.name, "log.txt"))
            if self.rank == 0
            else None
        )

        self.log(f"Trainer initialized on rank {self.rank} with device {self.device}.")
        self.log(f"Full config:\n{self.config.model_dump_json(indent=2)}")

        self.train_loader = self._prepare_dataloader()
        self.model, self.optimizer, self.lr_scheduler = (
            self._prepare_model_and_optimizer()
        )

        self.scaler = GradScaler()
        self.structure_loss = StructureLoss().to(self.device)
        self.dice_loss = EdgeDiceLoss().to(self.device)
        self.loss_log = AverageMeter()

    def log(self, message: str):
        if self.rank == 0:
            self.logger.info(message)

    def _prepare_dataloader(self) -> DataLoader:
        dataset = MyData(
            config,
            dataset_dir=self.config.train_dir,
            image_size=self.config.img_size,
            is_train=True,
        )
        sampler = DistributedSampler(dataset) if self.config.is_ddp else None

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=min(self.config.num_workers, self.config.batch_size),
            pin_memory=True,
            shuffle=(sampler is None),  # DDP模式下shuffle由sampler控制
            sampler=sampler,
            drop_last=True,
        )
        self.log(f"{len(loader)} batches of train dataloader created.")
        return loader

    def _prepare_model_and_optimizer(self):
        model = ESCNet(self.config, pretrained=True).to(self.device)
        if self.config.is_ddp:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.device])

        if self.config.compile:
            model = torch.compile(model, mode="reduce-overhead")
            self.log("Model compiled with torch.compile.")

        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=self.config.lr / 10
        )

        self.log("Model, optimizer, and scheduler have been initialized.")
        return model, optimizer, lr_scheduler

    def _save_checkpoint(self, epoch: int):
        if self.rank != 0:
            return  # 只有主进程保存模型

        if (
            epoch >= self.config.epochs - self.config.save_last
            and epoch % self.config.save_step == 0
        ):

            model_state = (
                self.model.module.state_dict()
                if self.config.is_ddp
                else self.model.state_dict()
            )
            save_path = os.path.join(self.config.save_model_dir, self.config.name, f"epoch_{epoch}.pth")
            torch.save(model_state, save_path)
            self.log(f"Checkpoint saved to {save_path}")

    def train_epoch(self, epoch: int):
        self.model.train()
        self.loss_log.reset()

        if self.config.is_ddp:
            self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, (inputs, gts, edges) in enumerate(self.train_loader):
            inputs, gts, edges = (
                inputs.to(self.device),
                gts.to(self.device),
                edges.to(self.device),
            )

            with autocast(device_type="cuda", dtype=torch.float32):
                out_edge, out_pred_masks = self.model(inputs)

                loss_dice = self.dice_loss(out_edge, edges)

                loss_structure = 0.0
                gts = torch.clamp(gts, 0, 1)
                factors = [0.5, 0.7, 0.9, 1.1]

                for i, pred_lvl in enumerate(out_pred_masks):
                    pred_lvl_resized = nn.functional.interpolate(
                        pred_lvl,
                        size=gts.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss_structure += (
                        self.structure_loss(pred_lvl_resized, gts) * factors[i]
                    )

                total_loss = loss_structure + loss_dice

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.loss_log.update(total_loss.item(), inputs.size(0))

            if self.rank == 0 and batch_idx % 50 == 0:
                log_msg = (
                    f"Epoch[{epoch}/{self.config.epochs}] Iter[{batch_idx}/{len(self.train_loader)}] | "
                    f"Total Loss: {total_loss.item():.3f} | "
                    f"Structure Loss: {loss_structure.item():.3f} | "
                    f"Edge Loss: {loss_dice.item():.3f}"
                )
                self.log(log_msg)

        self.log(
            f"@==Final== Epoch[{epoch}/{self.config.epochs}] Avg Training Loss: {self.loss_log.avg:.3f}"
        )
        self.lr_scheduler.step()

    def train(self):
        self.log("Starting training process...")
        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(epoch)
            self._save_checkpoint(epoch)
        self.log("Training finished.")


def main(config: Config):
    if config.is_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = len(config.device_ids)
        setup_ddp(rank, world_size)
    else:
        rank, world_size = 0, 1

    trainer = Trainer(config, rank, world_size)
    trainer.train()

    if config.is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESCNet Training Script")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.rand_seed)

    if config.multi_GPU:
        os.environ["OMP_NUM_THREADS"] = "4"
        main(config)
    else:  # 单卡模式
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_ids[0])
        main(config)
