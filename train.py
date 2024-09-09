import torch
import yaml
import numpy as np
import random
import os
import sys
import time
import math
import datetime
import argparse

from data.voc_style_data import VOCDetection
from src.backbone import PResNet
from src.encoder import HybridEncoder
from src.decoder import RTDETRTransformerv2
from src.rtdetrv2 import RTDETR
from src.matcher import HungarianMatcher
from src.postprocessor import RTDETRPostProcessor
from src.collator import BatchImageCollateFuncion
from src.criterion import RTDETRCriterionv2
from src.optimizer import ModelEMA, AdamW, MultiStepLR, LinearWarmup
from torch.utils.data import DataLoader
import argparse
from torch.cuda.amp.grad_scaler import GradScaler
from src.utils import get_optim_params
from data.geo_OD_data import GeoImageryODdata


from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        default="./configs/rtdetrv2_r50vd.yml",
        help="path to config file",
    )
    return parser.parse_args()


def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def get_device():
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def make_outdirs(config):
    out_dir = config["output_dir"]
    log_dir = config["log_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return out_dir, log_dir


def get_model(config):
    """Build and return the model based on the configuration file."""
    backbone_params = config["PResNet"]
    encoder_params = config["HybridEncoder"]
    decoder_params = config["RTDETRTransformerv2"]
    num_classes = config["num_classes"]

    backbone = PResNet(
        depth=backbone_params["depth"],
        variant=backbone_params["variant"],
        num_stages=backbone_params["num_stages"],
        return_idx=backbone_params["return_idx"],
        freeze_at=backbone_params["freeze_at"],
        freeze_norm=backbone_params["freeze_norm"],
        pretrained=backbone_params["pretrained"],
    )
    encoder = HybridEncoder(
        in_channels=encoder_params["in_channels"],
        feat_strides=encoder_params["feat_strides"],
        hidden_dim=encoder_params["feat_strides"],
        nhead=encoder_params["feat_strides"],
        dim_feedforward=encoder_params["feat_strides"],
        dropout=encoder_params["feat_strides"],
        enc_act=encoder_params["feat_strides"],
        use_encoder_idx=encoder_params["feat_strides"],
        num_encoder_layers=encoder_params["feat_strides"],
        expansion=encoder_params["feat_strides"],
        depth_mult=encoder_params["feat_strides"],
        act=encoder_params["feat_strides"],
        # eval_spatial_size = [640,640],
    )
    decoder = RTDETRTransformerv2(
        num_classes=num_classes,
        hidden_dim=decoder_params["hidden_dim"],
        num_queries=decoder_params["num_queries"],
        feat_channels=decoder_params["feat_channels"],
        feat_strides=decoder_params["feat_strides"],
        num_levels=decoder_params["num_levels"],
        num_points=decoder_params["num_points"],
        num_layers=decoder_params["num_layers"],
        num_denoising=decoder_params["num_denoising"],
        label_noise_ratio=decoder_params["label_noise_ratio"],
        box_noise_scale=decoder_params["box_noise_scale"],
        eval_idx=decoder_params["eval_idx"],
        aux_loss=decoder_params["aux_loss"],
        cross_attn_method=decoder_params["cross_attn_method"],
        query_select_method=decoder_params["query_select_method"],
        # eval_spatial_size=[640,640],
    )

    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def get_criterion(config):
    """Build and return the criterion based on the configuration file."""
    num_classes = config["num_classes"]
    criterion_params = config["RTDETRCriterionv2"]
    matcher_params = criterion_params["matcher"]
    matcher_weight_dict = matcher_params["weight_dict"]
    matcher_alpha = matcher_params["alpha"]
    matcher_gamma = matcher_params["gamma"]
    criterion_weight_dict = criterion_params["weight_dict"]
    criterion_losses = criterion_params["losses"]
    criterion_alpha = criterion_params["alpha"]
    criterion_gamma = criterion_params["gamma"]
    use_focal_loss = config["use_focal_loss"]
    matcher = HungarianMatcher(
        weight_dict=matcher_weight_dict,
        use_focal_loss=use_focal_loss,
        alpha=matcher_alpha,
        gamma=matcher_gamma,
    )
    return RTDETRCriterionv2(
        matcher=matcher,
        weight_dict=criterion_weight_dict,
        losses=criterion_losses,
        alpha=criterion_alpha,
        gamma=criterion_gamma,
        num_classes=num_classes,
    )


def get_postprocessor(config):
    """Build and return the postprocessor based on the configuration file."""
    num_classes = config["num_classes"]
    num_top_queries = config["num_top_queries"]
    use_focal_loss = config["use_focal_loss"]
    return RTDETRPostProcessor(
        num_classes=num_classes,
        use_focal_loss=use_focal_loss,
        num_top_queries=num_top_queries,
    )


def get_ema(config, model):
    """Build and return the ema based on the configuration file."""
    ema_params = config["ema"]
    ema_decay = ema_params["decay"]
    ema_warmups = ema_params["warmups"]
    return ModelEMA(model=model, decay=ema_decay, warmups=ema_warmups)


def get_scaler():
    return GradScaler


def get_optimizer(config, model):
    optim_conf_params = config["optimizer"]
    model_params = get_optim_params(optim_conf_params, model)
    return AdamW(params=model_params)


def get_lr_schedulers(config, optimizer):
    lr_scheduler_params = config["lr_scheduler"]
    lr_warmup_scheduler_params = config["lr_warmup_scheduler"]
    lr_milestones = lr_scheduler_params["milestones"]
    lr_gamma = lr_scheduler_params["gamma"]
    warmup_duration = lr_warmup_scheduler_params["warmup_duration"]
    lr_scheduler = MultiStepLR(
        optimizer=optimizer, milestones=lr_milestones, gamma=lr_gamma
    )
    lr_warmup_scheduler = LinearWarmup(
        lr_scheduler=lr_scheduler, warmup_duration=warmup_duration
    )
    return lr_scheduler, lr_warmup_scheduler


def get_dataloaders(config):

    train_dataloader_params = config["train_dataloader"]
    train_shuffle = train_dataloader_params["shuffle"]
    train_batch_size = train_dataloader_params["total_batch_size"]
    train_num_workers = train_dataloader_params["num_workers"]
    train_drop_last = train_dataloader_params["drop_last"]
    train_datarooot = train_dataloader_params["dataset_root"]
    train_mode = train_dataloader_params["mode"]
    train_num_imgs_per_folder = train_dataloader_params["num_imgs_per_folder"]

    train_dataset = GeoImageryODdata(
        train_datarooot, train_mode, train_num_imgs_per_folder
    )

    val_dataloader_params = config["val_dataloader"]
    val_shuffle = val_dataloader_params["shuffle"]
    val_batch_size = val_dataloader_params["total_batch_size"]
    val_num_workers = val_dataloader_params["num_workers"]
    val_drop_last = val_dataloader_params["drop_last"]
    val_datarooot = val_dataloader_params["dataset_root"]
    val_mode = val_dataloader_params["mode"]
    val_num_imgs_per_folder = val_dataloader_params["num_imgs_per_folder"]

    val_dataset = GeoImageryODdata(val_datarooot, val_mode, val_num_imgs_per_folder)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        num_workers=train_num_workers,
        drop_last=train_drop_last,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=val_num_workers,
        drop_last=val_drop_last,
    )

    return train_dataloader, val_dataloader


def save_model(model, optimizer, out_dir, **kwargs):

    val_loss = kwargs.get("val_loss", 1e8)
    expt_name = str(kwargs.get("expt_name", "trial"))
    sv_type = str(kwargs.get("svtype", "REGULAR"))
    epoch = kwargs.get("epoch", 1e8)

    save_pth = os.path.join(
        out_dir,
        "rtdetrv2_{0}_{1}_{2:.4f}|{3}.pth".format(expt_name, sv_type, val_loss, epoch),
    )

    print(
        "\nSaving {0} model for epoch: {1} ; Val loss : {2:.4f}\n".format(
            sv_type, epoch, val_loss
        )
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        },
        save_pth,
    )


def train(config_path):

    config = load_config(config_path)
    set_seeds(seed=config["seed"])
    device = get_device()
    out_dir, log_dir = make_outdirs(config)
    max_norm = config["clip_max_norm"]
    resume_path = getattr(config, "resume_path", None)
    start_epoch = getattr(config, "last_epoch", 0)
    epochs = config["epochs"]
    expt_name = config["expt_name"]
    checkpoint_freq = getattr(config, "checkpoint_freq", 10)
    # plot_freq = getattr(config, "plot_freq", 10)
    log_dir = getattr(config, "log_dir", None)
    writer = SummaryWriter(log_dir)

    model = get_model(config)
    criterion = get_criterion(config)
    postprocessor = get_postprocessor(config)
    # ema = get_ema(config) if config["use_ema"] else None
    # scaler = get_scaler(config) if config["use_amp"] else None
    optimizer = get_optimizer(config)
    lr_scheduler, lr_warmup_scheduler = get_lr_schedulers(config)
    train_dataloader, val_dataloader = get_dataloaders(config)

    model.to(device)
    criterion.to(device)
    if resume_path is not None:
        model.load_state_dict(torch.load(resume_path)["state_dict"], strict=True)

    n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"number of trainable parameters: {n_parameters}")

    start_time = time.time()

    best_epoch_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):

        ### training loop ###
        model.train()
        criterion.train()

        for i, (samples, targets) in enumerate(train_dataloader):

            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            global_step = epoch * len(train_dataloader) + i
            metas = dict(epoch=epoch, step=i, global_step=global_step)
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                print(loss_dict)
                sys.exit(1)

            # --------------
            #  Log Progress
            # --------------
            writer.add_scalar("Loss/train_total", loss.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/train_{k}", v.item(), global_step)

            # Print log
            sys.stdout.write(
                "\r Training : [Epoch %d/%d] [Batch %d/%d] [lr %f] [total_loss: %f, vfl_loss: %f, boxes_loss: %f]"
                % (
                    epoch,
                    epochs,
                    i,
                    len(train_dataloader),
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    loss_dict["vfl"].item(),
                    loss_dict["boxes"].item(),
                )
            )

        ### validation loop ###
        model.eval()
        criterion.eval()

        epoch_val_loss = 0.0
        for i, (samples, targets) in enumerate(val_dataloader):

            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                global_step = epoch * len(val_dataloader) + i
                metas = dict(epoch=epoch, step=i, global_step=global_step)
                outputs = model(samples, targets=targets)
                loss_dict = criterion(outputs, targets, **metas)
                loss: torch.Tensor = sum(loss_dict.values())

            epoch_val_loss += loss.item()

            # --------------
            #  Log Progress
            # --------------
            writer.add_scalar("loss/val_total", loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/val_{k}", v.item(), global_step)

            # Print log
            sys.stdout.write(
                "\r Validating : [Epoch %d/%d] [Batch %d/%d] [total_loss: %f, vfl_loss: %f, boxes_loss: %f]"
                % (
                    epoch,
                    epochs,
                    i,
                    len(val_dataloader),
                    loss.item(),
                    loss_dict["vfl"].item(),
                    loss_dict["boxes"].item(),
                )
            )

            # TODO
            # --------------
            #   Plot Samples ???
            # --------------

        epoch_val_loss /= len(val_dataloader)

        if epoch_val_loss < best_epoch_val_loss:

            best_epoch_val_loss = epoch_val_loss

            save_model(
                model,
                optimizer,
                val_loss=epoch_val_loss,
                expt_name=expt_name,
                sv_type="BEST",
                epoch=epoch,
            )

        elif epoch % checkpoint_freq == 0:
            save_model(
                model,
                optimizer,
                val_loss=epoch_val_loss,
                expt_name=expt_name,
                sv_type="REGULAR",
                epoch=epoch,
            )

        if lr_warmup_scheduler is None or lr_warmup_scheduler.finished():
            lr_scheduler.step()

    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    args = parse_args()
    config_path = args.conf
    train(config_path)
