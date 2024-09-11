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
from data.geo_OD_data import GeoImageryODdata, batch_image_collate_fn
from PIL import Image

import torchvision.utils as vutils
import torchvision


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
        hidden_dim=encoder_params["hidden_dim"],
        nhead=encoder_params["nhead"],
        dim_feedforward=encoder_params["dim_feedforward"],
        dropout=encoder_params["dropout"],
        enc_act=encoder_params["enc_act"],
        use_encoder_idx=encoder_params["use_encoder_idx"],
        num_encoder_layers=encoder_params["num_encoder_layers"],
        expansion=encoder_params["expansion"],
        depth_mult=encoder_params["depth_mult"],
        act=encoder_params["act"],
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
        # aux_loss=decoder_params["aux_loss"],
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
    num_top_queries = config["RTDETRPostProcessor"]["num_top_queries"]
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
    tile_size = config["tile_size"]
    train_shuffle = train_dataloader_params["shuffle"]
    train_batch_size = train_dataloader_params["total_batch_size"]
    train_num_workers = train_dataloader_params["num_workers"]
    train_drop_last = train_dataloader_params["drop_last"]
    train_datarooot = train_dataloader_params["dataset_root"]
    train_mode = train_dataloader_params["mode"]
    train_num_imgs_per_folder = train_dataloader_params["num_imgs_per_folder"]

    class_mapping_path = config["class_mapping_path"]

    train_dataset = GeoImageryODdata(
        train_datarooot,
        train_mode,
        train_num_imgs_per_folder,
        class_mapping_path,
        tile_size,
    )

    val_dataloader_params = config["val_dataloader"]
    val_shuffle = val_dataloader_params["shuffle"]
    val_batch_size = val_dataloader_params["total_batch_size"]
    val_num_workers = val_dataloader_params["num_workers"]
    val_drop_last = val_dataloader_params["drop_last"]
    val_datarooot = val_dataloader_params["dataset_root"]
    val_mode = val_dataloader_params["mode"]
    val_num_imgs_per_folder = val_dataloader_params["num_imgs_per_folder"]

    val_dataset = GeoImageryODdata(
        val_datarooot, val_mode, val_num_imgs_per_folder, class_mapping_path, tile_size
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        num_workers=train_num_workers,
        drop_last=train_drop_last,
        generator=torch.Generator(device="cuda"),
        collate_fn=batch_image_collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=val_num_workers,
        drop_last=val_drop_last,
        generator=torch.Generator(device="cuda"),
        collate_fn=batch_image_collate_fn,
    )

    return train_dataloader, val_dataloader


def save_model(model, optimizer, out_dir, **kwargs):

    val_loss = kwargs.get("val_loss", 1e8)
    expt_name = str(kwargs.get("expt_name", "trial"))
    sv_type = str(kwargs.get("sv_type", "REGULAR"))
    epoch = kwargs.get("epoch", 1e8)

    save_pth = os.path.join(
        out_dir,
        "rtdetrv2_{0}_{1}_valloss_{2:.4f}_epoch_{3}.pth".format(
            expt_name, sv_type, val_loss, epoch
        ),
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


def plot_and_save_batch(
    image_batch,
    results,
    targets,
    output_dir,
    batch_id,
    epoch,
):
    """
    Plots and saves a batch of images with bounding boxes and labels.

    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for drawing
    pred_color = (255, 0, 0)  # Red for predictions
    gt_color = (0, 255, 0)  # Green for ground truth

    # Initialize list to store processed images
    processed_images = []

    for i in range(image_batch.size(0)):

        image_tensor = image_batch[i]

        image_tensor = image_tensor.mul(255).byte()

        result_tensor = results[i]
        pred_boxes = result_tensor["boxes"]
        pred_labels = result_tensor["labels"]
        pred_confidences = result_tensor["scores"]

        target_tensor = targets[i]
        gt_boxes = target_tensor["boxes"]
        gt_labels = target_tensor["labels"]

        # Draw bounding boxes
        image_with_pred_boxes = vutils.draw_bounding_boxes(
            image_tensor,
            pred_boxes,
            colors=pred_color,
            labels=[
                f"Pred: {label.item()}__{conf:.2f}"
                for label, conf in zip(pred_labels, pred_confidences)
            ],
        )

        image_with_all_boxes = vutils.draw_bounding_boxes(
            image_with_pred_boxes,
            gt_boxes,
            colors=gt_color,
            labels=[f"GT: {label.item()}" for label in gt_labels],
        )

        processed_images.append(image_with_all_boxes.cpu())

    # Determine grid size
    num_images = len(processed_images)
    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_images)).float()).item())
    total_images = grid_size**2

    # Create a black image for padding
    height, width = processed_images[0].shape[1:]
    black_image = torch.zeros((3, height, width), dtype=torch.uint8).cpu()

    # Pad the list of images with black images if needed
    while len(processed_images) < total_images:
        processed_images.append(black_image)

    # Stack images in a grid
    grid_image = vutils.make_grid(processed_images, nrow=grid_size, padding=2)

    grid_image_array = grid_image.permute(1, 2, 0).detach().cpu().numpy()
    result = Image.fromarray(grid_image_array.astype(np.uint8))

    output_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_id}.jpg")

    result.save(output_path)


def train(config_path):

    config = load_config(config_path)
    print(config)
    set_seeds(seed=config["seed"])
    device = get_device()
    out_dir, log_dir = make_outdirs(config)
    plot_dir = config["plot_dir"]
    max_norm = config["clip_max_norm"]
    resume_path = config["resume_path"]
    start_epoch = config["start_epoch"]
    epochs = config["epochs"]
    expt_name = config["expt_name"]
    checkpoint_freq = config["checkpoint_freq"]
    plot_freq = getattr(config, "plot_freq", 10)
    log_dir = config["log_dir"]
    tile_size = config["tile_size"]
    writer = SummaryWriter(log_dir=log_dir)

    model = get_model(config)
    criterion = get_criterion(config)
    postprocessor = get_postprocessor(config)
    # ema = get_ema(config) if config["use_ema"] else None
    # scaler = get_scaler(config) if config["use_amp"] else None
    optimizer = get_optimizer(config, model)
    lr_scheduler, lr_warmup_scheduler = get_lr_schedulers(config, optimizer)
    train_dataloader, val_dataloader = get_dataloaders(config)

    if resume_path is not None:
        model.load_state_dict(torch.load(resume_path)["state_dict"], strict=True)

    model.to(device)
    criterion.to(device)

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

            print(
                f"Training : Epoch {epoch}/{epochs}, Batch {i}/{len(train_dataloader)}, lr { optimizer.param_groups[0]['lr'] }, total_loss: {loss.item()}"
            )

        print(
            "###################################### VALIDATING ... ###############################################"
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

            print(
                f"Validating : Epoch {epoch}/{epochs}, Batch {i}/{len(val_dataloader)}, total_loss: {loss.item()}"
            )

            # # -------------------
            # #   Plot Samples
            # # -------------------
            if i % plot_freq == 0:
                sample_wh = torch.stack(
                    [torch.Tensor([tile_size, tile_size]) for t in targets],
                    dim=0,
                )
                results = postprocessor(outputs, sample_wh)

                plot_and_save_batch(
                    samples,
                    results,
                    targets,
                    output_dir=plot_dir,
                    batch_id=i,
                    epoch=epoch,
                )

        epoch_val_loss /= len(val_dataloader)

        if epoch_val_loss < best_epoch_val_loss:

            best_epoch_val_loss = epoch_val_loss

            save_model(
                model,
                optimizer,
                out_dir,
                val_loss=epoch_val_loss,
                expt_name=expt_name,
                sv_type="BEST",
                epoch=epoch,
            )

        elif epoch % checkpoint_freq == 0:
            save_model(
                model,
                optimizer,
                out_dir,
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
