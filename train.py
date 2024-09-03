import torch
import yaml

from data.voc_style_data import VOCDetection
from src.backbone import PResNet
from src.encoder import HybridEncoder
from src.decoder import RTDETRTransformerv2
from src.rtdetrv2 import RTDETR
from src.matcher import HungarianMatcher
from src.postprocessor import RTDETRPostProcessor
from src.collator import BatchImageCollateFuncion
from src.criterion import RTDETRCriterionv2
from torch.utils.data import DataLoader
import argparse


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_model(config_path):
    """Build and return the model based on the configuration file."""
    config = load_config(config_path)
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


def get_criterion(config_path):
    """Build and return the criterion based on the configuration file."""
    config = load_config(config_path)
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
    matcher = HungarianMatcher(
        weight_dict=matcher_weight_dict, alpha=matcher_alpha, gamma=matcher_gamma
    )
    return RTDETRCriterionv2(
        matcher=matcher,
        weight_dict=criterion_weight_dict,
        losses=criterion_losses,
        alpha=criterion_alpha,
        gamma=criterion_gamma,
        num_classes=num_classes,
    )


def get_postprocessor(config_path):
    """Build and return the postprocessor based on the configuration file."""
    config = load_config(config_path)
    num_classes = config["num_classes"]
    num_top_queries = config["num_top_queries"]
    return RTDETRPostProcessor(num_classes=num_classes, num_top_queries=num_top_queries)
