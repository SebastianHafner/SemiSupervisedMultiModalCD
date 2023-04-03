import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import experiment_manager, networks, datasets, evaluation


def qualitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)
    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits_change = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        y_pred_change = torch.sigmoid(logits_change).squeeze().detach()

        gt_change = item['y_change'].squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(x_t1.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])
        axs[1, 0].imshow(x_t2.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])

        axs[0, 1].imshow(gt_change.numpy(), cmap='gray')
        axs[1, 1].imshow(y_pred_change.numpy(), cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME / f'change_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def qualitative_assessment_dualtask(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)
    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        y_pred_change = torch.sigmoid(logits[0]).squeeze().detach()
        gt_change = item['y_change'].squeeze()


        logits_stream1_sem_t1, logits_stream1_sem_t2, logits_stream2_sem_t1, logits_stream2_sem_t2 = logits[3:]


        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        # s2 images

        axs[0, 0].imshow(s2_t1.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])
        axs[1, 0].imshow(s2_t2.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])




        axs[0, 3].imshow(gt_change.numpy(), cmap='gray')
        axs[1, 3].imshow(y_pred_change.numpy(), cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME / f'change_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def qualitative_comparison(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)
    for item in ds:
        aoi_id = item['aoi_id']
        x_t1 = item['x_t1']
        x_t2 = item['x_t2']
        logits_change = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        y_pred_change = torch.sigmoid(logits_change).squeeze().detach()

        gt_change = item['y_change'].squeeze()

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(x_t1.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])
        axs[1, 0].imshow(x_t2.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]])

        axs[0, 1].imshow(gt_change.numpy(), cmap='gray')
        axs[1, 1].imshow(y_pred_change.numpy(), cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME / f'change_{aoi_id}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)







if __name__ == '__main__':
    args = assessment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_assessment(cfg, run_type=args.run_type)
