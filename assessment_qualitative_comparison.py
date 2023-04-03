import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import experiment_manager, networks, datasets, parsers, geofiles

BASELINE_CONFIGS = ['baseline_dualstream_gamma', 'siamesedt_gamma']


def qualitative_comparison(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)

    plot_folder = Path(cfg.PATHS.OUTPUT) / 'plots' / 'qualitative_comparison'
    plot_folder.mkdir(exist_ok=True)

    inference_folder = Path(cfg.PATHS.OUTPUT) / 'inference'

    def plot_baseline(ax, baseline: str, aoi_id: str):
        pred_file = inference_folder / baseline / f'pred_{baseline}_{aoi_id}.tif'
        pred, *_ = geofiles.read_tif(pred_file)
        pred = pred > 0.5
        ax.imshow(pred, cmap='gray')

    with torch.no_grad():
        for item in ds:
            aoi_id = item['aoi_id']
            x_t1 = item['x_t1']
            x_t2 = item['x_t2']
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            logits_change = logits[0] if cfg.MODEL.TYPE == 'whatevernet3' else logits
            y_pred_change = torch.sigmoid(logits_change).squeeze().detach().cpu()
            gt_change = item['y_change'].squeeze()

            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            plt.tight_layout()

            rgb_t1 = ds.load_s2_rgb(aoi_id, item['year_t1'], item['month_t1'])
            axs[0, 0].imshow(rgb_t1)
            rgb_t2 = ds.load_s2_rgb(aoi_id, item['year_t2'], item['month_t2'])
            axs[0, 1].imshow(rgb_t2)
            axs[0, 2].imshow(gt_change.numpy(), cmap='gray')

            axs[1, 2].imshow(y_pred_change.numpy() > 0.5, cmap='gray')

            plot_baseline(axs[1, 0], 'baseline_dualstream_gamma', aoi_id)
            plot_baseline(axs[1, 1], 'siamesedt_gamma', aoi_id)

            index = 0
            for _, ax in np.ndenumerate(axs):
                char = chr(ord('a') + index)
                ax.xaxis.set_label_coords(0.5, -0.025)
                ax.set_xlabel(f'({char})', fontsize=16, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                index += 1

            out_file = plot_folder / f'qualitative_comparison_{aoi_id}.png'
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    qualitative_comparison(cfg)