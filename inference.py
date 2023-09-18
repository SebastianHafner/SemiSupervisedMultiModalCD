import torch
from pathlib import Path
from utils import experiment_manager, networks, datasets, parsers, geofiles


def inference_change(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    pred_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / cfg.NAME
    pred_folder.mkdir(exist_ok=True)

    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)

    with torch.no_grad():
        for item in ds:
            aoi_id = item['aoi_id']
            x_t1 = item['x_t1']
            x_t2 = item['x_t2']
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            logits_change = logits[0] if isinstance(logits, tuple) else logits
            y_pred_change = torch.sigmoid(logits_change).squeeze().detach().cpu().numpy()

            transform, crs = ds.get_geo(aoi_id)
            pred_file = pred_folder / f'pred_{cfg.NAME}_{aoi_id}.tif'
            geofiles.write_tif(pred_file, y_pred_change[:, :, None], transform, crs)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    inference_change(cfg)
