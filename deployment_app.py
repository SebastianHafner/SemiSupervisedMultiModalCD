import torch
import numpy as np
from pathlib import Path
from utils import experiment_manager, networks, datasets, parsers, geofiles


def deploy_model(cfg: experiment_manager.CfgNode, site: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    out_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / cfg.NAME
    out_folder.mkdir(exist_ok=True)

    ds = datasets.DeploymentAppDataset(cfg, site)
    arr = ds.get_arr(2)
    transform, crs = ds.get_geo()

    for item in ds:
        x_t1 = item['x_t1'].to(device)
        x_t2 = item['x_t2'].to(device)
        with torch.no_grad():
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
        assert(isinstance(logits, tuple))
        logits_change, logits_sem_t1 = logits[0], logits[5]
        y_pred_change = torch.sigmoid(logits_change).squeeze().detach().cpu().numpy()
        y_pred_sem_t1 = torch.sigmoid(logits_sem_t1).squeeze().detach().cpu().numpy()

        i, j = item['i'], item['j']
        arr[i:i + ds.tile_size, j:j + ds.tile_size, 0] = (y_pred_change * 100).astype(np.uint8)
        arr[i:i + ds.tile_size, j:j + ds.tile_size, 1] = (y_pred_sem_t1 * 100).astype(np.uint8)

    pred_file = out_folder / f'pred_{site}_{cfg.NAME}.tif'
    geofiles.write_tif(pred_file, arr, transform, crs)


if __name__ == '__main__':
    args = parsers.deployment_app_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    deploy_model(cfg, args.site)
