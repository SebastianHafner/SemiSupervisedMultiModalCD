import torch
from utils import experiment_manager, networks, datasets, parsers, geofiles, evaluation
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantitative_assessment_change(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    measurer = evaluation.Measurer(run_type, 'change')

    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)
    with torch.no_grad():
        for item in ds:
            x_t1 = item['x_t1'].to(device)
            x_t2 = item['x_t2'].to(device)
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            logits = logits[0] if isinstance(logits, tuple) else logits
            y_hat = torch.sigmoid(logits).squeeze().detach()
            y = item['y_change'].to(device).squeeze().detach()

            measurer.add_sample(y, y_hat)

    file = Path(cfg.PATHS.OUTPUT) / 'testing' / f'quantitative_results_change_{run_type}.json'
    if not file.exists():
        data = {}
    else:
        data = geofiles.load_json(file)

    data[cfg.NAME] = {
        'f1_score': measurer.f1().item(),
        'precision': measurer.precision().item(),
        'recall': measurer.recall().item(),
        'iou': measurer.iou().item(),
    }

    geofiles.write_json(file, data)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_assessment_change(cfg, run_type='test')
