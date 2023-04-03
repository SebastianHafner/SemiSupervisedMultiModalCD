import torch
from torch import optim
from torch.utils import data as torch_data

import timeit

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers

# https://github.com/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb
if __name__ == '__main__':
    args = parsers.sweep_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Runnning on device: p', device)

    def run_training(sweep_cfg=None):

        with wandb.init(config=sweep_cfg):
            sweep_cfg = wandb.config

            # make training deterministic
            torch.manual_seed(cfg.SEED)
            np.random.seed(cfg.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            net = networks.create_network(cfg)
            net.to(device)
            optimizer = optim.AdamW(net.parameters(), lr=sweep_cfg.lr, weight_decay=0.01)

            criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

            # reset the generators
            dataset = datasets.MultimodalCDDataset(cfg=cfg, run_type='train')
            print(dataset)

            dataloader_kwargs = {
                'batch_size': sweep_cfg.batch_size,
                'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
                'shuffle': cfg.DATALOADER.SHUFFLE,
                'drop_last': True,
                'pin_memory': True,
            }
            dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

            # unpacking cfg
            epochs = cfg.TRAINER.EPOCHS
            steps_per_epoch = len(dataloader)

            # tracking variables
            global_step = epoch_float = 0

            # early stopping
            best_f1_val, trigger_times = 0, 0
            stop_training = False

            for epoch in range(1, epochs + 1):
                print(f'Starting epoch {epoch}/{epochs}.')

                start = timeit.default_timer()
                loss_set = []

                for i, batch in enumerate(dataloader):

                    net.train()
                    optimizer.zero_grad()

                    x_t1 = batch['x_t1'].to(device)
                    x_t2 = batch['x_t2'].to(device)

                    logits = net(x_t1, x_t2)

                    gt_change = batch['y_change'].to(device)

                    loss = criterion(logits, gt_change)
                    loss.backward()
                    optimizer.step()

                    loss_set.append(loss.item())

                    global_step += 1
                    epoch_float = global_step / steps_per_epoch

                    if global_step % cfg.LOGGING.FREQUENCY == 0:
                        # print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                        time = timeit.default_timer() - start
                        wandb.log({
                            'loss': np.mean(loss_set),
                            'labeled_percentage': 100,
                            'time': time,
                            'step': global_step,
                            'epoch': epoch_float,
                        })
                        start = timeit.default_timer()
                        loss_set = []
                    # end of batch

                assert (epoch == epoch_float)
                _ = evaluation.model_evaluation(net, cfg, 'train', epoch_float, global_step)
                f1_val = evaluation.model_evaluation(net, cfg, 'val', epoch_float, global_step)

                if f1_val <= best_f1_val:
                    trigger_times += 1
                    if trigger_times > cfg.TRAINER.PATIENCE:
                        stop_training = True
                else:
                    best_f1_val = f1_val
                    wandb.log({
                        'best val change F1': best_f1_val,
                        'step': global_step,
                        'epoch': epoch_float,
                    })
                    print(f'saving network (F1 {f1_val:.3f})', flush=True)
                    networks.save_checkpoint(net, optimizer, epoch, cfg)
                    trigger_times = 0

                if stop_training:
                    break  # end of training by early stopping

            net, *_ = networks.load_checkpoint(cfg, device)
            _ = evaluation.model_evaluation(net, cfg, 'test', epoch_float, global_step)

    if args.sweep_id is None:
        # Step 2: Define sweep config
        sweep_config = {
            'method': 'grid',
            'name': cfg.NAME,
            'metric': {'goal': 'maximize', 'name': 'best val change F1'},
            'parameters':
                {
                    'lr': {'values': [0.0001, 0.00005, 0.00001]},
                    'batch_size': {'values': [16, 8]},
                }
        }
        # Step 3: Initialize sweep by passing in config or resume sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity='population_mapping')
        # Step 4: Call to `wandb.agent` to start a sweep
        wandb.agent(sweep_id, function=run_training)
    else:
        # Or resume existing sweep via its id
        # https://github.com/wandb/wandb/issues/1501
        sweep_id = args.sweep_id
        wandb.agent(sweep_id, project=args.project, function=run_training)

