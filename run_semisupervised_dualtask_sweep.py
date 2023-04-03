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

        with wandb.init(config=sweep_cfg, mode='online' if not cfg.DEBUG else 'disabled'):
            sweep_cfg = wandb.config

            # make training deterministic
            torch.manual_seed(cfg.SEED)
            np.random.seed(cfg.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            net = networks.create_network(cfg)
            net.to(device)
            optimizer = optim.AdamW(net.parameters(), lr=sweep_cfg.lr, weight_decay=0.01)

            sup_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
            cons_criterion = loss_functions.get_criterion(cfg.CONSISTENCY_TRAINER.LOSS_TYPE)

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
                change_loss_set, sem_loss_set, sup_loss_set, cons_loss_set, loss_set = [], [], [], [], []

                n_labeled, n_notlabeled = 0, 0

                for i, batch in enumerate(dataloader):

                    net.train()
                    optimizer.zero_grad()

                    x_t1 = batch['x_t1'].to(device)
                    x_t2 = batch['x_t2'].to(device)

                    logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1, x_t2)
                    logits_change_sem = net.module.outc_change_sem(torch.cat((logits_sem_t1, logits_sem_t2), dim=1))
                    y_hat_change_sem = torch.sigmoid(logits_change_sem)

                    sup_loss, cons_loss = None, None

                    is_labeled = batch['is_labeled']
                    n_labeled += torch.sum(is_labeled).item()
                    if is_labeled.any():
                        # change detection
                        y_change = batch['y_change'].to(device)
                        change_loss = sup_criterion(logits_change[is_labeled], y_change[is_labeled])
                        change_sem_loss = sup_criterion(logits_change_sem[is_labeled], y_change[is_labeled])

                        # semantic segmentation
                        y_sem_t1 = batch['y_sem_t1'].to(device)
                        y_sem_t2 = batch['y_sem_t2'].to(device)

                        sem_t1_loss = sup_criterion(logits_sem_t1[is_labeled], y_sem_t1[is_labeled,])
                        sem_t2_loss = sup_criterion(logits_sem_t2[is_labeled], y_sem_t2[is_labeled,])

                        change_loss = change_loss + change_sem_loss
                        sem_loss = (sem_t1_loss + sem_t2_loss) / 2
                        sup_loss = change_loss + sem_loss

                        change_loss_set.append(change_loss.item())
                        sem_loss_set.append(sem_loss.item())
                        sup_loss_set.append(sup_loss.item())

                    if not is_labeled.all():
                        is_not_labeled = torch.logical_not(is_labeled)
                        n_notlabeled += torch.sum(is_not_labeled).item()

                        if cfg.CONSISTENCY_TRAINER.LOSS_TYPE == 'L2':
                            y_hat_change = torch.sigmoid(logits_change)
                            cons_loss = cons_criterion(y_hat_change[is_not_labeled], y_hat_change_sem[is_not_labeled])
                        else:
                            cons_loss = cons_criterion(logits_change[is_not_labeled], y_hat_change_sem[is_not_labeled,])
                        cons_loss = cons_loss * cfg.CONSISTENCY_TRAINER.LOSS_FACTOR
                        cons_loss_set.append(cons_loss.item())

                    if sup_loss is None and cons_loss is not None:
                        loss = cons_loss
                    elif sup_loss is not None and cons_loss is not None:
                        loss = sup_loss + cons_loss
                    else:
                        loss = sup_loss

                    loss_set.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    global_step += 1
                    epoch_float = global_step / steps_per_epoch

                    if global_step % cfg.LOGGING.FREQUENCY == 0:
                        print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                        time = timeit.default_timer() - start
                        wandb.log({
                            'change_loss': np.mean(change_loss_set) if len(change_loss_set) > 0 else 0,
                            'sem_loss': np.mean(sem_loss_set) if len(sem_loss_set) > 0 else 0,
                            'sup_loss': np.mean(sup_loss_set) if len(sup_loss_set) > 0 else 0,
                            'cons_loss': np.mean(cons_loss_set) if len(cons_loss_set) > 0 else 0,
                            'loss': np.mean(loss_set),
                            'labeled_percentage': n_labeled / (n_labeled + n_notlabeled) * 100,
                            'time': time,
                            'step': global_step,
                            'epoch': epoch_float,
                        })
                        start = timeit.default_timer()
                        n_labeled, n_notlabeled = 0, 0
                        change_loss_set, sem_loss_set, sup_loss_set, cons_loss_set, loss_set = [], [], [], [], []
                    # end of batch

                assert (epoch == epoch_float)
                print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
                # evaluation at the end of an epoch
                _ = evaluation.model_evaluation_dt(net, cfg, 'train', epoch_float, global_step)
                f1_val = evaluation.model_evaluation_dt(net, cfg, 'val', epoch_float, global_step)

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
            _ = evaluation.model_evaluation_dt(net, cfg, 'test', epoch_float, global_step)

    if args.sweep_id is None:
        # Step 2: Define sweep config
        sweep_config = {
            'method': 'grid',
            'name': cfg.NAME,
            'metric': {'goal': 'maximize', 'name': 'best val change F1'},
            'parameters':
                {
                    'loss_factor': {'values': [0.1, 0.01]},
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