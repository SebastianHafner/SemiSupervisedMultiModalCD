import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from pathlib import Path

from utils import experiment_manager


def create_network(cfg):
    if cfg.MODEL.TYPE == 'unet':
        model = UNet(cfg)
    elif cfg.MODEL.TYPE == 'dualstreamunet':
        model = DualStreamUNet(cfg)
    elif cfg.MODEL.TYPE == 'siameseunet':
        model = SiameseUNet(cfg)
    elif cfg.MODEL.TYPE == 'dtsiameseunet':
        model = DualTaskSiameseUNet(cfg)
    elif cfg.MODEL.TYPE == 'dtlatefusionsiameseunet':
        model = DualTaskLateFusionSiameseUnet(cfg)
    elif cfg.MODEL.TYPE == 'multimodalsiameseunet':
        model = MultiModalSiameseUnet(cfg)
    elif cfg.MODEL.TYPE == 'dtmultimodalsiameseunet':
        model = DualTaskMultiModalSiameseUnet(cfg)
    else:
        raise Exception(f'Unknown network ({cfg.MODEL.TYPE}).')
    return nn.DataParallel(model)


def save_checkpoint(network, optimizer, epoch, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(cfg: experiment_manager.CfgNode, device: torch.device):
    net = create_network(cfg)
    net.to(device)
    net_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'

    checkpoint = torch.load(net_file, map_location=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['epoch']


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS * 2
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.inc = InConv(n_channels, topology[0], DoubleConv)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.outc = OutConv(topology[0], n_classes)

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:
        x = torch.cat((x_t1, x_t2), dim=1)
        x = self.inc(x)
        features = self.encoder(x)
        x = self.decoder(features)
        out = self.outc(x)
        return out


class DualStreamUNet(nn.Module):
    def __init__(self, cfg):
        super(DualStreamUNet, self).__init__()
        self.cfg = cfg

        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        # stream 1 (S1)
        self.inc_stream1 = InConv(2*len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_stream1 = Encoder(cfg)
        self.decoder_stream1 = Decoder(cfg)

        # stream 2 (S2)
        self.inc_stream2 = InConv(2 * len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_stream2 = Encoder(cfg)
        self.decoder_stream2 = Decoder(cfg)

        self.outc = OutConv(2*topology[0], n_classes)

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:
        # stream1 (S1)
        s1_t1, s1_t2 = x_t1[:, :len(self.cfg.DATALOADER.S1_BANDS), ], x_t2[:, :len(self.cfg.DATALOADER.S1_BANDS), ]
        x_stream1 = torch.concat((s1_t1, s1_t2), dim=1)
        x_stream1 = self.inc_stream1(x_stream1)
        features_stream1 = self.encoder_stream1(x_stream1)
        x_stream1 = self.decoder_stream1(features_stream1)

        # stream2 (S2)
        s2_t1, s2_t2 = x_t1[:, len(self.cfg.DATALOADER.S1_BANDS):, ], x_t2[:, len(self.cfg.DATALOADER.S1_BANDS):, ]
        x_stream2 = torch.concat((s2_t1, s2_t2), dim=1)
        x_stream2 = self.inc_stream2(x_stream2)
        features_stream2 = self.encoder_stream2(x_stream2)
        x_stream2 = self.decoder_stream2(features_stream2)

        x_out = torch.concat((x_stream1, x_stream2), dim=1)
        out = self.outc(x_out)
        return out


class SiameseUNet(nn.Module):
    def __init__(self, cfg):
        super(SiameseUNet, self).__init__()
        self.cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.inc = InConv(n_channels, topology[0], DoubleConv)

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.outc = OutConv(topology[0], n_classes)

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:

        x1_t1 = self.inc(x_t1)
        features_t1 = self.encoder(x1_t1)

        x1_t2 = self.inc(x_t2)
        features_t2 = self.encoder(x1_t2)

        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)
        x2 = self.decoder(features_diff)
        out = self.outc(x2)

        return out


class DualTaskSiameseUNet(nn.Module):
    def __init__(self, cfg):
        super(DualTaskSiameseUNet, self).__init__()
        self.cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.inc = InConv(n_channels, topology[0], DoubleConv)

        self.encoder = Encoder(cfg)
        self.decoder_change = Decoder(cfg)
        self.decoder_sem = Decoder(cfg)

        self.outc_change = OutConv(topology[0], n_classes)
        self.outc_change_sem = OutConv(2, 1)
        self.outc_sem = OutConv(topology[0], n_classes)

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> list:
        x1_t1 = self.inc(x_t1)
        features_t1 = self.encoder(x1_t1)

        x1_t2 = self.inc(x_t2)
        features_t2 = self.encoder(x1_t2)

        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)

        x2 = self.decoder_change(features_diff)
        out_change = self.outc_change(x2)

        x2_t2 = self.decoder_sem(features_t2)
        out_sem_t2 = self.outc_sem(x2_t2)

        x2_t1 = self.decoder_sem(features_t1)
        out_sem_t1 = self.outc_sem(x2_t1)

        return out_change, out_sem_t1, out_sem_t2


class DualTaskLateFusionSiameseUnet(nn.Module):
    def __init__(self, cfg):
        super(DualTaskLateFusionSiameseUnet, self).__init__()
        self.cfg = cfg

        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        # sar
        self.inc_sar = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_sar = Encoder(cfg)
        self.decoder_sar_change = Decoder(cfg)
        self.decoder_sar_sem = Decoder(cfg)
        self.outc_sar_change = OutConv(topology[0], n_classes)
        self.outc_sar_sem = OutConv(topology[0], n_classes)

        # optical
        self.inc_optical = InConv(len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_optical = Encoder(cfg)
        self.decoder_optical_change = Decoder(cfg)
        self.decoder_optical_sem = Decoder(cfg)
        self.outc_optical_change = OutConv(topology[0], n_classes)
        self.outc_optical_sem = OutConv(topology[0], n_classes)

        # fusion
        self.outc_fusion_change = OutConv(2 * topology[0], n_classes)
        self.outc_fusion_sem = OutConv(2 * topology[0], n_classes)

    @ staticmethod
    def difference_features(features_t1: torch.Tensor, features_t2: torch.Tensor):
        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)
        return features_diff

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:

        # sar
        # encoding
        s1_t1, s1_t2 = x_t1[:, :len(self.cfg.DATALOADER.S1_BANDS), ], x_t2[:, :len(self.cfg.DATALOADER.S1_BANDS), ]
        x1_sar_t1 = self.inc_sar(s1_t1)
        features_sar_t1 = self.encoder_sar(x1_sar_t1)
        x1_sar_t2 = self.inc_sar(s1_t2)
        features_sar_t2 = self.encoder_sar(x1_sar_t2)
        features_sar_diff = self.difference_features(features_sar_t1, features_sar_t2)

        # decoding change
        x2_sar_change = self.decoder_sar_change(features_sar_diff)
        out_sar_change = self.outc_sar_change(x2_sar_change)

        # deconding semantics
        x2_sar_sem_t1 = self.decoder_sar_sem(features_sar_t1)
        out_sar_sem_t1 = self.outc_sar_sem(x2_sar_sem_t1)

        x2_sar_sem_t2 = self.decoder_sar_sem(features_sar_t2)
        out_sar_sem_t2 = self.outc_sar_sem(x2_sar_sem_t2)

        # optical
        # encoding
        s2_t1, s2_t2 = x_t1[:, len(self.cfg.DATALOADER.S1_BANDS):, ], x_t2[:, len(self.cfg.DATALOADER.S1_BANDS):, ]
        x1_optical_t1 = self.inc_optical(s2_t1)
        features_optical_t1 = self.encoder_optical(x1_optical_t1)
        x1_optical_t2 = self.inc_optical(s2_t2)
        features_optical_t2 = self.encoder_optical(x1_optical_t2)
        features_optical_diff = self.difference_features(features_optical_t1, features_optical_t2)

        # decoding change
        x2_optical_change = self.decoder_optical_change(features_optical_diff)
        out_optical_change = self.outc_optical_change(x2_optical_change)

        # deconding semantics
        x2_optical_sem_t1 = self.decoder_optical_sem(features_optical_t1)
        out_optical_sem_t1 = self.outc_optical_sem(x2_optical_sem_t1)

        x2_optical_sem_t2 = self.decoder_optical_sem(features_optical_t2)
        out_optical_sem_t2 = self.outc_optical_sem(x2_optical_sem_t2)

        # fusion
        x2_fusion_change = torch.concat((x2_sar_change, x2_optical_change), dim=1)
        out_fusion_change = self.outc_fusion_change(x2_fusion_change)

        # fusion semantic decoding
        x2_fusion_sem_t1 = torch.concat((x2_sar_sem_t1, x2_optical_sem_t1), dim=1)
        out_fusion_sem_t1 = self.outc_fusion_sem(x2_fusion_sem_t1)

        x2_fusion_sem_t2 = torch.concat((x2_sar_sem_t2, x2_optical_sem_t2), dim=1)
        out_fusion_sem_t2 = self.outc_fusion_sem(x2_fusion_sem_t2)

        return out_fusion_change, out_sar_sem_t1, out_sar_sem_t2, out_optical_sem_t1, out_optical_sem_t2,\
            out_fusion_sem_t1, out_fusion_sem_t2


class MultiModalSiameseUnet(nn.Module):
    def __init__(self, cfg):
        super(MultiModalSiameseUnet, self).__init__()
        self.cfg = cfg

        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = list(cfg.MODEL.TOPOLOGY)

        # sar encoder
        self.inc_sar = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_sar = Encoder(cfg)

        # optical encoder
        self.inc_optical = InConv(len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_optical = Encoder(cfg)

        # fusion decoder
        decoder_topology = list(2 * np.array(topology))
        self.decoder_fusion = Decoder(cfg, decoder_topology)
        self.outc_fusion = OutConv(2 * topology[0], n_classes)

    @staticmethod
    def difference_features(features_t1: torch.Tensor, features_t2: torch.Tensor):
        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)
        return features_diff

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:
        # sar encoding
        s1_t1, s1_t2 = x_t1[:, :len(self.cfg.DATALOADER.S1_BANDS), ], x_t2[:, :len(self.cfg.DATALOADER.S1_BANDS), ]
        x1_sar_t1 = self.inc_sar(s1_t1)
        features_sar_t1 = self.encoder_sar(x1_sar_t1)
        x1_sar_t2 = self.inc_sar(s1_t2)
        features_sar_t2 = self.encoder_sar(x1_sar_t2)
        features_sar_diff = self.difference_features(features_sar_t1, features_sar_t2)

        # optical encoding
        s2_t1, s2_t2 = x_t1[:, len(self.cfg.DATALOADER.S1_BANDS):, ], x_t2[:, len(self.cfg.DATALOADER.S1_BANDS):, ]
        x1_optical_t1 = self.inc_optical(s2_t1)
        features_optical_t1 = self.encoder_optical(x1_optical_t1)
        x1_optical_t2 = self.inc_optical(s2_t2)
        features_optical_t2 = self.encoder_optical(x1_optical_t2)
        features_optical_diff = self.difference_features(features_optical_t1, features_optical_t2)

        # features concat
        features_fusion_diff = []
        for f_sar, f_optical in zip(features_sar_diff, features_optical_diff):
            f_fusion = torch.concat((f_sar, f_optical), dim=1)
            features_fusion_diff.append(f_fusion)

        x2_fusion = self.decoder_fusion(features_fusion_diff)
        out_fusion = self.outc_fusion(x2_fusion)

        return out_fusion


class DualTaskMultiModalSiameseUnet(nn.Module):
    def __init__(self, cfg):
        super(DualTaskMultiModalSiameseUnet, self).__init__()
        self.cfg = cfg

        n_classes = cfg.MODEL.OUT_CHANNELS
        topology = list(cfg.MODEL.TOPOLOGY)

        # sar encoder
        self.inc_sar = InConv(len(cfg.DATALOADER.S1_BANDS), topology[0], DoubleConv)
        self.encoder_sar = Encoder(cfg)
        self.decoder_sar_sem = Decoder(cfg)
        self.outc_sar_sem = OutConv(topology[0], n_classes)

        # optical encoder
        self.inc_optical = InConv(len(cfg.DATALOADER.S2_BANDS), topology[0], DoubleConv)
        self.encoder_optical = Encoder(cfg)
        self.decoder_optical_sem = Decoder(cfg)
        self.outc_optical_sem = OutConv(topology[0], n_classes)

        # fusion decoder
        decoder_change_topology = list(2 * np.array(topology))
        self.decoder_fusion_change = Decoder(cfg, decoder_change_topology)
        self.outc_fusion_change = OutConv(2 * topology[0], n_classes)

        self.outc_fusion_sem = OutConv(2 * topology[0], n_classes)

    @staticmethod
    def difference_features(features_t1: torch.Tensor, features_t2: torch.Tensor):
        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)
        return features_diff

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:
        # sar encoding
        s1_t1, s1_t2 = x_t1[:, :len(self.cfg.DATALOADER.S1_BANDS), ], x_t2[:, :len(self.cfg.DATALOADER.S1_BANDS), ]
        x1_sar_t1 = self.inc_sar(s1_t1)
        features_sar_t1 = self.encoder_sar(x1_sar_t1)
        x1_sar_t2 = self.inc_sar(s1_t2)
        features_sar_t2 = self.encoder_sar(x1_sar_t2)
        features_sar_diff = self.difference_features(features_sar_t1, features_sar_t2)

        # optical encoding
        s2_t1, s2_t2 = x_t1[:, len(self.cfg.DATALOADER.S1_BANDS):, ], x_t2[:, len(self.cfg.DATALOADER.S1_BANDS):, ]
        x1_optical_t1 = self.inc_optical(s2_t1)
        features_optical_t1 = self.encoder_optical(x1_optical_t1)
        x1_optical_t2 = self.inc_optical(s2_t2)
        features_optical_t2 = self.encoder_optical(x1_optical_t2)
        features_optical_diff = self.difference_features(features_optical_t1, features_optical_t2)

        # features concat and change decoding
        features_fusion_diff = []
        for f_sar, f_optical in zip(features_sar_diff, features_optical_diff):
            f_fusion = torch.concat((f_sar, f_optical), dim=1)
            features_fusion_diff.append(f_fusion)

        x2_fusion_change = self.decoder_fusion_change(features_fusion_diff)
        out_fusion_change = self.outc_fusion_change(x2_fusion_change)

        # sar semantic decoding
        x2_sar_sem_t1 = self.decoder_sar_sem(features_sar_t1)
        out_sar_sem_t1 = self.outc_sar_sem(x2_sar_sem_t1)

        x2_sar_sem_t2 = self.decoder_sar_sem(features_sar_t2)
        out_sar_sem_t2 = self.outc_sar_sem(x2_sar_sem_t2)

        # optical semantic decoding
        x2_optical_sem_t1 = self.decoder_optical_sem(features_optical_t1)
        out_optical_sem_t1 = self.outc_optical_sem(x2_optical_sem_t1)

        x2_optical_sem_t2 = self.decoder_optical_sem(features_optical_t2)
        out_optical_sem_t2 = self.outc_optical_sem(x2_optical_sem_t2)

        # fusion semantic decoding
        x2_fusion_sem_t1 = torch.concat((x2_sar_sem_t1, x2_optical_sem_t1), dim=1)
        out_fusion_sem_t1 = self.outc_fusion_sem(x2_fusion_sem_t1)

        x2_fusion_sem_t2 = torch.concat((x2_sar_sem_t2, x2_optical_sem_t2), dim=1)
        out_fusion_sem_t2 = self.outc_fusion_sem(x2_fusion_sem_t2)

        return out_fusion_change, out_sar_sem_t1, out_sar_sem_t2, out_optical_sem_t1, out_optical_sem_t2,\
            out_fusion_sem_t1, out_fusion_sem_t2


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.cfg = cfg
        topology = cfg.MODEL.TOPOLOGY

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()
        return inputs


class Decoder(nn.Module):
    def __init__(self, cfg: experiment_manager.CfgNode, topology: list = None):
        super(Decoder, self).__init__()
        self.cfg = cfg

        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        # Variable scale
        n_layers = len(topology)
        up_topo = [topology[0]]  # topography upwards
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]  # last layer
            up_topo.append(out_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:

        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        return x1


# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
