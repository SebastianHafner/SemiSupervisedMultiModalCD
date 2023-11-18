import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles


class AbstractMultimodalCDDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type
        self.root_path = Path(cfg.PATHS.DATASET)

        self.metadata = geofiles.load_json(self.root_path / f'metadata.json')

        self.s1_band_indices = cfg.DATALOADER.S1_BANDS
        self.s2_band_indices = cfg.DATALOADER.S2_BANDS

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _load_s1_img(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s1_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_s2_img(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, self.s2_band_indices], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 'buildings' / f'buildings_{aoi_id}_{year}_{month:02d}.tif'
        label, _, _ = geofiles.read_tif(file)
        label = label > 0
        return np.nan_to_num(label).astype(np.float32)

    def _load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def get_geo(self, aoi_id: str) -> tuple:
        timestamps = self.metadata[aoi_id]
        timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1']]
        year, month = timestamps[0]
        file = self.root_path / aoi_id / 's1' / f's1_{aoi_id}_{year}_{month:02d}.tif'
        _, transform, crs = geofiles.read_tif(file)
        return transform, crs

    def load_s2_rgb(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        file = self.root_path / aoi_id / 's2' / f's2_{aoi_id}_{year}_{month:02d}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, [2, 1, 0]] / 0.3, 0, 1)
        return np.nan_to_num(img).astype(np.float32)


# dataset for urban extraction with building footprints
class MultimodalCDDataset(AbstractMultimodalCDDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 dataset_mode: str = None, disable_multiplier: bool = False, disable_unlabeled: bool = False):
        super().__init__(cfg, run_type)

        self.dataset_mode = cfg.DATALOADER.DATASET_MODE if dataset_mode is None else dataset_mode
        self.include_building_labels = cfg.DATALOADER.INCLUDE_BUILDING_LABELS

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        if run_type == 'train':
            if cfg.DATALOADER.TRAIN_PERCENTAGE == 100:
                self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
            elif cfg.DATALOADER.TRAIN_PERCENTAGE == 10:
                self.aoi_ids = list(cfg.DATASET.TRAIN_10_IDS)
            elif cfg.DATALOADER.TRAIN_PERCENTAGE == 20:
                self.aoi_ids = list(cfg.DATASET.TRAIN_20_IDS)
            elif cfg.DATALOADER.TRAIN_PERCENTAGE == 40:
                self.aoi_ids = list(cfg.DATASET.TRAIN_40_IDS)
        elif run_type == 'val':
            self.aoi_ids = list(cfg.DATASET.VALIDATION_IDS)
        elif run_type == 'test':
            self.aoi_ids = list(cfg.DATASET.TEST_IDS)
        else:
            raise Exception('Unknown run type!')

        self.labeled = [True] * len(self.aoi_ids)

        if cfg.DATALOADER.INCLUDE_UNLABELED and not disable_unlabeled:
            assert(run_type == 'train')
            aoi_ids_unlabelled = list(cfg.DATASET.UNLABELED_IDS)
            if cfg.DATALOADER.USE_TRAIN_AS_UNLABELED and cfg.DATALOADER.TRAIN_PERCENTAGE != 100:
                for aoi_id in cfg.DATASET.TRAIN_IDS:
                    if aoi_id not in self.aoi_ids:
                        aoi_ids_unlabelled.append(aoi_id)
            aoi_ids_unlabelled = sorted(aoi_ids_unlabelled)
            self.aoi_ids.extend(aoi_ids_unlabelled)
            self.labeled.extend([False] * len(aoi_ids_unlabelled))

        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_MULTIPLIER
            self.labeled = self.labeled * cfg.DATALOADER.TRAINING_MULTIPLIER

        manager = multiprocessing.Manager()
        self.unlabeled_ids = manager.list(list(self.cfg.DATASET.UNLABELED_IDS))
        self.aoi_ids = manager.list(self.aoi_ids)
        self.labeled = manager.list(self.labeled)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]
        labeled = self.labeled[index]
        timestamps = self.metadata[aoi_id]
        if labeled:
            timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1'] and ts['s2'] and ts['buildings'] and not ts['masked']]
        else:
            timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1'] and ts['s2']]

        if self.dataset_mode == 'first_last':
            indices = [0, -1]
        else:
            indices = sorted(np.random.randint(0, len(timestamps), size=2))

        # t1
        year_t1, month_t1 = timestamps[indices[0]]
        img_s1_t1 = self._load_s1_img(aoi_id, year_t1, month_t1)
        img_s2_t1 = self._load_s2_img(aoi_id, year_t1, month_t1)

        # t2
        year_t2, month_t2 = timestamps[indices[1]]
        img_s1_t2 = self._load_s1_img(aoi_id, year_t2, month_t2)
        img_s2_t2 = self._load_s2_img(aoi_id, year_t2, month_t2)

        if labeled:
            change = self._load_change_label(aoi_id, year_t1, month_t1, year_t2, month_t2)
            if self.include_building_labels:
                buildings_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
                buildings_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
                buildings = np.concatenate((buildings_t1, buildings_t2), axis=-1).astype(np.float32)
            else:
                buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)
        else:
            change = np.zeros((img_s1_t1.shape[0], img_s1_t1.shape[1], 1), dtype=np.float32)
            buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)

        # transformation, but this ain't pretty
        imgs = np.concatenate((img_s1_t1, img_s1_t2, img_s2_t1, img_s2_t2), axis=-1)
        imgs, buildings, change = self.transform((imgs, buildings, change))

        imgs_s1 = imgs[:2*len(self.s1_band_indices), ]
        img_s1_t1, img_s1_t2 = imgs_s1[:len(self.s1_band_indices), ], imgs_s1[len(self.s1_band_indices):, ]
        imgs_s2 = imgs[2*len(self.s1_band_indices):, ]
        img_s2_t1, img_s2_t2 = imgs_s2[:len(self.s2_band_indices), ], imgs_s2[len(self.s2_band_indices):, ]

        if self.cfg.DATALOADER.INPUT_MODE == 's1':
            x_t1, x_t2 = img_s1_t1, img_s1_t2
        elif self.cfg.DATALOADER.INPUT_MODE == 's2':
            x_t1, x_t2 = img_s2_t1, img_s2_t2
        else:
            x_t1 = torch.concat((img_s1_t1, img_s2_t1), dim=0)
            x_t2 = torch.concat((img_s1_t2, img_s2_t2), dim=0)

        item = {
            'x_t1': x_t1,
            'x_t2': x_t2,
            'y_change': change,
            'aoi_id': aoi_id,
            'year_t1': year_t1,
            'month_t1': month_t1,
            'year_t2': year_t2,
            'month_t2': month_t2,
            'is_labeled': labeled,
        }

        if self.include_building_labels:
            buildings_t1, buildings_t2 = buildings[0, ], buildings[1, ]
            item['y_sem_t1'] = buildings_t1.unsqueeze(0)
            item['y_sem_t2'] = buildings_t2.unsqueeze(0)

        return item

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class DeploymentDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, site: str):
        super().__init__()

        self.site = site
        self.data_path = Path(cfg.PATHS.DATASET)
        self.cfg = cfg
        self.s1_bands = [0, 1]
        self.s2_bands = [2, 1, 0, 3]

        self.tile_size = cfg.AUGMENTATION.CROP_SIZE

        # making sure that all data is available
        s1_t1_file = self.data_path / f'sentinel1_{site}_t1.tif'
        s1_t2_file = self.data_path / f'sentinel1_{site}_t2.tif'
        s2_t1_file = self.data_path / f'sentinel2_{site}_t1.tif'
        s2_t2_file = self.data_path / f'sentinel2_{site}_t2.tif'
        assert (s1_t1_file.exists() and s1_t2_file.exists() and s2_t1_file.exists() and s2_t2_file.exists())

        self.s1_t1, self.transform, self.crs = geofiles.read_tif(s1_t1_file)
        self.s1_t2, *_ = geofiles.read_tif(s1_t2_file)
        self.s2_t1, *_ = geofiles.read_tif(s2_t1_file)
        self.s2_t2, *_ = geofiles.read_tif(s2_t2_file)

        m, n, _ = self.s1_t1.shape
        self.m, self.n = (m // self.tile_size) * self.tile_size, (n // self.tile_size) * self.tile_size
        self.tiles = []
        for i in range(0, self.m , self.tile_size):
            for j in range(0, self.n, self.tile_size):
                tile = {
                    'i': i,
                    'j': j,
                }
                self.tiles.append(tile)

        manager = multiprocessing.Manager()
        self.tiles = manager.list(self.tiles)

        self.length = len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        i, j = tile['i'], tile['j']

        tile_s1_t1 = self.s1_t1[i:i + self.tile_size, j:j + self.tile_size, self.s1_bands]
        tile_s1_t2 = self.s1_t2[i:i + self.tile_size, j:j + self.tile_size, self.s1_bands]
        tile_s2_t1 = self.s2_t1[i:i + self.tile_size, j:j + self.tile_size, self.s2_bands]
        tile_s2_t2 = self.s2_t2[i:i + self.tile_size, j:j + self.tile_size, self.s2_bands]

        tile_s1_t1, tile_s1_t2 =  TF.to_tensor(tile_s1_t1), TF.to_tensor(tile_s1_t2)
        tile_s2_t1, tile_s2_t2 = TF.to_tensor(tile_s2_t1), TF.to_tensor(tile_s2_t2)

        if self.cfg.DATALOADER.INPUT_MODE == 's1':
            x_t1, x_t2 = tile_s1_t1, tile_s1_t2
        elif self.cfg.DATALOADER.INPUT_MODE == 's2':
            x_t1, x_t2 = tile_s2_t1, tile_s2_t2
        else:
            x_t1 = torch.concat((tile_s1_t1, tile_s2_t1), dim=0)
            x_t2 = torch.concat((tile_s1_t2, tile_s2_t2), dim=0)

        item = {
            'x_t1': x_t1,
            'x_t2': x_t2,
            'i': i,
            'j': j,
        }

        return item

    def get_arr(self, c: int = 1):
        return np.zeros((self.m, self.n), dtype=np.uint8) if c == 1 else np.zeros((self.m, self.n, c), dtype=np.uint8)

    def get_geo(self):
        return self.transform, self.crs

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class MultimodalCDDatasetStockholm(AbstractMultimodalCDDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 dataset_mode: str = None, disable_multiplier: bool = False, disable_unlabeled: bool = False):
        super().__init__(cfg, run_type)

        self.dataset_mode = cfg.DATALOADER.DATASET_MODE if dataset_mode is None else dataset_mode
        self.include_building_labels = cfg.DATALOADER.INCLUDE_BUILDING_LABELS

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        if run_type == 'train':
            if cfg.DATALOADER.TRAIN_PERCENTAGE == 100:
                self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
            elif cfg.DATALOADER.TRAIN_PERCENTAGE == 10:
                self.aoi_ids = list(cfg.DATASET.TRAIN_10_IDS)
            elif cfg.DATALOADER.TRAIN_PERCENTAGE == 20:
                self.aoi_ids = list(cfg.DATASET.TRAIN_20_IDS)
            elif cfg.DATALOADER.TRAIN_PERCENTAGE == 40:
                self.aoi_ids = list(cfg.DATASET.TRAIN_40_IDS)
        elif run_type == 'val':
            self.aoi_ids = list(cfg.DATASET.VALIDATION_IDS)
        elif run_type == 'test':
            self.aoi_ids = list(cfg.DATASET.TEST_IDS)
        else:
            raise Exception('Unknown run type!')

        self.labeled = [True] * len(self.aoi_ids)

        if cfg.DATALOADER.INCLUDE_UNLABELED and not disable_unlabeled:
            assert (run_type == 'train')
            aoi_ids_unlabelled = list(cfg.DATASET.UNLABELED_IDS)
            if cfg.DATALOADER.USE_TRAIN_AS_UNLABELED and cfg.DATALOADER.TRAIN_PERCENTAGE != 100:
                for aoi_id in cfg.DATASET.TRAIN_IDS:
                    if aoi_id not in self.aoi_ids:
                        aoi_ids_unlabelled.append(aoi_id)
            aoi_ids_unlabelled = sorted(aoi_ids_unlabelled)
            self.aoi_ids.extend(aoi_ids_unlabelled)
            self.labeled.extend([False] * len(aoi_ids_unlabelled))

        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_MULTIPLIER
            self.labeled = self.labeled * cfg.DATALOADER.TRAINING_MULTIPLIER

        if cfg.DATALOADER.FINETUNE_STOCKHOLM:
            self.sthlm_data_path = Path(cfg.PATHS.STHLM_DATA)
            tiles = geofiles.load_json(self.sthlm_data_path / 'tiles.json')
            self.aoi_ids.extend(tiles)
            self.labeled.extend([False] * len(tiles))
            self.sthlm_years = cfg.DATALOADER.STOCKHOLM_YEARS
            assert(self.s1_band_indices == [0, 1] and self.s2_band_indices == [2, 1, 0, 6])
            self.sthlm_buildings = self.cfg.DATALOADER.STOCKHOLM_BUILDINGS

        manager = multiprocessing.Manager()
        self.unlabeled_ids = manager.list(list(self.cfg.DATASET.UNLABELED_IDS))
        self.aoi_ids = manager.list(self.aoi_ids)
        self.labeled = manager.list(self.labeled)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]
        labeled = self.labeled[index]

        if aoi_id[:9] == 'stockholm':
            indices = sorted(np.random.randint(0, len(self.sthlm_years), size=2))
            year_t1, year_t2 = np.array(self.sthlm_years)[indices]
            month_t1 = month_t2 = 0
            tile_id = aoi_id[10:]
            img_s1_t1 = self._load_s1_img_sthlm(tile_id, year_t1)
            img_s2_t1 = self._load_s2_img_sthlm(tile_id, year_t1)
            img_s1_t2 = self._load_s1_img_sthlm(tile_id, year_t2)
            img_s2_t2 = self._load_s2_img_sthlm(tile_id, year_t2)

            change = np.zeros((img_s1_t1.shape[0], img_s1_t1.shape[1], 1), dtype=np.float32)
            buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)
            if self.sthlm_buildings.INCLUDE:
                sthlm_buildings = self._load_sthlm_buildings(tile_id)[:, :, 0]
                if self.sthlm_buildings.YEAR == year_t1:
                    buildings[:, :, 0] = sthlm_buildings
                if self.sthlm_buildings.YEAR == year_t2:
                    buildings[:, :, 1] = sthlm_buildings
        else:
            timestamps = self.metadata[aoi_id]
            if labeled:
                timestamps = [(ts['year'], ts['month']) for ts in timestamps if
                              ts['s1'] and ts['s2'] and ts['buildings'] and not ts['masked']]
            else:
                timestamps = [(ts['year'], ts['month']) for ts in timestamps if ts['s1'] and ts['s2']]

            if self.dataset_mode == 'first_last':
                indices = [0, -1]
            else:
                indices = sorted(np.random.randint(0, len(timestamps), size=2))

            # t1
            year_t1, month_t1 = timestamps[indices[0]]
            img_s1_t1 = self._load_s1_img(aoi_id, year_t1, month_t1)
            img_s2_t1 = self._load_s2_img(aoi_id, year_t1, month_t1)

            # t2
            year_t2, month_t2 = timestamps[indices[1]]
            img_s1_t2 = self._load_s1_img(aoi_id, year_t2, month_t2)
            img_s2_t2 = self._load_s2_img(aoi_id, year_t2, month_t2)

            if labeled:
                change = self._load_change_label(aoi_id, year_t1, month_t1, year_t2, month_t2)
                if self.include_building_labels:
                    buildings_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
                    buildings_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
                    buildings = np.concatenate((buildings_t1, buildings_t2), axis=-1).astype(np.float32)
                else:
                    buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)
            else:
                change = np.zeros((img_s1_t1.shape[0], img_s1_t1.shape[1], 1), dtype=np.float32)
                buildings = np.zeros((change.shape[0], change.shape[1], 2), dtype=np.float32)

        # transformation, but this ain't pretty
        imgs = np.concatenate((img_s1_t1, img_s1_t2, img_s2_t1, img_s2_t2), axis=-1)
        imgs, buildings, change = self.transform((imgs, buildings, change))

        imgs_s1 = imgs[:2 * len(self.s1_band_indices), ]
        img_s1_t1, img_s1_t2 = imgs_s1[:len(self.s1_band_indices), ], imgs_s1[len(self.s1_band_indices):, ]
        imgs_s2 = imgs[2 * len(self.s1_band_indices):, ]
        img_s2_t1, img_s2_t2 = imgs_s2[:len(self.s2_band_indices), ], imgs_s2[len(self.s2_band_indices):, ]

        if self.cfg.DATALOADER.INPUT_MODE == 's1':
            x_t1, x_t2 = img_s1_t1, img_s1_t2
        elif self.cfg.DATALOADER.INPUT_MODE == 's2':
            x_t1, x_t2 = img_s2_t1, img_s2_t2
        else:
            x_t1 = torch.concat((img_s1_t1, img_s2_t1), dim=0)
            x_t2 = torch.concat((img_s1_t2, img_s2_t2), dim=0)

        item = {
            'x_t1': x_t1,
            'x_t2': x_t2,
            'y_change': change,
            'aoi_id': aoi_id,
            'year_t1': year_t1,
            'month_t1': month_t1,
            'year_t2': year_t2,
            'month_t2': month_t2,
            'is_labeled': labeled,
            'is_sthlm': True if aoi_id[:9] == 'stockholm' else False,
        }

        if self.include_building_labels:
            buildings_t1, buildings_t2 = buildings[0,], buildings[1,]
            item['y_sem_t1'] = buildings_t1.unsqueeze(0)
            item['y_sem_t2'] = buildings_t2.unsqueeze(0)

        return item

    def _load_s1_img_sthlm(self, tile_id: str, year: int) -> np.ndarray:
        file = self.sthlm_data_path / str(year) / 's1' / f's1_stockholm_{year}-{tile_id}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img, 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_s2_img_sthlm(self, tile_id: str, year: int) -> np.ndarray:
        file = self.sthlm_data_path / str(year) / 's2' / f's2_stockholm_{year}-{tile_id}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = np.clip(img[:, :, [2, 1, 0, 3]], 0, 1)
        return np.nan_to_num(img).astype(np.float32)

    def _load_sthlm_buildings(self, tile_id: str) -> np.ndarray:
        file = self.sthlm_data_path / 'buildings' / f'buildings_stockholm-{tile_id}.tif'
        label, _, _ = geofiles.read_tif(file)
        label = label > 0
        return np.nan_to_num(label).astype(np.float32)

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'