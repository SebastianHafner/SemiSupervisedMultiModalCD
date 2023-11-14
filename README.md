
# Semi-Supervised Urban Change Detection Using Multi-Modal Sentinel-1 SAR and Sentinel-2 MSI Data 


This repository contains the official code for the following paper:

Hafner, S., Ban, Y. and Nascetti, A., 2023. Semi-Supervised Urban Change Detection Using Multi-Modal Sentinel-1 SAR and Sentinel-2 MSI Data. *Remote Sensing, 15*(21), p.5135.

[[Paper](https://doi.org/10.3390/rs15215135)] [[Dataset](https://doi.org/10.5281/zenodo.7794693)]

# Detecting urban changes in any location worldwide

We provide tools to detect urban changes in any location and for any time period (only restricted by the availability of Sentinel-1 and Sentinel-2 data). All you need is a Google account with access to [Google Earth Engine](https://earthengine.google.com/) and Google Drive:

1. Create a folder named `urban_cd_app` in your Google Drive

2. Download the model from [here]() (Google Drive) and place it in the `urban_cd_app` folder. Also make a copy of this [Colab notebook]() in the same folder. Your folder should now contain the following files:

    ```
    $ Your Google Drive setup
    Your Google Drive
    └── urban_cd_app
        ├── urban_cd_app.ipynb # this is the Colab notebook you can copied
        └── mmcr_train100.pt # this is the model you downloaded
    
    ```

3. Download satellite data for your region of interest with the UI in this [GEE script](https://code.earthengine.google.com/92a32c060345f31f643e1dacb347a4bb?hideCode=true).

4. Run the Colab notebook to detect urban changes for your region of interest.



# Replicating our results


## Dataset

The SEN12 Multi-Temporal Urban Mapping Dataset is comprised of monthly mean Sentinel-1 SAR and cloud-free Sentinel-2 MSI images for the SpaceNet 7 training and test sites. The dataset also includes monthly rasterized built-up area labels for the 60 training sites.


The dataset can be downloaded from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6914898.svg)](https://doi.org/10.5281/zenodo.6914898)

We also provide the jupyter notebook `data_download.ipynb` to recreate the dataset. The notebook contains functions to pre-process Sentinel-1 SAR data, Sentinel-2 MSI data, and building footprint data for the SpaceNet7 sites.


## Network training

To train the proposed network using semi-supervised learning from scratch, follow these steps:

### 1 Setup

Set up a virtual environment using Anaconda an install the required packages. For reference, our setup uses Ubuntu 18.04.6 LTS, Python 3.9.7, PyTorch 1.10.0, and CUDA 11.4. Additionally, rasterio (1.2.10) is required to handle GeoTIFF files. To install the `rasterio` package on Windows, consider using the [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal).


### 2 Network training

Run the ``train_mmrc.py`` file a config file of your choice. For example, to train the network using only 10 % of the labeled training sites run the following:

````
python train_mmcr.py -c mmcr_train10 -o 'path to output directory' -d 'path to dataset'
````


### 3 Model evaluation and inference

Run the files ``assessment_change.py`` and ``assessment_semantic.py`` with a config of choice and the path settings from above to assess network performance. For inference, use the file ``inference.py``.


### 4 (optional) Adding custom unlabeled data to the training

Upcoming


# Credits

If you find this work useful, please consider citing:


  ```bibtex
@article{hafner2023semi,
      title={Semi-Supervised Urban Change Detection Using Multi-Modal Sentinel-1 SAR and Sentinel-2 MSI Data},
      author={Hafner, Sebastian and Ban, Yifang and Nascetti, Andrea},
      journal={Remote Sensing},
      volume={15},
      number={21},
      pages={5135},
      year={2023},
      publisher={MDPI}
}
  ```
  