# Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments
## [Website](https://csiro-robotics.github.io/Wild-Places/) | [Paper](https://arxiv.org/abs/2211.12732) | [Data Download Portal](https://data.csiro.au/collection/csiro:56372?q=wild-places&_st=keyword&_str=1&_si=1)
![](./utils/docs/teaser_image.png)


This repository contains the code implementation used in the paper *Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments*, which has been accepted for publication at ICRA2023.  

If you find this dataset helpful for your research, please cite our paper using the following reference:
```
@inproceedings{2023wildplaces,
  title={Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments},
  author={Knights, Joshua and Vidanapathirana, Kavisha and Ramezani, Milad and Sridharan, Sridha and Fookes, Clinton and Moghadam, Peyman},
  year={2023},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  eprint={arXiv preprint arXiv:2211.12732}
}
```

### Evaluation
We provide evaluation scripts for both inter and intra-run evaluation on our dataset.

#### __Updated Intra-run Evaluation__
Example to run the Intra-run evaluation
```
python scripts/eval/intra-sequence_topK.py  \
      --databases /cluster/scratch/haozhu1/Testing/V-04.pickle \
      --database_features None --pcd_path /cluster/scratch/haozhu1/56372v003/data/ \
      --logg3d_cpt LoGG3D-Net/training/checkpoints/LoGG3D-Net.pth \
      --logg3d_outdim 32 \
      --run_names V03_logg3d \
      --save_dir Wild-Places/result 
```

#### __Inter-run Evaluation__

To perform inter-run evaluation on the Wild-Places dataset, run the following command:
```
python eval/inter-sequence.py \
    --queries $_PATH_TO_QUERIES_PICKLES \
    --databases $_PATH_TO_DATABASES_PICKLES \
    --query_features $_PATH_TO_QUERY_FEATURES \ 
    --database_features $_PATH_TO_DATABASE_FEATURES \
    --location_names $_LOCATION_NAMES \
```

Where:
- `$_PATH_TO_QUERIES_PICKLES` is a string pointing to the location of the generated query set pickle for an environment
- `$_PATH_TO_DATABASES_PICKLES` is a string pointing to the location of the generated database set pickle for an environment
- `$_PATH_TO_QUERY_FEATURES` is a string pointing towards a pickle file containing the query set features to be used in evaluation.  These features should be a list of Nxd numpy arrays or tensors, where N is the number of point cloud frames in the query set of each sequence in the environment.
- `$_PATH_TO_DATABASE_FEATURES` is a string pointing towards a pickle file containing the database set features to be used in evaluation.  These features should be a list of Nxd numpy arrays or tensors, where N is the number of point cloud frames in the database set of each sequence in the environment.
- `$_LOCATION_NAMES` is a string containing the name of the environment being evaluated

#### __Intra-run Evaluation__
To perform intra-run evaluation on the Wild-Places dataset, run the following command:
```
python eval/intra-sequence.py \
    --databases $_PATH_TO_DATABASES_PICKLES \
    --database_features $_PATH_TO_DATABASE_FEATURES \
    --run_names $_LOCATION_NAMES \
```
Where:
- `$_PATH_TO_DATABASES_PICKLES` is a string pointing to the location of the generated database set pickle for a single sequence
- `$_PATH_TO_DATABASE_FEATURES` is a string pointing towards a pickle file containing the run features to be used in evaluation.  These features should be a single Nxd numpy array or tensor, where N is the number of point cloud frames in that sequence
- `$_LOCATION_NAMES` is a string containing the name of the sequence being evaluated 

## 4. Thanks
Special thanks to the authors of the [PointNetVLAD](https://github.com/mikacuy/pointnetvlad) and [MinkLoc3D](https://github.com/jac99/MinkLoc3D), whose excellent code was used as a basis for the generation and evaluation scripts used in this repository. 

