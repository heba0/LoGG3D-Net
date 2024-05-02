# How to Run The Training Script

manually change the pickle path in train_config.py because it can't load a dictionary using command line
--train_pickles '{"wildplaces": "PATH_TO_PICKLE.pickle"}' \

```
torchpack dist-run -np ${_NGPU} python train.py \
       --wildplaces_dir {PATH_TO_WILD-PLACES_DIR} \
       --resume_training True \
       --resume_checkpoint {PATH_TO_PRETRAINED_CHECKPOINT} \
       --dataset GeneralPointSparseTupleDataset \
```