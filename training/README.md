# How to Run The Training Script

```
!torchpack dist-run -np ${_NGPU} python train.py \
       --train_pickles '{"wildplaces": "PATH_TO_PICKLE.pickle"}' \
       --wildplaces_dir {PATH_TO_WILD-PLACES_DIR} \
       --resume_training True \
       --resume_checkpoint {PATH_TO_PRETRAINED_CHECKPOINT} \
       --dataset GeneralPointSparseTupleDataset \
```