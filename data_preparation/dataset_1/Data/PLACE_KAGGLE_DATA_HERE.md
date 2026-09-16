# Place Kaggle data here

Not in the repo - too big and the licence on it is listed as "Unknown".

Grab it from either:

- OneDrive (ask Jack for the link) - easiest, it's already unzipped
- Kaggle: https://www.kaggle.com/datasets/mohiburrahmanrifat/alzheimer

You want the `Alzheimer_Split` folder, so this ends up looking like:

```
Data/
└── Alzheimer_Split/
    └── Alzheimer_Split/
        ├── train/
        │   ├── Mild_Demented/
        │   ├── Non_Demented/
        │   └── Very_Mild_Demented/
        ├── val/
        └── test/
```

Then run `01_data_preparation.ipynb`.

Ignore the `Alzheimer_Split -with Augment` folder if you see it. Those are augmented copies
of the training images and they'll wreck your results if any land in the test set.
