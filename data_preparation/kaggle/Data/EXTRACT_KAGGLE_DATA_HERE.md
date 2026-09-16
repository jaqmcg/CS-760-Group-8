# Extract Kaggle data here

The data isn't in the repo - too big, and the licence on it is listed as "Unknown".

Two ways to get it:

**OneDrive (easiest)** - grab `kaggle_data.zip` from the shared folder:
https://uoa-my.sharepoint.com/:f:/r/personal/jmcg509_uoa_auckland_ac_nz/Documents/data_preparation

**Or from Kaggle:** https://www.kaggle.com/datasets/mohiburrahmanrifat/alzheimer

Either way, unzip it into this folder so you end up with:

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

If you just want to get modelling and don't care about re-running the prep, the output is
already in `prepped_data/` in the repo, and also on the OneDrive.

Ignore the `Alzheimer_Split -with Augment` folder if you see it. Those are augmented copies of
the training images and they'll wreck your results if any end up in the test set.
