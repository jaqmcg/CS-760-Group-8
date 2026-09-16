# The datasets

We've prepped two. Dataset 1 is the Kaggle one we started with, dataset 2 is the proper OASIS
data. Both end up in the same shape so you can swap between them.

All numbers come from `dataset_1/01_data_preparation.ipynb` and
`dataset_2/02_oasis_preparation.ipynb`.

## Quick comparison

| | dataset 1 (Kaggle) | dataset 2 (OASIS-1) |
|---|---|---|
| people | 198 | 233 |
| images | 6,333 | 4,660 |
| image size | 128x128 | 208x176 |
| where the labels come from | folder names | actual clinical CDR scores |
| patient IDs | worked out from filenames | come with the data |
| scans lined up to a template | no | yes |
| age / MMSE / brain volume | none | yes |
| can we share it | yes | no, data agreement |
| download size | 200MB | 68GB |

**Use dataset 2 if you can.** Dataset 1 is fine for prototyping and it's much easier to pass
around, but dataset 2 is the one to report results on.

---

# Dataset 1 - Kaggle

Kaggle, `mohiburrahmanrifat/alzheimer`.

- **6,333 images** we can use (6,336 in the download, 3 have no number in the filename so we skip them)
- **198 patients**, about 32 MRI slices each
- 3 classes: Non_Demented 3,199 / Very_Mild_Demented 2,239 / Mild_Demented 895
- 128x128, greyscale, JPEG

Note it's 6,336 images, not 25k like our proposal said. The download has the same images
copied three times plus 6,000 augmented ones.

## Problems

### The train/test split it comes with is no good

Each patient has ~32 slices which are the same brain at different depths, so they look nearly
identical. The original split shuffled the *images*, so slices of the same patient ended up in
both train and test.

That means a model can score really well just by recognising a brain it's already seen, rather
than learning anything about Alzheimer's. We looked into how bad this was and it affects every
patient in the dataset.

**What we did:** threw the original split away and split by patient instead. The old split is
kept in the `original_split` column but don't train on it.

### We don't know where it came from

The original Kaggle dataset it's derived from has been deleted, and the copy that's still up
says "No description available" and "License: Unknown".

**What we did:** nothing much, but see the note at the bottom - the patient counts line up with
OASIS almost exactly, so we've got a decent idea now.

### It's not balanced

3.57 to 1. Always guessing Non_Demented already gets you ~50% accuracy.

**What we did:** worked out class weights for the loss, and we report macro-F1 and per-class
recall rather than just accuracy.

### Heavy compression

About 2.2 bits per pixel vs 8 for uncompressed, on images already shrunk to 128x128. Whatever
detail was lost is gone.

### Can't check if XAI heatmaps land on the right bit of brain

Properly checking a Grad-CAM heatmap means lining a brain atlas up with the image and seeing if
it lands on the hippocampus. That needs scans with spatial info in the file. These are flat
JPEGs with none of that.

**What we did:** made a brain mask per image instead. Only tells us brain vs background, which
is weaker, but a heatmap pointing at the black background is definitely wrong so it's still
worth having.

---

# Dataset 2 - OASIS-1

The real thing, from [oasisbrains.org](https://sites.wustl.edu/oasisbrains/). Needs
registration and a data use agreement.

- **233 people** with a usable clinical score
- 3 classes by CDR: Non_Demented 135 / Very_Mild_Demented 70 / Mild_Demented 28
- 20 slices each = **4,660 images**, 208x176
- comes with age, sex, MMSE and brain volume measurements

## Why it's better

**The labels are real.** OASIS gives you each person's Clinical Dementia Rating, which is what
a doctor actually assessed them as. CDR 0 / 0.5 / 1 maps straight onto our three classes. With
dataset 1 we're trusting a folder name.

**The patient IDs come with it.** No working them out from filenames.

**The scans are already lined up to a standard brain template.** This is the big one. It means
slice 70 is the same part of the brain in every person, so we can actually put an atlas over a
Grad-CAM heatmap and measure whether it's looking at the hippocampus. That's the whole point of
our project and dataset 1 can't do it.

OASIS also did the averaging and skull stripping for us, which saved a fair bit of work.

**We get age, MMSE and brain volume**, which lets us check the model isn't cheating (see below).

## Problems

### The classes are different ages

| class | average age |
|---|---|
| Non_Demented | 69.1 |
| Very_Mild_Demented | 76.2 |
| Mild_Demented | 77.8 |

About 9 years between the ends. A model could get a decent score just by spotting old brains
rather than sick ones.

**What we did:** kept `age`, `mmse` and `nwbv` in the output. If the predictions line up with
age better than with CDR, we've found a confound rather than a diagnosis. Someone should
actually check this.

### We had to drop a lot of people

436 rows in the spreadsheet, only 233 usable. 201 have no CDR at all - those are the younger
subjects who were never assessed, and "no score recorded" isn't the same as "healthy". Another
2 have CDR 2 (moderate), which is nowhere near enough to train on.

### Still only 28 people in the smallest class

Same as dataset 1. OASIS doesn't help here because dataset 1's Mild class *is* these 28 people.
Test ends up with 4 of them.

**What we did:** made 5 cross-validation folds as well as the normal split, so we can report an
average instead of trusting one small test set.

### Skull stripping might cause its own problems

There's a 2025 paper showing that removing the skull can make a model latch onto the edge left
behind rather than the actual brain tissue. The un-stripped version is in the tars
(`_t88_gfc` instead of `_t88_masked_gfc`) if anyone wants to compare.

### It's huge and we can't share it

68GB of tar files. Can't go in the repo, and the data agreement says we can't redistribute it
anyway. Everyone who wants it has to register themselves and run the notebook. The prepped
output is small (84MB) but it's still OASIS images so same rules apply.

---

# The two are related

Worth knowing: dataset 1 is almost certainly cut out of OASIS.

| class | OASIS patients | dataset 1 patients |
|---|---|---|
| Non_Demented | 135 | 100 |
| Very_Mild_Demented | **70** | **70** |
| Mild_Demented | **28** | **28** |

Two of the three match exactly. Whoever made the Kaggle version took all the very mild and mild
people and a subset of the healthy ones. So they're not really independent datasets - don't
treat getting a similar score on both as proof anything generalises.

# What to expect from a model

Always guessing the biggest class gets about 50% on dataset 1 and 58% on dataset 2, so anything
near that is no better than guessing.

Don't expect 99% like you see in Kaggle notebooks using this data. Once you split by patient
properly the numbers drop a lot. A review of 44 Alzheimer's papers found the ones that split
correctly get 66-90%, and the ones that don't claim 95-99%.
