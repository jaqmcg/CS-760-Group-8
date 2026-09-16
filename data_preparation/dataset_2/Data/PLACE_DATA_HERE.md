# Put the OASIS data in this folder

Not in the repo, and it can't be. It's 68GB, and the data use agreement we signed says we
can't redistribute it - that includes public code repos and includes the prepped output.

You have to register yourself, it takes about 20 minutes:

1. Go to https://sites.wustl.edu/oasisbrains/ and request access to OASIS-1
2. You'll need a research statement (there's a copy of ours if you want to crib it)
3. Download from the OASIS-1 section:
   - **CSV File with Demographic, and Clinical Data** - the spreadsheet, tiny, has the labels
   - **oasis_cross-sectional_disc1.tar.gz** through **disc12.tar.gz** - the scans
   - skip the FreeSurfer downloads, they're 9.5GB each and we don't need them

Unzip the .tar.gz files so you end up with .tar, and put everything in an `official_data`
folder:

```
Data/
└── official_data/
    ├── oasis_cross-sectional_disc1.tar
    ├── ... (through disc12)
    └── oasis_cross-sectional-<something>.xlsx
```

Then run `02_oasis_preparation.ipynb`. It pulls the one file it needs out of each tar, so you
don't have to extract the whole 50GB.

Fair warning: reading through the tars takes a while the first time. It skips anything it's
already pulled out, so running it again is quick.
