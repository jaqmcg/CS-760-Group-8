# Extract OASIS-1 data here

Not in the repo, and it can't be. It's 68GB, and the data use agreement we signed says we
can't redistribute it - that covers public code repos and covers the prepped output too.

**If you're in the group:** ask Jack for the OneDrive link. The zipped data and the
`prepped_data_oasis` output are both on there. Deliberately not linked here because this repo
is public.

**Otherwise you need to register yourself.** Takes about 20 minutes:

1. Go to https://sites.wustl.edu/oasisbrains/ and request access to OASIS-1
2. You'll need a research statement covering aims, methods and variables of interest
3. Download from the OASIS-1 section:
   - **CSV File with Demographic, and Clinical Data** - the spreadsheet, tiny, has the labels
   - **oasis_cross-sectional_disc1.tar.gz** through **disc12.tar.gz** - the scans
   - skip the FreeSurfer downloads, they're 9.5GB each and we don't use them

Unzip the .tar.gz files so you end up with .tar, and put everything straight in this folder:

```
Data/
├── oasis_cross-sectional_disc1.tar
├── ... (through disc12)
└── oasis_cross-sectional-<something>.xlsx
```

Then run `02_oasis_preparation.ipynb`. It pulls the one file it needs out of each tar, so you
don't have to extract the whole 50GB.

Fair warning: reading through the tars takes a while the first time. It skips anything it's
already pulled out, so running it again is quick.
