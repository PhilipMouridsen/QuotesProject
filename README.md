# QuotesProject
Research Project on Quote Extraction and Analysis by Lasse Funder Andersen & Philip Cronquist Mouridsen.
This repo contains code and (sample) data files
## Main Code

### quotebert.py
Class for generating BERT vectors

### segmentizer.py
Class that splits strings into segments divided by "."

### prototype.py
Main class predicting on unknown text and producing a visualization of result.
Using the data files in the models folder for illustration. These models are NOT the full scale versions as they are too large for version control. Result from report cannot be reproduced.

## Data

### models folder
folder holding the files with the BERT vectors after conversion.
Please note that this repo only contains small sample files to make prototype.py. 

## Misc Scripts
### dataexploration.py
Script for generating charts and figures describing the data

### extract_ftdata.py
Script to extract the speeches from Folketinget

### learningcurve.py
Script for generating learning curves

### modeltester.py
Script for testing different models and generating data

### negativedata.py
Script for preparing the data containing negative samples

### positivedata.py
Script for preparing the data containing positive samples


### QuoteExtractor.py
Script for extracting quotes from articles

### Scraper.py
Script for scraping dr.dk articles based on urls

