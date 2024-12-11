## AAC Data for the Paper 'Just Read the Codebook! Make Use of Quality Codebooks in Zero-Shot Classification of Multilabel Frame Datasets'

There are several files in this folder that we have created for our experiments.

### Full csv files
We have recreated the entire dataset as csv file, which was not the orignal release format (the dataset was released in conll format).
`ab/mw/mj/ne.csv` contain all data from the dataset.

### Original test files
for the three tested topics, we recreated the test splits from https://github.com/Leibniz-HBI/argument-aspect-corpus-v1 to properly compare with the baseline. 
These files are called `<dataset>_original_test.csv`

### small test files
These are for debugging and testing purposes.

### icl samples
For all dataset we have created `<dataset>_icl_sample.csv`. These are sampled from the test/dev splits, so from the full csv files minus the original test splits.  
`abortion_original_test.csv` does not refer to an original split, but to our split whch can be found in the subfolder `ab_dataset`
