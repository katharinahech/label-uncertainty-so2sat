# label-uncertainty-so2sat

## Categorising the world into local climate zones: towards quantifying labelling uncertainty for machine learning models
Katharina Hechinger, Xiao Xiang Zhu, Göran Kauermann 

### Abstract 

Image classification is often prone to labelling uncertainty. To generate suitable training data, images are labelled according to evaluations of human experts. This can result in ambiguities, which will affect subsequent models. In this work, we aim to model the labelling uncertainty in the context of remote sensing and the classification of satellite images. We construct a multinomial mixture model given the evaluations of multiple experts. This is based on the assumption that there is no ambiguity of the image class, but apparently in the experts’ opinion about it. The model parameters can be estimated by a stochastic expectation maximisation algorithm. Analysing the estimates gives insights into sources of label uncertainty. Here, we focus on the general class ambiguity, the heterogeneity of experts, and the origin city of the images. The results are relevant for all machine learning applications where image classification is pursued and labelling is subject to humans.

### Repository 

The repository contains the code to perform the analyses described in the paper. It is structured as follows.
The folder `data` contains the estimation results based on the proposed model for the full dataset, the expert-specific results and the results for separate cities. 
The folder `src` contains scripts to define the functions needed for specific tasks.
The folder `notebooks` contains the computations: 
- `prepare_datasets`: code to prepare the datasets as needed
- `run_sem`: run the model and compute estimates
- `results`: perform analyses and evaluate results
- `supplementary_material`: contains analyses for the supplementary material



