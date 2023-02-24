# Requirements
Python 3.7+ (Development in 3.10.9)

# Files
To access the directory. Please follow the link here: https://gatech.box.com/s/r5jomjljhfrkhj7u41iiodbv1yz2qyf9

> ModelAnalysis.py
> utils.py
> run_analysis.py
> environment.yml

# Environment Setup
conda env create -f environment.yml --force
conda activate CS7641

# Example Usage
python run_analysis.py --datasets nfl income --models decision_tree mlp knn

This will run the decision_tree, knn, and mlp analysis across the nfl and income datasets.

The datasets allowed are: nfl income
Models allowed are: decision_tree, svm, mlp, ada_boost, knn

# Program Descriptions
ModelAnalysis.py
This file contains the ModelAnalysis object. This object does all the analysis.

utils.py
This file is needed to import the datasets needed.

run_analysis.py
This is the command line parser that runs the ModelAnalysis specified by the given params.
For additional analysis, you can alter the hyperparameters to optimize for and ranges by altering
the maps in the code.

