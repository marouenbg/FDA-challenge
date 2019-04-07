This is my solution for the FDA sample mislabeling challenge
https://precision.fda.gov/challenges/4

The main challenge with the prediction of mismatched samples and clinical data was the high
number of missing proteomic values. Consequently, the handling of missing values is of
paramount importance in correctly classifying samples.

My approach was to either i) impute the missing values by the mean or ii) consider ternary
classification trees that have a third branch for missing values. The approach for sex
classification was based on the identification of Y-linked
proteins and setting the missing values to zeros. The missing values in these columns are not
likely missing but rather "unmeasurable".

The classifier chosen for this task was xgboost because of its ability to handle missing values
optimally and because its hyperparameters are amenable to optimisation and tuning. A classifier
of sex and a classifier of msi were both developed and optimised and their results were
compared to assess the occurrence of mislabeling events. I had ten submissions using different
parameters of the classifier and data imputation approach.

In the two last submissions, a variant of the detection of mislabeling event was developed to
account for the higher reliability of sex prediction in comparison to msi prediction performance
on training data. The sex and msi predictors were ensembled in one classifier. In this case, the
detection of mislabeling gives a higher weight to the prediction of the sex classifier in detecting
mislabeling.

submission 1: Missing values were imputed by the column mean and missing values in Y-linked
proteins were set to 0.

submission 2: A variant of 1 with optimized hyperparameters.

submission 3: Missing values were treated as missing in the gradient boost tree and Y-linked
proteins were set to 0.

submission 4: Missing values were treated as missing in the gradient boost tree.

submission 5: Only the sex classifier from submission 1.

submission 6: Only the sex classifier from submission 4.

submission 7: Only the msi classifier from submission 2.

submission 8: Only the msi classifier from submission 4.

submission 9: Applied threshold for classification in submission 3.

submission 10: Applied threshold for classification in submission 2.
