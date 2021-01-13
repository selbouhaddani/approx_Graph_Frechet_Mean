# approx_Graph_Frechet_Mean

The code provided approximates the Frechet mean of a set of graphs when considering the euclidean distance between the spectra of two graphs. The theory is outlined in **Approximate Frechet mean of sparse graphs with applications to linear regression** with an arxiv link to come. 

Here we provide both the raw data and the results for the Frechet mean as two separate files. With these files provided, one may write their own implementation fo the algorithms present here on the data and compare to our results. Our implementation of the algorithm is also provided for use.

**Raw Data**
The raw data files are titled "BLANK_BLANK_Data.mat" with the exception of "exp_1_consistency.mat". Each of these .mat files contains the parameters used to generate the data and the sample set of adjcency matrices. In the case of "exp_1_consistency.mat", the ensemble used to generate the data was the stochastic block model. 

**Converged Data**
The converged data serves as a way to compare and reproduce results of the paper. The files "BLANK_BLANK_Data_converged.mat" are the result of running approxFrechetMean.m given the raw data.

The data provided is associated with the above paper and the code can be used to generate the figures in the following way.

**Tutorial: Frechet Mean**
Step 1: Download the matlab file titled approxFrechetMean.m
Step 2: Download the data file(s)
Step 3: Load Barabasi_Albert_Data.mat into matlab (or any other raw data file)
        -This file contains 
              sampleAdjSet: the sample adjacency set of graphs
              m, m0: parameters used to generate the set from the Barabasi Albert ensemble
              n: the number of nodes
              N: the number of graphs in the set
Step 4: Run approxFrechetMean.m 
        -We suggest saving the data after this step.
Step 5: Run displayMean.m
        -This file will display the approximate Frechet mean of the set of graphs and the first graph in the observed sample set

**Tutorial: Regression**
Step 1: Download the matlab file titled 
