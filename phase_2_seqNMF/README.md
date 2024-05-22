# seqNMF

## Description
SeqNMF is an algorithm which uses regularized convolutional non-negative matrix factorization to extract repeated sequential patterns from high-dimensional data. It has been validated using neural calcium imaging, spike data, and spectrograms, and allows the discovery of patterns directly from timeseries data without reference to external markers.




## Usage
The main function is seqNMF.m and it can be called 
```matlab
[W,H,cost,loadings,power] = seqNMF(X,'K',K,'L',L,'lambda',0.01)
```
Where X is the data matrix, K and L are the factorization parameters and lambda is a parameter controling the strength of regularization.

Specifically seqNMF factorizes the NxT data matrix X into K factors. Factor exemplars are returned in the NxKxL tensor W. Factor timecourses are returned in the KxT matrix H

                                    ----------    
                                L  /         /|
                                  /         / |
        ----------------         /---------/  |          ----------------
        |              |         |         |  |          |              |
      N |      X       |   =   N |    W    |  /   (*)  K |      H       |           
        |              |         |         | /           |              |
        ----------------         /----------/            ----------------
               T                      K                         T

### Demo preparation
1. For the **Color** and **Driving** data, use the data generated from folder `phase_1_CItests`.
2. For the **Kitchen** data, use `python save_data_to_mat` to transform the original data to mat data for later use.

### Running the example
1. One way to run the example is run it in python mode by calling the `Matlab.engine`.
   ```python run_seqNMF.py```
2. Or you can do it directly in matlab by loading the corresponding data, and using `demo_color.m`, `demo_driving.m` and `demo_kitchen.m`.

### Tools
1. For calculation the precision, recall, F1 score on Color Dataset, call `python calcu_prec_recall_subtask.py`


* Credit to: https://github.com/FeeLab/seqNMF
For more information see their [**paper: https://elifesciences.org/articles/38471**](https://elifesciences.org/articles/38471); [**COSYNE talk**](https://www.youtube.com/watch?reload=9&v=XyWtCtZ_m-8); tutorial [**video**](https://cbmm.mit.edu/video/unsupervised-discovery-temporal-sequences-high-dimensional-datasets) and [**materials**](https://stellar.mit.edu/S/project/bcs-comp-tut/materials.html); and Simons foundation [**article**](https://www.simonsfoundation.org/2018/05/04/finding-neural-patterns-in-the-din/).



