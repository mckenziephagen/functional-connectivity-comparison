installing mpi4py on nersc: https://docs.nersc.gov/development/languages/python/parallel-python/


To get the data: 
`datalad clone https://github.com/datalad-datasets/hcp-functional-connectivity.git` 

While the HCP data isn't organized according to the BIDS standard, the derivatives I generate follow BEP-17 ("Relationship Matrices"). 

Parcellated timeseries created by `01-Processing/extract_timeseries.py` are saved to `derivatives/timeseries/...`

Generated connectivity matrices live in `hcp-functional-connectivity/derivatives/connectivity-matrices/...` according to their atlas. 











