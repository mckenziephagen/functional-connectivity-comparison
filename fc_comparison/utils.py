import pyuoi
from pyuoi.linear_model import UoI_Lasso
from mpi4py import MPI

from nilearn.connectome import ConnectivityMeasure
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LassoLarsIC


def init_model(model_str): 
    if model_str == 'uoi-lasso': 
        uoi_lasso = UoI_Lasso(estimation_score="BIC")
        comm = MPI.COMM_WORLD
        uoi_lasso.copy_X = True
        uoi_lasso.estimation_target = None
        uoi_lasso.logger = None
        uoi_lasso.warm_start = False
        uoi_lasso.comm = comm
        uoi_lasso.random_state = 1
        uoi_lasso.n_lambdas = 100

        model = uoi_lasso

    elif model_str == 'lasso-cv': 
        lasso = LassoCV(fit_intercept = True,
                        cv = 5, 
                        n_jobs=-1, 
                        max_iter=2000)

        model = lasso

    elif model_str == 'lasso-bic': 
        lasso = LassoLarsIC(criterion='bic',
                            fit_intercept = True,
                            max_iter=2000)

        model = lasso

    elif model_str == 'enet':
        enet = ElasticNetCV(fit_intercept = True,
                            cv = 5, 
                            n_jobs=-1, 
                            max_iter=2000)
        model = enet

    elif model_str in ['correlation', 'tangent']: 
        model = ConnectivityMeasure(
                kind=model_str)
        

    return model