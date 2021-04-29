
import numpy as np
import pandas as pd
import anomalias.log as log
from anomalias.tsmodels import SSM_AD
from anomalias.adtk import Adtk_AD
logger = log.logger('AnomalyDetector')


class Detector:
    def __init__(self, len):

        # Series
        self.__len = len
        #self.endog = np.zeros(self.__len)
        self.endog = pd.DataFrame([])
        #self.idx_anom = [False] * len
        self.idx_anom = pd.DataFrame([])

        self.__training = False
        self.__paused = False

    def ssm_ad(self, th, endog, model_type, **kwargs):
        # Model
        logger.info('Creating Anomaly Detector...')
        self.__model = SSM_AD(th, endog, model_type, **kwargs)

    def adtk_ad(self, model_type, **kargs):
        self.__model = Adtk_AD(model_type, **kargs)

    def train(self, serie):
        self.__model.train(serie)

    def anom_detect(self, observations):
        # Series Update
        self.endog = pd.concat([self.endog, observations]).iloc[-self.__len:]
        # Detection
        idx_anom = self.__model.detect(observations)
        self.idx_anom = pd.concat([self.idx_anom, idx_anom]).iloc[-self.__len:]
