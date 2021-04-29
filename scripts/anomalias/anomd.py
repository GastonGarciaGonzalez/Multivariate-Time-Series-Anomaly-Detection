import anomalias.log as log
import anomalias.series as series

logger = log.logger( 'core' )


class Anomd():
    def __init__(self):
        self.__series = []
        self.__series_id = []
        self.__ts_id = 0

    def new(self, len):
        try:
            self.__series.append(series.Series(id=self.__ts_id, len=len))
            self.__series_id.append(self.__ts_id)
            id = self.__ts_id
            self.__ts_id += 1
            return id
        except Exception as e:
            logger.error('%s', e)
            raise

    def ssm_ad(self, id, th, endog, model_type, **kwargs):
        try:
            if self.__exist_id(id):
                (self.__series[self.__series_id.index(id)]).ad.ssm_ad(th, endog, model_type, **kwargs)
        except Exception as e:
            logger.error('%s', e)
            return None

    def adtk_ad(self, id, model_type, **kwargs):
        try:
            if self.__exist_id(id):
                (self.__series[self.__series_id.index(id)]).ad.adtk_ad(model_type, **kwargs)
        except Exception as e:
            logger.error('%s', e)
            return None

    def remove(self, id):
        try:
            if self.__exist_id(id):
                index = self.__series_id.index(id)
                self.__series[index].exit()
                del self.__series[index]
                self.__series_id.remove(id)
        except Exception as e:
            logger.error('%s', e)
            return None

    def list_id(self):
        return self.__series_id

    def start(self,id):
        try:
            if self.__exist_id(id):
                if not self.__series[self.__series_id.index(id)].isAlive():
                    self.__series[self.__series_id.index(id)].start()
                else:
                    logger.warning('Series is running, id: %s', id)
        except Exception as e:
            logger.error('%s', e)

    def append(self, obs, id):
        if self.__exist_id(id):
            self.__series[self.__series_id.index(id)].append(obs)

    def train(self, series, id):
        if self.__exist_id(id):
            self.__series[self.__series_id.index(id)].ad.train(series)


    def get_detection(self, id):
        try:
            if self.__exist_id(id):
                if self.__series[self.__series_id.index(id)].isAlive():
                    endog = (self.__series[self.__series_id.index(id)]).ad.endog
                    idx_anom = (self.__series[self.__series_id.index(id)]).ad.idx_anom
                    return endog, idx_anom
                else:
                    logger.info('Series is not running, id: %s', id)
                    return None
        except Exception as e:
            logger.error('%s', e)
            return None

    def __exist_id(self, id):
        try:
            if self.__series_id.__contains__(id):
                return True
            else:
                logger.warning('Series not found, id: %s', id)
                return False
        except Exception as e:
            logger.error('%s', e)
            return False
