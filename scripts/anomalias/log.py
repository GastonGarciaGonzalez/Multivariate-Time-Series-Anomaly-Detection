import logging
import os

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s %(process)d: [%(threadName)s] %(message)s",
                    filename=os.path.dirname(__file__) + '/../anomalias.log',
                    filemode='w')

logging.captureWarnings(True)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s: [%(threadName)s] %(message)s')
console.setFormatter(formatter)

logging.getLogger('').addHandler(console)

def logger(s):
    return logging.getLogger(s)
