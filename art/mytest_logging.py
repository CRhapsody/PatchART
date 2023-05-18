import logging

logging.basicConfig(filename='example.log',format='[%(asctime)s][%(levelname)s][%(message)s]',level=logging.DEBUG)
logging.debug('This message should go to the file')
logging.info('you')
logging.warning('warning')
logging.error('error')

