import time
import logging

def get_logger( class_name, verbose = True ):
    logger = logging.getLogger( f'{class_name}' )

    logger.propagate = True

    if not logger.hasHandlers():
        log_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt="[%(levelname)s/%(processName)s/%(name)s] %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
        log_handler.setFormatter(log_formatter)

        logger.addHandler(log_handler)

    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel( level )

    return logger

class Timer:
    def __init__(self, text = '', logger = None):
        self.text = text
        self.logger = logger
        self.elapsed_time = None

    def __enter__(self):
        self.start = time.perf_counter()
        if self.logger:
            self.logger.info( f'Start: "{self.text}"' )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = time.perf_counter()
        time_count = end - self.start
        if self.logger:
            self.logger.info( f'Completed: "{self.text}" in {(time_count):.04f}s')
        self.elapsed_time = time_count

    @property
    def time( self ):
        curr_time = time.perf_counter()
        return curr_time - self.start
