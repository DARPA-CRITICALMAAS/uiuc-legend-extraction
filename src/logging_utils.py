import os
import sys
import logging

log = logging.getLogger('DARPA_CMASS')

def setup_logger(filepath, debuglvl):
    # Create directory if necessary
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(os.path.dirname(filepath))

    # Create Formatter
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    #log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:(%(lineno)d) - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    # Setup File handler
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(debuglvl)

    # Setup Stream handler (i.e. console)
    #stream_handler = logging.StreamHandler(stream=sys.stdout)
    #stream_handler.setFormatter(log_formatter)
    #stream_handler.setLevel(debuglvl)

    # Add Handlers to logger
    log.addHandler(file_handler)
    #log.addHandler(stream_handler)
    log.setLevel(debuglvl)

def set_log_file(filename):
    for h in log.handlers:
        if isinstance(h, logging.FileHandler):
            old_formatter = h.formatter
            old_level = h.level
            old_filename = h.baseFilename
            h.flush()
            h.close()
            log.removeHandler(h)
            os.rename(old_filename,filename)
            new_handler = logging.FileHandler(filename)
            new_handler.setFormatter(old_formatter)
            new_handler.setLevel(old_level)
            log.addHandler(new_handler)

def set_log_level(loglvl):
    for h in log.handlers:
        h.setLevel(loglvl)
    log.setLevel(loglvl)