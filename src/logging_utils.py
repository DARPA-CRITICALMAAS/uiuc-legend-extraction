import os
import sys
import logging

log = logging.getLogger('DARPA_CMASS')

def start_logger(filepath, debuglvl, writemode='a'):
    # Create directory if necessary
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(os.path.dirname(filepath))

    # Special handle for writing to 'latest' file
    if os.path.splitext(os.path.basename(filepath.lower()))[0] == 'latest':
        # Rename previous latest log 
        with open(filepath) as fh:
            newfilename = '{}_{}.log'.format(*(fh.readline().split(' ')[0:2]))
            newfilename = newfilename.replace('/','-').replace(':','-')            
        os.rename(filepath, os.path.join(dirname, newfilename))

    # Formatter
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    #log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:(%(lineno)d) - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    # Setup File handler
    file_handler = logging.FileHandler(filepath, mode=writemode)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(debuglvl)

    # Setup Stream handler (i.e. console)
    #stream_handler = logging.StreamHandler(stream=sys.stdout)
    #stream_handler.setFormatter(log_formatter)
    #stream_handler.setLevel(logging.INFO)

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