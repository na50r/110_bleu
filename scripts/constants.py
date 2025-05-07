N = 0  # no error, no rejection
R1 = 1 # rejection, outside acceptable range
E = 2  # error, no output, nothing to log
R2 = 3 # rejection, detected wrong language
R3 = 4 # rejection, failed to detect language (error but output exists, so we log)