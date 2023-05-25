from datetime import datetime
now = datetime.now()
date_string = now.strftime("%b-%d-%Y-%H-%M-%S") # for the name of folder (log)



Labels = 150
Threshold = 2
Top_Items = 10
n_sample = 0     # Number of samples. if it's 0, all data will be used.
