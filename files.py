from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(r'C:\Users\davyd\Documents\cancer-project\descriptions') if isfile(join(r'C:\Users\davyd\Documents\cancer-project\descriptions', f))]

print(onlyfiles)