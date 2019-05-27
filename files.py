from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(r'F:\Documents\ISIC-Archive-Downloader-master\Data\Descriptions') if isfile(join(r'F:\Documents\ISIC-Archive-Downloader-master\Data\Descriptions', f))]

print(onlyfiles)