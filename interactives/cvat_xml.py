# %% markdown
# # Scratchpad temporary code

# %% codecell
import sys
import os
from os.path import join
sys.path.append(join(os.getcwd(), 'vframe_cli'))
from vframe.settings import app_cfg

# %% codecell
import xmltodict
from pprint import pprint
from vframe.models.geometry import BBox

# %% codecell
fp_xml = '/media/adam/hamlet_2tb/data_store/vframe/cvat/ao25rt/annotations.xml'
with open(fp_xml) as fp:
    cvat_data = xmltodict.parse(fp.read())

# %% codecell
tracks = cvat_data.get('annotations').get('track')
print(len(tracks))

# %% codecell
size = cvat_data.get('annotations').get('meta').get('task').get('original_size')
w,h = (float(size.get('width')), float(size.get('height')))

# %% codecell
for track in tracks:
    for b in track.get('box'):
        xyxy = (b.get('@xtl'), b.get('@ytl'), b.get('@xbr'), b.get('@ybr'))

        xyxy = list(map(int,map(float, xyxy)))
        bbox = BBox(*xyxy,w,h)
        print(bbox)
        break
    break
