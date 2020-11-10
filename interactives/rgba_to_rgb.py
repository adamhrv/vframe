# %% markdown
# # Scratchpad temporary code

# %% codecell
import sys
import os
from os.path import join
sys.path.append(join(os.getcwd(), 'vframe_cli'))
from vframe.settings import app_cfg

# %% codecell
from PIL import Image

# %% codecell
fp = '/data_store_vframe/vframe/renders/ao25rt/ao25rt_02/mask/ao25rt_02_000000.png'
fp_out = '/home/adam/Downloads/mask.png'
Image.open(fp).convert('RGB').save(fp_out)
