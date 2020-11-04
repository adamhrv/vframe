# %% markdown
# # Scratchpad temporary code

# %% codecell
import sys
import os
from os.path import join
sys.path.append(join(os.getcwd(), 'vframe_cli'))
from vframe.settings import app_cfg

# %% codecell
print(app_cfg.DIR_MODELS)
