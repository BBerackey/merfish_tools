import os
import pandas as pd

from mftools import barcodes
from mftools import config
from mftools import fileio


import pdb
pdb.set_trace()

# initialize merlin results object
merlin_folder = r'D:\Bereket\Merfish_tools\MERlin_test_run\DrBintu_Data_decode_test\results\XXBBL1_05_26_2023_Scope3BB'
merlin_result = fileio.MerlinOutput(merlin_folder)
codebook = merlin_result.load_codebook()
bcs = barcodes.make_table(merlin_result, codebook)

bcs.to_csv(r'D:\Bereket\Merfish_tools\MERlin_test_run\DrBintu_Data_decode_test\detected_transcripts.csv')