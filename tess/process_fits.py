import fitsio as fits
import numpy as np
import glob
from tqdm import tqdm
import os
from scipy.ndimage.filters import maximum_filter1d
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input')
args = parser.parse_args()

indir = os.path.join(args.input)
outdir = os.path.join('processed', indir)
os.makedirs(outdir, exist_ok=True)
files = glob.glob(os.path.join(indir,'*.fits'))

window = 50
max_len = 400

for filename in tqdm(files):
    try:
        data = fits.read(filename, ext=None)
    except:
        continue
    data = data.astype([(x, '<f8') for x in data.dtype.names])
    time = data['TIME']
    bool_time = ~np.isnan(time)
    sap_flux_o_err = data['SAP_FLUX']/data['SAP_FLUX_ERR']
    sap_flux_o_err = sap_flux_o_err/np.nanmax(sap_flux_o_err)
    bool_flux = ~np.isnan(sap_flux_o_err)
    valid = (bool_flux == bool_time) & (bool_time == True)
    time = time[valid]
#     time = np.round(time, 2)
    sap_flux_o_err = sap_flux_o_err[valid]
    assert(time.shape == sap_flux_o_err.shape)
    sap_flux_o_err = maximum_filter1d(sap_flux_o_err, size=window)
    res = np.stack([time, sap_flux_o_err])
    res = res[:, ::window]
    res[0] = np.round(res[0], 2)
    try:
        res = res[:, :max_len]
    except:
        continue
    outname = os.path.basename(filename).replace('.fits','.npy')
    outname = os.path.join(outdir, outname)
    np.save(outname, res)
