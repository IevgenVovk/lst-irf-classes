import logging
import numpy as np
import pandas as pd


def mkmarkup(input_fname: str, key: str, ebinsdec: float, cuts: str = '') -> pd.DataFrame:
    """
    Marks up the PSF classes within the MC file.

    Parameters
    ----------
    input_fname: str
        input Monte Carlo file name
    key: str
        input HDF5 file key to read from
    ebinsdec: float
        number of true energy bins per dec to assume
    cuts: str
        event cuts to apply

    Returns
    -------
    df: pd.DataFrame
        MC event list with the "psf_class" column
    """

    log = logging.getLogger(__name__)
    data = pd.read_hdf(input_fname, key=key)

    if cuts:
        data = data.query(cuts)

    data['reco_offset'] = data.eval(
        'sqrt((reco_src_x - src_x)**2 + (reco_src_y - src_y)**2)'
    )
    data['psf_class'] = -1

    energy_edges = 10**np.arange(
        np.log10(data['mc_energy'].min()),
        np.log10(data['mc_energy'].max()),
        step=1 / ebinsdec
    )

    energy_ids = np.digitize(data['mc_energy'], energy_edges)

    for energy_id in np.unique(energy_ids):
        selection = energy_ids == energy_id
        mid_edges = np.percentile(
            data['reco_offset'][selection],
            [25, 50, 75]
        )
        offset_edges = np.concatenate(
            ([0], mid_edges, [np.inf])
        )
        psf_class = np.digitize(
            data['reco_offset'][selection],
            offset_edges
        )
        data.loc[selection, 'psf_class'] = psf_class

    if any(data['psf_class'].values == -1):
        log.warning(
            "not marked events found and will be dropped; "
            "this may indicate reconstructed offsets were "
            "outside the [0;inf] range"
        )
        data = data.query('psf_class != -1')

    return data