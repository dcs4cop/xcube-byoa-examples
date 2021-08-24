import unittest

import numpy as np
import xarray as xr

from xcube.util.jsonschema import JsonObjectSchema
from ..xcube_byoa_s2l2a.mci import MciDatasetProcessor

SCF_FLAGS = (
    (0, 'no_data'),
    (1, 'saturated_or_defective'),
    (2, 'dark_area_pixels'),
    (3, 'cloud_shadows'),
    (4, 'vegetation'),
    (5, 'bare_soils'),
    (6, 'water'),
    (7, 'clouds_low_probability_or_unclassified'),
    (8, 'clouds_medium_probability'),
    (9, 'clouds_high_probability'),
    (10, 'cirrus'),
    (11, 'snow_or_ice'),
)
SCF_ATTRS = {
    'flag_values': ','.join(map(lambda f: str(f[0]), SCF_FLAGS)),
    'flag_meanings': ' '.join(map(lambda f: f[1], SCF_FLAGS)),
}


class MciTest(unittest.TestCase):
    def test_params_schema(self):
        processor = MciDatasetProcessor()
        self.assertIsInstance(processor.get_process_params_schema(),
                              JsonObjectSchema)

    def test_process_dataset_no_params(self):
        processor = MciDatasetProcessor()
        dims = ('time', 'lat', 'lon')
        input_dataset = xr.Dataset(
            data_vars=dict(
                B04=xr.DataArray(
                    np.array([[[0.3, 0.4, 0.5, 0.6],
                               [0.4, 0.6, 0.8, 1.0],
                               [0.5, 0.6, 0.7, 0.8]]], dtype=np.float32),
                    dims=dims,
                    attrs=dict(wavelength=510)
                ),
                B05=xr.DataArray(
                    np.array([[[0.2, 0.3, 0.4, 0.5],
                               [0.3, 0.4, 0.5, 0.6],
                               [0.4, 0.6, 0.8, 1.0]]], dtype=np.float32),
                    dims=dims,
                    attrs=dict(wavelength=590)
                ),
                B06=xr.DataArray(
                    np.array([[[0.1, 0.3, 0.6, 0.9],
                               [0.2, 0.3, 0.4, 0.5],
                               [0.3, 0.4, 0.5, 0.6]]], dtype=np.float32),
                    dims=dims,
                    attrs=dict(wavelength=600)
                ),
                SCL=xr.DataArray(
                    np.array([[[6, 6, 0, 6],
                               [6, 6, 5, 6],
                               [6, 1, 6, 6]]], dtype=np.uint8),
                    dims=dims,
                    attrs=SCF_ATTRS,
                )
            ),
            coords=dict(
                time=xr.DataArray(np.array(['2021-06-07 10:30:00'],
                                           dtype='datetime64[ns]'),
                                  dims='time'),
                lat=xr.DataArray(np.array([53.1, 53.2, 53.3]),
                                 dims='lat'),
                lon=xr.DataArray(np.array([11.3, 11.4, 11.5, 11.6]),
                                 dims='lon'),
            )
        )

        input_dataset = input_dataset.chunk(dict(lon=4, lat=3))

        output_dataset = processor.process_dataset(input_dataset)

        self.assertIn('B04', output_dataset)
        self.assertIn('B05', output_dataset)
        self.assertIn('B06', output_dataset)
        self.assertIn('MCI', output_dataset)
        self.assertEqual('Maximum Chlorophyll Index (MCI)',
                         output_dataset.attrs.get('title'))

        mci_values = output_dataset.MCI.values
        np.testing.assert_almost_equal(
            mci_values,
            np.array([[[0.0777778, -0.0111111, np.nan, -0.3666666],
                       [0.0777778, 0.0666667, np.nan, 0.0444445],
                       [0.0777778, np.nan, 0.2777778, 0.3777778]]]),
        )
