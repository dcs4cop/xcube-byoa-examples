from typing import Any, Dict

import numpy as np
import xarray as xr

from xcube.core.compute import compute_dataset
from xcube.core.gen2 import DatasetProcessor
from xcube.core.maskset import MaskSet
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

MCI_VAR_NAME = 'MCI'


class MciDatasetProcessor(DatasetProcessor):

    @classmethod
    def get_process_params_schema(cls) -> JsonObjectSchema:
        """
        Get the JSON Object Schema for this processor's
        processing parameters.
        :return: a JSON Object Schema
        """
        return JsonObjectSchema(
            properties=dict(
                band_from=JsonStringSchema(min_length=1, default='B04'),
                band_peek=JsonStringSchema(min_length=1, default='B05'),
                band_to=JsonStringSchema(min_length=1, default='B06'),
                wlen_from=JsonNumberSchema(exclusive_minimum=0),
                wlen_peek=JsonNumberSchema(exclusive_minimum=0),
                wlen_to=JsonNumberSchema(exclusive_minimum=0),
            ),
            required=[],
            additional_properties=False,
        )

    def process_dataset(self,
                        source_cube: xr.Dataset,
                        band_from: str = None,
                        band_peek: str = None,
                        band_to: str = None,
                        wlen_from: float = None,
                        wlen_peek: float = None,
                        wlen_to: float = None) -> xr.Dataset:
        """
        Compute a new data cube with a single variable
        'MCI' (Maximum Chlorophyll Index) from three reflectances
        *band_from*, *band_peek*, *band_to* using optional corresponding band
        wavelengths *wlen_from*, *wlen_peek*, *wlen_to*.
        """

        # Get the 3 band names to be used
        bands = [
            band_from or 'B04',
            band_peek or 'B05',
            band_to or 'B06'
        ]

        # Get the corresponding 3 wavelengths to be used
        wavelengths = dict(
            wlen_from=wlen_from,
            wlen_peek=wlen_peek,
            wlen_to=wlen_to
        )

        # Validate inputs and fill in missing wavelengths
        for band_name, wlen_name in zip(bands, wavelengths.keys()):
            if band_name not in source_cube:
                raise ValueError(
                    f'variable {band_name!r} not found in dataset'
                )
            if wavelengths[wlen_name] is None:
                wavelengths[wlen_name] = source_cube[band_name].attrs.get(
                    'wavelength'
                )
            if wavelengths[wlen_name] is None:
                raise ValueError(
                    f'missing wavelength {wlen_name!r}'
                    f' for variable {band_name!r}'
                )

        # Mask out non-water and cloudy pixels
        scene_classif = MaskSet(source_cube.SCL)
        water_cube = source_cube.where(scene_classif.water)

        # concurrently compute the cube; chunks are processed in parallel
        mci_cube = compute_dataset(self.compute_mci_var,
                                   water_cube,
                                   input_var_names=bands,
                                   input_params=wavelengths,
                                   output_var_name=MCI_VAR_NAME)
        # the MCI variable
        mci = mci_cube[MCI_VAR_NAME]

        # Return a copy of the source cube plus the MCI variable
        mci.attrs['long_name'] = 'Maximum Chlorophyll Index'
        mci.attrs['units'] = 'unitless'
        # Allows image display to use initial value display range
        mci.attrs['valid_min'] = -0.5
        mci.attrs['valid_max'] = 1.0

        # Get a copy of the source cube plus the new MCI variable
        result_cube = source_cube.assign(MCI=mci)

        # Set a new title
        result_cube.attrs['title'] = 'Maximum Chlorophyll Index (MCI)'

        # Return result
        return result_cube

    @staticmethod
    def compute_mci_var(b_from: np.ndarray,
                        b_peek: np.ndarray,
                        b_to: np.ndarray,
                        input_params: Dict[str, Any],
                        dim_coords: Dict[str, np.ndarray]):
        """
        Compute Maximum Chlorophyll Index (MCI) from three reflectances.
        """

        # The first three arguments are chunks of the three input variables
        # we define below.
        # You can name them as you like. They are pure 3D numpy arrays.

        # The 'input_params' argument is a standard parameter that we
        # define in the call below.
        wlen_from = input_params['wlen_from']
        wlen_peek = input_params['wlen_peek']
        wlen_to = input_params['wlen_to']

        # The 'dim_coords' argument is optional and provides the
        # coordinate values for all dimension of the current chunk.
        # We don't use it here, but for many algorithms this is important
        # information (e.g. looking up aux data).
        lon, lat = (dim_coords[dim] for dim in ('lon', 'lat'))
        # print('dim_coords from', lon[0], lat[0], 'to', lon[-1], lat[-1])

        # You can use any popular data packages such as
        # numpy, scipy, dask here, or we can use ML packages such as
        # scikitlearn!
        # For simplicity, we do some simple array math here:

        f = (wlen_peek - wlen_from) / (wlen_to - wlen_from)
        mci = (b_peek - b_from) - f * (b_to - b_from)

        return mci
