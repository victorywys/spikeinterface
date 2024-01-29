import unittest
import numpy as np


from spikeinterface.postprocessing.tests.common_extension_tests import ResultExtensionCommonTestSuite

from spikeinterface.postprocessing import ComputeAmplitudeScalings



class AmplitudeScalingsExtensionTest(ResultExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputeAmplitudeScalings
    extension_function_kwargs_list = [
        dict(handle_collisions=True),
        dict(handle_collisions=False),
    ]

    def test_scaling_values(self):
        sorting_result = self._prepare_sorting_result("memory", True)
        sorting_result.compute("amplitude_scalings", handle_collisions=False)

        spikes = sorting_result.sorting.to_spike_vector()

        ext = sorting_result.get_extension("amplitude_scalings")
        
        for unit_index, unit_id in enumerate(sorting_result.unit_ids):
            mask = spikes["unit_index"] == unit_index
            scalings = ext.data["amplitude_scalings"][mask]
            median_scaling = np.median(scalings)
            # print(unit_index, median_scaling)
            np.testing.assert_array_equal(np.round(median_scaling), 1)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.hist(ext.data["amplitude_scalings"])
        # plt.show()


if __name__ == "__main__":
    test = AmplitudeScalingsExtensionTest()
    test.setUp()
    test.test_extension()
    # test.test_scaling_values()
