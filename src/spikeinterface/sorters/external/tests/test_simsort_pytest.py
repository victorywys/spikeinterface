import os
import sys
from pathlib import Path
import unittest
import pytest

# Add the local spikeinterface and simsort paths to Python path
SPIKEINTERFACE_PATH = Path("/home/v-yimuzhang/spikeinterface/src")
if str(SPIKEINTERFACE_PATH) not in sys.path:
    sys.path.insert(0, str(SPIKEINTERFACE_PATH))

from spikeinterface.sorters import SimSortSorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


@pytest.mark.skipif(not SimSortSorter.is_installed(), reason='simsort not installed')
class SimSortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = SimSortSorter

if __name__ == '__main__':
    # Test run, `test` here will be a test case object managed by unittest/pytest
    pytest.main(["-v", __file__])  # This will let pytest run your tests correctly
