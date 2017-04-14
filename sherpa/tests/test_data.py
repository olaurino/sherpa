#
#  Copyright (C) 2017  Smithsonian Astrophysical Observatory
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import numpy as np
from numpy.testing import assert_array_equal

try:
    from unittest import mock  # Py3
except ImportError:
    import mock  # Py2

import pytest

from sherpa.data import BaseData, GroupingManager, Data1D


def test_base_class_abstract():
    with pytest.raises(TypeError) as excinfo:
        BaseData()
    assert "Can't instantiate abstract class" in str(excinfo.value)


def test_call_base_data_init():
    class MyData(BaseData):
        def __init__(self, foo):
            BaseData.__init__(self)

        def _get_dep(self):
            pass

        def get_indep(self, filter=True):
            pass

    my_data = MyData("bar")
    assert "bar" == my_data.foo


def test_no_new_attrs_after_init():
    class MyData(BaseData):
        def __init__(self):
            BaseData.__init__(self)

        def _get_dep(self):
            pass

        def get_indep(self, filter=True):
            pass

    my_data = MyData()
    with pytest.raises(AttributeError) as excinfo:
        my_data.foo = "bar"
    assert "has no attribute 'foo'" in str(excinfo.value)


def test_grouper_counts():
    data_set = mock.MagicMock(y=np.array([1, 2, 3, 4, 5, 6, 7]))
    grouper = GroupingManager(data_set)
    grouper.group_counts(4, max_length=None, tab_stops=None)
    actual = grouper.apply(data_set.y)

    expected = [6, 4, 5, 6, 7]
    assert_array_equal(expected, actual)


def test_grouper_counts_inverse():
    counts = [1, 2, 3, 4, 5, 6, 7]
    counts.reverse()
    data_set = mock.MagicMock(y=np.array(counts))
    grouper = GroupingManager(data_set)
    grouper.group_counts(4, max_length=None, tab_stops=None)
    actual = grouper.apply(data_set.y)

    expected = [7, 6, 5, 4, 5]
    assert_array_equal(expected, actual)


def test_grouper_counts_ones():
    counts = [1, 1, 1, 1, 1, 1, 1]
    counts.reverse()
    data_set = mock.MagicMock(y=np.array(counts))
    grouper = GroupingManager(data_set)
    grouper.group_counts(4, max_length=None, tab_stops=None)
    actual = grouper.apply(data_set.y)

    expected = [4, ]
    assert_array_equal(expected, actual)


def test_grouper_counts_ones_and_zeros():
    counts = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    counts.reverse()
    data_set = mock.MagicMock(y=np.array(counts))
    grouper = GroupingManager(data_set)
    grouper.group_counts(4, max_length=None, tab_stops=None)
    actual = grouper.apply(data_set.y)

    expected = [4, 4]
    assert_array_equal(expected, actual)


def test_quality_array():
    channels = np.array([1, 2, 3, 4, 5])
    counts = np.array([1, 1, 1, 1, 1])
    data_set = Data1D(name="", x=channels, y=counts)
    data_set.group_counts(4, maxLength=None, tabStops=None)
    expected = [0, 0, 0, 0, 2]

    assert_array_equal(expected, data_set.quality)


def test_two_quality_arrays():
    channels = np.array([1, 2, 3, 4, 5])
    counts = np.array([1, 1, 1, 1, 1])
    data_set = Data1D(name="", x=channels, y=counts)
    data_set.quality = [1, 0, 0, 0, 0]
    data_set.group_counts(3, maxLength=None, tabStops=None)
    expected_quality = [1, 0, 0, 0, 2]

    assert_array_equal(expected_quality, data_set.quality)


def test_two_quality_arrays_grouping():
    channels = np.array([1, 2, 3, 4, 5])
    counts = np.array([1, 1, 1, 1, 1])
    data_set = Data1D(name="", x=channels, y=counts)
    data_set.quality = [1, 0, 0, 0, 0]
    data_set.group_counts(3, maxLength=None, tabStops=None)
    expected_grouping = [1, 1, -1, -1, 1]

    assert_array_equal(expected_grouping, data_set._grouping_manager.grouping)