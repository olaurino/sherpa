#
#  Copyright (C) 2008, 2015, 2016, 2017  Smithsonian Astrophysical Observatory
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
import numpy
import pytest

from sherpa.data import Data, BaseData, Data1D
from sherpa.models import Polynom1D
from sherpa.utils.err import NotImplementedErr, DataErr

NAME = "data_test"
X_ARRAY = numpy.arange(0, 10, 1)
Y_ARRAY = numpy.arange(100, 110, 1)
SYSTEMATIC_ERROR_ARRAY = numpy.arange(0, 0.10, 0.01)
STATISTICAL_ERROR_ARRAY = numpy.arange(0, 1, 0.1)
X_THRESHOLD = 3
MULTIPLIER = 2

DATA_1D_CLASSES = (Data, Data1D)


@pytest.fixture
def data(request):
    data_class = request.param
    return data_class(NAME, X_ARRAY, Y_ARRAY, STATISTICAL_ERROR_ARRAY, SYSTEMATIC_ERROR_ARRAY)


@pytest.fixture
def data_no_errors():
    return Data(NAME, X_ARRAY, Y_ARRAY)


def test_base_data_instantiation():
    with pytest.raises(NotImplementedErr):
        BaseData()


@pytest.mark.parametrize("data", (Data,), indirect=True)
def test_data_str_repr(data):
    assert repr(data) == "<Data data set instance 'data_test'>"
    assert str(data) == 'name      = data_test\nindep     = Int64[10]\ndep       = Int64[10]\nstaterror = ' \
                        'Float64[10]\nsyserror  = Float64[10]'


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data_str_repr(data):
    assert repr(data) == "<Data1D data set instance 'data_test'>"
    assert str(data) == 'name      = data_test\nx         = Int64[10]\ny         = Int64[10]\nstaterror = ' \
                        'Float64[10]\nsyserror  = Float64[10]'


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep(data):
    numpy.testing.assert_array_equal(data.get_indep(), [X_ARRAY, ])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[:X_THRESHOLD + 1], ])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_ignore(data):
    data.ignore(0, X_THRESHOLD)
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[X_THRESHOLD + 1:], ])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_ignore_string_lower(data):
    with pytest.raises(DataErr):
        data.ignore("0", 1)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_ignore_string_upper(data):
    with pytest.raises(DataErr):
        data.ignore(0, "1")


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_callable_filter(data):
    data.filter = lambda x: x <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[:X_THRESHOLD + 1], ])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_mask(data):
    data.mask = X_ARRAY == 0
    numpy.testing.assert_array_equal(data.get_indep(filter=True), X_ARRAY[0])  # Why is this not an array?


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_filter_mask(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    data.mask = X_ARRAY == 0
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [[X_ARRAY[0]]])  # Why is this an array then?


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_filter_null_mask(data):
    data.mask = False
    with pytest.raises(DataErr):
        data.get_indep(filter=True)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_dep_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_dep(filter=True), Y_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_staterror(data):
    numpy.testing.assert_array_equal(data.get_staterror(), STATISTICAL_ERROR_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_staterror_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_staterror(filter=True), STATISTICAL_ERROR_ARRAY[:X_THRESHOLD + 1])


def test_data_get_staterror_func(data_no_errors):
    data_no_errors.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data_no_errors.get_staterror(filter=False,
                                                                  staterrfunc=lambda x: MULTIPLIER * x),
                                     MULTIPLIER * Y_ARRAY)


def test_data_get_staterror_filter_func(data_no_errors):
    data_no_errors.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data_no_errors.get_staterror(filter=True,
                                                                  staterrfunc=lambda x: MULTIPLIER * x),
                                     MULTIPLIER * Y_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_syserror(data):
    numpy.testing.assert_array_equal(data.get_syserror(), SYSTEMATIC_ERROR_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_syserror_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_syserror(filter=True), SYSTEMATIC_ERROR_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_dep(data):
    numpy.testing.assert_array_equal(data.get_dep(), Y_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_eval_model(data):
    model = Polynom1D()
    model.c0 = 0
    model.c1 = MULTIPLIER
    evaluated_data = data.eval_model(model)
    numpy.testing.assert_array_equal(evaluated_data, MULTIPLIER * X_ARRAY)
