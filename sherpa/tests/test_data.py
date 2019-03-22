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

from sherpa.data import Data, BaseData, Data1D, DataSimulFit, Data1DInt, Data2D
from sherpa.models import Polynom1D
from sherpa.utils.err import NotImplementedErr, DataErr

NAME = "data_test"
X_ARRAY = numpy.arange(0, 10, 1)
Y_ARRAY = numpy.arange(100, 110, 1)
SYSTEMATIC_ERROR_ARRAY = numpy.arange(0, 0.10, 0.01)
STATISTICAL_ERROR_ARRAY = numpy.arange(0, 1, 0.1)
X_THRESHOLD = 3
MULTIPLIER = 2

DATA_1D_CLASSES = (Data1D, Data, Data1DInt)


@pytest.fixture
def data(request):
    data_class = request.param

    instance_args = {
        Data1D: (NAME, X_ARRAY, Y_ARRAY, STATISTICAL_ERROR_ARRAY, SYSTEMATIC_ERROR_ARRAY),
        Data: (NAME, (X_ARRAY, ), Y_ARRAY, STATISTICAL_ERROR_ARRAY, SYSTEMATIC_ERROR_ARRAY),
        Data1DInt: (NAME, X_ARRAY-0.5, X_ARRAY+0.5, Y_ARRAY, STATISTICAL_ERROR_ARRAY, SYSTEMATIC_ERROR_ARRAY)
    }

    return data_class(*instance_args[data_class])


@pytest.fixture
def data_no_errors():
    return Data(NAME, X_ARRAY, Y_ARRAY)


@pytest.fixture
def data_simul_fit():
    data_one = Data1D("data_one", X_ARRAY, Y_ARRAY, STATISTICAL_ERROR_ARRAY, SYSTEMATIC_ERROR_ARRAY)
    data_two = Data1D("data_two", MULTIPLIER * X_ARRAY, MULTIPLIER * Y_ARRAY,
                      MULTIPLIER * STATISTICAL_ERROR_ARRAY, MULTIPLIER * SYSTEMATIC_ERROR_ARRAY)
    return DataSimulFit(NAME, (data_one, data_two))


@pytest.fixture
def data_simul_fit_no_errors():
    data_one = Data1D("data_one", X_ARRAY, Y_ARRAY)
    data_two = Data1D("data_two", MULTIPLIER * X_ARRAY, MULTIPLIER * Y_ARRAY)
    return DataSimulFit(NAME, (data_one, data_two))


@pytest.fixture
def data_simul_fit_some_errors():
    data_one = Data1D("data_one", X_ARRAY, Y_ARRAY, STATISTICAL_ERROR_ARRAY, SYSTEMATIC_ERROR_ARRAY)
    data_two = Data1D("data_two", MULTIPLIER * X_ARRAY, MULTIPLIER * Y_ARRAY)
    return DataSimulFit(NAME, (data_one, data_two))


def test_base_data_instantiation():
    with pytest.raises(NotImplementedErr):
        BaseData()


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_get_x(data):
    with pytest.raises(DataErr):
        data.get_x()


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_x0(data):
    with pytest.raises(DataErr):
        data.get_x0()


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_x1(data):
    with pytest.raises(DataErr):
        data.get_x1()


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_xlabel(data):
    assert data.get_xlabel() == "x"


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_x0label(data):
    assert data.get_x0label() == "x0"


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_x1label(data):
    assert data.get_x1label() == "x1"


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_ylabel(data):
    assert data.get_ylabel() == "y"


@pytest.mark.parametrize("data", (Data,), indirect=True)
def test_data_get_dims(data):
    with pytest.raises(DataErr):
        data.get_dims()


@pytest.mark.parametrize("data", (Data,), indirect=True)
def test_data_get_img(data):
    with pytest.raises(DataErr):
        data.get_img()


@pytest.mark.parametrize("data", (Data,), indirect=True)
def test_data_get_imgerr(data):
    with pytest.raises(DataErr):
        data.get_imgerr()


@pytest.mark.parametrize("data", (Data,), indirect=True)
def test_data_get_xerr(data):
    assert data.get_xerr() is None


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data_get_xerr(data):
    assert data.get_xerr() is None


@pytest.mark.parametrize("data", (Data,), indirect=True)
def test_data_str_repr(data):
    assert repr(data) == "<Data data set instance 'data_test'>"
    assert str(data) == 'name      = data_test\nindep     = (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),)\ndep       ' \
                        '= Int64[10]\nstaterror = Float64[10]\nsyserror  = Float64[10]'


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_str_repr(data):
    assert repr(data) == "<Data1D data set instance 'data_test'>"
    assert str(data) == 'name      = data_test\nx         = Int64[10]\ny         = Int64[10]\nstaterror = ' \
                        'Float64[10]\nsyserror  = Float64[10]'


@pytest.mark.parametrize("data", (Data, Data1D), indirect=True)
def test_data_get_indep(data):
    numpy.testing.assert_array_equal(data.get_indep(), [X_ARRAY, ])


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_get_indep(data):
    numpy.testing.assert_array_equal(data.get_indep(), (X_ARRAY-0.5, X_ARRAY+0.5))


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_get_indep_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[:X_THRESHOLD + 1], ])


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_get_indep_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    expected = (X_ARRAY-0.5)[:X_THRESHOLD + 1], (X_ARRAY+0.5)[:X_THRESHOLD + 1]
    numpy.testing.assert_array_equal(data.get_indep(filter=True), expected)


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_1d_get_indep_ignore(data):
    data.ignore(0, X_THRESHOLD)
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[X_THRESHOLD + 1:], ])


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_1d_get_indep_ignore(data):
    data.ignore((0, ), (X_THRESHOLD, ), data.get_indep())
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[X_THRESHOLD + 1:], ])


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_1d_get_indep_ignore_string_lower(data):
    with pytest.raises(DataErr):
        data.ignore("0", 1)


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_get_indep_ignore_string_lower(data):
    with pytest.raises(DataErr):
        data.ignore(("0", ), (1, ), data.get_indep())


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_1d_get_indep_ignore_string_upper(data):
    with pytest.raises(DataErr):
        data.ignore(0, "1")


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_get_indep_ignore_string_upper(data):
    with pytest.raises(DataErr):
        data.ignore((0, ), ("1", ), data.get_indep())


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_1d_get_indep_notice(data):
    data.notice(0, X_THRESHOLD)
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[:X_THRESHOLD + 1], ])


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_1d_int_get_indep_notice(data):
    data.notice(0, X_THRESHOLD)
    expected = [(X_ARRAY-0.5)[:X_THRESHOLD + 1], (X_ARRAY+0.5)[:X_THRESHOLD + 1]]
    actual = data.get_indep(filter=True)
    numpy.testing.assert_array_equal(actual[0], expected[0])
    numpy.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_get_indep_notice(data):
    data.notice((0, ), (X_THRESHOLD, ), data.get_indep())
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [X_ARRAY[:X_THRESHOLD + 1], ])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_indep_callable_filter(data):
    with pytest.raises(AttributeError):
        data.filter = lambda x: x <= X_THRESHOLD


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_get_indep_mask(data):
    data.mask = X_ARRAY == 0
    numpy.testing.assert_array_equal(data.get_indep(filter=True), X_ARRAY[0])  # Why is this not an array?


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_get_indep_mask(data):
    data.mask = X_ARRAY == 0
    numpy.testing.assert_array_equal(data.get_indep(filter=True), ([(X_ARRAY-0.5)[0]], [(X_ARRAY+0.5)[0]]))


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_get_indep_filter_mask(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    data.mask = X_ARRAY == 0
    numpy.testing.assert_array_equal(data.get_indep(filter=True), [[X_ARRAY[0]]])  # Why is this an array then?


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_get_indep_filter_mask(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    data.mask = X_ARRAY == 0
    numpy.testing.assert_array_equal(data.get_indep(filter=True), ([(X_ARRAY-0.5)[0]], [(X_ARRAY+0.5)[0]]))


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_get_indep_filter_null_mask(data):
    data.mask = False
    with pytest.raises(DataErr):
        data.get_indep(filter=True)


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_get_indep_filter_null_mask(data):
    data.mask = False
    with pytest.raises(DataErr):
        data.get_indep(filter=True)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_dep_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_dep(filter=True), Y_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_set_dep_filter(data):
    data.set_dep([0, 1])
    numpy.testing.assert_array_equal(data.get_dep(filter=True), [0, 1])
    data.set_dep(0)
    numpy.testing.assert_array_equal(data.get_dep(filter=True), [0] * Y_ARRAY.size)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_staterror(data):
    numpy.testing.assert_array_equal(data.get_staterror(), STATISTICAL_ERROR_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_staterror_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_staterror(filter=True), STATISTICAL_ERROR_ARRAY[:X_THRESHOLD + 1])


def test_data_get_staterror_func(data_no_errors):
    data_no_errors.filter = X_ARRAY <= X_THRESHOLD
    stat_error = data_no_errors.get_staterror(filter=False, staterrfunc=lambda x: MULTIPLIER * x)
    numpy.testing.assert_array_equal(stat_error, MULTIPLIER * Y_ARRAY)


def test_data_get_staterror_filter_func(data_no_errors):
    data_no_errors.filter = X_ARRAY <= X_THRESHOLD
    stat_error = data_no_errors.get_staterror(filter=True, staterrfunc=lambda x: MULTIPLIER * x)
    numpy.testing.assert_array_equal(stat_error, MULTIPLIER * Y_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_syserror(data):
    numpy.testing.assert_array_equal(data.get_syserror(), SYSTEMATIC_ERROR_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_syserror_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_syserror(filter=True), SYSTEMATIC_ERROR_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_error(data):
    error = data.get_error()
    expected_error = numpy.sqrt(SYSTEMATIC_ERROR_ARRAY ** 2 + STATISTICAL_ERROR_ARRAY ** 2)
    numpy.testing.assert_array_equal(error, expected_error)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_yerr(data):
    error = data.get_yerr()
    expected_error = numpy.sqrt(SYSTEMATIC_ERROR_ARRAY ** 2 + STATISTICAL_ERROR_ARRAY ** 2)
    numpy.testing.assert_array_equal(error, expected_error)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_dep(data):
    numpy.testing.assert_array_equal(data.get_dep(), Y_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_y(data):
    numpy.testing.assert_array_equal(data.get_y(), Y_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_get_y_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    numpy.testing.assert_array_equal(data.get_y(filter=True), Y_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_get_y_filter_func(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    y = data.get_y(filter=True, yfunc=lambda x: MULTIPLIER*x)
    expected_y = (Y_ARRAY[:X_THRESHOLD + 1], MULTIPLIER*X_ARRAY[:X_THRESHOLD + 1])
    numpy.testing.assert_array_equal(y, expected_y)


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_get_y_filter_func(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    y = data.get_y(filter=True, yfunc=lambda x, y: (MULTIPLIER*x, MULTIPLIER*y))
    expected_y = (Y_ARRAY[:X_THRESHOLD + 1], (MULTIPLIER*(X_ARRAY-0.5)[:X_THRESHOLD + 1],
                  MULTIPLIER*(X_ARRAY+0.5)[:X_THRESHOLD + 1]))
    numpy.testing.assert_array_equal(y[0], expected_y[0])
    numpy.testing.assert_array_equal(y[1], expected_y[1])


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_get_y_func(data):
    y = data.get_y(filter=True, yfunc=lambda x: MULTIPLIER*x)
    expected_y = (Y_ARRAY, MULTIPLIER*X_ARRAY)
    numpy.testing.assert_array_equal(y, expected_y)


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_1d_int_get_y_func(data):
    y = data.get_y(filter=True, yfunc=lambda x, y: (MULTIPLIER*x, MULTIPLIER*y))
    expected_y = (Y_ARRAY, (MULTIPLIER*(X_ARRAY-0.5), MULTIPLIER*(X_ARRAY+0.5)))
    numpy.testing.assert_array_equal(y[0], expected_y[0])
    numpy.testing.assert_array_equal(y[1], expected_y[1])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_eval_model(data):
    model = Polynom1D()
    model.c0 = 0
    model.c1 = MULTIPLIER
    evaluated_data = data.eval_model(model)
    numpy.testing.assert_array_equal(evaluated_data, MULTIPLIER * X_ARRAY)


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_eval_model_to_fit_no_filter(data):
    model = Polynom1D()
    model.c0 = 0
    model.c1 = MULTIPLIER
    evaluated_data = data.eval_model_to_fit(model)
    numpy.testing.assert_array_equal(evaluated_data, MULTIPLIER * X_ARRAY)


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_eval_model_to_fit_filter(data):
    model = Polynom1D()
    model.c0 = 0
    model.c1 = MULTIPLIER
    data.filter = X_ARRAY <= X_THRESHOLD
    evaluated_data = data.eval_model_to_fit(model)
    numpy.testing.assert_array_equal(evaluated_data, MULTIPLIER * X_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_1d_int_eval_model_to_fit_filter(data):
    model = Polynom1D()
    model.c0 = 0
    model.c1 = MULTIPLIER
    data.filter = X_ARRAY <= X_THRESHOLD
    evaluated_data = data.eval_model_to_fit(model)
    numpy.testing.assert_array_equal(evaluated_data, MULTIPLIER * X_ARRAY[:X_THRESHOLD + 1])


@pytest.mark.parametrize("data", (Data1D, Data), indirect=True)
def test_data_to_guess(data):
    actual = data.to_guess()
    expected = [Y_ARRAY, X_ARRAY]
    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data_1d_int_to_guess(data):
    actual = data.to_guess()
    expected = [Y_ARRAY, X_ARRAY-0.5]
    numpy.testing.assert_array_equal(actual[0], expected[0])
    numpy.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("data", DATA_1D_CLASSES, indirect=True)
def test_data_1d_to_fit(data):
    actual = data.to_fit()
    expected = [Y_ARRAY, STATISTICAL_ERROR_ARRAY, SYSTEMATIC_ERROR_ARRAY]
    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_1d_to_plot(data):
    actual = data.to_plot()
    yerr = numpy.sqrt(SYSTEMATIC_ERROR_ARRAY ** 2 + STATISTICAL_ERROR_ARRAY ** 2)
    expected = [X_ARRAY, Y_ARRAY, yerr, None, "x", "y"]
    numpy.testing.assert_array_equal(actual[0], expected[0])
    numpy.testing.assert_array_equal(actual[1], expected[1])
    numpy.testing.assert_array_equal(actual[2], expected[2])
    numpy.testing.assert_array_equal(actual[3], expected[3])
    numpy.testing.assert_array_equal(actual[4], expected[4])
    numpy.testing.assert_array_equal(actual[5], expected[5])


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_1d_to_component_plot(data):
    actual = data.to_component_plot()
    yerr = numpy.sqrt(SYSTEMATIC_ERROR_ARRAY ** 2 + STATISTICAL_ERROR_ARRAY ** 2)
    expected = [X_ARRAY, Y_ARRAY, yerr, None, "x", "y"]
    numpy.testing.assert_array_equal(actual[0], expected[0])
    numpy.testing.assert_array_equal(actual[1], expected[1])
    numpy.testing.assert_array_equal(actual[2], expected[2])
    numpy.testing.assert_array_equal(actual[3], expected[3])
    numpy.testing.assert_array_equal(actual[4], expected[4])
    numpy.testing.assert_array_equal(actual[5], expected[5])


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_to_contour(data):
    with pytest.raises(DataErr):
        data.to_contour()


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_to_plot(data):
    with pytest.raises(DataErr):
        data.to_plot()


@pytest.mark.parametrize("data", (Data, ), indirect=True)
def test_data_to_component_plot(data):
    with pytest.raises(DataErr):
        data.to_component_plot()


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data_1d_to_contour(data):
    with pytest.raises(DataErr):
        data.to_contour()


def test_data_simul_fit(data_simul_fit):
    y, stat_error, systematic_error = data_simul_fit.to_fit()
    expected_y = numpy.concatenate((Y_ARRAY, MULTIPLIER * Y_ARRAY))
    expected_stat_error = numpy.concatenate((STATISTICAL_ERROR_ARRAY, MULTIPLIER * STATISTICAL_ERROR_ARRAY))
    expected_sys_error = numpy.concatenate((SYSTEMATIC_ERROR_ARRAY, MULTIPLIER * SYSTEMATIC_ERROR_ARRAY))
    numpy.testing.assert_array_equal(y, expected_y)
    numpy.testing.assert_array_equal(stat_error, expected_stat_error)
    numpy.testing.assert_array_equal(systematic_error, expected_sys_error)


def test_data_simul_fit_to_plot(data_simul_fit):
    actual = data_simul_fit.to_fit()
    expected_y = numpy.concatenate((Y_ARRAY, MULTIPLIER * Y_ARRAY))
    expected_stat_error = numpy.concatenate((STATISTICAL_ERROR_ARRAY, MULTIPLIER * STATISTICAL_ERROR_ARRAY))
    expected_sys_error = numpy.concatenate((SYSTEMATIC_ERROR_ARRAY, MULTIPLIER * SYSTEMATIC_ERROR_ARRAY))
    numpy.testing.assert_array_equal(actual[0], expected_y)
    numpy.testing.assert_array_equal(actual[1], expected_stat_error)
    numpy.testing.assert_array_equal(actual[2], expected_sys_error)


def test_data_simul_fit_no_errors(data_simul_fit_no_errors):
    y, stat_error, systematic_error = data_simul_fit_no_errors.to_fit()
    expected_y = numpy.concatenate((Y_ARRAY, MULTIPLIER * Y_ARRAY))
    expected_stat_error = None
    expected_sys_error = None
    numpy.testing.assert_array_equal(y, expected_y)
    numpy.testing.assert_array_equal(stat_error, expected_stat_error)
    numpy.testing.assert_array_equal(systematic_error, expected_sys_error)


def test_data_simul_fit_some_errors(data_simul_fit_some_errors):
    with pytest.raises(DataErr):
        data_simul_fit_some_errors.to_fit()


def test_data_simul_fit_eval_model_to_fit(data_simul_fit):
    model = Polynom1D()
    model.c0 = 0
    model.c1 = MULTIPLIER
    data_simul_fit.datasets[0].filter = X_ARRAY <= X_THRESHOLD
    data_simul_fit.datasets[1].filter = X_ARRAY <= X_THRESHOLD
    evaluated_data = data_simul_fit.eval_model_to_fit((model, model))
    expected_data = numpy.concatenate((MULTIPLIER * X_ARRAY[:X_THRESHOLD+1], MULTIPLIER **2 * X_ARRAY[:X_THRESHOLD+1]))
    numpy.testing.assert_array_equal(evaluated_data, expected_data)


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_dims(data):
    assert data.get_dims() == (X_ARRAY.size, )


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_filter(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    assert data.get_filter() == '0.0000:3.0000'


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_filter_mask(data):
    data.mask = X_ARRAY <= X_THRESHOLD
    assert data.get_filter() == '0.0000:3.0000'


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_filter_expr(data):
    data.filter = X_ARRAY <= X_THRESHOLD
    assert data.get_filter_expr() == '0.0000-3.0000 x'


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_bounding_mask_filter(data):
    mask = X_ARRAY <= X_THRESHOLD
    data.filter = mask
    assert data.get_bounding_mask() == (True, None)


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_bounding_mask(data):
    mask = X_ARRAY <= X_THRESHOLD
    data.mask = mask
    actual = data.get_bounding_mask()
    numpy.testing.assert_array_equal(actual[0], mask)
    numpy.testing.assert_array_equal(actual[1], X_ARRAY.size)


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_img(data):
    numpy.testing.assert_array_equal(data.get_img(), [Y_ARRAY, ])


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_img_yfunc(data):
    actual = data.get_img(yfunc=lambda x: MULTIPLIER * x)
    expected = ([Y_ARRAY, ], [MULTIPLIER * X_ARRAY, ], )
    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("data", (Data1D,), indirect=True)
def test_data1d_get_imgerr(data):
    expected_error = numpy.sqrt(SYSTEMATIC_ERROR_ARRAY ** 2 + STATISTICAL_ERROR_ARRAY ** 2)
    numpy.testing.assert_array_equal(data.get_imgerr(), [expected_error, ])


@pytest.mark.parametrize("data", (Data1D, Data1DInt), indirect=True)
def test_data1d_get_x(data):
    numpy.testing.assert_array_equal(data.get_x(), X_ARRAY)


@pytest.mark.parametrize("data", (Data1D, ), indirect=True)
def test_data1d_get_xerr(data):
    assert data.get_xerr() is None


@pytest.mark.parametrize("data", (Data1DInt, ), indirect=True)
def test_data1d_get_xerr(data):
    numpy.testing.assert_array_equal(data.get_xerr(), [1] * X_ARRAY.size)


@pytest.mark.parametrize("data", (Data1D, Data1DInt), indirect=True)
def test_data1d_get_y(data):
    numpy.testing.assert_array_equal(data.get_y(), Y_ARRAY)
