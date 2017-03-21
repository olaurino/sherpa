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

from behave import *

import numpy as np
from numpy.testing import assert_array_equal
from sherpa.astro.data import DataPHA
from sherpa.astro.ui.utils import Session


@given("a sherpa session")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    context.session = Session()


@given("a simple array of ones in channels from 1 to 100 loaded in session as PHA")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    x = np.arange(1, 101)
    y = np.ones_like(x)
    context.session.load_arrays(1, x, y, DataPHA)


@when("I group group data with 20 counts each")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    context.session.group_counts(20)


@then("the filtered dependent axis has 5 channels with 20 counts each")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    expected = np.full(5, 20, dtype='float64')
    actual = context.session.get_data().get_dep(filter=True)
    assert_array_equal(expected, actual)
