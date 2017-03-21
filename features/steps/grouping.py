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


@given("the following x and y arrays")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    exec(context.text, locals(), globals())
    context.session.load_arrays(1, globals()['x'], globals()['y'], DataPHA)


@when("I group data with {number} counts each")
def step_impl(context, number):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    number = int(number)
    context.session.group_counts(number)


@then("the filtered dependent axis has {final_counts} counts in channels")
def step_impl(context, final_counts):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    expected = list(map(int, final_counts.split(',')))
    actual = context.session.get_data().get_dep(filter=True)
    assert_array_equal(expected, actual)
