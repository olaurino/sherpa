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

from numpy.testing import assert_array_equal

from sherpa.astro.data import DataPHA
from sherpa.astro.ui.utils import Session

# def trace(frame, event, arg):
#     if event == "call":
#         if frame.f_code.co_name == "_set_mask":
#             the_frame = frame.f_back.f_back.f_back.f_back
#             print("call")
#             print(the_frame.f_code.co_filename, the_frame.f_code.co_firstlineno)
#             print("val:")
#             print(frame.f_locals['val'])
#             print()


@given("a sherpa session")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    import sys
    # sys.settrace(trace)
    context.session = Session()


@given("{xArray} and {yArray} as x and y arrays")
def step_impl(context, xArray, yArray):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    glob = globals()
    loc = locals()
    exec("x="+xArray, glob, loc)
    exec("y="+yArray, glob, loc)
    glob['x'] = loc['x']
    context.session.load_arrays(1, loc['x'], loc['y'], DataPHA)


@when("I group data with {number} counts each")
def step_impl(context, number):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    number = int(number)
    context.session.group_counts(number)
    # print(context.session.get_data().get_noticed_channels())


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


@then("the dependent axis has a {quality} quality array")
def step_impl(context, quality):
    """
    Parameters
    ----------
    context : behave.runner.Context
    quality : str
    """
    exec("q="+quality, globals(), locals())
    expected = locals()['q']
    actual = context.session.get_data().quality
    assert_array_equal(expected, actual)


@step("I ignore the bad channels")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    context.session.ignore_bad()


@when("I notice {what} from {min} to {max}")
def step_impl(context, what, min, max):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    from sherpa.utils.err import DataErr
    if what == 'channels':
        try:
            context.session.set_analysis("chan")
        except DataErr:
            pass  # this is in case there is no response matrix
    elif what == 'energy':
        context.session.set_analysis("energy")  # will fail if there is no response matrix
    context.session.notice(float(min), float(max))
    # print(context.session.get_data().get_noticed_channels())


@step('I type {command}')
def step_impl(context, command):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    exec(command, globals(), locals())


@when("I set the analysis to {analysis}")
def step_impl(context, analysis):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    context.session.set_analysis(analysis)


@step("I import {module} as {alias}")
def step_impl(context, module, alias):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    exec("import {} as {}".format(module, alias), globals(), globals())


@given("a simple response matrix")
def step_impl(context):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    import mock
    from sherpa.astro.data import DataRMF

    # def apply_rmf(*args):
    #     foo
    #     return args[0]/2

    # def apply_arf(*args):
    #     return args[0]

    channels = globals()['x'].size
    np = globals()['np']
    e_lo = np.linspace(1, channels, channels, endpoint=False)
    e_hi = np.linspace(e_lo[1], channels, channels)

    # def get_indep():
    #     foo
    #     return e_lo, e_hi

    rmf_dict = {
        'detchans': channels,
        # 'energ_lo': 0,
        # 'energ_hi': 0,
        # 'get_indep': get_indep,
        'e_min': e_lo,
        'e_max': e_hi,
        # 'apply_rmf': apply_rmf,
    }

    rmf = mock.MagicMock(spec=DataRMF, **rmf_dict)

    context.session.set_rmf(rmf)


@step("I load {filename}")
def step_impl(context, filename):
    """
    Parameters
    ----------
    context : behave.runner.Context
    """
    context.session.load_data(filename)