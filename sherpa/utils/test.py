#
#  Copyright (C) 2007, 2015, 2016  Smithsonian Astrophysical Observatory
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

"""Utility routines for the Sherpa test suite.
"""

import importlib
import os

from unittest import skipIf

import numpy
import numpytest

from sherpa.utils._utils import sao_fcmp

__all__ = ('SherpaTest', 'SherpaTestCase',
           'requires_data', 'requires_fits', 'requires_package')


def _get_datadir():
    try:
        import sherpatest
        datadir = os.path.dirname(sherpatest.__file__)
    except ImportError:
        try:
            import sherpa
            datadir = os.path.join(os.path.dirname(sherpa.__file__), os.pardir,
                                   'sherpa-test-data', 'sherpatest')
            if not os.path.exists(datadir) or not os.listdir(datadir):
                # The dir is empty, maybe the submodule was not initialized
                datadir = None
        except ImportError:
            # neither sherpatest nor sherpa can be found, falling back to None
            datadir = None
    return datadir


class SherpaTestCase(numpytest.NumpyTestCase):
    "Base class for Sherpa unit tests"

    # The location of the Sherpa test data (it is optional)
    datadir = _get_datadir()

    def make_path(self, *segments):
        """Add the segments onto the test data location.

        Parameters
        ----------
        *segments
           Path segments to combine together with the location of the
           test data.

        Returns
        -------
        fullpath : None or string
           The full path to the repository, or None if the
           data directory is not set.

        """
        if self.datadir is None:
            return None
        return os.path.join(self.datadir, *segments)

    # What is the benefit of this over numpy.testing.assert_allclose(),
    # which was added in version 1.5 of NumPy?
    def assertEqualWithinTol(self, first, second, tol=1e-7, msg=None):
        """Check that the values are equal within an absolute tolerance.

        Parameters
        ----------
        first : number or array_like
           The expected value, or values.
        second : number or array_like
           The value, or values, to check. If first is an array, then
           second must be an array of the same size. If first is
           a scalar then second can be a scalar or an array.
        tol : number
           The absolute tolerance used for comparison.
        msg : string
           The message to display if the check fails.

        """

        self.assertFalse(numpy.any(sao_fcmp(first, second, tol)), msg)

    def assertNotEqualWithinTol(self, first, second, tol=1e-7, msg=None):
        """Check that the values are not equal within an absolute tolerance.

        Parameters
        ----------
        first : number or array_like
           The expected value, or values.
        second : number or array_like
           The value, or values, to check. If first is an array, then
           second must be an array of the same size. If first is
           a scalar then second can be a scalar or an array.
        tol : number
           The absolute tolerance used for comparison.
        msg : string
           The message to display if the check fails.

        """

        self.assertTrue(numpy.all(sao_fcmp(first, second, tol)), msg)


def requires_data(test_function):
    """Decorator to skip a test if the external data is not available.

    This decorator causes tests to be skipped if the `sherpa-test-data`
    directory of the Sherpa distribution has not been initialized (it
    is a git submodule).
    """
    condition = SherpaTestCase.datadir is None
    msg = "required test data missing"
    return skipIf(condition, msg)(test_function)


def _has_package_from_list(*packages):
    """Returns True if at least one of the ``packages`` args is importable.
    """
    for package in packages:
        try:
            importlib.import_module(package)
            return True
        except:
            pass
    return False


def requires_package(msg=None, *packages):
    """Decorator for test functions requiring specific packages.
    """
    condition = _has_package_from_list(*packages)
    msg = msg or "required module missing among {}.".format(", ".join(packages))

    def decorator(test_function):
        return skipIf(not condition, msg)(test_function)
    return decorator


def requires_xspec(test_function):
    """Decorator for test functions requiring XSPEC"""
    return requires_package('xspec required', 'sherpa.astro.xspec')(test_function)


def requires_ds9(test_function):
    """Decorator for test functions requiring ds9"""
    return requires_package('ds9 required', 'sherpa.image.ds9_backend')(test_function)


def requires_fits(test_function):
    """Returns True if there is an importable backend for FITS I/O.

    Used to skip tests requiring fits_io
    """
    packages = ('pyfits',
                'astropy.io.fits',
                'pycrates',
                )
    msg = "FITS backend required"
    return requires_package(msg, *packages)(test_function)


def requires_pylab(test_function):
    """Returns True if the pylab module is available (pylab).

    Used to skip tests requiring matplotlib
    """
    packages = ('pylab',
                )
    msg = "matplotlib backend required"
    return requires_package(msg, *packages)(test_function)


class SherpaTest(numpytest.NumpyTest):
    "Sherpa test suite manager"

    def test(self, level=1, verbosity=1, datadir=None):
        old_datadir = SherpaTestCase.datadir
        SherpaTestCase.datadir = datadir

        try:
            result = numpytest.NumpyTest.test(self, level, verbosity)
            if result is None or result.failures or result.errors or result.unexpectedSuccesses:
                raise Exception("Test failures were detected")
        finally:
            SherpaTestCase.datadir = old_datadir
