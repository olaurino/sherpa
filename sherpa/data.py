#
#  Copyright (C) 2008, 2015, 2016  Smithsonian Astrophysical Observatory
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

"""
Tools for creating, storing, inspecting, and manipulating data sets
"""
import abc

import logging
from six import add_metaclass
from six.moves import zip as izip
import inspect
import numpy
from sherpa.utils.err import DataErr, ImportErr
from sherpa.utils import SherpaFloat, NoNewAttributesAfterInit, print_fields, calc_total_error, bool_cast


__all__ = ('Data', 'DataSimulFit', 'Data1D', 'Data1DInt', 'Data2D', 'Data2DInt')

warning = logging.getLogger(__name__).warning

group_status = False
try:
    import group as pygroup
    group_status = True
except ImportError:
    group_status = False
    pygroup = None
    warning('the group module (from the CIAO tools package) is not ' +
            'installed.\nDynamic grouping functions will not be available.')


class GroupingManager(object):
    def __init__(self, data_set):
        self.data_set = data_set
        self.grouping = None
        self.quality = None

    def group_counts(self, num, max_length, tab_stops):
        if not group_status:
            raise ImportErr('importfailed', 'group', 'dynamic grouping')

        self._group(pygroup.grpNumCounts, self.data_set.y, num,
                    maxLength=max_length, tabStops=tab_stops)

        if hasattr(self.data_set, "background_ids"):
            for bkg_id in self.data_set.background_ids:
                bkg = self.data_set.get_background(bkg_id)
                if hasattr(bkg, "group_counts"):
                    bkg.group_counts(num, maxLength=max_length, tabStops=tab_stops)

    def apply(self, data, groupfunc=sum):
        if data is None:
            return data

        group_flags = self.grouping

        quality_filter = None
        if self.quality is not None:
            quality_filter = ~numpy.asarray(self.quality, bool)

        if quality_filter is None:
            return self._reduce_groups(data, group_flags, groupfunc)

        if len(data) != len(quality_filter) or len(group_flags) != len(quality_filter):
            raise DataErr('mismatch', "quality filter", "data array")

        filtered_data = numpy.asarray(data)[quality_filter]
        group_flags = numpy.asarray(group_flags)[quality_filter]
        grouped_data = self._reduce_groups(filtered_data, group_flags, groupfunc)

        # if data is self.channel and groupfunc is self._make_groups:
        #     return numpy.arange(1, len(grouped_data) + 1, dtype=int)

        return grouped_data

    def _group(self, group_func, *args, **kwargs):
        kwargs = self._extract_args(**kwargs)
        self.grouping, self.quality = group_func(*args, **kwargs)

    @staticmethod
    def _reduce_groups(data, group_flags, func):
        grouped_data = []
        group_indexes = []

        for idx, group_flag in enumerate(group_flags):
            if group_flag >= 1:
                group_indexes.append(idx)

        # None as the last index makes sure the array is consumed entirely, below
        group_indexes.append(None)

        for idx, group_index in enumerate(group_indexes[:-1]):
            counts = func(data[group_index:group_indexes[idx+1]])
            grouped_data.append(counts)

        return grouped_data

    @staticmethod
    def _extract_args(**kwargs):
        keys = list(kwargs.keys())[:]
        for key in keys:
            if kwargs[key] is None:
                kwargs.pop(key)
        return kwargs


class QualityManager(object):
    def __init__(self, grouping_manager):
        # type: (GroupingManager) -> None
        self.grouping_manager = grouping_manager
        self._quality = None

    @property
    def quality(self):
        grouping_quality = self.grouping_manager.quality

        if grouping_quality is not None and self._quality is not None:
            return numpy.maximum(grouping_quality, self._quality)
        elif grouping_quality is not None:
            return grouping_quality
        else:
            return self._quality

    @quality.setter
    def quality(self, quality_array):
        self._quality = quality_array


@add_metaclass(abc.ABCMeta)
class BaseData(NoNewAttributesAfterInit):
    """Base class for all data set types"""

    def __new__(cls, *args, **kwags):
        """

        Initialize a data object.  This method can only be called from
        a derived class constructor.  Attempts to create a BaseData
        instance will raise NotImplementedErr.

        Derived class constructors must call this method directly (and
        not indirectly through a superclass constructor).  When thus
        invoked, this method will extract the argument names and
        values from the derived class constructor invocation and set
        corresponding attributes on the instance (thereby eliminating
        the need for the derived class constructor to do its own
        attribute setting).  If the name of an argument matches the
        name of a DataProperty of the derived class, then the
        corresponding attribute name will have an underscore prepended
        (meaning the property will use the value directly instead of
        relying on _get_*/_set_* methods).

        """

        # frame = inspect.currentframe().f_back
        # cond = (frame.f_code is self.__init__.__func__.__code__)
        # assert cond, (('%s constructor must call BaseData constructor ' +
        #                'directly') % type(self).__name__)
        # args = inspect.getargvalues(frame)
        #
        # self._fields = tuple(args[0][1:])
        # for f in self._fields:
        #     cond = (f not in vars(self))
        #     assert cond, "'%s' object already has attribute '%s'" % (type(self).__name__, f)
        #     setattr(self, f, args[3][f])

        obj = object.__new__(cls)
        obj._grouping_manager = GroupingManager(obj)
        obj._quality_manager = QualityManager(obj._grouping_manager)

        return obj

        # NoNewAttributesAfterInit.__init__(self)

    def get_dep(self, filter=True):
        dep = self._get_dep()
        if filter:
            return self._grouping_manager.apply(dep)
        return dep

    @abc.abstractmethod
    def _get_dep(self):
        return numpy.array([])

    @abc.abstractmethod
    def get_indep(self, filter=True):
        return numpy.array([])

    @property
    def quality(self):
        return self._quality_manager.quality

    @quality.setter
    def quality(self, quality_array):
        self._quality_manager.quality = quality_array

    def group_counts(self, num, maxLength=None, tabStops=None):
        self._grouping_manager.group_counts(num, maxLength, tabStops)

    def __str__(self):
        """

        Return a listing of the attributes listed in self._fields and,
        if present, self._extra_fields.

        """

        fields = self._fields + getattr(self, '_extra_fields', ())
        fdict = dict(izip(fields, [getattr(self, f) for f in fields]))
        return print_fields(fields, fdict)


class Data(BaseData):
    "Generic data set"

    def __init__(self, name, indep, dep, staterror=None, syserror=None):
        """

        Initialize a Data instance.  indep should be a tuple of
        independent axis arrays, dep should be an array of dependent
        variable values, and staterror and syserror should be arrays
        of statistical and systematic errors, respectively, in the
        dependent variable (or None).

        """

        BaseData.__init__(self)

    def __repr__(self):
        r = '<%s data set instance' % type(self).__name__
        if hasattr(self, 'name'):
            r += " '%s'" % self.name
        r += '>'
        return r

    def eval_model(self, modelfunc):
        return modelfunc(*self.get_indep())

    def eval_model_to_fit(self, modelfunc):
        return modelfunc(*self.get_indep(filter=True))

    #
    # Primary properties.  These can depend only on normal attributes (and not
    # other properties).
    #

    def get_indep(self, filter=False):
        """Return the independent axes of a data set.

        Parameters
        ----------
        filter : bool, optional
           Should the filter attached to the data set be applied to
           the return value or not. The default is `False`.

        Returns
        -------
        axis: tuple of arrays
           The independent axis values for the data set. This gives
           the coordinates of each point in the data set.

        See Also
        --------
        get_dep : Return the dependent axis of a data set.

        """
        indep = getattr(self, 'indep', None)
        filter=bool_cast(filter)
        if filter:
            indep = tuple([self.apply_filter(x) for x in indep])
        return indep

    def get_staterror(self, filter=False, staterrfunc=None):
        """Return the statistical error on the dependent axis of a data set.

        Parameters
        ----------
        filter : bool, optional
           Should the filter attached to the data set be applied to
           the return value or not. The default is `False`.
        staterrfunc : function
           If no statistical error has been set, the errors will
           be calculated by applying this function to the
           dependent axis of the data set.

        Returns
        -------
        axis : array or `None`
           The statistical error for each data point. A value of
           `None` is returned if the data set has no statistical error
           array and `staterrfunc` is `None`.

        See Also
        --------
        get_error : Return the errors on the dependent axis of a data set.
        get_indep : Return the independent axis of a data set.
        get_syserror : Return the systematic errors on the dependent axis of a data set.

        """
        staterror = getattr(self, 'staterror', None)
        filter=bool_cast(filter)
        if filter:
            staterror = self.apply_filter(staterror)

        if (staterror is None) and (staterrfunc is not None):
            dep = self.get_dep()
            if filter:
                dep = self.apply_filter(dep)
            staterror = staterrfunc(dep)
        return staterror

    def get_syserror(self, filter=False):
        """Return the statistical error on the dependent axis of a data set.

        Parameters
        ----------
        filter : bool, optional
           Should the filter attached to the data set be applied to
           the return value or not. The default is `False`.

        Returns
        -------
        axis : array or `None`
           The systematic error for each data point. A value of
           `None` is returned if the data set has no systematic
           errors.

        See Also
        --------
        get_error : Return the errors on the dependent axis of a data set.
        get_indep : Return the independent axis of a data set.
        get_staterror : Return the statistical errors on the dependent axis of a data set.

        """
        syserr = getattr(self, 'syserror', None)
        filter=bool_cast(filter)
        if filter:
            syserr = self.apply_filter(syserr)
        return syserr

    #
    # Utility methods
    #

    def _wrong_dim_error(self, baddim):
        raise DataErr('wrongdim', self.name, baddim)

    def _no_image_error(self):
        raise DataErr('notimage', self.name)

    def _no_dim_error(self):
        raise DataErr('nodim', self.name)

    #
    # Secondary properties.  To best support subclasses, these should depend
    # only on the primary properties whenever possible, though there may be
    # instances when they depend on normal attributes.
    #

    def get_dims(self):
        self._no_dim_error()

    def get_error(self, filter=False, staterrfunc=None):
        """Return the total error on the dependent variable.

        Parameters
        ----------
        filter : bool, optional
           Should the filter attached to the data set be applied to
           the return value or not. The default is `False`.
        staterrfunc : function
           If no statistical error has been set, the errors will
           be calculated by applying this function to the
           dependent axis of the data set.

        Returns
        -------
        axis : array or `None`
           The error for each data point, formed by adding the
           statistical and systematic errors in quadrature.

        See Also
        --------
        get_dep : Return the independent axis of a data set.
        get_staterror : Return the statistical errors on the dependent axis of a data set.
        get_syserror : Return the systematic errors on the dependent axis of a data set.

        """
        return calc_total_error(self.get_staterror(filter, staterrfunc),
                                self.get_syserror(filter))

    def get_x(self, filter=False):
        "Return linear view of independent axis/axes"
        self._wrong_dim_error(1)

    def get_xerr(self, filter=False):
        "Return linear view of bin size in independent axis/axes"
        return None

    def get_xlabel(self):
        "Return label for linear view ofindependent axis/axes"
        return 'x'

    def get_y(self, filter=False, yfunc=None):
        "Return dependent axis in N-D view of dependent variable"
        y = self.get_dep(filter)

        if yfunc is not None:
            if filter:
                yfunc = self.eval_model_to_fit(yfunc)
            else:
                yfunc = self.eval_model(yfunc)
            y = (y, yfunc)

        return y

    def get_yerr(self, filter=False, staterrfunc=None):
        "Return errors in dependent axis in N-D view of dependent variable"
        return self.get_error(filter, staterrfunc)

    def get_ylabel(self, yfunc=None):
        "Return label for dependent axis in N-D view of dependent variable"
        return 'y'

    def get_x0(self, filter=False):
        "Return first dimension in 2-D view of independent axis/axes"
        self._wrong_dim_error(2)

    def get_x0label(self):
        "Return label for first dimension in 2-D view of independent axis/axes"
        return 'x0'

    def get_x1(self, filter=False):
        "Return second dimension in 2-D view of independent axis/axes"
        self._wrong_dim_error(2)

    def get_x1label(self):
        """

        Return label for second dimension in 2-D view of independent axis/axes

        """
        return 'x1'

    # For images, only need y-array
    # Also, we do not filter, as imager needs M x N (or
    # L x M x N) array
    def get_img(self, yfunc=None):
        "Return dependent variable as an image"
        self._no_image_error()

    def get_imgerr(self, yfunc=None):
        "Return total error in dependent variable as an image"
        self._no_image_error()

    def to_guess(self):
        arrays = [self.get_y(True)]
        arrays.extend(self.get_indep(True))
        return tuple(arrays)

    def to_fit(self, staterrfunc=None):
        return (self.get_dep(True),
                self.get_staterror(True, staterrfunc),
                self.get_syserror(True))

    def to_plot(self, yfunc=None, staterrfunc=None):
        return (self.get_x(True),
                self.get_y(True, yfunc),
                self.get_yerr(True, staterrfunc),
                self.get_xerr(True),
                self.get_xlabel(),
                self.get_ylabel())

    def to_contour(self, yfunc=None):
        return (self.get_x0(True),
                self.get_x1(True),
                self.get_y(True, yfunc),
                self.get_x0label(),
                self.get_x1label())


class DataSimulFit(Data):
    """Store multiple data sets.

    This class lets multiple data sets be treated as a single
    dataset by concatenation. That is, if two data sets have lengths
    n1 and n2 then they can be considered as a single data set of
    length n1 + n2.

    Parameters
    ----------
    name : str
        The name for the collection of data.
    datasets : sequence of Data objects
        The datasets to be stored; there must be at least one. They are
        assumed to behave as sherpa.data.Data objects, but there is no
        check for this condition.

    Attributes
    ----------
    datasets : sequence of Data

    See Also
    --------
    sherpa.models.model.SimulFitModel

    Examples
    --------

    >>> d1 = Data1D('d1', [1, 2, 3], [10, 12, 15])
    >>> d2 = Data1D('d2', [1, 2, 5, 7], [4, 15, 9, 24])
    >>> dall = DataSimulFit('comb', (d1, d2))
    >>> yvals, _, _ = dall.to_fit()
    >>> print(yvals)
    [10 12 15  4 15  9 24]

    """

    def __init__(self, name, datasets):
        if len(datasets) == 0:
            raise DataErr('zerodatasimulfit', type(self).__name__)
        datasets = tuple(datasets)
        BaseData.__init__(self)

    def eval_model_to_fit(self, modelfuncs):
        total_model = []

        for func, data in izip(modelfuncs, self.datasets):
            total_model.append(data.eval_model_to_fit(func))

        return numpy.concatenate(total_model)

    def to_fit(self, staterrfunc=None):
        total_dep = []
        total_staterror = []
        total_syserror = []

        no_staterror = True
        no_syserror  = True

        for data in self.datasets:
            dep, staterror, syserror = data.to_fit(staterrfunc)

            total_dep.append(dep)

            if staterror is not None:
                no_staterror = False
            total_staterror.append(staterror)

            if syserror is not None:
                no_syserror = False
            else:
                syserror = numpy.zeros_like(dep)
            total_syserror.append(syserror)

        total_dep = numpy.concatenate(total_dep)

        if no_staterror:
            total_staterror = None
        elif numpy.any([numpy.equal(array, None).any() for array in total_staterror]):
            raise DataErr('staterrsimulfit')
        else:
            total_staterror = numpy.concatenate(total_staterror)

        if no_syserror:
            total_syserror = None
        else:
            total_syserror = numpy.concatenate(total_syserror)

        return (total_dep, total_staterror, total_syserror)

    def to_plot(self, yfunc=None, staterrfunc=None):
        return self.datasets[0].to_plot(yfunc.parts[0], staterrfunc)


class DataND(Data):
    "Base class for Data1D, Data2D, etc."

    def _get_dep(self):
        y = self.y
        return y

    def set_dep(self, val):
        "Set the dependent variable values"
        dep = None
        if numpy.iterable(val):
            dep = numpy.asarray(val, SherpaFloat)
        else:
            val = SherpaFloat(val)
            dep = numpy.array([val]*len(self.get_indep()[0]))
        setattr(self, 'y', dep)


class Data1D(DataND):
    "1-D data set"

    def __init__(self, name, x, y, staterror=None, syserror=None):
        self.name = name
        self._x = x
        self.y = y
        self.staterror = staterror
        self.syserror = syserror
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter=bool_cast(filter)
        if filter:
            return (self._x,)
        return (self.x,)

    def get_x(self, filter=False):
        return self.get_indep(filter)[0]

    def get_dims(self, filter=False):
        return (len(self.get_x(filter)),)

    def get_img(self, yfunc=None):
        "Return 1D dependent variable as a 1 x N image"
        y_img = self.get_y(False, yfunc)
        if yfunc is not None:
            y_img = (y_img[0].reshape(1,y_img[0].size),
                     y_img[1].reshape(1,y_img[1].size))
        else:
            y_img = y_img.reshape(1,y_img.size)
        return y_img

    def get_imgerr(self):
        err = self.get_error()
        if err is not None:
            err = err.reshape(1,err.size)
        return err

    def notice(self, xlo=None, xhi=None, ignore=False):
        BaseData.notice(self, (xlo,), (xhi,), self.get_indep(), ignore)


class Data1DInt(Data1D):
    "1-D integrated data set"

    def __init__(self, name, xlo, xhi, y, staterror=None, syserror=None):
        self._lo = xlo
        self._hi = xhi
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter=bool_cast(filter)
        if filter:
            return (self._lo, self._hi)
        return (self.xlo, self.xhi)

    def get_x(self, filter=False):
        indep = self.get_indep(filter)
        return (indep[0] + indep[1]) / 2.0

    def get_xerr(self, filter=False):
        xlo,xhi = self.get_indep(filter)
        return xhi-xlo


class Data2D(DataND):
    "2-D data set"

    def __init__(self, name, x0, x1, y, shape=None, staterror=None,
                 syserror=None):
        self._x0 = x0
        self._x1 = x1
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter=bool_cast(filter)
        if filter:
            return (self._x0, self._x1)
        return (self.x0, self.x1)

    def get_x0(self, filter=False):
        return self.get_indep(filter)[0]


    def get_x1(self, filter=False):
        return self.get_indep(filter)[1]

    def get_axes(self):
        self._check_shape()
        # FIXME: how to filter an axis when self.mask is size of self.y?
        return (numpy.arange(self.shape[1])+1, numpy.arange(self.shape[0])+1)

    def get_dims(self, filter=False):
        #self._check_shape()
        if self.shape is not None:
            return self.shape[::-1]
        return (len(self.get_x0(filter)), len(self.get_x1(filter)))

    def _check_shape(self):
        if self.shape is None:
            raise DataErr('shape',self.name)

    def get_max_pos(self, dep=None):
        if dep is None:
            dep = self.get_dep(True)
        x0 = self.get_x0(True)
        x1 = self.get_x1(True)

        pos = numpy.asarray(numpy.where(dep == dep.max())).squeeze()
        if pos.ndim == 0:
            pos = int(pos)
            return (x0[pos], x1[pos])

        return [(x0[index], x1[index]) for index in pos]

    def get_img(self, yfunc=None):
        self._check_shape()
        y_img = self.get_y(False, yfunc)
        if yfunc is not None:
            y_img = (y_img[0].reshape(*self.shape),
                     y_img[1].reshape(*self.shape))
        else:
            y_img = y_img.reshape(*self.shape)
        return y_img

    def get_imgerr(self):
        self._check_shape()
        err = self.get_error()
        if err is not None:
            err = err.reshape(*self.shape)
        return err


class Data2DInt(Data2D):
    "2-D integrated data set"

    def __init__(self, name, x0lo, x1lo, x0hi, x1hi, y, shape=None,
                 staterror=None, syserror=None):
        self._x0lo = x0lo
        self._x1lo = x1lo
        self._x0hi = x0hi
        self._x1hi = x1hi
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter=bool_cast(filter)
        if filter:
            return (self._x0lo, self._x1lo, self._x0hi, self._x1hi)
        return (self.x0lo, self.x1lo, self.x0hi, self.x1hi)

    def get_x0(self, filter=False):
        indep = self.get_indep(filter)
        return (indep[0] + indep[2]) / 2.0

    def get_x1(self, filter=False):
        indep = self.get_indep(filter)
        return (indep[1] + indep[3]) / 2.0
