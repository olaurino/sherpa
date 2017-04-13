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
import pytest

from sherpa.data import BaseData


def test_base_class_abstract():
    with pytest.raises(TypeError) as excinfo:
        BaseData()
    assert "Can't instantiate abstract class" in str(excinfo.value)


def test_call_base_data_init():
    class MyData(BaseData):
        def __init__(self, foo):
            BaseData.__init__(self)

        def get_dep(self, filter=True):
            pass

        def get_indep(self, filter=True):
            pass

    my_data = MyData("bar")
    assert "bar" == my_data.foo
