#!/usr/bin/env bash

# Tarballs to fetch
ds9_site=http://ds9.si.edu/download/centos6/
xpa_site=http://ds9.si.edu/download/centos6/
ds9_tar=ds9.centos6.7.5.tar.gz
xpa_tar=xpa.centos6.2.1.17.tar.gz

# Fetch them
wget --quiet $ds9_site$ds9_tar
wget --quiet $xpa_site$xpa_tar

# untar them
main_dir=$(pwd)
cd $miniconda/bin
tar xf $main_dir/$ds9_tar
tar xf $main_dir/$xpa_tar
cd -

# install build dependencies
if [[ $TRAVIS_OS_NAME == "linux" ]];
then
    sudo apt-get update
    sudo apt-get install -qq libwcs4 wcslib-dev libx11-dev libsm-dev libxrender-dev

    # Special requirement for the linux build, we need to pin down the exact version of the library xspec was
    # built with
    sed -i.orig "s|#gfortran_libraries = gfortran|gfortran_libraries= :libgfortran.so.3|g" setup.cfg
else  # osx
    # It looks like xvfb doesn't "just work" on osx travis, so...
    sudo Xvfb :99 -ac -screen 0 1024x768x8 &
fi

# Set HEADAS environment variables
export HEADAS=$miniconda/envs/build/Xspec/spectral

xspec_library_path=$miniconda/envs/build/Xspec/x86_64-unknown-linux-gnu-libc2.15-0/lib/
xspec_include_path=$miniconda/envs/build/Xspec/x86_64-unknown-linux-gnu-libc2.15-0/include/
wcs_library_path=/usr/lib/x86_64-linux-gnu/

# Change build configuration
sed -i.orig "s/#with-xspec=True/with-xspec=True/g" setup.cfg
sed -i.orig "s|#xspec_lib_dirs = None|xspec_lib_dirs=${xspec_library_path}|g" setup.cfg
sed -i.orig "s|#xspec_include_dirs = None|xspec_include_dirs=${xspec_include_path}|g" setup.cfg
sed -i.orig "s|#wcslib_lib_dirs = None|wcslib_lib_dirs=${wcs_library_path}|g" setup.cfg
