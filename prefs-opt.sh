#----------------------------------*-sh-*--------------------------------------
# =========                 |
# \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
#  \\    /   O peration     |
#   \\  /    A nd           | Copyright (C) 2010-2011 OpenCFD Ltd.
#    \\/     M anipulation  |
#------------------------------------------------------------------------------
#                           | Copyright (C) 2011-2016 OpenFOAM Foundation
#------------------------------------------------------------------------------
# License
#     This file is part of OpenFOAM, licensed under GNU General Public License
#     <http://www.gnu.org/licenses/>.
#
# File
#     config.csh/example/prefs.csh
#     - sourced by OpenFOAM-*/etc/cshrc
#
# Description
#     Example of preset variables for the OpenFOAM configuration (C-Shell shell)
#
# See also
#     'foamEtcFile -help' or 'foamEtcFile -list' for information about the
#     paths searched
#
#------------------------------------------------------------------------------

source etc/bashrc
export WM_COMPILE_OPTION=Opt
export WM_NCOMPPROCS=1
export PV_PLUGIN_PATH=$FOAM_LIBBIN/paraview-5.4
export PATH=/work/vitalbre/local_installs/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin:$PATH

#------------------------------------------------------------------------------
