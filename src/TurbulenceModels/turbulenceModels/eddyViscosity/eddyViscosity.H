/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::eddyViscosity

Group
    grpTurbulence

Description
    Eddy viscosity turbulence model base class

SourceFiles
    eddyViscosity.C

\*---------------------------------------------------------------------------*/

#ifndef eddyViscosity_H
#define eddyViscosity_H

#include "linearViscousStress.H"
#include "inferenceEngine.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class eddyViscosity Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class eddyViscosity
:
    public linearViscousStress<BasicTurbulenceModel>
{

protected:

    // Protected data

        // Fields

            volScalarField nut_;


    // Protected Member Functions

        virtual void correctNut() = 0;


    // Scaling

        mutable std::vector<double> scaler_X_max_abs_;
        mutable std::vector<double> scaler_y_max_abs_;
        mutable bool scalers_loaded_;

        void loadScalers() const;
        void normalizeInput(double* buffer, const std::vector<int64_t>& dims) const;
        void denormalizeOutput(double* buffer, const int64_t N) const;
        void writeTensorDebugCompact(const volSymmTensorField& R) const;
        // Fonctions de debug
        void writeInputBeforeNormalization(const double* buffer, const std::vector<int64_t>& dims) const;
        void writeInputAfterNormalization(const double* buffer, const std::vector<int64_t>& dims) const;
        void writeModelOutput(const double* buffer, const int64_t N) const;
        void writeVelocityDebug(const volVectorField& U) const;
        void writeGradientComponents(const volTensorField& gradU) const;

        void writeModelOutputCompactBeforeDenormalization(const double* buffer, int64_t N) const;
        void writeModelOutputCompactAfterDenormalization(const double* buffer, int64_t N) const;


        //- Debug parallel information (processors, mesh distribution)
        void debugParallelInfo() const;


        //- Write parallel-specific debug data for each processor
        void writeParallelDebugData(const double* buffer, 
                                   const std::vector<int64_t>& dims) const;

        //- Compare ML results across processors
        void compareResultsAcrossProcessors(const double* buffer, int64_t N) const;

        //- Broadcast scalers to all processors
        void broadcastScalers() const;  // NOUVELLE FONCTION



public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    mutable infEngine::inferenceEngine infEngine_;


    // Constructors

        //- Construct from components
        eddyViscosity
        (
            const word& modelName,
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName
        );


    //- Destructor
    virtual ~eddyViscosity() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read() = 0;

        //- Return the turbulence viscosity
        virtual tmp<volScalarField> nut() const
        {
            return nut_;
        }

        //- Return the turbulence viscosity on patch
        virtual tmp<scalarField> nut(const label patchi) const
        {
            return nut_.boundaryField()[patchi];
        }

        //- Return the turbulence kinetic energy
        virtual tmp<volScalarField> k() const = 0;



        std::tuple<double*, std::vector<int64_t>, int64_t> loadData() const;
        double* runInference(double* buffer, const std::vector<int64_t> dims) const;
        tmp<Field<symmTensor>> convert2field (double* _MlRBuffer, const int64_t N) const;

        //- Return the Reynolds stress tensor
        virtual tmp<volSymmTensorField> R() const;

        //- Validate the turbulence fields after construction
        //  Update turbulence viscosity and other derived fields as requires
        virtual void validate();

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct() = 0;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "eddyViscosity.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
