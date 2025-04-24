/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2017 OpenFOAM Foundation
    Copyright (C) 2023 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "eddyViscosity.H"
#include "fvc.H"
#include "fvm.H"
// #include "inferenceEngine.h"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Foam::eddyViscosity<BasicTurbulenceModel>::eddyViscosity
(
    const word& type,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    linearViscousStress<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    nut_
    (
        IOobject
        (
            IOobject::groupName("nut", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE,
            IOobject::REGISTER
        ),
        this->mesh_
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool Foam::eddyViscosity<BasicTurbulenceModel>::read()
{
    return BasicTurbulenceModel::read();
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::eddyViscosity<BasicTurbulenceModel>::R() const
{
    // std::cout<<"EDDYVISCOSITY::R-> Creating infEngine"<<std::endl;
    // infEngine::inferenceEngine inf_engine;
    // std::cout<<"EDDYVISCOSITY::R-> Init"<<std::endl;
    // inf_engine.init("/opt/InfEngine/ml4turb.pt", "TorchScript", true, 1, 1);
    // std::cout<<"EDDYVISCOSITY::R-> Getting buffer data"<<std::endl;
    // const double* dataPtr = this->U_.internalField().cdata()->v_;
    // std::cout<<"EDDYVISCOSITY::R-> Getting dimensions"<<std::endl;
    // const label N = this->U_.internalField().size();  // Number of points (cells, faces, etc.)
    // const label D = Vector<double>::nComponents;      // Should be 3 (x, y, z) for the field U
    // const std::vector<int64_t> dims = {static_cast<int64_t>(N), static_cast<int64_t>(D)};
    // std::cout<<"EDDYVISCOSITY::R-> Loading data"<<std::endl;
    // inf_engine.loadData(dataPtr, dims);
    // std::cout<<"EDDYVISCOSITY::R-> Computing inference"<<std::endl;
    // inf_engine.compute();
    // std::cout<<"EDDYVISCOSITY::R-> Getting results"<<std::endl;
    // double* ml_R = inf_engine.getResults<double>();
    // std::cout<<"EDDYVISCOSITY::R-> ML inference done in openfoam!"<<std::endl;

    std::cout<<"EDDYVISCOSITY::R"<<std::endl ;
    tmp<volScalarField> tk(k());

    // Get list of patchField type names from k
    wordList patchFieldTypes(tk().boundaryField().types());

    // For k patchField types which do not have an equivalent for symmTensor
    // set to calculated
    forAll(patchFieldTypes, i)
    {
        if
        (
           !fvPatchField<symmTensor>::patchConstructorTablePtr_
                ->contains(patchFieldTypes[i])
        )
        {
            patchFieldTypes[i] = fvPatchFieldBase::calculatedType();
        }
    }

    return volSymmTensorField::New
    (
        IOobject::groupName("R", this->alphaRhoPhi_.group()),
        IOobject::NO_REGISTER,
        ((2.0/3.0)*I)*tk() - (nut_)*devTwoSymm(fvc::grad(this->U_)),
        patchFieldTypes
    );
}


template<class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::validate()
{
    correctNut();
}


template<class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::correct()
{
    std::cout<<"eddyViscosity::correct()"<<std::endl ;
    BasicTurbulenceModel::correct();
}


// ************************************************************************* //
