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
#include "fvCFD.H"
#include "fvc.H"
#include "fvm.H"
#include "inferenceEngine.h"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template <class BasicTurbulenceModel>
Foam::eddyViscosity<BasicTurbulenceModel>::eddyViscosity(const word &type, const alphaField &alpha,
                                                         const rhoField &rho, const volVectorField &U,
                                                         const surfaceScalarField &alphaRhoPhi,
                                                         const surfaceScalarField &phi,
                                                         const transportModel &transport,
                                                         const word &propertiesName)
    : linearViscousStress<BasicTurbulenceModel>(type, alpha, rho, U, alphaRhoPhi, phi, transport,
                                                propertiesName),

      nut_(IOobject(IOobject::groupName("nut", alphaRhoPhi.group()), this->runTime_.timeName(), this->mesh_,
                    IOobject::MUST_READ, IOobject::AUTO_WRITE, IOobject::REGISTER),
           this->mesh_) {
  std::cout << "EDDYVISCOSITY::EDDYVISCOSITY-> Initializing infEngine\n";
  // this->infEngine_ = std::make_unique<infEngine::inferenceEngine>();
  this->infEngine_.init("/opt/InfEngine/ml4turb.pt", "TorchScript", true, 1, 1);
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template <class BasicTurbulenceModel> bool Foam::eddyViscosity<BasicTurbulenceModel>::read() {
  return BasicTurbulenceModel::read();
}

template <class BasicTurbulenceModel>
std::tuple<double *, std::vector<int64_t>, int64_t>
Foam::eddyViscosity<BasicTurbulenceModel>::loadData() const {
  std::cout << "EDDYVISCOSITY::loadData-> loading data to input buffer\n";
  tmp<volTensorField> tgradU = fvc::grad(this->U_);
  Field<tensor> &gradU = tgradU.ref().internalFieldRef();
  double *_buffer = gradU.data()->data();

  const int64_t N = gradU.size(), D = 9;
  const std::vector<int64_t> dims = {N, D};
  
  // Copy buffer for ownership, so we don't have a double free or a dangling pointer if Field is deleted
  // before Ideally, we'd have a move to transfer the buffer data ownership and safely delete Field
  double *buffer = new double[N * D];
  std::copy(_buffer, _buffer + N * D, buffer);

  std::cout << "EDDYVISCOSITY::loadData-> data loaded to input buffer\n";
  return {buffer, dims, N};
}

template <class BasicTurbulenceModel>
double *Foam::eddyViscosity<BasicTurbulenceModel>::runInference(double *buffer,
                                                                const std::vector<int64_t> dims) const {

  std::cout << "EDDYVISCOSITY::runInference-> running infEngine\n";
  this->infEngine_.loadData(buffer, dims);
  this->infEngine_.compute();

  double* _MlRBuffer = this->infEngine_.getResults<double>();

  this->infEngine_.clearInputs();

  std::cout << "EDDYVISCOSITY::runInference-> inference done\n";

  return _MlRBuffer;
}

template <class BasicTurbulenceModel>
tmp<Field<symmTensor>> Foam::eddyViscosity<BasicTurbulenceModel>::convert2field(double *MlRBuffer,
                                                                                const int64_t N) const {
  std::cout << "EDDYVISCOSITY::convert2field-> converting buffer to field\n";

  // Copy buffer to prevent segFault (takeResults and transfers ownership not possible).
  int64_t N6 = N * 6;

  // build Field
  List<symmTensor> MlRList(N);
  for (label i = 0; i < N; ++i) {
    MlRList[i] = symmTensor(MlRBuffer[6 * i + 0], MlRBuffer[6 * i + 1], MlRBuffer[6 * i + 2],
                            MlRBuffer[6 * i + 3], MlRBuffer[6 * i + 4], MlRBuffer[6 * i + 5]);
  }
  Field<symmTensor> MlRInternalField(std::move(MlRList));
  tmp<Field<symmTensor>> tMlRInternalField(new Field<symmTensor>(std::move(MlRInternalField)));

  // Info << "EDDYVISCOSITY::convert2field-> Field first element: " << tMlRInternalField.ref()[0] << Foam::nl;
  std::cout << "EDDYVISCOSITY::convert2field-> buffer converted to field\n";

  return tMlRInternalField;
}

template <class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField> Foam::eddyViscosity<BasicTurbulenceModel>::R() const {
  auto [buffer, dims, N] = loadData();
  double *_MlRBuffer = runInference(buffer, dims);
  tmp<Field<symmTensor>> tMlRInternalField = convert2field(_MlRBuffer, N);

  auto result = volSymmTensorField::New(IOobject::groupName("R", this->alphaRhoPhi_.group()),
                                        IOobject::NO_REGISTER, this->mesh_, dimensionSet(0, 2, -2, 0, 0),
                                        tMlRInternalField(), fvPatchFieldBase::calculatedType());

  return result;
}

template <class BasicTurbulenceModel> void Foam::eddyViscosity<BasicTurbulenceModel>::validate() {
  correctNut();
}

template <class BasicTurbulenceModel> void Foam::eddyViscosity<BasicTurbulenceModel>::correct() {
  BasicTurbulenceModel::correct();
}

// ************************************************************************* //
