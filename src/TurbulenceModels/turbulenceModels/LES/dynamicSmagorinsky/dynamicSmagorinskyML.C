/*---------------------------------------------------------------------------*\
dynamicSmagorinskyML - Implementation of the dynamic Smagorinsky
                     SGS model.

Copyright Information
    Copyright (C) 1991-2009 OpenCFD Ltd.
    Copyright (C) 2010-2024 Alberto Passalacqua 

License
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
\*---------------------------------------------------------------------------*/

#include "dynamicSmagorinskyML.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void dynamicSmagorinskyML<BasicTurbulenceModel>::correctNut
(
    const tmp<volTensorField>& gradU
)
{
    const volSymmTensorField S(dev(symm(gradU)));

    // The SGS viscosity is bounded so that nuEff cannot become negative.
    // Values are limited here, and not in nuEff, for consistency in stored
    // data and in submodels using nuSgs().
    // No warning message is printed when this limitation is applied.
    this->nut_ = max(cD_*sqr(this->delta())*sqrt(magSqr(S)), -this->nu());

    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void dynamicSmagorinskyML<BasicTurbulenceModel>::correctNut()
{
    correctNut(fvc::grad(this->U_));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
dynamicSmagorinskyML<BasicTurbulenceModel>::dynamicSmagorinskyML
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    LESeddyViscosity<BasicTurbulenceModel>
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

    cD_
    (
        IOobject
        (
            IOobject::groupName("cD_", this->alphaRhoPhi_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("cD", dimless, 0.0)
    ),

    cI_
    (
        IOobject
        (
            IOobject::groupName("cI_", this->alphaRhoPhi_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("cI", dimless, 0.0)
    ),
    filterPtr_(LESfilter::New(U.mesh(), this->coeffDict())),
    filter_(filterPtr_()),
    TSGS_
    (
        IOobject
        (
            IOobject::groupName("TSGS", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    )
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
#ifdef USE_ML4TURB
    ml4turb_enabled_ = this->LESDict_.template getOrDefault<Switch>("ml4turb",false);
    if(ml4turb_enabled_)
    {
      ml4trub_model_ = this->LESDict_.template getOrDefault<word>("MLModelConfig","ml4turb-config.json") ;
      clip_ = this->LESDict_.template getOrDefault<double>("clip",0.) ;
      std::cout<<"ML4TURB INFO : MODEL PATH "<<ml4trub_model_<<std::endl ;
      std::cout<<"               CLIP VALUE "<<clip_<<std::endl ;
    }
#endif
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void dynamicSmagorinskyML<BasicTurbulenceModel>::calcCD
(
    const volSymmTensorField& S
)
{
    const volVectorField& U = this->U_;
    tmp<volSymmTensorField> LL = dev(filter_(sqr(U)) - (sqr(filter_(U))));

    const volSymmTensorField MM
    (
        sqr(this->delta())
       *(filter_(mag(S)*(S)) - 4.0*mag(filter_(S))*filter_(S))
    );

    // Locally averaging MMMM on cell faces
    volScalarField MMMM(fvc::average(magSqr(MM)));

    MMMM.max(VSMALL);

    // Performing local average on cell faces on return
    cD_ = 0.5*fvc::average(LL && MM)/MMMM;

    if (debug)
    {
        Info<< "min(cD) = " << min(cD_) 
            << ", max(cD) = " << max(cD_)
            << ", average(cD) = " << average(cD_) << endl;
    }
}

template<class BasicTurbulenceModel>
void dynamicSmagorinskyML<BasicTurbulenceModel>::calcCI
(
    const volSymmTensorField& S
)
{
    const volVectorField& U = this->U_;
    tmp<volScalarField> KK = 0.5*(filter_(magSqr(U)) - magSqr(filter_(U)));

    const volScalarField mm
    (
        sqr(this->delta())*(4*sqr(mag(filter_(S))) - filter_(sqr(mag(S))))
    );

    // Locally averaging mmmm on cell faces
    volScalarField mmmm(fvc::average(magSqr(mm)));

    mmmm.max(VSMALL);

    // Performing local average on cell faces on return
    cI_ = fvc::average(KK*mm)/mmmm;

    if (debug)
    {
        Info<< "min(cI) = " << min(cI_) 
            << ", max(cI) = " << max(cI_)
            << ", average(cI) = " << average(cI_) << endl;
    }
}


template<class BasicTurbulenceModel>
bool dynamicSmagorinskyML<BasicTurbulenceModel>::read()
{
    if (LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        filter_.read(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void dynamicSmagorinskyML<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const volVectorField& U = this->U_;

    LESeddyViscosity<BasicTurbulenceModel>::correct();

    tmp<volTensorField> tgradU(fvc::grad(U));
    const volTensorField& gradU = tgradU();

    volSymmTensorField S(dev(symm(gradU)));

    calcCD(S);
    calcCI(S);
    correctNut(gradU);

#ifdef USE_ML4TURB
    if(!ml4turb_api_initialized_ && ml4turb_enabled_)
    {
      mlturb_api_ptr_.reset(new ml4turb::ML4Turb()) ;
      mlturb_api_ptr_->initFromConfigFile(ml4trub_model_) ;
      ml4turb_api_initialized_ = true;
    }
#endif

    std::cout<<"BUILD T SGS"<<std::endl ;
    volTensorField turb_t((this->alpha_*this->rho_*this->nut())*dev2(T(fvc::grad(this->U_)))) ;
    volTensorField nu_t((this->alpha_*this->rho_*this->nu())*dev2(T(fvc::grad(this->U_)))) ;
    volTensorField nueff_t((this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(this->U_)))) ;
    volSymmTensorField D(symm(fvc::grad(this->U_)));
    std::size_t icount = 0;
#ifdef USE_ML4TURB
    auto nb_cells = this->mesh_.cells().size() ;
    std::vector<bool> filter(nb_cells) ;
    double mag_d_max = 0. ;
    if(ml4turb_enabled_)
    {
      mlturb_api_ptr_->startCompute(nb_cells) ;
      forAll(TSGS_, celli)
      {
          const symmTensor& d = D[celli];
          double mag_d = mag(d) ;
          mag_d_max = std::max(mag_d_max,mag_d) ;
          if(mag_d>clip_)
          {
            std::vector<float> grad_vit = {  d.xx(), d.xy(), d.xz(),
                                             d.yx(), d.yy(), d.yz(),
                                             d.xz(), d.yz(), d.zz() };
            mlturb_api_ptr_->asynchCompute(grad_vit) ;
            filter[icount] = true ;
          }
          else
            filter[icount] = false ;
          ++icount ;
      }
      mlturb_api_ptr_->endCompute() ;
    }
    std::cout<<"MAG D MAX : "<<mag_d_max<<std::endl ;
    std::vector<float> tau_ij(6) ;
    icount = 0 ;
 #endif

    forAll(TSGS_, celli)
    {
      symmTensor& tsgs = TSGS_[celli];
      auto& t = nu_t[celli] ;
#ifdef USE_ML4TURB
      if(filter[icount] && ml4turb_enabled_)
      {
        mlturb_api_ptr_->getNextResult(tau_ij) ;
        tsgs = symmTensor(tau_ij[0]+t[0],tau_ij[1]+t[1],tau_ij[2]+t[2],
                                         tau_ij[3]+t[3],tau_ij[4]+t[4],
                                                        tau_ij[5]+t[5]) ;
      }
      else
#endif
      {
        auto& t = nueff_t[celli] ;
        tsgs = symmTensor(t.xx(),t.xy(),t.xz(),
                                 t.yy(),t.yz(),
                                        t.zz()) ;
      }
      ++icount ;
    }
    std::cout<<"FIN dynamicSmagorinskyML::correct()"<<typeid(*this).name()<<std::endl;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //
