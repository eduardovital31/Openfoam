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

#include "processorFvPatch.H"  // Pour processorFvPatch
#include "Pstream.H"           // Pour les communications parallèles


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
           this->mesh_),
      scalers_loaded_(false) {
  std::cout << "EDDYVISCOSITY::EDDYVISCOSITY-> Initializing infEngine\n";
  // this->infEngine_ = std::make_unique<infEngine::inferenceEngine>();
  this->infEngine_.init("/opt/InfEngine/ml4turb.pt", "TorchScript", false, 1, 1);
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::loadScalers() const {
    if (scalers_loaded_) return;
    
    std::cout << "EDDYVISCOSITY::loadScalers-> Loading normalization parameters\n";
        
    std::ifstream scaler_X_file("/opt/InfEngine/scaler_X.txt");
    std::ifstream scaler_y_file("/opt/InfEngine/scaler_y.txt");

    if (!scaler_X_file.is_open()) {
      std::cout << "ERROR: Cannot open /opt/InfEngine/scaler_X.txt" << std::endl;
      return;
    }
    if (!scaler_y_file.is_open()) {
        std::cout << "ERROR: Cannot open /opt/InfEngine/scaler_y.txt" << std::endl;
        return;
    }
  
  // Charger et vérifier scaler_X
  scaler_X_max_abs_.resize(9);
  for (int i = 0; i < 9; ++i) {
      scaler_X_file >> scaler_X_max_abs_[i];
      std::cout << "scaler_X[" << i << "] = " << scaler_X_max_abs_[i] << std::endl;
      if (scaler_X_max_abs_[i] == 0.0) {
          std::cout << "WARNING: scaler_X[" << i << "] is zero!" << std::endl;
      }
  }
    
    scaler_X_max_abs_.resize(9);
    for (int i = 0; i < 9; ++i) {
        scaler_X_file >> scaler_X_max_abs_[i];
    }
    
    scaler_y_max_abs_.resize(6);
    for (int i = 0; i < 6; ++i) {
        scaler_y_file >> scaler_y_max_abs_[i];
    }
    
    scalers_loaded_ = true;
    std::cout << "EDDYVISCOSITY::loadScalers-> Scalers loaded successfully\n";
}


template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::normalizeInput(double* buffer, 
                                                              const std::vector<int64_t>& dims) const {
    const int64_t N = dims[0];
    const int64_t D = dims[1]; 
    
    std::cout << "EDDYVISCOSITY::normalizeInput-> Normalizing input data\n";
    
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < D; ++j) {
            int64_t idx = i * D + j;
            buffer[idx] = buffer[idx] / scaler_X_max_abs_[j];
        }
    }
}

template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::denormalizeOutput(double* buffer, 
                                                                 const int64_t N) const {
    std::cout << "EDDYVISCOSITY::denormalizeOutput-> Denormalizing output data\n";
    
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < 6; ++j) {
            int64_t idx = i * 6 + j;
            buffer[idx] = buffer[idx] * scaler_y_max_abs_[j];
        }
    }
}

template <class BasicTurbulenceModel> bool Foam::eddyViscosity<BasicTurbulenceModel>::read() {
  return BasicTurbulenceModel::read();
}

template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::writeGradientComponents(const volTensorField& gradU) const {
    std::string filename = "debug_gradient_components_t" + std::to_string(this->runTime_.timeOutputValue()) + ".txt";
    
    std::ofstream debugFile(filename);
    const Field<tensor>& gradUInternal = gradU.internalField();
    
    debugFile << "Temps: " << this->runTime_.timeOutputValue() << std::endl;
    debugFile << "Cellules: " << gradUInternal.size() << std::endl;
    debugFile << "Ordre OpenFOAM: xx xy xz yx yy yz zx zy zz" << std::endl;
    
    label step = max(1, gradUInternal.size() / 50);
    
    for (label cellI = 0; cellI < gradUInternal.size(); cellI += step) {
        const tensor& grad = gradUInternal[cellI];
        debugFile << "Cell " << cellI << ": "
                  << grad.xx() << " " << grad.xy() << " " << grad.xz() << " "
                  << grad.yx() << " " << grad.yy() << " " << grad.yz() << " "
                  << grad.zx() << " " << grad.zy() << " " << grad.zz() << std::endl;
    }
    
    debugFile.close();
    std::cout << "DEBUG: Composantes du gradient écrites dans " << filename << std::endl;
}


template <class BasicTurbulenceModel>
std::tuple<double *, std::vector<int64_t>, int64_t>
Foam::eddyViscosity<BasicTurbulenceModel>::loadData() const {
  std::cout << "EDDYVISCOSITY::loadData-> loading data to input buffer\n";
  // NOUVEAU: Debug du champ de vitesse avant calcul du gradient
//   writeVelocityDebug(this->U_);
  tmp<volTensorField> tgradU = fvc::grad(this->U_);

   // NOUVEAU: Debug des composantes du gradient
//    writeGradientComponents(tgradU.ref());

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


template <class BasicTurbulenceModel> void Foam::eddyViscosity<BasicTurbulenceModel>::validate() {
  correctNut();
}

template <class BasicTurbulenceModel> void Foam::eddyViscosity<BasicTurbulenceModel>::correct() {
  BasicTurbulenceModel::correct();
}



// Fonction pour écrire les entrées avant normalisation
template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::writeInputBeforeNormalization(const double* buffer, 
                                                                              const std::vector<int64_t>& dims) const {
    const int64_t N = dims[0];
    const int64_t D = dims[1]; // Devrait être 9 pour les 9 composantes du gradient
    
    std::string filename = "debug_input_before_norm_t" + std::to_string(this->runTime_.timeOutputValue()) + ".txt";
    
    std::ofstream debugFile(filename);
    
    if (!debugFile.is_open()) {
        std::cout << "ERROR: Cannot open debug file " << filename << std::endl;
        return;
    }
    
    debugFile << "Temps: " << this->runTime_.timeOutputValue() << std::endl;
    debugFile << "Points: " << N << std::endl;
    debugFile << "Dimensions: " << D << " (composantes du gradient de vitesse)" << std::endl;
    debugFile << "Format: Point | dU/dx_x dU/dx_y dU/dx_z dU/dy_x dU/dy_y dU/dy_z dU/dz_x dU/dz_y dU/dz_z" << std::endl;
    
    // Échantillonner ~100 points
    label step = max(1, N / 100);
    
    for (int64_t i = 0; i < N; i += step) {
        debugFile << "Point " << i << ": ";
        for (int64_t j = 0; j < D; ++j) {
            int64_t idx = i * D + j;
            debugFile << buffer[idx];
            if (j < D - 1) debugFile << " ";
        }
        debugFile << std::endl;
    }
    
    debugFile.close();
    std::cout << "DEBUG: Entrées avant normalisation écrites dans " << filename << std::endl;
}

// Fonction pour écrire les entrées après normalisation
template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::writeInputAfterNormalization(const double* buffer, 
                                                                             const std::vector<int64_t>& dims) const {
    const int64_t N = dims[0];
    const int64_t D = dims[1];
    
    std::string filename = "debug_input_after_norm_t" + std::to_string(this->runTime_.timeOutputValue()) + ".txt";
    
    std::ofstream debugFile(filename);
    
    if (!debugFile.is_open()) {
        std::cout << "ERROR: Cannot open debug file " << filename << std::endl;
        return;
    }
    
    debugFile << "Temps: " << this->runTime_.timeOutputValue() << std::endl;
    debugFile << "Points: " << N << std::endl;
    debugFile << "Dimensions: " << D << " (composantes du gradient de vitesse normalisées)" << std::endl;
    debugFile << "Format: Point | dU/dx_x_norm dU/dx_y_norm dU/dx_z_norm dU/dy_x_norm dU/dy_y_norm dU/dy_z_norm dU/dz_x_norm dU/dz_y_norm dU/dz_z_norm" << std::endl;
    
    // Échantillonner ~100 points
    label step = max(1, N / 100);
    
    for (int64_t i = 0; i < N; i += step) {
        debugFile << "Point " << i << ": ";
        for (int64_t j = 0; j < D; ++j) {
            int64_t idx = i * D + j;
            debugFile << buffer[idx];
            if (j < D - 1) debugFile << " ";
        }
        debugFile << std::endl;
    }
    
    debugFile.close();
    std::cout << "DEBUG: Entrées après normalisation écrites dans " << filename << std::endl;
}



// Implémentation dans le fichier .C
template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::writeVelocityDebug(const volVectorField& U) const {
    std::string filename = "debug_velocity_t" + std::to_string(this->runTime_.timeOutputValue()) + ".txt";
    
    std::ofstream debugFile(filename);
    
    if (!debugFile.is_open()) {
        std::cout << "ERROR: Cannot open debug file " << filename << std::endl;
        return;
    }
    
    const Field<vector>& UInternal = U.internalField();
    
    debugFile << "Temps: " << this->runTime_.timeOutputValue() << std::endl;
    debugFile << "Cellules: " << UInternal.size() << std::endl;
    debugFile << "Format: Cell | U_x U_y U_z |U| angle_xy" << std::endl;
    debugFile << "=========================================" << std::endl;
    
    // Échantillonner ~200 cellules pour éviter des fichiers trop volumineux
    label step = max(1, UInternal.size() / 100);
    
    for (label cellI = 0; cellI < UInternal.size(); cellI += step) {
        const vector& velocity = UInternal[cellI];
        scalar magnitude = mag(velocity);
        scalar angle_xy = atan2(velocity.y(), velocity.x()) * 180.0 / M_PI; // angle dans le plan xy en degrés
        
        debugFile << "Cell " << cellI << ": "
                  << velocity.x() << " " 
                  << velocity.y() << " " 
                  << velocity.z() << " "
                  << magnitude << " " 
                  << angle_xy << std::endl;
        
    }

}


template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::writeModelOutputCompactBeforeDenormalization(const double* buffer, int64_t N) const {
    std::string filename = "debug_output_before_norm" + std::to_string(this->runTime_.timeOutputValue()) + ".txt";
    std::ofstream debugFile(filename);

    if (!debugFile.is_open()) {
        std::cout << "ERROR: Cannot open debug file " << filename << std::endl;
        return;
    }

    debugFile << "Temps: " << this->runTime_.timeOutputValue() << std::endl;
    debugFile << "Cellules (compact, sorties normalisées): " << N << std::endl;

    int64_t step = std::max<int64_t>(1, N / 100);

    for (int64_t i = 0; i < N; i += step) {
        const double* t = &buffer[i * 6];
        double trace = t[0] + t[3] + t[5];
        debugFile << "Cell " << i << ", xx=" << t[0]
                  << ", xy=" << t[1]
                  << ", xz=" << t[2]
                  << ", yy=" << t[3]
                  << ", yz=" << t[4]
                  << ", zz=" << t[5] << std::endl;
    }

    debugFile.close();
    std::cout << "DEBUG: Sortie modèle normalisée (compacte) écrite dans " << filename << std::endl;
}


template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::writeModelOutputCompactAfterDenormalization(const double* buffer, int64_t N) const {
    std::string filename = "debug_output_compact_after_norm" + std::to_string(this->runTime_.timeOutputValue()) + ".txt";
    std::ofstream debugFile(filename);

    if (!debugFile.is_open()) {
        std::cout << "ERROR: Cannot open debug file " << filename << std::endl;
        return;
    }

    debugFile << "Temps: " << this->runTime_.timeOutputValue() << std::endl;
    debugFile << "Cellules (compact, sorties dénormalisées): " << N << std::endl;

    int64_t step = std::max<int64_t>(1, N / 100);

    for (int64_t i = 0; i < N; i += step) {
        const double* t = &buffer[i * 6];
        double trace = t[0] + t[3] + t[5];
        debugFile << "Cell " << i << ", xx=" << t[0]
                  << ", xy=" << t[1]
                  << ", xz=" << t[2]
                  << ", yy=" << t[3]
                  << ", yz=" << t[4]
                  << ", zz=" << t[5] << std::endl;
    }

    debugFile.close();
    std::cout << "DEBUG: Sortie modèle dénormalisée (compacte) écrite dans " << filename << std::endl;
}





// template <class BasicTurbulenceModel>
// Foam::tmp<Foam::volSymmTensorField> Foam::eddyViscosity<BasicTurbulenceModel>::R() const {
//   loadScalers();

//   // Récupération des données - renommé pour éviter le conflit
//   auto loadResult = loadData();
//   double* buffer = std::get<0>(loadResult);
//   std::vector<int64_t> dims = std::get<1>(loadResult);
//   int64_t N = std::get<2>(loadResult);

// //   writeInputBeforeNormalization(buffer, dims);
//   normalizeInput(buffer, dims);
// //   writeInputAfterNormalization(buffer, dims);

//   double *_MlRBuffer = runInference(buffer, dims);

//   // Debug: Écrire les sorties du modèle (avant dénormalisation)
// //   writeModelOutputCompactBeforeDenormalization(_MlRBuffer, N);
//   denormalizeOutput(_MlRBuffer, N);
// //   writeModelOutputCompactAfterDenormalization(_MlRBuffer, N);
  
//   tmp<Field<symmTensor>> tMlRInternalField = convert2field(_MlRBuffer, N);

//   // Création du champ résultat - renommé pour éviter le conflit
//   auto resultField = volSymmTensorField::New(IOobject::groupName("R", this->alphaRhoPhi_.group()),
//                                             IOobject::NO_REGISTER, this->mesh_, dimensionSet(0, 2, -2, 0, 0),
//                                             tMlRInternalField(), fvPatchFieldBase::calculatedType());

//   return resultField;
// }




// 1. Correction de la fonction debugParallelInfo
template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::debugParallelInfo() const {
    int rank = Pstream::myProcNo();
    int nProcs = Pstream::nProcs();
    
    std::cout << "PROC[" << rank << "/" << nProcs << "] - Mesh info:" << std::endl;
    std::cout << "  - Local cells: " << this->mesh_.nCells() << std::endl;
    std::cout << "  - Total cells: " << returnReduce(this->mesh_.nCells(), sumOp<label>()) << std::endl;
    std::cout << "  - Boundary patches: " << this->mesh_.boundary().size() << std::endl;
    
    // Vérifier les communications inter-processeurs - CORRECTION ICI
    forAll(this->mesh_.boundary(), patchI) {
        const fvPatch& patch = this->mesh_.boundary()[patchI];
        if (isType<processorFvPatch>(patch)) {  // Changé de isA à isType
            const processorFvPatch& procPatch = refCast<const processorFvPatch>(patch);
            std::cout << "  - Processor patch " << patchI 
                      << " connects to proc " << procPatch.neighbProcNo()
                      << " with " << patch.size() << " faces" << std::endl;
        }
    }
}


template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::writeParallelDebugData(const double* buffer, 
                                                                       const std::vector<int64_t>& dims) const {
    int rank = Pstream::myProcNo();
    const int64_t N = dims[0];
    const int64_t D = dims[1];
    
    std::string filename = "debug_parallel_proc" + std::to_string(rank) + 
                          "_t" + std::to_string(this->runTime_.timeOutputValue()) + ".txt";
    
    std::ofstream debugFile(filename);
    
    if (!debugFile.is_open()) {
        std::cout << "ERROR: Cannot open debug file " << filename << std::endl;
        return;
    }
    
    debugFile << "Processeur: " << rank << "/" << Pstream::nProcs() << std::endl;
    debugFile << "Temps: " << this->runTime_.timeOutputValue() << std::endl;
    debugFile << "Points locaux: " << N << std::endl;
    debugFile << "Dimensions: " << D << std::endl;
    
    // Écrire quelques échantillons
    label step = max(1, N / 20);  // Moins d'échantillons pour éviter des fichiers trop gros
    
    for (int64_t i = 0; i < N; i += step) {
        debugFile << "LocalCell " << i << ": ";
        for (int64_t j = 0; j < D; ++j) {
            int64_t idx = i * D + j;
            debugFile << buffer[idx];
            if (j < D - 1) debugFile << " ";
        }
        debugFile << std::endl;
    }
    
    debugFile.close();
    std::cout << "PROC[" << rank << "] Debug data written to " << filename << std::endl;
}

// 2. Correction de la fonction compareResultsAcrossProcessors 
template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::compareResultsAcrossProcessors(const double* buffer, int64_t N) const {
    int rank = Pstream::myProcNo();
    // int nProcs = Pstream::nProcs();  // SUPPRIMÉ car non utilisé
    
    if (N == 0) return;
    
    // Calculer quelques statistiques locales sur les résultats
    std::vector<scalar> localStats(18, 0.0); // 6 composantes * 3 stats (min, max, avg)
    
    for (int comp = 0; comp < 6; ++comp) {
        scalar minVal = GREAT;
        scalar maxVal = -GREAT;
        scalar sumVal = 0.0;
        
        for (int64_t i = 0; i < N; ++i) {
            scalar val = buffer[i * 6 + comp];
            minVal = min(minVal, val);
            maxVal = max(maxVal, val);
            sumVal += val;
        }
        
        localStats[comp * 3 + 0] = minVal;     // min
        localStats[comp * 3 + 1] = maxVal;     // max
        localStats[comp * 3 + 2] = sumVal / N; // avg
    }
    
    // Afficher les statistiques locales
    std::cout << "PROC[" << rank << "] ML Results stats:" << std::endl;
    for (int comp = 0; comp < 6; ++comp) {
        std::cout << "  Comp[" << comp << "] - Min: " << localStats[comp * 3 + 0]
                  << ", Max: " << localStats[comp * 3 + 1]
                  << ", Avg: " << localStats[comp * 3 + 2] << std::endl;
    }
}


// 4. Correction des fonctions broadcast - FONCTION CORRIGÉE
template <class BasicTurbulenceModel>
void Foam::eddyViscosity<BasicTurbulenceModel>::broadcastScalers() const {
    // Convertir std::vector en List pour OpenFOAM
    if (Pstream::master()) {
        // Le maître diffuse les scalers
        List<double> scalerXList(scaler_X_max_abs_.size());
        List<double> scalerYList(scaler_y_max_abs_.size());
        
        forAll(scalerXList, i) {
            scalerXList[i] = scaler_X_max_abs_[i];
        }
        forAll(scalerYList, i) {
            scalerYList[i] = scaler_y_max_abs_[i];
        }
        
        Pstream::scatter(scalerXList);
        Pstream::scatter(scalerYList);
    } else {
        // Les esclaves reçoivent
        List<double> scalerXList;
        List<double> scalerYList;
        
        Pstream::scatter(scalerXList);
        Pstream::scatter(scalerYList);
        
        // Convertir back en std::vector
        scaler_X_max_abs_.resize(scalerXList.size());
        scaler_y_max_abs_.resize(scalerYList.size());
        
        forAll(scalerXList, i) {
            scaler_X_max_abs_[i] = scalerXList[i];
        }
        forAll(scalerYList, i) {
            scaler_y_max_abs_[i] = scalerYList[i];
        }
    }
}

// 5. FONCTION R() CORRIGÉE POUR LE PARALLÈLE
template <class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField> Foam::eddyViscosity<BasicTurbulenceModel>::R() const {
    // DEBUG: Afficher les informations de parallélisation
    debugParallelInfo();
    
    // Charger les scalers seulement sur le processeur maître, puis diffuser
    if (Pstream::master()) {
        loadScalers();
    }
    
    // Diffuser les scalers à tous les processeurs
    broadcastScalers();
    scalers_loaded_ = true;
    
    // Chaque processeur fait l'inférence sur ses données locales
    auto loadResult = loadData();
    double* buffer = std::get<0>(loadResult);
    std::vector<int64_t> dims = std::get<1>(loadResult);
    int64_t N = std::get<2>(loadResult);
    
    // DEBUG: Écrire les données de chaque processeur
    // writeParallelDebugData(buffer, dims);
    
    normalizeInput(buffer, dims);
    double* mlResults = runInference(buffer, dims);
    denormalizeOutput(mlResults, N);
    
    // DEBUG: Comparer les résultats entre processeurs
    // compareResultsAcrossProcessors(mlResults, N);
    
    tmp<Field<symmTensor>> tMlRInternalField = convert2field(mlResults, N);
    
    auto resultField = volSymmTensorField::New(
        IOobject::groupName("R", this->alphaRhoPhi_.group()),
        IOobject::NO_REGISTER, 
        this->mesh_, 
        dimensionSet(0, 2, -2, 0, 0),
        tMlRInternalField(), 
        fvPatchFieldBase::calculatedType()
    );
    
    // Important: Synchroniser les conditions aux limites
    resultField.ref().correctBoundaryConditions();
    
    return resultField;
}

// template <class BasicTurbulenceModel>
// Foam::tmp<Foam::volSymmTensorField> Foam::eddyViscosity<BasicTurbulenceModel>::R() const {
//   loadScalers();

//   // Récupération des données - renommé pour éviter le conflit
//   auto loadResult = loadData();
//   double* buffer = std::get<0>(loadResult);
//   std::vector<int64_t> dims = std::get<1>(loadResult);
//   int64_t N = std::get<2>(loadResult);

// //   writeInputBeforeNormalization(buffer, dims);
//   normalizeInput(buffer, dims);
// //   writeInputAfterNormalization(buffer, dims);

//   double *_MlRBuffer = runInference(buffer, dims);

//   // Debug: Écrire les sorties du modèle (avant dénormalisation)
// //   writeModelOutputCompactBeforeDenormalization(_MlRBuffer, N);
//   denormalizeOutput(_MlRBuffer, N);
// //   writeModelOutputCompactAfterDenormalization(_MlRBuffer, N);
  
//   tmp<Field<symmTensor>> tMlRInternalField = convert2field(_MlRBuffer, N);

//   // Création du champ résultat - renommé pour éviter le conflit
//   auto resultField = volSymmTensorField::New(IOobject::groupName("R", this->alphaRhoPhi_.group()),
//                                             IOobject::NO_REGISTER, this->mesh_, dimensionSet(0, 2, -2, 0, 0),
//                                             tMlRInternalField(), fvPatchFieldBase::calculatedType());

//   return resultField;
// }

// ************************************************************************* //
