// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FPLS_BASE_H__
#define __FPLS_BASE_H__

#include <fdaPDE/utils.h>
#include <Eigen/SVD>

#include "../../calibration/calibration_base.h"
#include "../../calibration/off.h"
#include "../../calibration/gcv.h"
using fdapde::calibration::Calibrator;
#include "functional_base.h"
#include "center.h"

// regularized_svd
#include "regularized_svd.h"
using fdapde::models::RSVDType;

namespace fdapde {
namespace models {

// FPLS (Functional Partial Least Square regression) model signature
template <typename RegularizationType_, typename FPLS_MODEL>
class FPLS_BASE : public FunctionalBase<FPLS_BASE<RegularizationType_, FPLS_MODEL>, RegularizationType_> {
   public:
    using RegularizationType = std::decay_t<RegularizationType_>;
    using This = FPLS_BASE<RegularizationType, FPLS_MODEL>;
    using Base = FunctionalBase<This, RegularizationType>;
    using SmootherType = std::conditional_t<is_space_only<This>::value, SRPDE, STRPDE<RegularizationType, monolithic>>;
    IMPORT_MODEL_SYMBOLS;
    using Base::df_;
    using Base::n_basis;
    using Base::n_obs;
    using Base::n_stat_units;
    using Base::X;   // n_stat_units \times n_locs data matrix

    // constructors
    FPLS_BASE() = default;
    fdapde_enable_constructor_if(is_space_only, This)
      FPLS_BASE(const pde_ptr& pde, Sampling s, RegularizedSVD<sequential> rsvd = RegularizedSVD<fdapde::sequential>{}) :
        Base(pde, s), rsvd_(rsvd) {};
    fdapde_enable_constructor_if(is_space_time_separable, This)
      FPLS_BASE(const pde_ptr& space_penalty, const pde_ptr& time_penalty, Sampling s,
           RegularizedSVD<sequential> rsvd = RegularizedSVD<fdapde::sequential>{}) :
        Base(space_penalty, time_penalty, s), rsvd_(rsvd) {};

    void init_model() {
        // initialize smoothing solver for regression step
        if constexpr (is_space_only<SmootherType>::value) { smoother_ = SmootherType(Base::pde(), Base::sampling()); }
	      else {
            smoother_ = SmootherType(Base::pde(), Base::time_pde(), Base::sampling());
            smoother_.set_temporal_locations(Base::time_locs());
        }
        smoother_.set_spatial_locations(Base::locs());
        if (!calibrator_) {   // smoothing solver's calibration strategy fallback
            if (rsvd_.calibration() == Calibration::off) {
                calibrator_ = calibration::Off {}(Base::lambda());
            } else {
                calibrator_ = calibration::GCV {core::Grid<Dynamic> {}, StochasticEDF(100)}(rsvd_.lambda_grid());
	          }
        }
        return;
    }
    void solve() {
        // allocate space
        X_space_directions_.resize(n_basis(), n_comp_);        // optimal direction in X space
        X_loadings_.resize(n_basis(), n_comp_);                // optimal X loadings
        Y_space_directions_.resize(Y().cols(), n_comp_);       // optimal direction in Y space
        Y_loadings_.resize(Y().cols(), n_comp_);               // optimal Y loadings
        X_latent_scores_.resize(n_stat_units(), n_comp_);      // X latent scores
        Y_latent_scores_.resize(n_stat_units(), n_comp_);      // Y latent scores

        // copy original data to avoid side effects
        DMatrix<double> X_h = X(), Y_h = Y();

        for (std::size_t h = 0; h < n_comp_; ++h) {
            // directions estimation step:
            model().directions_estimation(X_h, Y_h, h, rsvd_);

            // regression step:
            model().regression(X_h, Y_h, h, smoother_, calibrator_);

            // deflation
            model().deflation(X_h, Y_h, h);
        }

        return;
    }

    FPLS_MODEL & model() {
        return static_cast<FPLS_MODEL &> (*this);
    }

    // getters
    const std::size_t n_comp() const { return n_comp_; }
    const DMatrix<double>& Y() const { return df_.template get<double>(OBSERVATIONS_BLK); }
    const DMatrix<double>& X() const { return df_.template get<double>(DESIGN_MATRIX_BLK); }
    const DMatrix<double>& X_space_directions() const { return X_space_directions_; }
    const DMatrix<double>& Y_space_directions() const { return Y_space_directions_; }
    const DMatrix<double>& X_latent_scores() const { return X_latent_scores_; }
    const DMatrix<double>& Y_latent_scores() const { return Y_latent_scores_; }
    const DMatrix<double>& X_loadings() const { return X_loadings_; }
    const DMatrix<double>& Y_loadings() const { return Y_loadings_; }
    // setters
    void set_ncomp(std::size_t n_comp) { n_comp_ = n_comp; }
    void set_rsvd(const RSVDType<This>& rsvd) { rsvd_ = rsvd; }
    template <typename CalibratorType_> void set_regression_step_calibrator(CalibratorType_&& calibrator) {
        calibrator_ = calibrator;
    }
   protected:
    SmootherType smoother_;                 // smoothing algorithm used in regression step
    Calibrator<SmootherType> calibrator_;   // calibration strategy used in regression step
    RSVDType<This> rsvd_;                   // RSVD solver employed in correlation maximization step
    std::size_t n_comp_ = 3;                // number of latent components

    // problem solution
    DMatrix<double> X_space_directions_;   // optimal directions in X space
    DMatrix<double> Y_space_directions_;   // optimal directions in Y space
    DMatrix<double> X_latent_scores_;   // X latent scores
    DMatrix<double> Y_latent_scores_;   // Y latent scores
    DMatrix<double> X_loadings_;   // optimal X loadings
    DMatrix<double> Y_loadings_;   // optimal Y loadings
    // DMatrix<double> B_;
};

}   // namespace models
}   // namespace fdapde

#endif   // __FPLS_BASE_H__
