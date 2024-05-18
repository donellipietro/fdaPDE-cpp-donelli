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

#ifndef __FPLS_R_H__
#define __FPLS_R_H__

#include "fpls_base.h"

namespace fdapde {
namespace models {

// FPLS (Functional Partial Least Square regression) model signature
template <typename RegularizationType_>
class FPLS_R : public FPLS_BASE<RegularizationType_, FPLS_R<RegularizationType_>> {
   public:
    using RegularizationType = std::decay_t<RegularizationType_>;
    using Model = FPLS_R<RegularizationType>;
    using ModelBase = FPLS_BASE<RegularizationType, Model>;
    using SmootherType = std::conditional_t<is_space_only<ModelBase>::value, SRPDE, STRPDE<RegularizationType, monolithic>>;
    using ModelBase::X_space_directions_;
    using ModelBase::Y_space_directions_;
    using ModelBase::X_loadings_;
    using ModelBase::X_loadings;
    using ModelBase::Y_loadings_;
    using ModelBase::Y_loadings;
    using ModelBase::X_latent_scores_;
    using ModelBase::X_latent_scores;
    using ModelBase::Y_latent_scores_;
    using ModelBase::Y_latent_scores;
    using ModelBase::n_comp;
    using ModelBase::n_basis;
    using ModelBase::Psi;

    // constructors
    FPLS_R() = default;
    fdapde_enable_constructor_if(is_space_only, ModelBase)
      FPLS_R(const pde_ptr& pde, Sampling s, RegularizedSVD<sequential> rsvd = RegularizedSVD<fdapde::sequential>{}) :
        ModelBase(pde, s, rsvd) {};
    fdapde_enable_constructor_if(is_space_time_separable, ModelBase)
      FPLS_R(const pde_ptr& space_penalty, const pde_ptr& time_penalty, Sampling s,
           RegularizedSVD<sequential> rsvd = RegularizedSVD<fdapde::sequential>{}) :
        ModelBase(space_penalty, time_penalty, s, rsvd) {};

    void directions_estimation(DMatrix<double>& X_h, DMatrix<double> & Y_h, std::size_t h, RSVDType<ModelBase> rsvd_) {
        // correlation maximization
        // solves \argmin_{v,w} \norm_F{Y_h^\top*X_h - v^\top*w}^2 + (v^\top*v)*P_{\lambda}(w)
        rsvd_.compute(Y_h.transpose() * X_h, model_base(), 1);
        X_space_directions_.col(h) = rsvd_.loadings();
        Y_space_directions_.col(h) = rsvd_.scores() / rsvd_.loadings_norm()[0];
        return;
    }
    void regression(DMatrix<double>& X_h, DMatrix<double> & Y_h, std::size_t h, SmootherType smoother_, Calibrator<SmootherType> calibrator_) {
        // X regression: solves \argmin_{r} \norm_F{X_h - t*r^\top}^2 + P_{\lambda}(r)
        X_loadings_.col(h) = smooth_mean(X_h, X_latent_scores_.col(h), smoother_, calibrator_).second;
        // Y regression: solves \argmin_{c} \norm_F{Y_h - t*c^\top}^2
        Y_loadings_.col(h) = Y_h.transpose() * X_latent_scores_.col(h) / X_latent_scores_.col(h).squaredNorm();
        return;
    }
    void deflation(DMatrix<double>& X_h, DMatrix<double> & Y_h, std::size_t h) {
        X_h -= X_latent_scores_.col(h) * (Psi() * X_loadings_.col(h)).transpose();
        Y_h -= X_latent_scores_.col(h) * Y_loadings_.col(h).transpose();
        return;
    }

    // getters
    DMatrix<double> fitted(std::size_t h = 0) const {
        if(h == 0) h = n_comp();
        const DMatrix<double> & T_h = X_latent_scores_.leftCols(h);
        const DMatrix<double> & C_h = Y_loadings_.leftCols(h);
        return T_h * C_h.transpose();
    }
    DMatrix<double> reconstructed(std::size_t h = 0) const {
        if(h == 0) h = n_comp();
        const DMatrix<double> & T_h = X_latent_scores_.leftCols(h);
        const DMatrix<double> & R_h = X_loadings_.leftCols(h);
        return T_h * R_h.transpose();
    }
    const DMatrix<double> Beta(std::size_t h = 0) const { 
        if(h == 0) h = n_comp();
        const DMatrix<double> & W_h = X_space_directions_.leftCols(h);
        const DMatrix<double> & R_h = X_loadings_.leftCols(h);
        const DMatrix<double> & C_h = Y_loadings_.leftCols(h);
        return W_h * (R_h.transpose() * Psi().transpose() * Psi() * W_h).partialPivLu().solve(C_h.transpose());
    }
    ModelBase & model_base() {
        return static_cast<ModelBase &> (*this);
    }
};

}   // namespace models
}   // namespace fdapde

#endif   // __FPLS_R_H__
