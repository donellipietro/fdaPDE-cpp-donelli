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

#ifndef __CENTER_H__
#define __CENTER_H__

#include <fdaPDE/utils.h>

namespace fdapde {
namespace models {

// computes the smooth weighted mean field from a set of functional data (stored rowwise in a data matrix X)
// solves \argmin_{f} \| X*\frac{w}{\norm{w}_2^2} - f \|_2^2 + P_{\lambda}(f) (using a linear smoother)
template <typename SmootherType_, typename CalibratorType_>
std::pair<DVector<double>, DMatrix<double>> smooth_mean(
  const DMatrix<double>& X, const DVector<double>& w, SmootherType_&& smoother, CalibratorType_&& calibrator) {
    fdapde_assert(X.rows() == w.rows());
    BlockFrame<double, int> df;
    // let O_{p_i} the set of index where x_j is observed at location p_i, compute smoother data {y_i}_i
    // y_i = \sum_{j \in O_{p_i}} x_j(p_i)*w_j / \sum_{j \in O_{p_i}} w_j
    DMatrix<double> X_ = X.array().isNaN().select(0, X).transpose() * w;
    for (std::size_t i = 0; i < X.cols(); ++i) { X_(i, 0) /= X.col(i).array().isNaN().select(0, w).squaredNorm(); }
    df.insert<double>(OBSERVATIONS_BLK, X_);
    smoother.set_data(df);
    DVector<double> lambda_opt{calibrator.fit(smoother)};
    smoother.set_lambda(calibrator.fit(smoother));   // find optimal smoothing parameter
    smoother.init();
    smoother.solve();
    return {lambda_opt, smoother.f()};
}

// computes the smooth mean field from a set of functional data
template <typename SmootherType_, typename CalibratorType_>
std::pair<DVector<double>, DMatrix<double>> smooth_mean(const DMatrix<double>& X, SmootherType_&& smoother, CalibratorType_&& calibrator) {
    return smooth_mean(X, DVector<double>::Ones(X.rows()), smoother, calibrator);
}

// functional centering of a data matrix X
struct CenterReturnType {
    DVector<double> lambda_opt;
    DMatrix<double> fitted;   // centred data, X - \mu
    DMatrix<double> mean;     // mean field expansion coefficients
};
template <typename SmootherType_, typename CalibratorType_>
CenterReturnType
center(const DMatrix<double>& X, const DVector<double>& w, SmootherType_&& smoother, CalibratorType_&& calibrator) {
    auto [lambda_opt, mean_field] = smooth_mean(X, w, smoother, calibrator);
    // compute mean matrix and return
    return {lambda_opt, X - smoother.fitted().replicate(1, X.rows()).transpose(), smoother.f()};
}
template <typename SmootherType_, typename CalibratorType_>
CenterReturnType center(const DMatrix<double>& X, SmootherType_&& smoother, CalibratorType_&& calibrator) {
    return center(X, DVector<double>::Ones(X.rows()), smoother, calibrator);
}

}   // namespace models
}   // namespace fdapde

#endif   // __CENTER_H__
