#ifndef UTILS_SA_ESTIM_H
#define UTILS_SA_ESTIM_H

arma::vec dB(Dist_sa& Ydist,
             arma::cube& etas,
             arma::cube& rb,
             const bool flagEIM,
             double& ssB);

arma::mat newL(arma::vec& mu, arma::mat& L, arma::mat& rR, arma::mat& Z, double& ss);

void newpMR(arma::mat& Z, arma::mat& posM, arma::cube& posR, int iter);

arma::mat fyz(Dist_sa& Ydist, arma::cube& etas);

arma::mat fz(arma::mat& Z, arma::vec& mu, arma::mat& R);

arma::mat newZ_RWMH(Dist_sa& Ydist, arma::cube& etas, arma::mat& Z, arma::mat& X,
                    arma::cube& b, arma::vec& mu, arma::mat& L, double& ar,
                    const double h, const arma::mat fmat);

#endif
