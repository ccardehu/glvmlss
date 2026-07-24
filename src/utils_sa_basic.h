#ifndef UTILS_SA_BASIC_H
#define UTILS_SA_BASIC_H

arma::mat make_ZX(arma::mat& Z, arma::mat& X, const arma::mat fmat);

arma::vec digamma_vec(arma::vec& x);

arma::vec trigamma_vec(arma::vec& x);

arma::vec d1mu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d1sg_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d1nu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d2mu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d2sg_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d2nu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec dcms_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec dcmn_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec dcsn_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d2muF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d2sgF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec d2nuF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec dcmsF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec dcmnF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::vec dcsnF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu);

arma::mat rMVnorm(const int n, arma::vec& mu, arma::mat& R);

arma::mat rMVStnorm(const int n, const int q);

double dMVnorm(arma::vec& y, arma::vec& mu, arma::mat& R, const bool log);

arma::mat dMVStnorm(arma::mat& Z);

arma::cube make_etas(arma::mat& ZX, arma::cube& b);

void cube2eye(arma::cube& C);

arma::mat ProxL(arma::mat& L, arma::mat& rR);

#endif
