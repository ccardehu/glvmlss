#include <RcppArmadillo.h>
#include "utils_basic.h"
#include "utils_sa_basic.h"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

using namespace Rcpp;
using namespace arma;
using namespace boost::math;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]

// Create ZX
arma::mat make_ZX(arma::mat& Z, arma::mat& X, const arma::mat fmat){
  const int n = Z.n_rows;
  arma::vec inter(n, arma::fill::ones);
  arma::mat ZX = arma::join_rows(Z, X);
  ZX = ZX * fmat;
  ZX = arma::join_rows(inter, ZX);
  return ZX;
}

// Vectorized digamma function from boost
arma::vec digamma_vec(arma::vec& x){
  const int n = x.n_elem ;
  arma::vec out(n);
  for(int i = 0; i < n; i++){
    double xt = x(i);
    if(std::isnan(xt)){
      out(i) = arma::datum::nan;
    } else {
      out(i) = boost::math::digamma(xt);
    }
  }
  return(out);
}

// Vectorized trigamma function from boost
arma::vec trigamma_vec(arma::vec& x){
  const int n = x.n_elem ;
  arma::vec out(n);
  for(int i = 0; i < n; i++){
    double xt = x(i);
    if(std::isnan(xt)){
      out(i) = arma::datum::nan;
    } else {
      out(i) = boost::math::trigamma(xt);
    }
  }
  return(out);
}

// First derivative of SN wrt mu (used in SA)
arma::vec d1mu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = z/sigma - (nu/sigma)%zeta1(w);
  return out ;
}

// First derivative of SN wrt sigma (used in SA)
arma::vec d1sg_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = (-zeta1(w)%w)/sigma + (arma::square(z)-1.0)/sigma;
  return out ;
}

// First derivative of SN wrt nu (used in SA)
arma::vec d1nu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = zeta1(w) % z;
  return out ;
}

// Second derivative of SN wrt mu (observed)
arma::vec d2mu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = -1/arma::square(sigma) + arma::square(nu/sigma) % zeta2(w);
  return out ;
}

// Second derivative of SN wrt sigma
arma::vec d2sg_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = 1/arma::square(sigma) % (1 - 3*arma::square(z) + 2* w % zeta1(w) + arma::square(w)%zeta2(w)) ;
  return out ;
}

// Second derivative of SN wrt nu
arma::vec d2nu_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = zeta2(w) % arma::square(z) ;
  return out ;
}

// Cross-derivative of SN wrt mu-sigma
arma::vec dcms_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = 1/arma::square(sigma) % (-2*z + nu % zeta1(w) + arma::square(nu) % zeta2(w) % z) ;
  return out ;
}

// Cross-derivative of SN wrt mu-nu
arma::vec dcmn_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = -1/sigma % zeta1(w) - w%zeta2(w) % (1/sigma) ;
  return out ;
}

// Cross-derivative of SN wrt sigma-nu
arma::vec dcsn_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  arma::vec out = (-zeta1(w)%z)/sigma - (zeta2(w)%w%z)/sigma ;
  return out ;
}

// Second derivative of SN wrt mu (expected)
arma::vec d2muF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  double i1 = arma::as_scalar(arma::mean(arma::square(zeta1(w))));
  arma::vec out = -1/arma::square(sigma) - arma::square(nu/sigma) * i1 ;
  return out ;
}

// Second derivative of SN wrt sigma (expected)
arma::vec d2sgF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma ;
  arma::vec w = nu % z;
  double i1 = arma::as_scalar(arma::mean(arma::square(z % zeta1(w)))) ;
  arma::vec out = 1/arma::square(sigma) % (-2 - arma::square(nu)*i1) ;
  return out ;
}

// Second derivative of SN wrt nu (expected)
arma::vec d2nuF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma ;
  arma::vec w = nu % z;
  double i1 = arma::as_scalar(arma::mean(arma::square(z % zeta1(w))));
  arma::vec out(arma::size(w));
  out.fill(-i1);
  return out ;
}

// Cross-derivative of SN wrt mu-sigma (expected)
arma::vec dcmsF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  double i1 = arma::as_scalar(arma::mean(z % arma::square(zeta1(w))));
  arma::vec out = 1/arma::square(sigma) % (-log2pi*nu % (1 + 2*arma::square(nu))/arma::pow(1+arma::square(nu),1.5) -
    arma::square(nu)*i1) ;
  return out ;
}

// Cross-derivative of SN wrt mu-nu (expected)
arma::vec dcmnF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  double i1 = arma::as_scalar(arma::mean(z % arma::square(zeta1(w))));
  arma::vec out = 1/sigma % (-log2pi/arma::pow(1+arma::square(nu),1.5) + nu*i1) ;
  return out ;
}

// Cross-derivative of SN wrt sigma-nu (expected)
arma::vec dcsnF_vec(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  arma::vec z = (y - mu)/sigma;
  arma::vec w = nu % z;
  double i1 = arma::as_scalar(arma::mean(arma::square(z % zeta1(w))));
  arma::vec out = nu/sigma *i1; // Equation (A1.4) in dissertation missing sigma
  return out ;
}

// Draw from Multivariate Standard Normal distribution
arma::mat rMVStnorm(const int n, const int q){
  arma::mat Y(n,q) ;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < q; j++){
      Y(i,j) = R::rnorm(0,1);
    }
  }
  return (Y);
}

// Density of Multivariate Normal distribution
double dMVnorm(arma::vec& y, arma::vec& mu, arma::mat& R, const bool log = true){
  const int q = y.size() ;
  double lR = arma::log_det_sympd(R);
  double out = -0.5*(q*std::log(2*M_PI) + lR + arma::as_scalar((y - mu).t()*(arma::solve(R, y - mu,arma::solve_opts::likely_sympd)))) ;
  if(log) return(out);
  else return(std::exp(out));
}

// Density of Multivariate Standard Normal distribution (matrix output, entry-wise function)
arma::mat dMVStnorm(arma::mat& Z){
  const int n = Z.n_rows;
  const int q = Z.n_cols;
  arma::mat out(n,q) ;
  for(int j = 0; j < q; j++){
    for(int m = 0; m < n; m++){
      out(m,j) = R::dnorm(Z(m,j),0.0,1.0,0);
    }
  }
  return (out);
}

// Cube (n x p x d) for linear combinations eta_d = ZX %*% b
arma::cube make_etas(arma::mat& ZX, arma::cube& b){
  const int n = ZX.n_rows;
  const int p = b.n_rows;
  const int d = b.n_slices;
  arma::cube out(n, p, d);
  for(int k = 0; k < d; k++){
    arma::mat tmp(n,p);
    for(int j = 0; j < p; j++){
      arma::vec btheta = b.slice(k).row(j).t() ;
      tmp.col(j) = ZX * btheta ;
    }
    out.slice(k) = tmp;
  }
  return(out);
}

// Fixes posterior R holders starting as diag(q)
void cube2eye(arma::cube& C){
  const int q = C.n_rows;
  C.fill(0.0);
  C.each_slice() += arma::eye(q,q);
}

arma::mat ProxL(arma::mat& L, arma::mat& rR){
  int q = L.n_rows;
  arma::mat Lout(q,q);
  arma::vec dR = arma::diagvec(rR) ;
  for(int i = 0; i < q; i++){
    if(dR(i) != 1) continue ;
    double cross = arma::norm(L.row(i),2L);
    if(cross > 0){
      Lout.row(i) = L.row(i)/cross;
    } else {
      Rcpp::stop("L has norm zero");
    }
  }
  return(Lout);
}
