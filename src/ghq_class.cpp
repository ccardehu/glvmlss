#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
#include "utils_basic.h"
#include "ghq_class.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEnsmallen)]]

using namespace Rcpp;
using namespace arma;
using namespace ens;

// Computes univariate ghQ (inspect D. Rizopoulos ltm package)
Rcpp::List ghQ::ghq(const int &n){
  int m = trunc((n + 1) / 2) ;
  Rcpp::NumericVector x(n,-1.0), w(n,-1.0) ;
  double z = 0.0, p1, p2, p3 , z1 , pp ;
  for(int i = 0; i < m; i++){
    if(i == 0){
      z = std::sqrt(2.0 * n + 1.0) - 1.85575 * std::pow(2.0 * n + 1.0, -0.16667) ;
    } else if(i == 1){
      z = z - 1.14 * std::pow(n, 0.426) / z ;
    } else if(i == 2){
      z = 1.86 * z - 0.86 * x[0] ;
    } else if(i == 3){
      z = 1.91 * z - 0.91 * x[1] ;
    } else {
      z = 2.0 * z - x[i - 2] ;
    }
    for(int j = 0; j < 10; j++){
      p1 = 0.751125544464943 ; // pi^{-1/4}, the normalization for H_0
      p2 = 0.0 ;
      for(int ii = 1; ii <= n; ii++){
        p3 = p2 ;
        p2 = p1 ;
        p1 = z * std::sqrt(2.0 / ii) * p2 - std::sqrt((ii - 1.0) / ii) * p3 ;
      }
      pp = std::sqrt(2.0 * n) * p2 ;
      z1 = z ;
      z = z1 - p1/pp ;
      if (std::fabs(z - z1) <= 3e-14){
        break ;
      }
    }
    x[i] = z ;
    x[n - 1 - i] = z * -1.0 ; // symmetry: nodes come in +/- pairs
    w[i] = 2 / std::pow(pp,2.0) ;
    w[n - 1 - i] = w[i] ;
  }
  return Rcpp::List::create(Rcpp::Named("x") = x,
                            Rcpp::Named("w") = w);
}

// Replicates vecC (times)
Rcpp::List ghQ::newList(const int& times,
                        arma::colvec& vecC){

  typedef std::vector<double> stdvec;
  stdvec vecS = arma::conv_to< stdvec >::from(vecC);
  std::vector< std::vector<double> > v ;

  for(int i = 0; i < times; i++){
    v.push_back(vecS);
  }
  return Rcpp::wrap( v ) ;
}

// Data frame to arma::mat
arma::mat ghQ::df2mat(Rcpp::DataFrame& df){
  int nrows = df.nrows();
  int ncols = df.size();
  arma::mat mat(nrows, ncols);
  for (int i = 0; i < ncols; i++) {
    mat.col(i) = Rcpp::as<arma::colvec>(df[i]);
  }
  return mat;
}

// Builds multivariate ghQ grid, re-weighting against the N(0, I_q) prior,
// and rotating by the Cholesky-like factor of Sigma (via eigen-decomposition)
Rcpp::List ghQ::mvghq(const int& q, const int& nqp){

  Rcpp::Function expGrid("expand.grid");
  Rcpp::List ghQ1d = ghq(nqp);
  arma::colvec ghqP = Rcpp::as<arma::colvec>(ghQ1d["x"]);
  arma::colvec ghqW = Rcpp::as<arma::colvec>(ghQ1d["w"]);

  // Reweight 1D-weights so the product grid integrates against N(0,1)
  // Factor (2 pi)^{-1/2} * exp(x^2/2) converts weights (for integrand * exp(-x^2)) into weights for an N(0,1) prior.
  ghqW = ghqW * std::pow(2*arma::datum::pi, -0.5) % arma::exp(0.5*arma::pow(ghqP,2));

  Rcpp::List ghqPList = newList(q,ghqP);
  Rcpp::List ghqWList = newList(q,ghqW);

  Rcpp::DataFrame ghqP_grid = expGrid(ghqPList, Rcpp::Named("KEEP.OUT.ATTRS", false));
  Rcpp::DataFrame ghqW_grid = expGrid(ghqWList, Rcpp::Named("KEEP.OUT.ATTRS", false));

  arma::mat ghqPm = df2mat(ghqP_grid);
  int ii = ghqPm.n_cols - 1;
  arma::mat ghqWm = arma::cumprod(df2mat(ghqW_grid),1);
  ghqWm = ghqWm.col(ii);
  ghqWm += (1.0 - arma::accu(ghqWm))/ ghqWm.n_elem;

  arma::vec eigval;
  arma::mat eigvec;

  eig_sym(eigval, eigvec, Sigma);

  // Rotate grid from N(0, I_q) coordinates to N(0, Sigma) coordinates via Sigma^{1/2} = eigvec * diag(sqrt(eigval)) * eigvec^T,
  // Only left factor eigvec * diag(sqrt(eigval)) is applied (equivalent to a Cholesky-like square root).
  ghqPm = ((eigvec * arma::diagmat(arma::sqrt(eigval))) * ghqPm.t()).t();

  Rcpp::NumericMatrix ghqPo = Rcpp::wrap(ghqPm);
  Rcpp::CharacterVector nam(q);

  for (int i = 0; i < q; i++) {
    std::string id1 = "Z";
    std::string id2 = std::to_string(i + 1);
    nam[i] = id1.append(id2);
  }

  colnames(ghqPo) = nam;
  Rcpp::NumericVector ghqWo = Rcpp::wrap(ghqWm);

  return Rcpp::List::create(Rcpp::Named("points") = ghqPo,
                            Rcpp::Named("weights") = ghqWo);
}

// Draw n iid samples from N(mu, sigma)
arma::mat ghQ::mvrnorm_sim(const int n, arma::vec& mu, arma::mat& sigma){
  // int ncols = sigma.n_cols;
  // arma::vec eigenval;
  // arma::mat eigenvec;
  // arma::eig_sym(eigenval, eigenvec, sigma);
  // arma::mat Y = arma::randn(n, ncols);
  // return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);

  int q = sigma.n_cols;
  arma::mat Y(n,q), eigm;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < q; j++){
      Y(i,j) = R::rnorm(0,1);
    }
  }
  arma::vec eigv;
  eig_sym(eigv,eigm,sigma);
  arma::mat teigm = eigm.t();
  teigm.each_col() %= arma::sqrt(eigv);
  return arma::repmat(mu, 1, n).t() + Y * (eigm * teigm);
}

// An n x (q+1) matrix of latent variables + intercept
Rcpp::NumericMatrix ghQ::get_simZ(const int& q, const int& n){

  arma::vec mu(q);
  arma::mat res = mvrnorm_sim(n, mu, Sigma);
  arma::vec inter(n, arma::fill::ones);
  Rcpp::NumericMatrix out = Rcpp::wrap(join_rows(inter, res));

  Rcpp::CharacterVector nam(q+1);
  nam[0] = "(Intercept)";

  for (int i = 1; i < q+1; i++) {
    std::string id1 = "Z";
    std::string id2 = std::to_string(i);
    nam[i] = id1.append(id2);
  }

  colnames(out) = nam;

  return out;
}

void ghQ::set_grid_weig(const int& q, const int& nqp){
  Rcpp::List ghQP = mvghq(q,nqp);
  Qpi = Rcpp::as<arma::mat>(ghQP["points"]);
  Qw = Rcpp::as<arma::vec>(ghQP["weights"]);
  arma::vec inter(Qpi.n_rows, arma::fill::ones);
  Qp = join_rows(inter,Qpi);
}

arma::mat ghQ::get_grid(){
  return Qp;
}

arma::mat ghQ::get_poin(){
  return Qpi;
}

arma::mat ghQ::get_sigm(){
  return Sigma;
}

arma::vec ghQ::get_weig(){
  return Qw;
}

arma::vec ghQ::get_phis(){
  arma::uvec Li = arma::trimatl_ind(arma::size(Sigma),-1);
  arma::vec phis = Sigma(Li);
  return phis;
}

int ghQ::get_tqp(){
  return tqp ;
}

int ghQ::get_nqp(){
  return nqpo ;
}

int ghQ::get_q(){
  return qo;
}

void ghQ::set_sigm(arma::mat &S){
  Sigma = S;
}

void ghQ::update(){
  set_grid_weig(qo,nqpo);
}
