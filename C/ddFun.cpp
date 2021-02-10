#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// [[Rcpp::export]]

NumericVector timesTwo(NumericVector x) {
  NumericVector results = 2*x;
  return results;
}


// [[Rcpp::export]]
List f1 (NumericVector x) {
  NumericVector a = timesTwo(x) + pow(x,0.5);
  NumericVector b = timesTwo(x) - pow(x,0.5);
  return List::create(Named("a") = a, Named("b") = b);
}

double f (double mu) {
  double val = ((R::dnorm(-mu, 0, 1, false)) /
                (1 - R::pnorm(-mu, 0, 1, true, false))
  ) ;
  return(val) ;
}

double g (double mu) {
  double val = ((R::dnorm(-mu, 0, 1, false)) /
                (R::pnorm(-mu, 0, 1, true, false))
  ) ;
  return(val) ;
}

// [[Rcpp::export]]
List em3 (const arma::mat y,
          const arma::mat X,
          const int maxit = 10
) {
  // inputs
const int N = y.n_rows ;
const int K = X.n_cols ;

// containers
arma::mat beta(K, 1) ;
beta.fill(0.0) ; // initialize betas to 0
arma::mat eystar(N, 1) ;
eystar.fill(0) ;

// algorithm
for (int it = 0 ; it < maxit ; it++) {
  arma::mat mu = X * beta ;
  // augmentation step
// #pragma omp parallel for // un-comment for paralleling 
  for (int n = 0 ; n < N ; n++) {
    if (y(n, 0) == 1) { // y = 1
      eystar(n, 0) = mu(n, 0) + f(mu(n, 0)) ;
    }
    if (y(n, 0) == 0) { // y = 0
      eystar(n, 0) = mu(n, 0) - g(mu(n, 0)) ;
    }
  }
  // maximization step
  beta = (X.t() * X).i() * X.t() * eystar ;
}

// returns
List ret ;
ret["N"] = N ;
ret["K"] = K ;
ret["beta"] = beta ;
ret["eystar"] = eystar ;
return(ret) ;
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.

/*** R
timesTwo(42)
*/