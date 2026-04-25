#ifndef GHQClass_H
#define GHQClass_H

// Multivariate Gauss-Hermite Quadrature (ghQ)
// Stores covariance matrix Sigma, points, and weights,
class ghQ {
  // Private Helper functions
  // Univariate ghQ
  Rcpp::List ghq(const int &n);
  // Helper function: replicates vecC (times)
  Rcpp::List newList(const int& times,arma::colvec& vecC);
  // Helper function: data frame to arma::mat
  arma::mat df2mat(Rcpp::DataFrame& df);
  // Helper function: simulate n MVN (eigen-decomposition; used in get_simZ for simulation)
  arma::mat mvrnorm_sim(const int n, arma::vec& mu, arma::mat& sigma);
  // Helper function: builds multivariate ghQ
  Rcpp::List mvghq(const int & q, const int & nqp);
  // Helper function: builds weights and grid points (+ intercept)
  void set_grid_weig(const int& q, const int& nqp);

  // Private class members
  arma::mat Sigma, Qp, Qpi ;
  arma::vec Qw ;
  int qo, nqpo, tqp ;

public:

  // Constructors

  ghQ(const int &q, const int& nqp) {
    Sigma.eye(q,q);
    set_grid_weig(q,nqp);
    tqp = Qpi.n_rows;
    qo = q;
    nqpo = nqp;
  }

  ghQ(const int &q, const int& nqp, arma::mat& S) {
    Sigma = S;
    set_grid_weig(q,nqp);
    tqp = Qpi.n_rows;
    qo = q;
    nqpo = nqp;
  }

  ghQ(ghQ& ghQobj){
    Qp = ghQobj.get_grid();
    Qpi = ghQobj.get_poin();
    Qw = ghQobj.get_weig();
    qo = ghQobj.get_q();
    Sigma = ghQobj.get_sigm();
    tqp = ghQobj.get_tqp();
    nqpo = ghQobj.get_nqp();
  }

  // Member functions
  Rcpp::NumericMatrix get_simZ(const int& q, const int& n);
  void set_sigm(arma::mat& S);
  arma::mat get_grid();
  arma::mat get_poin();
  arma::vec get_weig();
  arma::mat get_sigm();
  arma::vec get_phis();
  int get_tqp();
  int get_nqp();
  int get_q();
  void update();
};

#endif
