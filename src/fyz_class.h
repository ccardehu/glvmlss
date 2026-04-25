#ifndef fYZClass_H
#define fYZClass_H

// Posterior density on the quadrature marginal llk
class fYZ {
  // Private class members
  // pD(m,i): posterior weight at quadrature point i for respondent m
  // Normalized: (sum_i pD(m,i)*Qw(i) = 1).
  arma::mat pD ;
  // marginal llk:
  double ll ;

public:

  // Constructors

  fYZ(){}

  fYZ(Rcpp::NumericMatrix& Yr,
      Rcpp::CharacterVector& famr,
      ghQ& Q,
      arma::cube& b) {

    int p = Yr.ncol(), n = Yr.nrow() ;
    arma::mat Yc(Yr.begin(),n,p,false);
    arma::mat Qp = Q.get_grid() ;
    arma::vec Qw = Q.get_weig() ;
    int tqp = Qp.n_rows ;

    pD.set_size(n,tqp);
    pD.fill(0.0);

    for(int i = 0; i < p; i++){
      std::string fam = Rcpp::as<std::string>(famr[i]);
      Dist Y(fam) ;
      arma::mat dYi = Y.dy(i,Yc,Qp,b);
      dYi.replace(arma::datum::nan, 0.0);
      pD += dYi;
    }

    arma::vec Cm = arma::max(pD, 1);
    arma::mat pD_shift = pD.each_col() - Cm;
    arma::vec llv = Cm + arma::log(arma::exp(pD_shift) * Qw);
    ll = arma::accu(llv);
    pD.each_col() -= llv ;
    pD = arma::exp(pD) ;

  }

  fYZ(fYZ& fYZobj){
    pD = fYZobj.get_pD();
    ll = fYZobj.get_ll();
  }

  // Member functions
  void update(Rcpp::NumericMatrix& Yr,
              Rcpp::CharacterVector& famr,
              ghQ& Q,
              arma::cube& b);
  arma::mat get_pD();
  double get_ll();
};

#endif
