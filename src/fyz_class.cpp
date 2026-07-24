#include <RcppArmadillo.h>
#include "utils_basic.h"
#include "ghq_class.h"
#include "dist_class.h"
#include "fyz_class.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

void fYZ::update(Rcpp::NumericMatrix& Yr,
                 Rcpp::CharacterVector& famr,
                 ghQ& Q,
                 arma::cube& b){

  int p = Yr.ncol(), n = Yr.nrow() ;
  arma::mat Yc(Yr.begin(),n,p,false);
  arma::mat Qp = Q.get_grid() ;
  arma::vec Qw = Q.get_weig() ;

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

arma::mat fYZ::get_pD(){
    return pD;
}

double fYZ::get_ll(){
    return ll;
}
