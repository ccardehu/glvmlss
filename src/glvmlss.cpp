#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
#include "ghq_class.h"
#include "dist_class.h"
#include "fyz_class.h"
#include "utils_basic.h"
#include "utils_estimation.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEnsmallen)]]

using namespace Rcpp;
using namespace arma;
using namespace ens;

// [[Rcpp::export]]
Rcpp::List glvmlss_rcpp(Rcpp::NumericMatrix& Yr,
                        Rcpp::CharacterVector& famr,
                        const int q,
                        arma::cube& b,
                        arma::cube& rb,
                        arma::mat& R,
                        arma::mat& rR,
                        Rcpp::List& control){

  int nqp = control["nQP"];
  bool flagVERBO = control["verbose"];
  bool flagCORLV = control["estim.R"];
  bool flagGD = control["EM.useGD"];
  int EM_iL = control["EM.maxit"];
  int DM_iL = control["DM.maxit"];

  // Build the quadrature at R
  ghQ Q(q,nqp,R);
  // Build the posterior density fYZ at (b,R); computes llk and pD.
  fYZ fyz(Yr,famr,Q,b);

  // EM-step
  if(EM_iL != 0){
    if(flagGD){
      EM_stepGD(Yr,famr,b,rb,rR,Q,fyz,control);
    } else {
      EM_step(Yr,famr,b,rb,rR,Q,fyz,control);
    }
  }
  // DM-step
  if(DM_iL != 0) DM_step(Yr,famr,b,rb,rR,Q,fyz,control);

  arma::mat pD = fyz.get_pD();
  if(flagVERBO) Rcpp::Rcout << "\n Computing standard errors ... " ;
  Rcpp::List se = SEs(Yr,famr,Q,b,rb,rR,pD,flagCORLV);
  if(flagVERBO) Rcpp::Rcout << "\r Computing standard errors ... Done!" ;

  arma::cube seb = se["seb"];

  arma::mat fsc = fscores(Q,pD);
  int n = Yr.nrow() ;
  int K = se["K"];
  double AIC = -2.0*fyz.get_ll() + 2.0*K;
  double BIC = -2.0*fyz.get_ll() + std::log(n)*K;

  if(flagCORLV){
    arma::mat seR = se["seR"];
    return Rcpp::List::create(Rcpp::Named("b") = b,
                              Rcpp::Named("R") = Q.get_sigm(),
                              Rcpp::Named("ll") = fyz.get_ll(),
                              Rcpp::Named("seb") = seb,
                              Rcpp::Named("seR") = seR,
                              Rcpp::Named("Z") = fsc,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("K") = K) ;
  } else{
    return Rcpp::List::create(Rcpp::Named("b") = b,
                              Rcpp::Named("R") = Q.get_sigm(),
                              Rcpp::Named("ll") = fyz.get_ll(),
                              Rcpp::Named("seb") = seb,
                              Rcpp::Named("Z") = fsc,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("K") = K) ;
  }
}

// [[Rcpp::export]]
Rcpp::List glvmlss_rcpp_sim(const int n,
                            Rcpp::CharacterVector& famr,
                            const int q,
                            arma::cube& b,
                            arma::mat& R,
                            Rcpp::List& control){

  int nqp = control["nQP"];
  int p = famr.length();
  arma::mat Yo(n,p);

  ghQ Q(q,nqp,R);
  Rcpp::NumericMatrix Zr = Q.get_simZ(q,n);
  arma::mat Z(Zr.begin(),n,q+1,false,false);

  for(int i = 0; i < p; i++){
    std::string fam = Rcpp::as<std::string>(famr(i));
    Dist Y(fam) ;
    arma::vec Yi = Y.sim(n,i,b,Z);
    Yo.col(i) = Yi;
  }

  return Rcpp::List::create(Rcpp::Named("Y") = Yo,
                            Rcpp::Named("Z") = Z,
                            Rcpp::Named("b") = b,
                            Rcpp::Named("R") = R) ;
}
