#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
#include "ghq_class.h"
#include "dist_class.h"
#include "dist_sa_class.h"
#include "fyz_class.h"
#include "utils_basic.h"
#include "utils_sa_basic.h"
#include "utils_estimation.h"
#include "utils_sa_estimation.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEnsmallen)]]

using namespace Rcpp;
using namespace arma;
using namespace ens;

// [[Rcpp::export]]
Rcpp::List glvmlss_fit_gq_rcpp(Rcpp::NumericMatrix& Yr,
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
Rcpp::List glvmlss_fit_sa_rcpp(Rcpp::NumericMatrix& Yr,
                               Rcpp::CharacterVector& famr,
                               arma::cube& b,
                               arma::cube& rb,
                               arma::mat& R,
                               arma::mat& rR,
                               Rcpp::NumericVector& mur,
                               Rcpp::List& control,
                               const arma::mat fmat,
                               arma::mat& Z,
                               Rcpp::Nullable<arma::mat> Xr = R_NilValue){

  const int burnin = control["burn.iter"];
  const int iterlim = control["max.iter"];
  const int tunelim = control["tune.iter"];
  const int window = control["window"];
  const double epsStop = control["tol"];
  const double gTune = control["tune.gamma"];
  const bool verbose = control["verbose"];
  const int every = control["verbose.every"];
  const bool stopFLAG = control["stop.atconv"];
  const bool traceFLAG = control["return.trace"];
  const bool hessFLAG = control["hess.fisher"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);
  const std::string algo = Rcpp::as<std::string>(control["algorithm"]);
  const double h = control["h"];
  const double g = control["gamma"];
  const double gB = control["gamma.b"];
  const double gR = control["gamma.R"];
  const double sap = control["sa.power"];

  const int p = Yr.ncol(), n = Yr.nrow() ;
  arma::mat Yc(Yr.begin(),n,p,false);
  // const int d = b.n_slices;
  const int q = R.n_cols;

  // Place-holders for parameters and latent variables
  arma::cube Bout(arma::size(b), arma::fill::zeros), Bn(b);
  arma::mat L = arma::chol(R,"lower");
  arma::mat Ln(L);
  arma::vec mu = Rcpp::as<arma::vec>(Rcpp::wrap(mur));
  arma::vec Mout(q, arma::fill::zeros), Mn(q);
  arma::mat Lout(arma::size(L), arma::fill::zeros);
  arma::mat Rout(arma::size(R), arma::fill::zeros);
  arma::cube pR(q,q,n), pRout(q,q,n, arma::fill::zeros);
  cube2eye(pR);
  arma::mat pM(arma::size(Z)), pMout(arma::size(Z), arma::fill::zeros);
  arma::mat Zout(arma::size(Z), arma::fill::zeros), Zn(Z);
  // arma::cube ETAS(n,p,d);
  arma::mat X(n, 0);
  if(Xr.isNotNull()){
    X = Rcpp::as<arma::mat>(Xr) ;
  }
  arma::mat ZX = make_ZX(Z,X,fmat);

  // Indices for free parameters
  // arma::uvec irb = arma::find_nan(rb);
  // arma::uvec irR = arma::find_nan(rR);

  arma::uvec Rld;
  if(q > 1){
    Rld = arma::trimatl_ind(arma::size(R));
  } else {
    Rld.resize(1);
    Rld.fill(0);
  }

  int tp = 0;
  tp = b.size() + mu.size() + Rld.size();
  arma::mat theta(window,tp);
  arma::mat patrace(iterlim/10,tp);
  arma::mat lltrace(iterlim/10,2);
  arma::vec artrace(iterlim/10);

  int iter = 1;
  double ar = 0;
  double ssB;
  double ssR;
  double ssZ;

  for(int ii = 1; ii <= iterlim + tunelim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();

    // Tuning phase: readjusting learning rates for SA
    if(ii <= tunelim){
      ssB = gTune/n * gB;
      ssR = gTune/n * gR;
      if(verbose){
        if(ii == 1) Rcpp::Rcout << "\n Iteration: " << std::setw(5) << ii << " (tuning phase)";
        if(ii % every == 0) Rcpp::Rcout << "\r Iteration: " << std::setw(5) << ii << " (tuning phase)";
      }
      ar = 0 ;
      ssZ = h ;
    } else {
      ssB = g/n * gB * std::pow(iter,sap) ;
      ssR = g/n * gR * std::pow(iter,sap) ;
      if(sampler == "ULA"){
        ssZ = h * std::pow(iter,-.333) ;
      } else if(sampler == "MALA"){
        ssZ = h * std::pow(q,-.333) ;
      } else if(sampler == "RWMH"){
        ssZ = h * std::pow(q,-1) ;
      }
    }

    arma::cube ETAS = make_etas(ZX,b);
    Dist_sa YDIST(Yc,famr,ZX,b);
    arma::vec gb = dB(YDIST,ETAS,rb,hessFLAG,ssB);
    arma::vec Vb = Cb2Vb(b,rb);
    Vb -= gb ;
    arma::cube bn = Vb2Cb(Vb,b,rb);
    // arma::vec Mn = newM(mu,R,Z,ssAC);
    // arma::cube bn = b;

    if(q > 1){
      Ln = newL(mu,L,rR,Z,ssR);
      // Ln = L;
      Ln = ProxL(Ln, rR);
    } else {
      Ln = L;
      Ln = ProxL(Ln, rR);
    }

    if(ii <= tunelim){
      Zn = Z;
    } else {
      if(sampler == "ULA"){
      // Zn = newZ_ULA(Y,PI,Z,A,C,mu,L,spD,ssZ);
      } else if(sampler == "MALA"){
      // Zn = newZ_MALA(Y,PI,Z,A,C,mu,L,ar,spD,ssZ,knots,degree,basis);
      } else if(sampler == "RWMH"){
        Zn = newZ_RWMH(YDIST,ETAS,Z,X,b,mu,L,ar,ssZ,fmat);
        // Zn = Z;
      }
      newpMR(Zn,pM,pR,iter);
    }

    b = bn;
    // // mu = Mn;
    L = Ln;
    R = Ln*Ln.t();
    Z = Zn;
    ZX = make_ZX(Z,X,fmat);
    Dist_sa YDISTu(Yc,famr,ZX,b);
    arma::cube ETASu = make_etas(ZX,b);

    for(int t1 = window - 1; t1 > 0; t1--){
      theta.row(t1) = theta.row(t1 - 1);
    }
    arma::rowvec bi = arma::vectorise(b).t();
    arma::rowvec mi = mu.t();
    arma::rowvec ri = R(Rld).t();
    arma::rowvec input = arma::join_rows(bi,mi,ri);
    theta.row(0) = input;
    arma::mat dtheta = arma::diff(theta,1,0);
    double maxvalue = arma::max(arma::abs(arma::vectorise(dtheta)));

    if(iter % 10 == 0){
      patrace.row(iter/10-1) = input;
      double fzll = arma::accu(fz(Z,mu,R));
      lltrace(iter/10-1,0) = fzll;
      double fyzll = arma::accu(fyz(YDISTu, ETASu));
      lltrace(iter/10-1,1) = fyzll;
      double cdll = fyzll + fzll;
      artrace(iter/10-1) = ar/(n*iter);

      if(verbose & (iter % every == 0)){
        if(iter >= burnin){
          // if(sampler == "ULA"){
          //   Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
          //     ", max (abs) change: " << std::to_string(maxvalue) << ")";
          // } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR (" << sampler << "): " << std::to_string(ar/(n*iter)) << ")";
          // }
        } else {
          // if(sampler == "ULA"){
          //   Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase, CD-llk: " << std::to_string(cdll) <<
          //     ", max (abs) change: " << std::to_string(maxvalue) << ")";
          // } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR (" << sampler << "): " << std::to_string(ar/(n*iter)) << ")";
          // }
        }
      }

      if(stopFLAG & (maxvalue < epsStop)) {
        iter++;
        break;
      }
    }

    if(iter >= burnin){
      Bout += b;
      Mout += mu;
      Lout += L;
      Rout += R;
      Zout += Z;
      pMout += pM;
      pRout += pR;
    }
    if(ii > tunelim) iter++;
  }

  Bout /= (iter - burnin);
  Mout /= (iter - burnin);
  Lout /= (iter - burnin);
  Rout /= (iter - burnin);
  Zout /= (iter - burnin);
  pMout /= (iter - burnin);
  pRout /= (iter - burnin);

  // arma::mat Lout_ = arma::chol(Rout, "lower");
  // Lout_ = ProxL(Lout_, rR);

  // const int nsim = control["nsim"];
  // double llk(0), BIC(0), AIC(0);
  // if(nsim != 0){
  //   llk = fy_gapmCDM_IS(Y,Aout,Cout,Mout,Lout_,Zout,
  //                       pMout,pRout,control);
  //   BIC = -2*llk + std::log(n)*Aout.n_elem;
  //   AIC = -2*llk + 2*Aout.n_elem;
  // }
  // arma::mat Uout = Z2U(Zout);
  // spObj = SpU_bsp(Uout,knots,degree);
  // spM = spObj.slice(0);
  // PI = prob(Aout,Cout,spM);
  if(traceFLAG){
    return Rcpp::List::create(Rcpp::Named("b") = Bout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("cdllk.trace") = lltrace,
                              Rcpp::Named("ar.trace") = artrace,
                              Rcpp::Named("theta.trace") = patrace,
                              Rcpp::Named("posMu") = pMout,
                              Rcpp::Named("posR") = pRout);
  } else {
    return Rcpp::List::create(Rcpp::Named("b") = Bout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("posMu") = pMout,
                              Rcpp::Named("posR") = pRout);
  }
 }

// [[Rcpp::export]]
Rcpp::List glvmlss_sim_rcpp(const int n,
                            Rcpp::CharacterVector& famr,
                            const int q,
                            arma::cube& b,
                            arma::mat& R,
                            arma::vec& mu,
                            const arma::mat fmat,
                            Rcpp::Nullable<arma::mat> Xr = R_NilValue){

 int p = famr.length();
 arma::mat Yo(n,p);

 arma::mat Z = rMVnorm(n, mu, R);
 arma::mat ZX(Z) ;
 arma::mat X(Z.n_rows, 0);
 if(Xr.isNotNull()){
  X = Rcpp::as<arma::mat>(Xr) ;
  ZX = arma::join_rows(Z, X);
 }
 ZX = ZX * fmat;
 arma::vec inter(n, arma::fill::ones);
 ZX = arma::join_rows(inter, ZX);
 arma::cube eta = make_etas(ZX, b);

 for(int j = 0; j < p; j++){
  arma::mat etaj = eta.col_as_mat(j);
  std::string fam = Rcpp::as<std::string>(famr(j));
  Dist Y(fam) ;
  arma::vec Yj = Y.sim(n,etaj);
  Yo.col(j) = Yj;
 }

 if(Xr.isNotNull()){
  return Rcpp::List::create(Rcpp::Named("Y") = Yo,
                            Rcpp::Named("Z") = Z,
                            Rcpp::Named("X") = X,
                            Rcpp::Named("b") = b,
                            Rcpp::Named("R") = R) ;
 } else {
  return Rcpp::List::create(Rcpp::Named("Y") = Yo,
                            Rcpp::Named("Z") = Z,
                            Rcpp::Named("b") = b,
                            Rcpp::Named("R") = R) ;
 }
}
