#include <RcppArmadillo.h>
#include "utils_basic.h"
#include "utils_sa_basic.h"
#include "dist_sa_class.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// Vector of (n) entries: log f_i(y_m | z_m); NaN for missing entries
arma::vec Dist_sa::dy_sa(const int i, arma::cube& etas){

  std::string fam = Rcpp::as<std::string>(famv(i));
  arma::vec Yc = Yr.col(i) ;
  arma::mat etac = etas.col_as_mat(i);
  const int n = Yc.n_elem ;
  arma::vec out(n, arma::fill::zeros);

  if(fam == "normal"){
    arma::vec mu = etac.col(0);
    arma::vec sg = arma::exp(etac.col(1));
    for(int m = 0; m < n; m++){
      if(std::isnan(Yc(m))){
        out(m) = arma::datum::nan ;
      } else {
        double mud = mu(m);
        double sgd = sg(m);
        out(m) = R::dnorm(Yc(m), mud, sgd, 1);
      }
    }
    return out;
  } else if(fam == "binomial"){
    arma::vec mu = expitvec(etac.col(0));
    for(int m = 0; m < n; m++){
      if(std::isnan(Yc(m))){
        out(m) = arma::datum::nan ;
      } else {
        double mud = mu(m);
        out(m) = R::dbinom(Yc(m), 1, mud, 1);
      }
    }
    return out;
  } else if(fam == "beta"){
    fixY(Yc);
    arma::vec mu = expitvec(etac.col(0));
    arma::vec sg = expitvec(etac.col(1));
    arma::vec al = geta(mu,sg);
    arma::vec be = getb(mu,sg);
    for(int m = 0; m < n; m++){
      if(std::isnan(Yc(m))){
        out(m) = arma::datum::nan ;
      } else {
        double ad = al(m);
        double bd = be(m);
        out(m) = R::dbeta(Yc(m), ad, bd, 1);
      }
    }
    return out;
  } else if(fam == "sn"){
    arma::vec mu = etac.col(0);
    arma::vec sg = exp(etac.col(1));
    arma::vec nu = expitvec(etac.col(2));

    arma::vec nuc = p2nuc(nu);
    arma::vec nud = nc2nd(nuc);
    arma::vec sgd = sc2sd(sg,nuc);
    arma::vec mud = mc2md(mu,sg,nuc);

    for(int m = 0; m < n; m++){
      if(std::isnan(Yc(m))){
        out(m) = arma::datum::nan ;
      } else {
        out(m) = dsn(Yc(m), mud(m), sgd(m), nud(m));
      }
    }
    return out;
  } else return out;
}

// Matrix of (n x D) 1st derivatives: d log f_i(y_m | z_m, x_m) / d eta_d; NaN for missing entries
arma::mat Dist_sa::d1y_sa(const int i, arma::cube& etas){

  std::string fam = Rcpp::as<std::string>(famv(i));
  arma::vec Yc = Yr.col(i) ;
  arma::mat etac = etas.col_as_mat(i);
  const int n = Yc.n_elem ;
  const int D = etac.n_cols ;

  arma::mat out(n,D) ;

  if(fam == "normal"){
    arma::vec mu = etac.col(0);
    arma::vec sg = arma::exp(etac.col(1));
    // First derivatives of mu: d1fdmu * dmudeta
    out.col(0) = (Yc - mu)/arma::square(sg); // * 1
    // First derivatives of sg: d1fdsg * dsgdeta
    out.col(1) = (arma::square(Yc - mu) - arma::square(sg))/arma::pow(sg,3) % sg;
    return out;
  } else if(fam == "binomial"){
    arma::vec mu = expitvec(etac.col(0));
    arma::vec dX = dMd1Evec(logitvec(mu));
    // First derivatives of mu: d1fdmu * dmudeta
    out.col(0) = (Yc - mu)/(mu % (1 - mu)) % dX ;
    return out;
  } else if(fam == "beta"){
    fixY(Yc);
    arma::vec mu = expitvec(etac.col(0));
    arma::vec sg = expitvec(etac.col(1));
    arma::vec al = geta(mu,sg);
    arma::vec be = getb(mu,sg);
    arma::vec albe = al + be;
    arma::vec dXm = dMd1Evec(logitvec(mu));
    arma::vec dXs = dMd1Evec(logitvec(sg));
    // First derivatives of mu: d1fdmu * dmudeta
    out.col(0) = ((1.0 - arma::square(sg))/arma::square(sg)) % (-1.0*digamma_vec(al) +
      digamma_vec(be) + arma::log(Yc) - arma::log(1.0-Yc)) % dXm ;
    // First derivatives of sg: d1fdsg * dsgdeta
    out.col(1) = -1.0*(2.0/(arma::pow(sg,3))) % (mu % (-1.0*digamma_vec(al) + digamma_vec(albe) +
      arma::log(Yc)) + (1-mu)%(-1.0*digamma_vec(be) + digamma_vec(albe) + arma::log(1.0 - Yc))) % dXs ;
    return out;
  } else if(fam == "sn"){
    arma::vec mu = etac.col(0);
    arma::vec sg = arma::exp(etac.col(1));
    arma::vec nu = expitvec(etac.col(2));
    arma::vec dXn = dMd1Evec(logitvec(nu));

    arma::vec nuc = p2nuc(nu);
    arma::vec nud = nc2nd(nuc);
    arma::vec sgd = sc2sd(sg,nuc);
    arma::vec mud = mc2md(mu,sg,nuc);

    double dnucdp = (2*std::sqrt(2)*(4-M_PI)/std::pow(M_PI-2,1.5));

    arma::vec D1M = d1mu_vec(Yc,mud,sgd,nud);
    arma::vec D1S = d1sg_vec(Yc,mud,sgd,nud);
    arma::vec D1N = d1nu_vec(Yc,mud,sgd,nud);

    arma::vec v_dmddsc = dmddsc(nuc) ;
    arma::vec v_dmddnc = dmddnc(sg,nuc) ;
    arma::vec v_dsddsc = dsddsc(nuc) ;
    arma::vec v_dsddnc = dsddnc(sg,nuc) ;
    arma::vec v_dnddnc = dnddnc(nuc);

    D1N %= v_dnddnc;
    arma::vec D1Ms = D1M % v_dmddsc ;
    arma::vec D1Ss = D1S % v_dsddsc ;
    arma::vec D1Mn = D1M % v_dmddnc ;
    arma::vec D1Sn = D1S % v_dsddnc ;
    arma::vec D1MsD1Ss = D1Ms + D1Ss ;
    arma::vec D1MnD1SnD1N = D1Mn + D1Sn + D1N ;

    out.col(0) = D1M ;
    out.col(1) = D1MsD1Ss % sg ;
    out.col(2) = D1MnD1SnD1N % (dnucdp * dXn);

    return out;
  } else return out;
}

// Matrix of (n x D) 2nd derivatives: d^2 logf_i(y_m | z_m, x_m) /(d eta_d)^2; NaN for missing entries
arma::mat Dist_sa::d2y_sa(const int i,
                          arma::cube& etas,
                          const bool flagEIM,
                          arma::mat& d1Yi){

  std::string fam = Rcpp::as<std::string>(famv(i));
  arma::vec Yc = Yr.col(i) ;
  arma::mat etac = etas.col_as_mat(i);
  const int n = Yc.n_elem ;
  const int D = etac.n_cols ;

  arma::mat out(n,D) ;

  if(fam == "normal"){
    arma::vec mu = etac.col(0);
    arma::vec sg = arma::exp(etac.col(1));

    if(flagEIM){
      // Second derivatives of mu: d2fdmu * dmudeta^2
      out.col(0) = -1/arma::square(sg); // * 1
      // Second derivatives of sg: d2fdsg * dsgdeta^2
      out.col(1) = -2/arma::square(sg) % arma::square(sg); // exp(log(sgd)) ^ 2 = sgd^2;
      return out;
    } else {
      // Second derivatives of mu: (d2fdmu * dmudeta^2) + d1fdmu
      out.col(0) = -1/arma::square(sg); // * 1
      // Second derivatives of sg: (d2fdsg * dsgdeta^2) + d1fdsg
      out.col(1) = (-3*arma::square(Yc - mu) / arma::pow(sg,4) + 1/arma::square(sg)) % arma::square(sg) + d1Yi.col(1) ; // exp(log(sgd)) ^ 2 = sgd^2;
      return out;
    }
  } else if(fam == "binomial"){
    arma::vec mu = expitvec(etac.col(0));
    arma::vec d1X = dMd1Evec(logitvec(mu));
    arma::vec d2X = dMd2Evec(mu);

    if(flagEIM){
      // Second derivatives of mu: d2fdmu * dmudeta^2
      out.col(0) = -1/(mu % (1-mu)) % arma::square(d1X) ;
      return out;
    } else {
      // Second derivatives of mu: (d2fdmu * dmudeta^2) + d1fdmu
      out.col(0) = (-(arma::square(mu) - 2*Yc%mu + Yc) / arma::square((mu-1)%mu) % arma::square(d1X) + d2X % d1Yi.col(0)) ;
      return out;
    }
  } else if(fam == "beta"){
    fixY(Yc);
    arma::vec mu = expitvec(etac.col(0));
    arma::vec sg = expitvec(etac.col(1));
    arma::vec al = geta(mu,sg);
    arma::vec be = getb(mu,sg);
    arma::vec albe = al + be;
    arma::vec d1Xm = dMd1Evec(logitvec(mu));
    arma::vec d1Xs = dMd1Evec(logitvec(sg));
    arma::vec d2Xm = dMd2Evec(mu);
    arma::vec d2Xs = dMd2Evec(sg);

    if(flagEIM){
      // Second derivatives of mu: d2fdmu * dmudeta^2
      out.col(0) = -1.0*(arma::square(1.0 - arma::square(sg))/arma::pow(sg,4)) % (trigamma_vec(al) +
        trigamma_vec(be)) % arma::square(d1Xm) ;
      // Second derivatives of sg: d2fdsg * dsgdeta^2
      out.col(1) = -1.0*(4.0/arma::pow(sg,6)) % (arma::square(mu) % trigamma_vec(al) +
        arma::square(1.0 - mu) % trigamma_vec(be) - trigamma_vec(albe)) % arma::square(d1Xs) ;
      return out;
    } else {
      // Second derivatives of mu: (d2fdmu * dmudeta^2) + d1fdmu
      out.col(0) = -1.0*(arma::square(1.0 - arma::square(sg))/ arma::pow(sg,4)) % (trigamma_vec(al) +
        trigamma_vec(be)) % arma::square(d1Xm) + d2Xm % d1Yi.col(0) ;
      // Second derivatives of sg: (d2fdsg * dsgdeta^2) + d1fdsg
      out.col(1) = -1.0*(4.0/arma::pow(sg,6)) % (arma::square(mu) % trigamma_vec(al) +
        arma::square(1.0 - mu) % trigamma_vec(be) - trigamma_vec(albe)) % arma::square(d1Xs) + d2Xs % d1Yi.col(1) ;
      return out;
    }
  } else if(fam == "sn"){
    arma::vec mu = etac.col(0);
    arma::vec sg = arma::exp(etac.col(1));
    arma::vec nu = expitvec(etac.col(2));
    arma::vec d1Xn = dMd1Evec(logitvec(nu));
    arma::vec d2Xn = dMd2Evec(nu);

    arma::vec nuc = p2nuc(nu);
    arma::vec nud = nc2nd(nuc);
    arma::vec sgd = sc2sd(sg,nuc);
    arma::vec mud = mc2md(mu,sg,nuc);

    double dnucdp = (2*std::sqrt(2)*(4-M_PI)/std::pow(M_PI-2,1.5));

    if(flagEIM){
      arma::vec D2M = d2muF_vec(Yc,mud,sgd,nud);
      arma::vec D2S = d2sgF_vec(Yc,mud,sgd,nud);
      arma::vec D2N = d2nuF_vec(Yc,mud,sgd,nud);
      arma::vec DcMS = dcmsF_vec(Yc,mud,sgd,nud);
      arma::vec DcMN = dcmnF_vec(Yc,mud,sgd,nud);
      arma::vec DcSN = dcsnF_vec(Yc,mud,sgd,nud);

      arma::vec v_dmddsc = dmddsc(nuc);
      arma::vec v_dmddnc = dmddnc(sg,nuc) ;
      arma::vec v_dsddsc = dsddsc(nuc) ;
      arma::vec v_dsddnc = dsddnc(sg,nuc) ;
      arma::vec v_dnddnc = dnddnc(nuc) ;

      arma::vec D2Ms = D2M % v_dmddsc ;
      arma::vec D2Mn = D2M % v_dmddnc ;
      arma::vec DcMSs1 = DcMS % v_dsddsc ;
      arma::vec DcMSs2 = DcMS % v_dmddsc ;
      arma::vec DcMSn1 = DcMS % v_dsddnc ;
      arma::vec DcMSn2 = DcMS % v_dmddnc ;
      arma::vec DcMNn1 = DcMN % v_dnddnc ;
      arma::vec DcMNn2 = DcMN % v_dmddnc ;

      arma::vec D2Ss = D2S % v_dsddsc ;
      arma::vec D2Sn = D2S % v_dsddnc ;
      arma::vec DcSNn1 = DcSN % v_dnddnc ;
      arma::vec DcSNn2 = DcSN % v_dsddnc ;

      D2N %= v_dnddnc ;

      arma::vec D2MsDcMSs1 = D2Ms + DcMSs1 ;
      arma::vec DcMSs2D2Ss = DcMSs2 + D2Ss ;
      arma::vec D2MnDcMSn1DcMNn1 = D2Mn + DcMSn1 + DcMNn1 ;
      arma::vec DcMSn2D2SnDcSNn1 = DcMSn2 + D2Sn + DcSNn1 ;
      arma::vec DcMNn2DcSNn2D2N = DcMNn2 + DcSNn2 + D2N ;

      out.col(0) = D2M ;
      out.col(1) = (D2MsDcMSs1 % v_dmddsc + DcMSs2D2Ss % v_dsddsc) % arma::square(sg) ;
      out.col(2) = (D2MnDcMSn1DcMNn1 % v_dmddnc + DcMSn2D2SnDcSNn1 % v_dsddnc +
        DcMNn2DcSNn2D2N % v_dnddnc) % arma::square(dnucdp * d1Xn) ;

      return out;
    } else {

      arma::vec D2M = d2mu_vec(Yc,mud,sgd,nud);
      arma::vec D2S = d2sg_vec(Yc,mud,sgd,nud);
      arma::vec D2N = d2nu_vec(Yc,mud,sgd,nud);
      arma::vec DcMS = dcms_vec(Yc,mud,sgd,nud);
      arma::vec DcMN = dcmn_vec(Yc,mud,sgd,nud);
      arma::vec DcSN = dcsn_vec(Yc,mud,sgd,nud);

      arma::vec v_dmddsc = dmddsc(nuc);
      arma::vec v_dmddnc = dmddnc(sg,nuc) ;
      arma::vec v_dsddsc = dsddsc(nuc) ;
      arma::vec v_dsddnc = dsddnc(sg,nuc) ;
      arma::vec v_dnddnc = dnddnc(nuc) ;

      arma::vec D2Ms = D2M % v_dmddsc ;
      arma::vec D2Mn = D2M % v_dmddnc ;
      arma::vec DcMSs1 = DcMS % v_dsddsc ;
      arma::vec DcMSs2 = DcMS % v_dmddsc ;
      arma::vec DcMSn1 = DcMS % v_dsddnc ;
      arma::vec DcMSn2 = DcMS % v_dmddnc ;
      arma::vec DcMNn1 = DcMN % v_dnddnc ;
      arma::vec DcMNn2 = DcMN % v_dmddnc ;

      arma::vec D2Ss = D2S % v_dsddsc ;
      arma::vec D2Sn = D2S % v_dsddnc ;
      arma::vec DcSNn1 = DcSN % v_dnddnc ;
      arma::vec DcSNn2 = DcSN % v_dsddnc ;

      D2N %= v_dnddnc ;

      arma::vec D2MsDcMSs1 = D2Ms + DcMSs1 ;
      arma::vec DcMSs2D2Ss = DcMSs2 + D2Ss ;
      arma::vec D2MnDcMSn1DcMNn1 = D2Mn + DcMSn1 + DcMNn1 ;
      arma::vec DcMSn2D2SnDcSNn1 = DcMSn2 + D2Sn + DcSNn1 ;
      arma::vec DcMNn2DcSNn2D2N = DcMNn2 + DcSNn2 + D2N ;

      out.col(0) = D2M ;
      out.col(1) = (D2MsDcMSs1 % v_dmddsc + DcMSs2D2Ss % v_dsddsc) % arma::square(sg) + d1Yi.col(1) ;
      out.col(2) = (D2MnDcMSn1DcMNn1 % v_dmddnc + DcMSn2D2SnDcSNn1 % v_dsddnc +
        DcMNn2DcSNn2D2N % v_dnddnc) % arma::square(dnucdp * d1Xn) + (d1Yi.col(2) % d2Xn) ;

      return out;
    }
  } else return out;
}

// arma::mat Dist_sa::dCy_sa
