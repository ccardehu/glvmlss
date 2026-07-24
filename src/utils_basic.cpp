#include <RcppArmadillo.h>
#include "utils_basic.h"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

using namespace Rcpp;
using namespace arma;
using namespace boost::math;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]

// Linear list of combinations of dist. parameter indices (k1,k2)
arma::uword ixd2ll(int& k1, int& k2, int& K){
  if(K == 2){
    if((k1 == 0) & (k2 == 1)){return 0;}
  } else if(K == 3){
    if((k1 == 0) & (k2 == 1)){return 0;}
    if((k1 == 0) & (k2 == 2)){return 1;}
    if((k1 == 1) & (k2 == 2)){return 2;}
  } else if(K == 4){
    if((k1 == 0) & (k2 == 1)){return 0;}
    if((k1 == 0) & (k2 == 2)){return 1;}
    if((k1 == 0) & (k2 == 3)){return 2;}
    if((k1 == 1) & (k2 == 2)){return 3;}
    if((k1 == 1) & (k2 == 3)){return 4;}
    if((k1 == 2) & (k2 == 3)){return 5;}
  }
  return 0;
}

// Counts free coefficients in rb for each item i and dist. parameter k
arma::mat count_b(arma::cube& rb){
  const int K = rb.n_slices;
  const int p = rb.n_rows;
  arma::mat out(p,K);
  for(int k = 0; k < K; k++){
    arma::mat rbtmp = rb.slice(k);
    for(int i = 0; i < p; i++){
      arma::rowvec rtmp = rbtmp.row(i);
      int q = rtmp.size();
      for(int j = 0; j < q; j++){
        out(i,k) += (std::isnan(rbtmp(i,j))) ? 1.0 : 0.0;
      }
    }
  }
  return out;
}

// Cube b to matrix b
arma::mat Cb2Mb(arma::cube& b){
  int K = b.n_slices, p = b.n_rows, q = b.n_cols ;
  arma::mat bt(b.slice(0).begin(),p,q,false) ;
  for(int k = 1; k  < K; k++){
    arma::mat bt1(b.slice(k).begin(),p,q,false);
    bt = arma::join_rows(bt,bt1);
  }
  return bt;
}

// Cube b to vector b (free parameters)
arma::vec Cb2Vb(arma::cube& b, arma::cube& rb){
  arma::vec rbV = arma::vectorise(Cb2Mb(rb), 1).t();
  arma::vec bV = arma::vectorise(Cb2Mb(b), 1).t();
  arma::uvec idx = arma::find_nan(rbV);
  arma::vec out = bV(idx);
  return out ;
}

// Matrix b to cube b
arma::cube Mb2Cb(arma::mat mb,
                 const int d1,
                 const int d2){
  arma::cube b(mb.n_rows,d1,d2);
  int mem = 0;
  for(int k = 0; k < d2; k++){
    arma::uvec id0 = arma::regspace<arma::uvec>(mem, (mem + (d1-1)));
    b.slice(k) = mb.cols(id0);
    mem += (d1);
  }
  return b;
}

// Vector b to cube b (fixes free parameters)
arma::cube Vb2Cb(arma::vec& vb,
                 arma::cube& b,
                 arma::cube& rb){
  arma::mat bM = Cb2Mb(b).t();
  arma::mat rbM = Cb2Mb(rb).t();
  arma::uvec id0 = arma::find_nan(rbM);
  bM(id0) = vb;
  int D = b.n_slices;
  int QS = b.n_cols;
  arma::cube bC = Mb2Cb(bM.t(),QS,D);
  return bC;
}

// Vector to matrix (column-major)
arma::mat vec2mat(arma::vec& x, int nrow, int ncol){
  arma::mat y(x);
  y.reshape(nrow, ncol);
  return y;
}

// Fixes values at 0 and 1 to avoid overflow
void fixY(arma::vec& y){
  y.elem(arma::find(y == 1.0)).fill(1.0 - std::sqrt(arma::datum::eps));
  y.elem(arma::find(y == 0.0)).fill(std::sqrt(arma::datum::eps));
}

// Elementwise logistic (expit) function
arma::vec expitvec(arma::vec eta){
  arma::vec expx = arma::exp(eta);
  return expx / (1.0 + expx);
}

// Elementwise logit function (inverse of expit)
arma::vec logitvec(arma::vec mu){
  return arma::log(mu / (1.0 - mu));
}

// First derivative dmu/deta for the logit link
arma::vec dMd1Evec(arma::vec eta){
  arma::vec expx = arma::exp(eta);
  return expx / arma::pow(1.0 + expx, 2);
}

// Second derivative for the logit link
arma::vec dMd2Evec(arma::vec mu){
  return 1.0 - 2.0*mu ;
}

// Maps (mu, sigma) to the shape parameter alpha in beta pdf
arma::vec geta(arma::vec& mu, arma::vec& sg){
  arma::vec a(mu.n_elem);
  a = mu % (1.0 - arma::pow(sg,2)) / arma::pow(sg,2);
  return a;
}

// Maps (mu, sigma) to the shape parameter beta in beta pdf
arma::vec getb(arma::vec& mu, arma::vec& sg){
  arma::vec b(mu.n_elem);
  b = (1.0 - mu) % (1.0 - arma::pow(sg,2)) / arma::pow(sg,2);
  return b;
}

// Mills-ratio used in SN derivatives: zeta1(y) = phi(y) / Phi(y)
arma::vec zeta1(arma::vec& y){
  int n = y.n_elem;
  arma::vec out(n);
  for(int i = 0; i < n; i++){
    if(std::isnan(y(i))){
      out(i) = arma::datum::nan ;
    } else {
      out(i) = R::dnorm(y(i),0,1,1) - R::pnorm(y(i),0,1,1,1);
    }
  }
  return arma::exp(out) ;
}

// Second derivative of log(2*Phi(y)) used in SN second derivatives: -y*zeta1(y) - zeta1(y)^2
arma::vec zeta2(arma::vec& y){
  arma::vec temp = zeta1(y);
  return -y % temp - arma::pow(temp,2) ;
}

// First derivative of SN wrt mu
arma::mat d1mu(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = z/sigma(i) - (nu(i)/sigma(i))*zeta1(w);
  }
  return out ;
}

// First derivative of SN wrt sigma
arma::mat d1sg(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = (-zeta1(w)%w)/sigma(i) + (arma::pow(z,2)-1.0)/sigma(i);
  }
  return out ;
}

// First derivative of SN wrt nu
arma::mat d1nu(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = zeta1(w) % z;
  }
  return out ;
}

// Second derivative of SN wrt mu (observed)
arma::mat d2mu(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = -std::pow(sigma(i),-2) + std::pow(nu(i)/sigma(i),2)*zeta2(w);
  }
  return out ;
}

// Second derivative of SN wrt sigma
arma::mat d2sg(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = std::pow(sigma(i),-2) * (1 - 3*arma::pow(z,2) + 2*w% zeta1(w) + arma::pow(w,2)%zeta2(w)) ;
  }
  return out ;
}

// Second derivative of SN wrt nu
arma::mat d2nu(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = zeta2(w) % arma::pow(z,2) ;
  }
  return out ;
}

// Cross-derivative of SN wrt mu-sigma
arma::mat dcms(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = std::pow(sigma(i),-2) * (-2*z + nu(i)*zeta1(w) + std::pow(nu(i),2)*zeta2(w) % z) ;
  }
  return out ;
}

// Cross-derivative of SN wrt mu-nu
arma::mat dcmn(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = -std::pow(sigma(i),-1) * zeta1(w) - w%zeta2(w)*std::pow(sigma(i),-1) ;
  }
  return out ;
}

// Cross-derivative of SN wrt sigma-nu
arma::mat dcsn(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    out.col(i) = -zeta1(w)%z/sigma(i) - zeta2(w)%w%z/sigma(i) ;
  }
  return out ;
}

// Second derivative of SN wrt mu (expected)
arma::mat d2muF(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    double i1 = arma::as_scalar(arma::mean(arma::pow(zeta1(w),2)));
    out.col(i).fill(-std::pow(sigma(i),-2) - std::pow(nu(i)/sigma(i),2)*i1) ;
  }
  return out ;
}

// Second derivative of SN wrt sigma (expected)
arma::mat d2sgF(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    double i1 = arma::as_scalar(arma::mean(arma::pow(z % zeta1(w),2)));
    out.col(i).fill(std::pow(sigma(i),-2) * (-2 - std::pow(nu(i),2)*i1)) ;
  }
  return out ;
}

// Second derivative of SN wrt nu (expected)
arma::mat d2nuF(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    double i1 = arma::as_scalar(arma::mean(arma::pow(z % zeta1(w),2)));
    out.col(i).fill(-i1);
  }
  return out ;
}

// Cross-derivative of SN wrt mu-sigma (expected)
arma::mat dcmsF(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  double log2pi = std::sqrt(2/M_PI);
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    double i1 = arma::as_scalar(arma::mean(z % arma::pow(zeta1(w),2)));
    double i2 = std::pow(sigma(i),-2) * (-log2pi*nu(i)*(1+
                         2*std::pow(nu(i),2))/std::pow(1+std::pow(nu(i),2),1.5) - std::pow(nu(i),2)*i1) ;
    out.col(i).fill(i2);
  }
  return out ;
}

// Cross-derivative of SN wrt mu-nu (expected)
arma::mat dcmnF(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  double log2pi = std::sqrt(2/M_PI);
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    double i1 = arma::as_scalar(arma::mean(z % arma::pow(zeta1(w),2)));
    double i2 = 1/sigma(i)*(-log2pi/std::pow(1+std::pow(nu(i),2),1.5) + nu(i)*i1) ;
    out.col(i).fill(i2);
  }
  return out ;
}

// Cross-derivative of SN wrt sigma-nu (expected)
arma::mat dcsnF(arma::vec& y, arma::vec& mu, arma::vec& sigma, arma::vec& nu){
  int n = y.n_elem, tqp = mu.n_elem;
  arma::mat out(n,tqp);
  arma::vec z(tqp), w(tqp);
  for(int i = 0; i < tqp; i++){
    z = (y - mu(i))/sigma(i);
    w = nu(i) * z;
    double i1 = arma::as_scalar(arma::mean(arma::pow(z % zeta1(w),2)));
    double i2 = nu(i)*i1/sigma(i) ;
    out.col(i).fill(i2);
  }
  return out ;
}

// Log-density of SN (direct parameterization)
double dsn(double& x, double& mu, double& sigma, double& nu){
  double z = (x-mu)/sigma ;
  double d1 = -std::log(2*M_PI)/2 - std::log(sigma) - std::pow(z,2)/2 ;
  double d2 = R::pnorm(nu*z,0,1,1,1) + std::log(2) ;
  return d1 + d2 ;
}

// Random draw from SN (direct parameterization)
double rsn(double& mu, double& sigma, double& nu){
  double delta = nu/std::sqrt(1+std::pow(nu,2));
  arma::vec tn(2);
  for(int i = 0; i < 2; i++){
    tn(i) = R::rnorm(0.0,1.0);
  }
  double z = delta * std::abs(tn(0)) + std::sqrt(1-std::pow(delta,2)) * tn(1) ;
  return mu + sigma*z  ;
}

// Maps nu_c (centered) to nu_d (direct) of SN distribution
arma::vec nc2nd(arma::vec& nuc){
  double log2pi = std::sqrt(2/M_PI);
  double OoT = 1.0/3.0 ;
  arma::vec R = arma::pow((2*arma::abs(nuc))/(4-M_PI),OoT) % arma::sign(nuc);
  return R / arma::sqrt(std::pow(log2pi,2) - (1-std::pow(log2pi,2)) * arma::square(R));
}

// Maps sg_c (centered) to sg_d (direct) of SN distribution
arma::vec sc2sd(arma::vec& sgc, arma::vec& nuc){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec nud = nc2nd(nuc) ;
  arma::vec R = nud / arma::sqrt(1+arma::pow(nud,2)) ;
  return sgc / arma::sqrt(1-std::pow(log2pi,2) * arma::square(R)) ;
}

// Maps mu_c (centered) to mu_d (direct) of SN distribution
arma::vec mc2md(arma::vec& muc, arma::vec& sgc, arma::vec& nuc){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec nud = nc2nd(nuc) ;
  arma::vec R = nud / arma::sqrt(1+arma::square(nud)) ;
  arma::vec sgd = sc2sd(sgc,nuc);
  return muc - log2pi * sgd % R;
}

// Maps (0,1) of proposed SN parametrization to nu_c (centered)
arma::vec p2nuc(arma::vec& p){
  double g1max = std::sqrt(2)*(4-M_PI)/std::pow(M_PI-2,1.5);
  return g1max*(2*p - 1);
}

// Jacobian d sigma_d / d sigma_c in the centred-to-direct SN reparametrization.
arma::vec dsddsc(arma::vec& nuc){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec nud = nc2nd(nuc);
  arma::vec R = nud / arma::sqrt(1+arma::pow(nud,2)) ;
  return 1 / arma::sqrt(1-std::pow(log2pi,2) * arma::square(R)) ;
}

// Jacobian d nu_d / d nu_c in the centred-to-direct SN reparametrization.
arma::vec dnddnc(arma::vec& nuc){
  double a = 2/(3*(4-M_PI));
  double log2pi = std::sqrt(2/M_PI);
  double OoT = 1.0/3.0 ;
  arma::vec R = arma::pow((2*arma::abs(nuc))/(4-M_PI),OoT) % arma::sign(nuc);
  arma::vec t = arma::sqrt(std::pow(log2pi,2) - (1-std::pow(log2pi,2)) * arma::square(R));
  return a * (1/(t % arma::square(R)) + (1-std::pow(log2pi,2))/arma::pow(t,3));
}

// Jacobian d mu_d / d sigma_c in the centred-to-direct SN reparametrization.
arma::vec dmddsc(arma::vec& nuc){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec nud = nc2nd(nuc);
  arma::vec R = nud / arma::sqrt(1+arma::square(nud)) ;
  return - (log2pi * R) / arma::sqrt(1 - std::pow(log2pi,2) * arma::square(R)) ;
}

// Jacobian d mu_d / d nu_c in the centred-to-direct SN reparametrization.
arma::vec dmddnc(arma::vec& sgc, arma::vec& nuc){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec nud = nc2nd(nuc);
  arma::vec R = nud / arma::sqrt(1+arma::square(nud)) ;
  return - (sgc % (log2pi * R)) / (3*arma::sqrt(1 - std::pow(log2pi,2) * arma::square(R)) % nuc) ;
}

// Jacobian d sigma_d / d nu_c in the centred-to-direct SN reparametrization
arma::vec dsddnc(arma::vec& sgc, arma::vec& nuc){
  double log2pi = std::sqrt(2/M_PI);
  arma::vec nud = nc2nd(nuc);
  arma::vec R = nud / arma::sqrt(1+arma::square(nud)) ;
  arma::vec mz = log2pi * R ;
  arma::vec sz = arma::sqrt(1 - std::pow(log2pi,2) * arma::square(R)) ;
  arma::vec d1 = -sgc / arma::square(sz);
  arma::vec d2 = -mz/sz % (log2pi / arma::pow(1+arma::square(nud), 1.5)) ;
  arma::vec d3 = dnddnc(nuc);
  return d1 % d2 % d3;
}

// Posterior mean of Zm
arma::mat Ezm(int q,
              arma::mat& Qpi,
              arma::vec& Qw,
              arma::mat& pD){

  int n = pD.n_rows;
  int tqp = pD.n_cols;

  arma::mat zm(n,q);
  for(int m = 0; m < n; m++){
    for(int i = 0; i < tqp; i++){
      zm.row(m) += Qpi.row(i) * arma::as_scalar(pD(m,i)) * arma::as_scalar(Qw(i)) ;
    }
  }
  return zm ;
}

// Posterior covariance of Zm
arma::cube Vzm(int q,
               arma::mat& Qpi,
               arma::vec& Qw,
               arma::mat& pD,
               arma::mat& EZM){

  int n = pD.n_rows;
  int tqp = pD.n_cols;

  arma::cube Vm(q,q,n);
  for(int m = 0; m < n; m++){
    for(int i = 0; i < tqp; i++){
      arma::rowvec tmp = Qpi.row(i) - EZM.row(m) ;
      Vm.slice(m) += (tmp.t()*tmp * arma::as_scalar(pD(m,i)) * arma::as_scalar(Qw(i)));
    }
  }
  return Vm;
}

// Sum over m the quadratic form tr(G * Vm) + E[z_m]' G E[z_m] in SigmaGrad
double EVzm(arma::mat& G,
            arma::mat& EZM,
            arma::cube& VZM){
  int n = EZM.n_rows;
  double out = 0.0 ;
  for(int m = 0; m < n; m++){
    out += arma::trace(G*VZM.slice(m)) + arma::as_scalar(EZM.row(m)*G*EZM.row(m).t());
  }
  return out ;
}

// Log-density of N(0, S) evaluated at each row of y; used inside fZY.
arma::vec log_mvnN(arma::mat& y,
                   arma::mat& S){

  int tqp = y.n_rows ;
  int q = y.n_cols ;
  double lS;
  double sign;
  bool ok = arma::log_det(lS, sign, S);

  arma::vec out(tqp);

  if(ok){
    arma::mat iS = arma::inv_sympd(S);
    for(int m = 0; m < tqp ; m++){
      out(m) = -0.5*(q*std::log(2*M_PI) + lS + arma::as_scalar(y.row(m)*(iS*y.row(m).t()))) ;
    }
  }
  return out;
}

// Projection step for the Cholesky factor L
void fixL(arma::mat& L, arma::mat& R){
  int q = L.n_rows;
  arma::vec dR = R.diag();
  for(int i = 0; i < q; i++){
    if(dR(i) != 1.0){
      continue;
    } else {
      double cross = arma::dot(L.row(i),L.row(i));
      L.row(i) = L.row(i)/std::sqrt(cross);
    }
  }
}

// Maps the free lower-triangular entries of L to the corresponding entries of
// R = LL' (for numerical Jacobian use).
Rcpp::NumericVector chol2cor(Rcpp::NumericVector& vLi, int& q){
  int ncor = vLi.length();
  arma::vec vL(vLi.begin(),ncor,false,false);
  arma::mat L = eye(q,q);
  arma::uvec Li = arma::trimatl_ind(arma::size(L));
  L(Li) = vL;
  arma::mat R = L*L.t();
  arma::uvec Ri = arma::trimatl_ind(arma::size(R));
  arma::vec out = R(Ri);
  return Rcpp::wrap(out);
}

// Numerical Jacobian of chol2cor, used in SEs computation for R
Rcpp::NumericMatrix Jchol(Rcpp::NumericVector& vLi,
                          int q) {

  Rcpp::Environment numDerivR = Rcpp::Environment::namespace_env("numDeriv");
  Rcpp::Function jacobian = numDerivR["jacobian"];
  Rcpp::NumericMatrix out = jacobian(Rcpp::_["func"] = Rcpp::InternalFunction(chol2cor),
                                     Rcpp::_["x"] = vLi,
                                     Rcpp::_["q"] = q);

  return out;
}

// Fix user-specified sign restrictions on the first free loading of each latent variable
// flipping signs in b
void fixb(arma::cube& b,
          arma::cube& rb,
          Rcpp::Nullable<Rcpp::NumericVector> c2fi = R_NilValue){
  if(c2fi.isNotNull()){
    arma::vec  c2f = Rcpp::as<arma::vec>(c2fi);
    arma::uvec c2fp = arma::conv_to<arma::uvec>::from(arma::abs(c2f));
    arma::vec  c2fs = arma::sign(c2f);
    int nq = c2f.size();
    for(int i = 0; i < nq; i++){
      arma::uvec idx = arma::find_nan(rb.col(c2fp(i)));
      if(idx.size() == 0){
        continue;
      } else{
         if(arma::sign(b(idx(0), c2fp(i), 0)) !=  c2fs(i)){
          b.col(c2fp(i)) *= -1.0;
        }
      }
    }
  }
}

// Draw from Multivariate Normal distribution
arma::mat rMVnorm(const int n, arma::vec& mu, arma::mat& R){
 const int q = R.n_cols;
 arma::mat Y(n,q), eigm(q,q);
 for(int i = 0; i < n; i++){
  for(int j = 0; j < q; j++){
   Y(i,j) = R::rnorm(0,1);
  }
 }
 arma::vec eigv(q);
 arma::eig_sym(eigv,eigm,R);
 arma::mat teigm = eigm.t();
 teigm.each_col() %= arma::sqrt(eigv);
 return arma::repmat(mu, 1, n).t() + Y * (eigm * teigm);
}
