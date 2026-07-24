#include <RcppArmadillo.h>
#include "utils_basic.h"
#include "utils_sa_basic.h"
#include "dist_sa_class.h"
#include "utils_sa_estimation.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Score (scaled by Hessian) for b
arma::vec dB(Dist_sa& Ydist,
             arma::cube& etas,
             arma::cube& rb,
             const bool flagEIM,
             double& ssB){

  int p = rb.n_rows ;
  arma::mat nbs = count_b(rb);
  int npars = arma::accu(nbs);

  arma::mat ZX = Ydist.get_ZX();

  arma::vec sc(npars);
  int mem = 0 ;

  for(int i = 0; i < p; i++){
    arma::mat d1Yi = Ydist.d1y_sa(i, etas);
    arma::mat d2Yi = Ydist.d2y_sa(i, etas, flagEIM, d1Yi);
    int D = d1Yi.n_cols ;
    for(int d = 0; d < D; d++){
      arma::vec tmprb = rb.slice(d).row(i).t() ;
      arma::uvec idx1 = arma::find_nan(tmprb);
      arma::mat ZXi1 = ZX.each_col() % d1Yi.col(d);
      arma::mat ZXi2 = ZX.each_col() % d2Yi.col(d);
      ZXi1 = ZXi1.cols(idx1);
      ZXi2 = ZXi2.cols(idx1);
      arma::vec tmpsc = (arma::sum(ZXi1,0)).t();
      arma::vec tmphs = (1/(arma::sum(ZXi2,0))).t();
      arma::uvec idx2 = arma::regspace<arma::uvec>(mem, (mem + nbs(i,d) - 1));
      sc(idx2) = tmpsc % tmphs ;
      mem += nbs(i,d);
    }
  }
  return sc * ssB;
}

arma::mat newL(arma::vec& mu, arma::mat& L, arma::mat& rR, arma::mat& Z, double& ss){
  const int n = Z.n_rows;
  const int q = mu.n_elem;
  arma::uvec idL = arma::trimatl_ind(arma::size(L));
  arma::vec vL = L(idL);
  arma::uvec iL = arma::regspace<arma::uvec>(0,idL.n_elem-1);

  idL = idL(iL);
  vL = vL(iL);
  const int np = vL.size();
  arma::vec gL(np);
  arma::mat Ri = L*L.t();
  arma::mat iR = arma::inv_sympd(Ri);

  arma::mat Zc = Z.each_row() - mu.t();

  for(int i = 0; i < np; i++){
    arma::uword iLi = idL(i);
    arma::mat Djk(q,q);
    Djk(iLi) = 1;
    arma::mat Gjk = iR*Djk*L.t()*iR;
    double g1 = -n*arma::trace(L.t()*iR*Djk);
    double g2 = 0;
    for(int m = 0; m < n; m++){
      g2 += arma::as_scalar((Zc.row(m))*Gjk*(Zc.row(m)).t());
    }
    gL(i) = g1 + g2;
  }
  vL += ss*gL;
  arma::mat Lnew(L);
  Lnew(idL) = vL;
  Lnew = ProxL(Lnew, rR);
  return(Lnew);
}

void newpMR(arma::mat& Z, arma::mat& posM, arma::cube& posR, int iter){
  const int n = Z.n_rows;
  const int p = Z.n_cols;
  arma::rowvec diff_row(p);
  arma::mat outer_prod(p, p);
  double t1 = 1.0/(iter + 1.0);
  double t2 = iter/(iter + 1.0);
  double t3 = iter/std::pow(iter + 1.0,2);
  for(int i = 0; i < n; i++) {
    diff_row = Z.row(i) - posM.row(i);
    outer_prod = diff_row.t() * diff_row;
    posR.slice(i) = t3 * outer_prod + t2 * posR.slice(i);
  }
  posM += t1 * (Z - posM);
}

arma::mat fyz(Dist_sa& Ydist, arma::cube& etas){
  const int p = Ydist.get_p();
  const int n = Ydist.get_n();
  arma::mat out(n,p,arma::fill::zeros);
  for(int i = 0; i < p; i++){
    out.col(i) = Ydist.dy_sa(i, etas) ;
  }
  return(out);
}

arma::mat fz(arma::mat& Z, arma::vec& mu, arma::mat& R){
  const int n = Z.n_rows;
  arma::vec out(n);
  for(int i = 0; i < n; i++){
    arma::vec zrow = Z.row(i).t();
    out(i) = std::max(arma::datum::log_min, dMVnorm(zrow, mu, R, true)) ;
  }
  return(out);
}

arma::mat newZ_RWMH(Dist_sa& Ydist, arma::cube& etas, arma::mat& Z, arma::mat& X,
                    arma::cube& b, arma::vec& mu, arma::mat& L, double& ar,
                    const double h, const arma::mat fmat){
  const int n = Ydist.get_n();
  const int q = L.n_cols;
  arma::mat R = L*L.t();
  arma::mat Y = Ydist.get_Y();
  Rcpp::CharacterVector famv = Ydist.get_fam();

  arma::mat hD = h*arma::eye(arma::size(L));
  arma::vec mu0(q);
  arma::mat rE = rMVnorm(n,mu0,hD);
  arma::mat Zn = Z + rE;
  arma::mat ZXn = make_ZX(Zn,X,fmat);
  arma::cube etasn = make_etas(ZXn,b);
  // Dist_sa Ydistn(Y,famv,ZXn,b);

  arma::vec pdo = arma::sum(fyz(Ydist,etas),1) + arma::sum(fz(Z,mu,R),1);
  arma::vec pdn = arma::sum(fyz(Ydist,etasn),1) + arma::sum(fz(Zn,mu,R),1);

  arma::mat Zout(arma::size(Z));
  for(int i = 0; i < n; i++){
    double rej = R::runif(0,1);
    double com = std::min(1.0,std::exp(pdn(i) - pdo(i)));
    if(rej < com){
      Zout.row(i) = Zn.row(i);
      ar++;
    } else {
      Zout.row(i) = Z.row(i);
    }
  }
  return Zout;
}

