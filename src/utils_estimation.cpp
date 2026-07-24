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

// Computes score vector wrt b in EM step
arma::vec d1ll(Rcpp::NumericMatrix& Yr,
               Rcpp::CharacterVector& famr,
               ghQ& Q,
               arma::cube& b,
               arma::cube& rb,
               arma::mat& pD){

  int p = Yr.ncol(), n = Yr.nrow() ;
  arma::mat Yc(Yr.begin(),n,p,false);
  arma::mat Qp = Q.get_grid() ;
  arma::vec Qw = Q.get_weig() ;
  // int tqp = Qp.n_rows ;

  arma::mat nbs = count_b(rb);
  int npars = arma::accu(nbs);

  arma::vec sc(npars);
  int mem = 0;

  for(int i = 0; i < p; i++){
    std::string fam = Rcpp::as<std::string>(famr(i));
    Dist Y(fam) ;
    arma::cube d1Yi = Y.d1y(i,Yc,Qp,b);
    int K = d1Yi.n_slices;
    for(int kk = 0; kk < K; kk++){
      arma::mat tmprb = rb.slice(kk);
      arma::mat tmpM = d1Yi.slice(kk);
      tmpM.replace(arma::datum::nan, 0.0);
      arma::rowvec vec1 = arma::sum(tmpM % pD, 0);
      arma::uvec idx1 = arma::find_nan(tmprb.row(i));
      arma::vec vec2 = ((vec1.t() % Qw).t() * Qp.cols(idx1)).t() ;
      arma::uvec idx2 = arma::regspace<arma::uvec>(mem, (mem + nbs(i,kk) - 1));
      sc(idx2) = vec2 ;
      mem += nbs(i,kk);
    }
  }
  return sc;
}

// Computes Hessian wrt b in EM step
// (Currently not in use)
arma::mat d2ll_EM(Rcpp::NumericMatrix& Yr,
                  Rcpp::CharacterVector& famr,
                  ghQ& Q,
                  arma::cube& b,
                  arma::cube& rb,
                  arma::mat& pD,
                  const bool flagEIM){

  int p = Yr.ncol(), n = Yr.nrow() ;
  arma::mat Yc(Yr.begin(),n,p,false);
  arma::mat Qp = Q.get_grid() ;
  arma::vec Qw = Q.get_weig() ;
  int tqp = Qp.n_rows ;

  arma::mat nbs = count_b(rb);
  int npars = arma::accu(nbs);

  arma::mat hess(npars, npars);
  int mem1 = 0, mem2 = 0;

  for(int i = 0; i < p; i++){
    std::string fam = Rcpp::as<std::string>(famr(i));
    Dist Y(fam) ;
    arma::cube d1Yi, dCYi ;
    if(!flagEIM){
      d1Yi = Y.d1y(i,Yc,Qp,b);
    }
    arma::cube d2Yi = Y.d2y(i,Yc,Qp,b,flagEIM,d1Yi);
    int K = d2Yi.n_slices;
    if(K > 1){
      dCYi = Y.dCy(i,Yc,Qp,b,flagEIM);
    }

    for(int k1 = 0; k1 < K; k1++){
      arma::mat rbk1 = rb.slice(k1);
      arma::uvec idk1 = arma::find_nan(rbk1.row(i));
      arma::mat Qpk1 = Qp.cols(idk1) ;
      arma::uvec id1 = arma::regspace<arma::uvec>(mem1, (mem1 + nbs(i,k1) - 1));

      arma::mat dYk1 = d2Yi.slice(k1);
      arma::rowvec veck1a(tqp) ;
      for(int jj = 0; jj < n; jj++){
        if(std::isnan(dYk1(jj,0))){
          continue;
        } else {
          veck1a += dYk1.row(jj) % pD.row(jj) ;
        }
      }
      arma::vec veck1b = (veck1a.t() % Qw) ;

      for(int k2 = k1; k2 < K; k2++){
        if(k2 == k1){
          hess(id1,id1) += Qpk1.t() * arma::diagmat(veck1b) * Qpk1;
          mem1 += nbs(i,k2);
          mem2 = mem1;
        } else {
          arma::mat rbk2 = rb.slice(k2);
          arma::uvec idk2 = arma::find_nan(rbk2.row(i));
          arma::mat Qpk2 = Qp.cols(idk2) ;
          arma::uvec id2 = arma::regspace<arma::uvec>(mem2, (mem2 + nbs(i,k2) - 1));

          arma::uword idH = ixd2ll(k1,k2,K);
          arma::mat dYk2 = dCYi.slice(idH);
          arma::rowvec veck2a(tqp) ;
          for(int jj = 0; jj < n; jj++){
            if(std::isnan(dYk2(jj,0))){
              continue;
            } else {
              veck2a += dYk2.row(jj) % pD.row(jj) ;
            }
          }
          arma::vec veck2b = (veck2a.t() % Qw) ;
          hess(id1,id2) += Qpk1.t()*arma::diagmat(veck2b)*Qpk2;
          hess(id2,id1) += Qpk2.t()*arma::diagmat(veck2b)*Qpk1;
          mem2 += nbs(i,k2);
        }
      }
    }
  }
  return hess;
}

// Single Newton/Fisher-scoring update for b in EM step
arma::vec solve_EM(Rcpp::NumericMatrix& Yr,
                   Rcpp::CharacterVector& famr,
                   ghQ& Q,
                   arma::cube& b,
                   arma::cube& rb,
                   arma::mat& pD,
                   const bool flagEIM){

  int p = Yr.ncol(), n = Yr.nrow() ;
  arma::mat Yc(Yr.begin(),n,p,false);
  arma::mat Qp = Q.get_grid() ;
  arma::vec Qw = Q.get_weig() ;

  arma::mat nbs = count_b(rb);
  int npars = arma::accu(nbs);

  arma::vec sc(npars);
  arma::mat hess(npars, npars);
  int mem1 = 0, mem2 = 0;

  for(int i = 0; i < p; i++){
    std::string fam = Rcpp::as<std::string>(famr(i));
    Dist Y(fam) ;
    arma::cube d1Yi = Y.d1y(i,Yc,Qp,b);
    arma::cube d2Yi = Y.d2y(i,Yc,Qp,b,flagEIM,d1Yi);
    arma::cube dCYi ;
    int K = d1Yi.n_slices;
    if(K > 1){
      dCYi = Y.dCy(i,Yc,Qp,b,flagEIM);
    }

    for(int k1 = 0; k1 < K; k1++){
      arma::mat rbk1 = rb.slice(k1);
      arma::uvec idk1 = arma::find_nan(rbk1.row(i));
      arma::mat Qpk1 = Qp.cols(idk1) ;
      arma::uvec id1 = arma::regspace<arma::uvec>(mem1, (mem1 + nbs(i,k1) - 1));

      arma::mat dYk0 = d1Yi.slice(k1);
      arma::mat dYk1 = d2Yi.slice(k1);
      dYk0.replace(arma::datum::nan, 0.0);
      dYk1.replace(arma::datum::nan, 0.0);
      arma::rowvec veck0a = arma::sum(dYk0 % pD, 0);
      arma::rowvec veck1a = arma::sum(dYk1 % pD, 0);
      arma::vec veck0b = (veck0a.t() % Qw) ;
      arma::vec veck1b = (veck1a.t() % Qw) ;

      for(int k2 = k1; k2 < K; k2++){
        if(k2 == k1){
          sc(id1) += Qpk1.t()*veck0b;
          hess(id1,id1) += Qpk1.t()*arma::diagmat(veck1b)*Qpk1;
          mem1 += nbs(i,k2);
          mem2 = mem1;
        } else {
          arma::mat bk2 = b.slice(k2);
          arma::mat rbk2 = rb.slice(k2);
          arma::uvec idk2 = arma::find_nan(rbk2.row(i));
          arma::mat Qpk2 = Qp.cols(idk2) ;
          arma::uvec id2 = arma::regspace<arma::uvec>(mem2, (mem2 + nbs(i,k2) - 1));

          arma::uword idH = ixd2ll(k1,k2,K);
          arma::mat dYk2 = dCYi.slice(idH);
          dYk2.replace(arma::datum::nan, 0.0);
          arma::rowvec veck2a = arma::sum(dYk2 % pD, 0);
          arma::vec veck2b = (veck2a.t() % Qw) ;
          hess(id1,id2) += Qpk1.t()*arma::diagmat(veck2b)*Qpk2;
          hess(id2,id1) += Qpk2.t()*arma::diagmat(veck2b)*Qpk1;
          mem2 += nbs(i,k2);
        }
      }
    }
  }
  return arma::solve(hess,sc, solve_opts::fast + solve_opts::likely_sympd) ;
}

// Gradient wrt L (Cholesky decomp of Sigma)
arma::mat SigmaGrad(ghQ& Q, fYZ& fyz,
                    arma::uvec& Li, arma::mat& L){

  arma::mat Sig = Q.get_sigm();
  arma::mat Qpi = Q.get_poin();
  arma::vec Qw = Q.get_weig();
  arma::mat pD = fyz.get_pD();

  int q = Sig.n_rows;
  int n = pD.n_rows;

  arma::mat EZm = Ezm(q,Qpi,Qw,pD);
  arma::cube VZm = Vzm(q,Qpi,Qw,pD,EZm);

  int ncor = Li.n_elem;

  arma::mat iS = arma::inv_sympd(Sig);
  arma::vec grad(ncor) ;

  for(int i = 0; i < ncor; i++){
    arma::uword ii = Li(i);
    arma::mat D(q,q,arma::fill::zeros);
    D(ii) = 1.0 ;
    arma::mat SiD = arma::solve(Sig,D);
    arma::mat G = SiD*L.t()*iS;
    grad(i) = -n*arma::trace(L.t()*SiD) + EVzm(G,EZm,VZm);
  }
  arma::mat out(grad);
  return out ;
}

// Expected complete-data log-prior sum (used as objective for L-update)
double fZY(ghQ& Q, fYZ& fyz){

  arma::mat Sig = Q.get_sigm();
  arma::mat Qpi = Q.get_poin();
  arma::vec Qw = Q.get_weig();
  arma::mat pD = fyz.get_pD();

  int n = pD.n_rows ;
  double out = 0.0 ;
  arma::vec logQpS = log_mvnN(Qpi,Sig);
  for(int m = 0; m < n; m++){
    out += arma::accu(logQpS % pD.row(m).t() % Qw);
  }
  return out;
}

// L-BFGS wrapper for L; used by update_sigm
class opL {
  // Private class members
private:
  ghQ& Q ;
  fYZ& fyz;
  arma::mat rR;
  double objf;
  arma::mat SI, L;
  arma::uvec Li;
  arma::vec tvL;

public:
  // Constructor
  opL(ghQ& Qi, fYZ& fyzi, arma::mat& rRi) : Q(Qi), fyz(fyzi), rR(rRi), objf(0.0) {
    SI = Q.get_sigm();
    L = arma::chol(SI,"lower");
    Li = arma::trimatl_ind(arma::size(L));
    tvL = L(Li);
  }

  double EvaluateWithGradient(const arma::mat& vL, arma::mat& grad){

    Rcpp::checkUserInterrupt();
    tvL = vL;
    L(Li) = tvL;
    fixL(L,rR);
    SI = L*L.t() ;
    Q.set_sigm(SI);
    grad = -SigmaGrad(Q,fyz,Li,L) ;
    objf = fZY(Q,fyz);
    return -1.0*objf;
  }

  double get_fzy(){
    return objf;
  }

  arma::vec get_vL(){
    return tvL;
  }

};

// Updates Sigma = LL.t() via L-BFGS
void update_sigm(ghQ& Q, fYZ& fyz, arma::mat& rR){

  opL optL(Q,fyz,rR);
  arma::vec vL = optL.get_vL();
  ens::L_BFGS lbfgsL;
  lbfgsL.Factr() = std::sqrt(arma::datum::eps);
  lbfgsL.Optimize(optL, vL);
  Q.update();
}

// Updates Sigma = LL.t() via GD (single step)
void update_sigmGD(ghQ& Q, fYZ& fyz, arma::mat& rR, double& SS){

  arma::mat Sig = Q.get_sigm();
  arma::mat L = arma::chol(Sig,"lower");
  arma::uvec Li = arma::trimatl_ind(arma::size(L));
  arma::vec vL = L(Li);
  arma::uvec idx = arma::regspace<arma::uvec>(1,Li.n_elem-1);
  Li = Li(idx);
  vL = vL(idx);

  arma::vec gradR = SigmaGrad(Q,fyz,Li,L);
  int countEMr = 0;
  do{
    vL += SS*gradR;
    countEMr ++;
  } while (countEMr < 6);

  L(Li) = vL;
  fixL(L, rR);
  Sig = L*L.t() ;
  Q.set_sigm(Sig);
  Q.update();
}

// Computes SEs via outer product of gradients
Rcpp::List SEs(Rcpp::NumericMatrix& Yr,
               Rcpp::CharacterVector& famr,
               ghQ& Q,
               arma::cube& b,
               arma::cube& rb,
               arma::mat& rR,
               arma::mat& pD,
               bool FLAGCORLV){

  int p = Yr.ncol(), n = Yr.nrow() ;
  arma::mat Yc(Yr.begin(),n,p,false);
  arma::mat Qp = Q.get_grid() ;
  arma::vec Qw = Q.get_weig() ;
  int q = Q.get_q() ;

  arma::mat nbs = count_b(rb);
  int npars = arma::accu(nbs), ncor = 0 ;

  arma::mat SSt(npars,npars,arma::fill::zeros) ;
  arma::mat sc(npars,n);
  sc.fill(arma::datum::nan);
  arma::mat grad, scout, L;
  arma::vec vL;
  int mem = 0;

  for(int i = 0; i < p; i++){
    std::string fam = Rcpp::as<std::string>(famr(i));
    Dist Y(fam) ;
    arma::cube d1Yi = Y.d1y(i,Yc,Qp,b);
    int K = d1Yi.n_slices;
    for(int k = 0; k < K; k++){
      arma::mat tmprb = rb.slice(k);
      arma::mat tmpM = d1Yi.slice(k);
      arma::uvec idx1 = arma::find_nan(tmprb.row(i));
      arma::mat Zk = Qp.cols(idx1);
      arma::mat mat1 = tmpM % pD;
      mat1.each_row() %= Qw.t();
      arma::mat mat2 = (mat1 * Zk).t();
      sc(arma::span(mem, mem + nbs(i,k) - 1), arma::span::all) = mat2;
      mem += nbs(i,k);
    }
  }

  if(!FLAGCORLV){
    scout = sc;
  }
  else{
    arma::mat Sig = Q.get_sigm();
    arma::mat Qpi = Q.get_poin();

    L = arma::chol(Sig,"lower");
    arma::uvec Li = arma::trimatl_ind(arma::size(L));
    vL = L(Li);

    arma::mat EZm = Ezm(q,Qpi,Qw,pD);
    arma::cube VZm = Vzm(q,Qpi,Qw,pD,EZm);

    ncor = vL.n_elem;

    arma::mat iS = arma::inv_sympd(Sig);
    grad.zeros(ncor,n) ;
    scout.zeros(npars + ncor,n);

    for(int i = 0; i < ncor; i++){
      arma::uword ii = Li(i);
      arma::mat D(q,q,arma::fill::zeros);
      D(ii) = 1.0 ;
      arma::mat SiD = arma::solve(Sig,D);
      arma::mat G = SiD*L.t()*iS;
      for(int m = 0; m < n; m++){
        grad(i,m) = -n*arma::trace(L.t()*SiD) + arma::trace(G*VZm.slice(m)) + arma::as_scalar(EZm.row(m)*G*EZm.row(m).t());
      }
    }
    scout = arma::join_cols(sc,grad);
    SSt.set_size(npars+ncor,npars+ncor);
    SSt.zeros();
  }

  scout.replace(arma::datum::nan, 0.0);
  SSt = scout * scout.t();
  arma::uvec idxb = arma::regspace<arma::uvec>(0, npars-1);
  arma::mat iSStb = arma::inv(SSt(idxb,idxb));

  arma::vec seb = arma::sqrt(arma::diagvec(iSStb));

  if(FLAGCORLV){
    arma::cube SEb(arma::size(b),arma::fill::zeros);
    SEb = Vb2Cb(seb,SEb,rb);

    arma::mat SER(q,q, arma::fill::zeros);
    arma::uvec Ri = arma::trimatl_ind(arma::size(SER));
    arma::uvec idxR = arma::find_nan(rR(Ri));

    int nPhi = q*(q+1)/2;
    arma::uvec idxL = arma::regspace<arma::uvec>(npars, npars+ncor-1);
    arma::mat iSStL = arma::inv(SSt(idxL,idxL));

    Rcpp::NumericVector vLtemp = Rcpp::NumericVector(vL.begin(),vL.end());
    Rcpp::NumericMatrix dLdRtemp = Jchol(vLtemp,q);
    arma::mat dLdR(dLdRtemp.begin(),nPhi,ncor,false,false);

    arma::mat SEPhi = dLdR*iSStL*dLdR.t();
    arma::vec SEphi = arma::sqrt(arma::diagvec(SEPhi));
    SER(Ri(idxR)) = SEphi(idxR);
    SER = (SER + SER.t()) - arma::diagmat(SER);

    return Rcpp::List::create(Rcpp::Named("seb") = SEb,
                              Rcpp::Named("seR") = SER,
                              Rcpp::Named("K") = iSStb.n_rows + idxR.n_elem) ;
  } else {
    arma::cube SEb(arma::size(b),arma::fill::zeros);
    SEb = Vb2Cb(seb,SEb,rb);
    return Rcpp::List::create(Rcpp::Named("seb") = SEb,
                              Rcpp::Named("K") = iSStb.n_rows) ;
  }
}

// EM-step updates on b + L-BFGS updates on Sigma (if corLV).
void EM_step(Rcpp::NumericMatrix& Yr,
             Rcpp::CharacterVector& famr,
             arma::cube& b,
             arma::cube& rb,
             arma::mat& rR,
             ghQ& Q,
             fYZ& fyz,
             Rcpp::List& control){

  int iL = control["EM.maxit"] ;
  int upR = control["updt.R.every"] ;
  bool flagFISHER = control["Fisher"], flagVERBO = control["verbose"];
  bool flagCORLV = control["estim.R"] ;
  double tol = control["tolerance"] ;
  Rcpp::Nullable<Rcpp::NumericVector> c2fi = control["sign.restr.b"];

  // int q = Q.get_q();
  arma::vec llK(iL + 1, fill::zeros) ;
  llK(0) = fyz.get_ll();
  arma::mat pD = fyz.get_pD();
  arma::vec Vb = Cb2Vb(b,rb);

  for(int i = 0; i < iL; i++){

    if (i % 2 == 0) Rcpp::checkUserInterrupt();
    Vb -= solve_EM(Yr,famr,Q,b,rb,pD,flagFISHER);
    b = Vb2Cb(Vb,b,rb);
    fixb(b,rb,c2fi);

    if(flagCORLV & (i % upR == 0)) update_sigm(Q,fyz,rR);
    // if(flagCORLV & (i % upR == 0)) update_sigmGD(Q,fyz,rR,LR);

    fyz.update(Yr,famr,Q,b);
    llK(i+1) = fyz.get_ll();
    pD = fyz.get_pD();
    double eps0 = llK(i+1) - llK(i);

    if(flagVERBO){
      if(i == 0) Rcpp::Rcout << "\n EM iteration (NR-update): " << i+1 << " (llk: " << std::to_string(llK(i+1)) << ")";
      else Rcpp::Rcout << "\r EM iteration (NR-update): " << i+1 << " (llk: " << std::to_string(llK(i+1)) << ")";
    }
    if(eps0 < 0){
      Rcpp::Rcout << "\n Attention!: No increase in log-likelihood with EM update (trying DM)." ;
      break;
    }
    if(eps0 < tol) break;
  }
}

// EM-step updates on b and Sigma (if corLV).
void EM_stepGD(Rcpp::NumericMatrix& Yr,
               Rcpp::CharacterVector& famr,
               arma::cube& b,
               arma::cube& rb,
               arma::mat& rR,
               ghQ& Q,
               fYZ& fyz,
               Rcpp::List& control){

  int iL = control["EM.maxit"] ;
  int upR = control["updt.R.every"] ;
  bool flagVERBO = control["verbose"], flagCORLV = control["estim.R"] ;
  double tol = control["tolerance"] ;
  double LR = control["LR"];
  Rcpp::Nullable<Rcpp::NumericVector> c2fi = control["sign.restr.b"];

  // int q = Q.get_q();
  arma::vec llK(iL + 1, fill::zeros) ;
  llK(0) = fyz.get_ll();
  arma::mat pD = fyz.get_pD();

  arma::cube bold = b;
  int i = 0, LRup = 0;

  while(i < iL){

    if (i % 2 == 0) Rcpp::checkUserInterrupt();
    arma::vec Vb = Cb2Vb(bold,rb);
    arma::vec gradB = d1ll(Yr,famr,Q,bold,rb,pD);
    int countEM = 0;
    do{
      Vb += LR*gradB;
      countEM ++;
    } while (countEM < 6);

    arma::cube bnew = Vb2Cb(Vb,b,rb);
    fixb(bnew,rb,c2fi);

    if(flagCORLV && ((i % upR) == 0)) update_sigmGD(Q,fyz,rR,LR);

    fyz.update(Yr,famr,Q,bnew);
    llK(i+1) = fyz.get_ll();
    pD = fyz.get_pD();
    double eps0 = llK(i+1) - llK(i);
    i++;

    if(eps0 < 0){
      i--;
      LRup++;
      LR = LR/2;
      if(LR < tol){
        b = bold ;
        break;
      }
      continue;
    }

    if(flagVERBO && ((i % 10) == 0) ){
      if(i == 10) Rcpp::Rcout << "\n EM iteration (GD-update): " << i << " (llk: " << std::to_string(llK(i)) <<", LR updates: " << LRup << ")";
      else Rcpp::Rcout << "\r EM iteration (GD-update): " << i << " (llk: " << std::to_string(llK(i)) <<", LR updates: " << LRup << ")";
    }
    bold = bnew;
    b = bold;
    if(eps0 < tol){
      b = bnew ;
      break;
    }
  }
}

// L-BFGS wrapper for b; used in DM_step
class mLLK {
  // Private class members
private:
  Rcpp::NumericMatrix& Yr;
  Rcpp::CharacterVector& famr ;
  ghQ& Q ;
  arma::cube& b ;
  arma::cube& rb ;
  fYZ& fyz;
  double llkO;

public:
  // Constructor
  mLLK(Rcpp::NumericMatrix& Yi,
       Rcpp::CharacterVector& fami,
       ghQ& Qi, fYZ& fyzi,
       arma::cube& bi, arma::cube& rbi) : Yr(Yi), famr(fami), Q(Qi), b(bi), rb(rbi), fyz(fyzi), llkO(0.0) {
    fyz.update(Yr,famr,Q,b);
  }

  double EvaluateWithGradient(const arma::mat& Vb, arma::mat& grad){

    Rcpp::checkUserInterrupt();
    // int q = Q.get_q();
    arma::vec tVb(Vb);
    b = Vb2Cb(tVb,b,rb);
    fyz.update(Yr,famr,Q,b);
    arma::mat pD = fyz.get_pD();
    arma::vec tgr1 = -d1ll(Yr,famr,Q,b,rb,pD);
    grad = vec2mat(tgr1,tgr1.n_elem,1);
    llkO = fyz.get_ll();
    return -1.0*llkO ;
  }

  double get_ll(){
    return llkO;
  }

};

// Direct maximization step via L-BFGS
void DM_step(Rcpp::NumericMatrix& Yr,
             Rcpp::CharacterVector& famr,
             arma::cube& b,
             arma::cube& rb,
             arma::mat& rR,
             ghQ& Q,
             fYZ& fyz,
             Rcpp::List& control){

  int iL = control["DM.maxit"] ;
  bool flagCORLV = control["estim.R"] , flagVERBO = control["verbose"];
  double tol = control["tolerance"] ;
  Rcpp::Nullable<Rcpp::NumericVector> c2fi = control["sign.restr.b"];

  arma::vec Vb = Cb2Vb(b,rb);

  if(!flagCORLV){

    mLLK llk(Yr,famr,Q,fyz,b,rb);
    ens::L_BFGS lbfgs;
    lbfgs.MaxIterations() = iL;
    lbfgs.MinGradientNorm() = std::sqrt(arma::datum::eps);

    if(flagVERBO) Rcpp::Rcout << "\n Direct MML estimation (L-BFGS) ... " ;
    lbfgs.Optimize(llk, Vb);
    fixb(b,rb,c2fi);
    if(flagVERBO) Rcpp::Rcout << "\r Direct MML estimation (L-BFGS) ... Converged! (llk: " << std::to_string(llk.get_ll()) << ")" ;

  } else {

    int sigup = 0;
    int Rit = control["R.maxit"];
    Rit += 1 ;
    double eps = 1, llk0 ;

    while((std::abs(eps) > tol) & (sigup < Rit)){

      llk0 = fyz.get_ll();
      mLLK llk(Yr,famr,Q,fyz,b,rb);
      ens::L_BFGS lbfgs;
      lbfgs.MaxIterations() = iL;
      lbfgs.Factr() = std::sqrt(arma::datum::eps);

      if(flagVERBO & (sigup == 0)) Rcpp::Rcout << "\n Direct MML estimation (L-BFGS, Phi updates: " << sigup << ") ... " ;
      lbfgs.Optimize(llk, Vb);
      fixb(b,rb,c2fi);
      update_sigm(Q,fyz,rR);
      fyz.update(Yr,famr,Q,b);
      eps = fyz.get_ll() - llk0;
      if(flagVERBO) Rcpp::Rcout << "\r Direct MML estimation (L-BFGS, Phi updates: " << sigup << ") ... Converged! (llk: " << std::to_string(llk.get_ll()) << ")" ;
      sigup ++;

    }
  }
}

// Factor scores (EAP)
arma::mat fscores(ghQ& Q, arma::mat& pD){

  arma::mat Qpi = Q.get_poin();
  arma::vec Qw = Q.get_weig();
  Qpi.each_col() %= Qw ;
  arma::mat scores = pD * Qpi;
  return scores;
}
