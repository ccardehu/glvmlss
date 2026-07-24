#ifndef dist_SA_Class_H
#define dist_SA_Class_H

class Dist_sa {

private:

  // Private helper functions
  // Private class members
  arma::mat Yr;
  Rcpp::CharacterVector famv;
  arma::mat ZX;
  arma::cube b, etas;

public:

  // Constructors
  Dist_sa(arma::mat& Yri, Rcpp::CharacterVector& fami, arma::mat& ZXi,
          arma::cube& bi) : Yr(Yri), famv(fami), ZX(ZXi), b(bi) {
  };

  // Public helper functions

  // Vector of (n) entries: log f_i(y_m | z_m, x_m)
  arma::vec dy_sa(const int i, arma::cube& etas);

  // 1st derivative matrix (n x D): d log f(y_m | z_m, x_m) / d eta_d
  arma::mat d1y_sa(const int i, arma::cube& etas);

  // 2nd derivative matrix (n x D): d^2 logf_i(y_m | z_m, x_m) /(d eta_d)^2
  arma::mat d2y_sa(const int i,
                   arma::cube& etas,
                   const bool flagEIM,
                   arma::mat& d1Yi);

  arma::mat get_ZX(){
    return ZX;
  }

  const int get_p(){
    return Yr.n_cols ;
  }

  const int get_n(){
    return Yr.n_rows ;
  }

  arma::mat get_Y(){
    return Yr;
  }

  Rcpp::CharacterVector get_fam(){
    return famv;
  }

};

#endif
