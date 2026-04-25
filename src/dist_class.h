#ifndef distClass_H
#define distClass_H

// Class for item-distribution maths (sim, log-density, 1st and 2nd derivatives)
class Dist {
  std::string fam;

public:
  // Constructors
  Dist(std::string faminput){
    fam = faminput;
  };

  //Public member functions
  // Simulation member function
  arma::vec sim(const int n,
                int i,
                arma::cube& b,
                arma::mat & Z);

  // log-density matrix (n x tqp) f(y_m|z_qp)
  arma::mat dy(const int i,
               arma::mat& Yr,
               arma::mat& Z,
               arma::cube& b);

  // 1st derivative cube (n x tqp x D): d f(y_m|z_qp; theta_d) / d theta_d
  arma::cube d1y(const int i,
                 arma::mat& Yr,
                 arma::mat& Z,
                 arma::cube& b);

  // 2nd derivative cube (n x tqp x D): d^2 f(y_m|z_qp; theta_d) / (d theta_d)^2
  arma::cube d2y(const int i,
                 arma::mat& Yr,
                 arma::mat& Z,
                 arma::cube& b,
                 const bool flagEIM,
                 arma::cube& d1Yi);

  // Cross-derivative cube (n x tqp x D): d^2 f(y_m|z_qp; theta_d) / d theta_d1 d theta_d2
  arma::cube dCy(const int i,
                 arma::mat& Yr,
                 arma::mat& Z,
                 arma::cube& b,
                 const bool flagEIM);

};

#endif
