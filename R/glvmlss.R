#' Fit a Generalized latent variable model for location, scale, and shape parameters.
#'
#' @param data A matrix \code{(n x p)} of observed variables. Missing data should be coded as \code{NA}.
#' @param family A string vector of size \code{p} with family for each observed variable.
#' @param mu.eq A formula object for the location parameter e.g., \code{mu.eq = ~ Z1}.
#' @param sg.eq (optional) A formula object for the scale parameter.
#' @param nu.eq (optional) A formula object for the (1st) shape parameter.
#' @param ta.eq (optional) A formula object for the (2nd) shape parameter.
#' @param control (optional) A list of control parameters (see 'Details').
#' @param ... (optional) Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{b}: An array with the estimated intercept and slopes.
#'  \item \code{R}: Estimated factor correlation matrix (if \code{control$corLV = T}).
#'  \item \code{ll}: Marginal log-likelihood evaluated at \code{b} and \code{Rz}.
#'  \item \code{seb}: Estimated standard errors for \code{b}.
#'  \item \code{seR}: Estimated standard errors for \code{R} (if \code{control$corLV = T}).
#' }
#' @details The current implementation allows only for the following entries in the string vector \code{family} : \code{"normal"} (for Gaussian items),
#' \code{"binomial"} (for binary items, fixed \code{size = 1}), \code{"beta"} (for Beta distributed items), and \code{"sn"} (for Skew-Normal items).
#' More distributions will be continuously added as users require them.
#'
#' The \code{control} argument is a list that can supply any of the following components:
#' \itemize{
#'  \item \code{EM.maxit} (int): Maximum number of iterations for the EM (initialization) step. Defaults to \code{EM.maxit = 20} (set \code{EM.maxit = 0} to avoid this step).
#'  \item \code{DM.maxit} (int): Maximum number of iterations for the direct maximization step. Defaults to \code{DM.maxit = 500} (set \code{DM.maxit = 0} to avoid this step).
#'  \item \code{R.maxit} (int): Maximum number of iterations for the estimation of the latent variables covariance matrix. Defaults to \code{R.maxit = 50}.
#'  \item \code{nQP} (int): Number of quadrature points (per dimension). Defaults to \code{nQP = 25}.
#'  \item \code{Fisher} (bool): Use expected information matrix (\code{Fisher = T}) or observed information matrix (\code{Fisher = F}) in the update step
#'  of the EM-algorithm. Defaults to \code{Fisher = T}.
#'  \item \code{verbose} (bool): Print output at each iteration. Defaults to \code{verbose = T}.
#'  \item \code{EM.useGD} (bool): Use gradient descent update rule in the EM-algorithm (instead of Newton update rule). Defaults to \code{EM.useGD = F}.
#'  \item \code{LR} (float): Learning rate in the gradient descent update rule. Defaults to \code{LR = 1e-3}.
#'  \item \code{tolerance} (float): Tolerance for difference in the objective function determining the
#'  convergence of EM-algorithm and direct maximization steps. Defaults to \code{tolerance = 1e-4}.
#'  \item \code{corLV} (bool): The latent variables are correlated. Defaults to \code{corLV = F}.
#'  \item \code{start.b} (array): An array \code{(p x (q+1) x d)} of initial values for the intercepts and factor loadings. \code{p} refers
#'  to the number of items, \code{q} to the number of latent variables, and \code{d} to the total number of distributional parameters.
#'  If the argument is not provided (i.e., \code{start.b = NULL} by default),
#'  the elements of \code{start.b} are initialized using the \code{gamlss::gamlss} function, with explanatory variables given by the first \code{q} normalized principal components
#'  computed using the function \code{princomp}.
#'  \item \code{start.R} (matrix): A matrix \code{(q x q)} for the covariance matrix of the latent variables.
#'  If the argument is not provided (i.e., \code{start.R = NULL} by default),
#'  it defaults to \code{start.R = diag(q)} if \code{control$corLV = F}. If \code{start.R = NULL} and \code{control$corLV = T}, the diagonal elements are
#'  initialized to 1.1 and the off-diagonal elements are initialized to 0.05.
#'  \item \code{restr.b} (array): An array \code{(p x (q+1) x d)} of restrictions on the intercepts and factor loadings. Free parameters
#'  (i.e., to be estimated) are encoded by \code{NA}, while fixed parameters must take some value (e.g., fixed factor loading with value 1
#'  for item 1, latent variable 1, and location parameter is \code{start.b[1,2,1] = 1}).
#'  If the argument is not provided (i.e., \code{restr.b = NULL} by default),
#'  the elements of \code{restr.b} are fixed to comply with either an errors-in-variables identification restriction (if \code{control$corLV = T}),
#'  or with a recursive identification restriction (if \code{control$corLV = F}).
#'  \item \code{restr.R} (matrix): A matrix \code{(q x q)} of restrictions on the latent variable covariances. Typical restrictions include
#'  fixed variances with value equal to one (e.g., \code{start.R[1,1] = 1}), or fixed covariances with value equal to zero (e.g. \code{start.R[1,2] = 0}).
#'  If the argument is not provided (i.e., \code{restr.R = NULL} by default),
#'  the diagonal elements of \code{restr.R} are fixed to 1 and off-diagonal elements are \code{NA} (if \code{control$corLV} = T),
#'  or \code{restr.R = diag(q)} (if \code{control$corLV = F}).
#'  \item \code{sign.restr.b} (vector): A vector where each entry denotes a column corresponding to the \code{q}th latent variable.
#'  The sign of each entry defines the sign of the first free factor loading in column \code{q}. For example, in a model with \code{q = 2} latent variables,
#'  the vector \code{sign.restr.b = c(-1,2)} denotes that the factor loadings for the first latent variable should be negative, while the
#'  factor loadings for the second latent variable should be positive. NOTE: Introducing sign restrictions can affect the sign of the
#'  factor covariances, and might also result in numerical instabilities or slow convergence of the optimization algorithm in the estimation step.
#'  \item \code{updt.R.every} (int): A number referring to how often the covariance matrix \code{R} should be updated in the EM-step.
#'  It defaults to \code{updt.R.every = 1}. The user can increase the default number to gain computational time during the estimation process,
#'  but this is not necessary recommended. Use with caution.
#' }
#'
#' The argument \code{...} can be used to pass elements directly to the list \code{control}, without explicitly defining the latter.
#'
#' @usage glvmlss = function(data, family,
#'                     mu.eq = ~ Z1, sg.eq = NULL,
#'                     nu.eq = NULL, ta.eq = NULL,
#'                     control = list(), ...)
#'
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
#' @examples
#' library(glvmlss)
#' data = ANES2020[,1:8]
#' p = ncol(data)
#' famt = rep("beta",p)
#'
#' # A heteroscedastic Beta factor model
#' mod1 = glvmlss::glvmlss(data = data, family = famt, mu.eq = ~ Z1, sg.eq = ~ Z1,
#'                          EM.maxit = 30, DM.maxit = 300, nQP = 40,"verbose" = T)
#'
glvmlss = function(data, family,
                   mu.eq = ~ Z1, sg.eq = NULL,
                   nu.eq = NULL, ta.eq = NULL,
                   X = NULL,
                   control = list(), ...){

  Y = as.matrix(data)
  family = prep_family(family)
  form = prep_form(mu.eq, sg.eq, nu.eq, ta.eq, family)
  vars = prep_vars(form)
  control$q = vars$q
  control = prep_control_fit(control, ...)
  if(!is.null(X) && !is.matrix(X)) X = as.matrix(X)
  if(!is.null(X) && length(vars$xvars) == 0) X = NULL
  zn = prep_z_fit(Y,family,form,control,X)
  pars = prep_pars_fit(Y,family,form,control,vars,zn,X)

  if(!is.null(control$seed)) set.seed(control$seed)
  if(control$method == "sa"){
    out = glvmlss_fit_sa_rcpp(Yr = Y, famr = family,
                              b = pars$b[], rb = pars$rb[],
                              R = pars$R[], rR = pars$rR[],
                              mur = pars$mu[],
                              control = control,
                              fmat = vars$zxmat[],
                              Z = zn[], Xr = X)
  } else if(control$method == "ghq"){
    out = glvmlss_fit_gq_rcpp(Yr = Y,
                              famr = family,
                              q = vars$q,
                              b = pars$b[],
                              rb = pars$rb[],
                              R = pars$R[],
                              rR = pars$rR[],
                              control = control)
    dimnames(out$b) = dimnames(out$seb) = dimnames(pars$b)
    dimnames(out$R) = dimnames(pars$R)
    if(!is.null(out$seR)) dimnames(out$seR) = dimnames(pars$R)
    colnames(out$Z) = paste0("Z",1:vars$q)
    out$Z = as.data.frame(out$Z)
    # if(!is.null(X)){
    #   colnames(out$X) = vars$xvars
    #   out$X = as.data.frame(out$X)
    # }
  }
  class(out) = c("glvmlss")
  return(out)
}

#' Simulate data for a Generalized latent variable model for location, scale, and shape parameters
#'
#' @param n Number of simulated entries.
#' @param family String vector of size \code{p} with family for each observed variable.
#' @param mu.eq Formula for location parameter e.g.,  \code{mu.eq = ~ Z1}.
#' @param sg.eq (optional) Formula for scale parameter.
#' @param nu.eq (optional) Formula for shape parameter.
#' @param ta.eq (optional) Formula for shape parameter.
#' @param control List of control parameters (see 'Details').
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{Y}: A matrix \code{(n x p)} of observed variables, each column from a distribution specified in \code{family}.
#'  \item \code{Z}: Simulated latent variables (includes a column corresponding to the intercept).
#'  \item \code{b}: An array of true coefficients.
#'  \item \code{R}: A matrix \code{(q x q)} of true factor covariances.
#' }
#' @details The current implementation allows only for the following entries in the string vector \code{family} : \code{"normal"} (for Gaussian items),
#' \code{"binomial"} (for binary items, fixed \code{size = 1}), \code{"beta"} (for Beta distributed items), and \code{"sn"} (for Skew-Normal items).
#' More distributions will be continuously added as users require them.
#'
#' The \code{control} argument is a list that can supply any of the following components:
#' \itemize{
#'  \item \code{corLV} (bool): The latent variables are correlated. Defaults to \code{corLV = F}.
#'  \item \code{start.b} (array): \code{[MANDATORY]} An array \code{(p x (q+1) x d)} of true values for the intercepts and factor loadings. \code{p} refers
#'  to the number of items, \code{q} to the number of latent variables, and \code{d} to the total number of distributional parameters.
#'  \item \code{start.R} (matrix): A matrix \code{(q x q)} for the covariance matrix of the latent variables.
#'  If the argument is not provided (i.e., \code{start.R = NULL} by default),
#'  it defaults to \code{diag(start.R) = 1.1} if \code{control$corLV = F}. If \code{start.R = NULL} and \code{control$corLV = T}, the diagonal elements are
#'  initialized to 1.1 and the off-diagonal elements are initialized to 0.3.
#'  \item \code{restr.b} (array): An array \code{(p x (q+1) x d)} of restrictions on the intercepts and factor loadings. Free parameters
#'  (i.e., to be estimated) are encoded by \code{NA}, while fixed parameters must take some value (e.g., fixed factor loading with value 1
#'  for item 1, latent variable 1, and location parameter is \code{start.b[1,2,1] = 1}).
#'  If the argument is not provided, restrictions are implicitly derived from the argument \code{start.b}.
#'  \item \code{restr.R} (matrix): A matrix \code{(q x q)} of restrictions on the latent variable covariances. Typical restrictions include
#'  fixed variances with value equal to one (e.g., \code{start.R[1,1] = 1}), or fixed covariances with value equal to zero (e.g. \code{start.R[1,2] = 0}).
#' }
#'
#' The argument \code{...} can be used to pass elements directly to the list \code{control}, without explicitly defining the latter.
#'
#' @usage glvmlss_sim = function(n, family,
#'                         mu.eq = ~ Z1, sg.eq = NULL,
#'                         nu.eq = NULL, ta.eq = NULL,
#'                         control = list(), ...)
#'
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @examples
#' library(glvmlss)
#'
#' # Simulate data following the PISA2018 application
#' # in Cárdenas-Hurtado et al. (2025)
#' # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#'
#' # Eight (8) Skew-Normal response times and eight (8) binary items
#' p = 8
#' faml = c(rep("sn",p),rep("binomial",p))
#'
#' # Population (true) factor loadings
#' lc = array(0,dim = c(2*p,3,3))
#' lc[1:p,1,1] = c( .19,  .30,  .42,  .45, 1.00,  .16,  .65,  .58)
#' lc[1:p,2,1] = c(-.17, -.24, -.25, -.34, -.36, -.36, -.35, -.30)
#' lc[1:p,1,2] = c(-.95, -.89, -.61, -.87, -.68, -.97,-1.15, -.90)
#' lc[1:p,2,2] = c( .03,  .10,  .08,  .05, -.14, -.03,  .04,  .02)
#' lc[1:p,1,3] = c( .63, 1.56,-1.07,-1.45,-1.04,  .10,  .36,  .30)
#' lc[1:p,2,3] = c( .03,  .78, -.81,  .33,  .32,  .89,  .58, -.36)
#' lc[(p+1):(2*p),1,1] = c( .64, -.47, -.04, -.69,-2.85, -.91,-4.84,-2.73)
#' lc[(p+1):(2*p),3,1] = c( .78, 1.03, 1.95,  .96, 2.28,  .32, 2.53, 1.46)
#'
#' # Population (true) factor correlation matrix
#' R = matrix(c(1,-0.28,-0.28,1),2)
#'
#' # Identification restrictions
#' ires = array(NA,dim = c(2*p,3,3))
#' ires[1:p,3,1:3] = 0
#' ires[(p+1):(2*p),2,1] = 0
#' ires[(p+1):(2*p),,2:3] = 0
#'
#' # Sample size
#' n = 3000
#'
#' # Simulate data
#' mod0 = glvmlss::glvmlss_sim(n = n, family = faml, mu.eq = ~ Z1+Z2, sg.eq = ~ Z1+Z2, nu.eq = ~ Z1+Z2,
#'                              start.b = lc, start.R = R,
#'                              restr.b = ires, restr.R = matrix(c(1,NA,NA,1),2))
#' \dontrun{
#' # Fit the model:
#' mod1 = glvmlss::glvmlss(data = mod0$Y,family = faml,mu.eq = ~ Z1+Z2, sg.eq = ~ Z1+Z2, nu.eq = ~ Z1+Z2,
#'                          EM.useGD = T, LR = 0.0001,
#'                          EM.maxit = 500, DM.maxit = 1000, nQP = 35, verbose = T,
#'                          corLV = T, tolerance = 1e-5, start.b = mod0$b, start.R = mod0$R,
#'                          restr.b = ires, restr.R = matrix(c(1,NA,NA,1),2), sign.restr.b = c(-1,2))
#' }
#' @export
glvmlss_sim = function(n, family,
                       mu.eq = ~ Z1, sg.eq = NULL,
                       nu.eq = NULL, ta.eq = NULL,
                       X = NULL,
                       control = list(), ...){

  if(is.numeric(n)) n = as.integer(n)
  if(!is.integer(n)) stop("Argument n is not interger")
  family = prep_family(family)
  form = prep_form(mu.eq, sg.eq, nu.eq, ta.eq, family)
  vars = prep_vars(form)
  control = prep_control_sim(control, vars$q, ...)
  pars = prep_pars_sim(control, vars, length(family), form)
  if(!is.null(X) && !is.matrix(X)) X = as.matrix(X)
  if(!is.null(X) && length(vars$xvars) == 0) X = NULL
  out = glvmlss_sim_rcpp(n = n, famr = family, q = vars$q,
                         b = pars$b[], R = pars$R[],
                         mu = pars$mu[],
                         fmat = vars$zxmat[], Xr = X)
  dimnames(out$b) = dimnames(pars$b)
  colnames(out$Y) = rownames(pars$b)
  colnames(out$Z) = vars$zvars
  dimnames(out$R) = dimnames(pars$R)
  if(!is.null(X)){
    colnames(out$X) = vars$xvars
    out$X = as.data.frame(out$X)
  }
  out$Y = as.data.frame(out$Y)
  out$Z = as.data.frame(out$Z)
  class(out) = c("glvmlss")
  return(out)
}
