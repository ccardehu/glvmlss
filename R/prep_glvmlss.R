prep_family = function(family){
  for(i in 1:length(family)){
    family[i] = match.arg(family[i], c("normal","binomial","beta", "sn"))
  }
  return(family)
}

prep_form <- function(mu.eq, sg.eq, nu.eq, ta.eq, family){

  i2formula <- function(x){
    is.formula <- function(x){ inherits(x,"formula") }
    if(!is.formula(x) & !is.null(x)) x <- as.formula(x)
    return(x)
  }
  form <- lapply(list(mu.eq, sg.eq, nu.eq, ta.eq), i2formula)
  names(form) <- c("mu","sg","nu","ta")
  nD = fam4D(family)
  for(i in names(form)){
    if(is.null(form[[i]])){
      if(which(i == names(form)) == nD){
        form[[i]] <- ~ 1
      } else form[[i]] <- NULL
    }
  }
  return(form)
}

prep_control <- function(control,q,...){

  con <- list("EM.maxit" = 20, "DM.maxit" = 500, "R.maxit" = 50,
              "nQP" = 25, "Fisher" = T, "verbose" = T,
              "EM.useGD" = F, "LR" = 1e-3,
              "start.b" = NULL, "start.R" = NULL, "restr.b" = NULL, "restr.R" = NULL,
              "estim.R" = T, "sign.restr.b" = NULL, "updt.R.every" = 1,
              "corLV" = F, "tolerance" = 1e-4) # sqrt(.Machine$double.eps)
  control <- c(control, list(...))
  namC <- names(con)
  con[(namc <- names(control))] <- control
  if (length(namc[!namc %in% namC]) > 0)
    warning("Unknown names in control: ", paste(namc[!namc %in% namC], collapse = ", "))
  # Sanity checks:
  if(!is.null(con$start.R)){
    if(is.diag(con$start.R)) con$corLV <- F else con$corLV <- T
  }
  if(q == 1){
    con$estim.R <- F
    con$corLV <- F
  } else {
    con$nQP <- 15
  }
  if(!is.integer(con$updt.R.every)) con$updt.R.every <- as.integer(round(con$updt.R.every))
  return(con)
}

prep_control_fit <- function(control, ...){

  tmp <- list(...)
  if(is.null(tmp$method)) tmp$method = "sa"
  if(tmp$method == "ghq"){
    con <- list("EM.maxit" = 20, "DM.maxit" = 500, "R.maxit" = 50,
                "nQP" = 25, "Fisher" = T, "verbose" = T,
                "EM.useGD" = F, "LR" = 1e-3,
                "start.b" = NULL, "start.R" = NULL, "restr.b" = NULL, "restr.R" = NULL,
                "estim.R" = T, "sign.restr.b" = NULL, "updt.R.every" = 1,
                "corLV" = F, "tolerance" = 1e-4, "method" = "ghq") # sqrt(.Machine$double.eps)
  } else {
    con <- list("burn.iter" = 3e3, "max.iter" = 1e4, "tune.iter" = 50, "is.iter" = 1e4,
                "tol" = 1e-5, "sampler" = "RWMH",
                "gamma" = 1, "gamma.b" = 1, "gamma.R" = 1,
                "h" = 1e-2, "tune.gamma" = 1, "return.trace" = F,
                "verbose" = T, "verbose.every" = 10,
                "seed" = NULL,
                "start.b" = NULL, "start.R" = NULL,
                "restr.b" = NULL, "restr.R" = NULL,
                "sign.restr.b" = NULL, "corLV" = T, "hess.fisher" = T,
                "start.zn" = "fa",
                "window" = 10, "stop.atconv" = T,
                "algorithm" = "GD", "adam.b1" = .9, "adam.b2" = .999,
                "sa.power" = -.51, "q" = 1, "method" = "sa")
  }

  control <- c(control, list(...))
  namC <- names(con)
  namc <- names(control)
  con[namc] <- control
  if (length(namc[!namc %in% namC]) > 0)
    warning("Unknown names in control: ", paste(namc[!namc %in% namC], collapse = ", "))
  # Sanity checks:
  if(!is.null(control$start.zn) && is.character(con$start.zn)){
    con$start.zn = match.arg(con$start.zn, c("random", "fa"))
  }
  if(!is.null(con$start.R)){
    if(is.diag(con$start.R)) con$corLV <- F else con$corLV <- T
  }
  return(con)
}

prep_control_sim <- function(control,q,...){
  con <- list("verbose" = T,
              "start.b" = NULL, "start.R" = NULL, "restr.b" = NULL, "restr.R" = NULL,
              "corLV" = F)
  control <- c(control, list(...))
  namC <- names(con)
  con[(namc <- names(control))] <- control
  if (length(namc[!namc %in% namC]) > 0)
    warning("Unknown names in control: ", paste(namc[!namc %in% namC], collapse = ", "))
  # Sanity checks:
  if(!is.null(con$start.R)){
    if(is.diag(con$start.R)) con$corLV <- F else con$corLV <- T
  }
  if(q == 1){
    con$corLV <- F
  }
  return(con)
}

prep_vars <- function(form){
  lvar = unique(unlist(lapply(1:length(form), function(i) all.vars(form[[i]]))))
  lvar = lvar[order(rank_terms(lvar))]
  lter = unique(unlist(lapply(1:length(form), function(i) attr(terms(form[[i]]), "term.labels"))))
  lter = lter[order(rank_terms(lter))]
  lfac = make_lfac(lvar,lter)
  lter = c("(Intercept)", lter)
  Zvar = lvar[grepl("Z", lvar)]
  Xvar = lvar[!grepl("Z|~", lvar)]
  return(list(q = length(Zvar), vars = lvar, zvars = Zvar,
              xvars = Xvar, tvars = names(form), terms = lter,
              zxmat = lfac))
}

prep_pars_fit <- function(Y, family, form, control,
                          vars, Z, X = NULL){

  nostartb <- is.null(control$start.b)
  norestrb <- is.null(control$restr.b)
  nostartR <- is.null(control$start.R)
  norestrR <- is.null(control$restr.R)
  q = vars$q
  p = ncol(Y)

  if(all(!nostartb,!norestrb,!nostartR,!norestrR)){
    if(length(vars$terms) != ncol(control$start.b)) stop("\n Mismatch ncol(control$start.b) and terms in all formulas.")
    colnames(control$start.b) <- vars$terms
    rownames(control$start.b) <- paste0("Y",1:p)
    dimnames(control$start.b)[[3]] <- vars$tvars
    control$start.b = b_check(control$start.b, form)
    dimnames(control$restr.b) = dimnames(control$start.b)
    colnames(control$start.R) = rownames(control$start.R) = paste0("Z",1:q)
    dimnames(control$restr.R) = dimnames(control$start.R)
    bL <- rb2b(control$start.b, control$restr.b, vars)
    RL <- rR2R(control$start.R, control$restr.R)
    if(bL$irb + RL$irR < q^2) stop(paste0("\n Not enough identification restrictions (", bL$irb,
                                          " in control$restr.b + ", RL$irR,
                                          " in control$restr.R, ", q^2, " needed)"))
    return(list(b = bL$b, rb = bL$rb, R = RL$R, rR = RL$rR, mu = rep(0,q)))
  }

  if(all(nostartb,norestrb,nostartR,norestrR)){
    if(!control$corLV){
      control$start.R <- diag(q)
      colnames(control$start.R) = rownames(control$start.R) = paste0("Z",1:q)
      control$restr.R <- genrR(control$start.R)
    } else {
      control$start.R <- cor(Z); diag(control$start.R) <- 1
      colnames(control$start.R) = rownames(control$start.R) = paste0("Z",1:q)
      control$restr.R <- genrR(control$start.R)
    }
    if(control$verbose) cat("\n Defining starting values (via gamlss; missing control$start.b) ...")
    b <- suppressWarnings(b_ini(Y, family, form, vars, Z, X))
    if(control$verbose) cat("\r Defining starting values (via gamlss; missing control$start.b) ... (Done!)")
    if(q > 1){
      if(control$corLV){
        if(control$verbose) cat("\n Imposing errors-in-variables restrictions (missing control$restr.b; control$corLV == T).")
        idres = q:1
        for(r in 1:q){
          b[r, paste0("Z",idres[r]), 1] <- 0
        }
      } else {
        if(control$verbose) cat("\n Imposing recursive restrictions (missing control$restr.b; control$corLV == F).")
        b[, paste0("Z",1:q),1][upper.tri(b[,paste0("Z",1:q), 1])] <- 0
      }
    }
    control$start.b <- b_check(b, form)
    control$restr.b <- genrb(control$start.b)
    bL <- rb2b(control$start.b, control$restr.b, vars)
    RL <- rR2R(control$start.R, control$restr.R)
    if(bL$irb + RL$irR < q^2) stop(paste0("\n Not enough identification restrictions (", bL$irb,
                                          " in control$restr.b + ", RL$irR,
                                          " in control$restr.R, ", q^2, " needed)"))
    return(list(b = bL$b, rb = bL$rb, R = RL$R, rR = RL$rR, mu = rep(0,q)))
  }

  if(!nostartR & !nostartb){
    if(length(vars$terms) != ncol(control$start.b)) stop("\n Mismatch ncol(control$start.b) and terms in all formulas.")
    colnames(control$start.b) <- vars$terms
    rownames(control$start.b) <- paste0("Y",1:p)
    dimnames(control$start.b)[[3]] <- vars$tvars
    control$start.b = b_check(control$start.b, form)
    if(norestrb){
      control$restr.b <- genrb(control$start.b)
    } else {
      dimnames(control$restr.b) = dimnames(control$start.b)
    }
    colnames(control$start.R) = rownames(control$start.R) = paste0("Z",1:q)
    if(norestrR){
      control$restr.R <- genrR(control$start.R)
    } else {
      dimnames(control$restr.R) = dimnames(control$start.R)
    }
    bL <- rb2b(control$start.b, control$restr.b, vars)
    RL <- rR2R(control$start.R, control$restr.R)
    if(bL$irb + RL$irR < q^2) stop(paste0("\n Not enough identification restrictions (", bL$irb,
                                          " in control$restr.b + ", RL$irR,
                                          " in control$restr.R, ", q^2, " needed)"))
    return(list(b = bL$b, rb = bL$rb, R = RL$R, rR = RL$rR, mu = rep(0,q)))
  }

  if(nostartR & nostartb){
    if(!control$corLV){
      control$start.R <- diag(1.1,q)
    } else {
      control$start.R <- cor(Z); diag(control$start.R) <- 1.1
    }
    colnames(control$start.R) = rownames(control$start.R) = paste0("Z",1:q)
    if(norestrR){
      diag(control$start.R) <- 1
      control$restr.R <- genrR(control$start.R)
    } else {
      dimnames(control$restr.R) = dimnames(control$start.R)
    }
    if(control$verbose) cat("\n Defining starting values (via gamlss; missing control$start.b) ...")
    b <- suppressWarnings(b_ini(Y, family, form, vars, Z, X))
    if(control$verbose) cat("\r Defining starting values (via gamlss; missing control$start.b) ... (Done!)")
    if(q > 1){
      if(norestrb){
        if(control$corLV){
          if(control$verbose) cat("\n Imposing errors-in-variables restrictions (missing control$restr.b; control$corLV == T).")
          idres = q:1
          for(r in 1:q){
            b[r, paste0("Z",idres[r]), 1] <- 0
          }
        } else {
          if(control$verbose) cat("\n Imposing recursive restrictions (missing control$restr.b; control$corLV == F).")
          b[, paste0("Z",1:q),1][upper.tri(b[,paste0("Z",1:q), 1])] <- 0
        }
        b = b_check(b, form)
        control$restr.b <- genrb(b)

      } else {
        dimnames(control$restr.b) = dimnames(b)
      }
    } else {
      dimnames(control$restr.b) = dimnames(b)
    }
    control$start.b <- b
    bL <- rb2b(control$start.b, control$restr.b, vars)
    RL <- rR2R(control$start.R, control$restr.R)
    if(bL$irb + RL$irR < q^2) stop(paste0("\n Not enough identification restrictions (", bL$irb,
                                          " in control$restr.b + ", RL$irR,
                                          " in control$restr.R, ", q^2, " needed)"))
    return(list(b = bL$b, rb = bL$rb, R = RL$R, rR = RL$rR, mu = rep(0,q)))
  }

  if(nostartR & !nostartb){
    if(!control$corLV){
      control$start.R <- diag(1.1,q)
    } else {
      control$start.R <- cor(Z); diag(control$start.R) <- 1.1
    }
    colnames(control$start.R) = rownames(control$start.R) = paste0("Z",1:q)
    if(norestrR){
      diag(control$start.R) <- 1
      control$restr.R <- genrR(control$start.R)
    } else {
      dimnames(control$restr.R) = dimnames(control$start.R)
    }
    if(length(vars$terms) != ncol(control$start.b)) stop("\n Mismatch ncol(control$start.b) and terms in all formulas.")
    colnames(control$start.b) <- vars$terms
    rownames(control$start.b) <- paste0("Y",1:p)
    dimnames(control$start.b)[[3]] <- vars$tvars
    control$start.b = b_check(control$start.b, form)
    if(norestrb){
      control$restr.b <- genrb(control$start.b)
    } else {
      dimnames(control$restr.b) = dimnames(control$start.b)
    }
    bL <- rb2b(control$start.b, control$restr.b, vars)
    RL <- rR2R(control$start.R, control$restr.R)
    if(bL$irb + RL$irR < q^2) stop(paste0("\n Not enough identification restrictions (", bL$irb,
                                          " in control$restr.b + ", RL$irR,
                                          " in control$restr.R, ", q^2, " needed)"))
    return(list(b = bL$b, rb = bL$rb, R = RL$R, rR = RL$rR, mu = rep(0,q)))
  }

  if(!nostartR & nostartb){
    colnames(control$start.R) = rownames(control$start.R) = paste0("Z",1:q)
    if(norestrR){
      control$restr.R <- genrR(control$start.R)
    } else {
      dimnames(control$restr.R) = dimnames(control$start.R)
    }
    if(control$verbose) cat("\n Defining starting values (via gamlss; missing control$start.b) ...")
    b <- suppressWarnings(b_ini(Y, family, form, vars, Z, X))
    if(control$verbose) cat("\r Defining starting values (via gamlss; missing control$start.b) ... (Done!)")
    if(q > 1){
      if(norestrb){
        if(control$corLV){
          if(control$verbose) cat("\n Imposing errors-in-variables restrictions (missing control$restr.b; control$corLV == T).")
          idres = q:1
          for(r in 1:q){
            b[r, paste0("Z",idres[r]), 1] <- 0
          }
        } else {
          if(control$verbose) cat("\n Imposing recursive restrictions (missing control$restr.b; control$corLV == F).")
          b[, paste0("Z",1:q),1][upper.tri(b[,paste0("Z",1:q), 1])] <- 0
        }
        b = b_check(b, form)
        control$restr.b <- genrb(b)
      } else {
        dimnames(control$restr.b) = dimnames(b)
      }
    } else {
      dimnames(control$restr.b) = dimnames(b)
    }
    control$start.b <- b
    bL <- rb2b(control$start.b, control$restr.b, vars)
    RL <- rR2R(control$start.R, control$restr.R)
    if(bL$irb + RL$irR < q^2) stop(paste0("\n Not enough identification restrictions (", bL$irb,
                                          " in control$restr.b + ", RL$irR,
                                          " in control$restr.R, ", q^2, " needed)"))
    return(list(b = bL$b, rb = bL$rb, R = RL$R, rR = RL$rR, mu = rep(0,q)))
  }
}

prep_pars_sim <- function(control, vars, p, form){

  nostartb = is.null(control$start.b)
  norestrb = is.null(control$restr.b)
  nostartR = is.null(control$start.R)
  norestrR = is.null(control$restr.R)
  q = vars$q

  if(nostartb) stop("\n Missing argument: control$start.b")
  if(length(vars$terms) != ncol(control$start.b)) stop("\n Mismatch ncol(control$start.b) and terms in all formulas.")
  colnames(control$start.b) <- vars$terms
  rownames(control$start.b) <- paste0("Y",1:p)
  dimnames(control$start.b)[[3]] <- vars$tvars
  control$start.b = b_check(control$start.b, form)
  if(norestrb){
    control$restr.b <- genrb(control$start.b)
  } else {
    dimnames(control$restr.b) <- dimnames(control$start.b)
  }

  if(nostartR){
    if(!control$corLV){
      if(control$verbose) cat("\n Missing control$start.R: R set to diag(q) (control$corLV == F).")
      control$start.R <- diag(q)
    } else {
      if(control$verbose) cat("\n Missing control$start.R: Correlations in R set to 0.3 (control$corLV == T).")
      control$start.R <- matrix(0.3,q,q); diag(control$start.R) <- 1.0
    }
    colnames(control$start.R) <- rownames(control$start.R) <- paste0("Z",1:q)
    if(norestrR){
      diag(control$start.R) <- 1
      control$restr.R <- genrR(control$start.R)
    } else {
      dimnames(control$restr.R) = dimnames(control$start.R)
    }
  } else {
    colnames(control$start.R) <- rownames(control$start.R) <- paste0("Z",1:q)
    if(norestrR){
      control$restr.R <- genrR(control$start.R)
    } else {
      dimnames(control$restr.R) = dimnames(control$start.R)
    }
  }
  bL = rb2b(control$start.b, control$restr.b, vars)
  RL = rR2R(control$start.R, control$restr.R)
  if(bL$irb + RL$irR < q^2) stop(paste0("\n Not enough identification restrictions (", bL$irb,
                                        " in control$restr.b + ", RL$irR,
                                        " in control$restr.R, ", q^2, " needed)"))
  return(list(b = bL$b, rb = bL$rb, R = RL$R, rR = RL$rR, mu = rep(0,q)))
}

prep_z_fit = function(Y, family, form, control, X = NULL){
  if(!is.null(control$start.zn) && !is.character(control$start.zn)){
    zn = as.matrix(control$start.zn)
  } else {
    if(!is.null(control$seed)) set.seed(control$seed)
    if(!is.null(X)){
      Ytmp = ortho_Y(Y, family, form, control, X)
    } else {
      Ytmp = Y
    }
    if(control$start.zn == "random") {
      if(control$verbose) cat("\n Generating random starting values for latent variables ...")
      zn = mvtnorm::rmvnorm(nrow(Ytmp),mean = rep(0, control$q))
      if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!)")
    } else if(control$start.zn == "fa") {
      if(control$verbose) cat("\n Generating starting values for latent variables via Factor Analysis ...")
      tmp = suppressWarnings(psych::fa(r = Ytmp, nfactors = control$q, fm = "ml", rotate = "oblimin",
                                       missing = T))
      zn = tmp$scores
      if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!)")
    }
  }
  colnames(zn) <- paste0("Z", 1:control$q)
  return(zn)
}
