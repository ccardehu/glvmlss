fam4D = function(family){
  out = 1
  for(i in 1:length(out)){
    if(family[i] == "normal") out = max(out,2)
    if(family[i] == "binomial") out = max(out,1)
    if(family[i] == "beta") out = max(out,2)
    if(family[i] == "sn") out = max(out,3)
  }
  out
}

fam4bstart <- function(family){
  out <- vector(mode = "list", length = length(family))
  for(i in 1:length(out)){
    if(family[i] == "normal") out[[i]] <- quote(NO())
    if(family[i] == "binomial") out[[i]] <- quote(BI())
    if(family[i] == "beta") out[[i]] <- quote(BE())
    if(family[i] == "sn") out[[i]] <- quote(SN1())
  }
  return(out)
}

b_ini <- function(Y, family, form, vars, Z, X = NULL){

  q <- vars$q
  p <- ncol(Y)
  if(is.null(X)){
    ZX = Z
  } else {
    ZX = cbind(Z,X)
  }

  bstart <- array(0, dim = c(p, length(vars$terms), length(vars$tvars)))
  fstart <- fam4bstart(family)
  colnames(bstart) <- vars$terms
  rownames(bstart) <- colnames(Y)
  dimnames(bstart)[[3]] <- vars$tvars
  eq <- form

  for(i in 1:ncol(Y)){
    tmpY <- Y[,i]
    if(family[i] == "beta") tmpY <- fixy(tmpY)
    for(ii in 1:length(eq)){ eq[[ii]] <- update(eq[[ii]], tmpY ~ .)  }
    tmp <- try(gamlss::gamlss(eq$mu, sigma.formula = eq$sg, nu.formula = eq$nu, tau.formula = eq$ta,
                              family = fstart[[i]], data = as.data.frame(cbind(tmpY,ZX)),
                              control = gamlss::gamlss.control(trace = F)), silent = T)

    if(!inherits(tmp, "try-error")){
      tmpB <- gamlss::coefAll(tmp)
      K <- length(tmpB)
      for(k in 1:K){
        bstart[i,names(tmpB[[k]]),k] <- tmpB[[k]]
      }
    } else {
      bstart[i,,] <- 0.01
    }
  }
  return(bstart)
}

fixy <- function(x){
  if (any(x == 1, na.rm = T)) x[x == 1] <- 1 - sqrt(.Machine$double.eps)
  if (any(x == 0, na.rm = T)) x[x == 0] <- sqrt(.Machine$double.eps)
  return(x)
}

genrb <- function(b){
  rb <- ifelse(b == 0 | b == 1, b, NA)
  dimnames(rb) = dimnames(b)
  return(rb)
}

genrR <- function(R){
  rR <- ifelse(R == 0 | R == 1, R, NA)
  dimnames(rR) = dimnames(R)
  return(rR)
}

rb2b <- function(b,rb,vars){
  b[!is.na(rb)] <- rb[!is.na(rb)]
  zvars = vars$zvars
  irb = sum(!is.na(rb[,zvars,, drop = F]))
  return(list(b = b, rb = rb, irb = irb))
}

rR2R <- function(R,rR){
  if(!isSymmetric(rR)){
    rR[upper.tri(rR)] <- t(rR)[upper.tri(rR)]
  }
  R[!is.na(rR)] <- rR[!is.na(rR)]
  irR = sum(!is.na(rR[lower.tri(rR,diag = T)]))
  return(list(R = R, rR = rR, irR = irR))
}

is.diag <- function(m){
  all(m[lower.tri(m)] == 0, m[upper.tri(m)] == 0)
}

rank_terms <- function(x) {
  ifelse(grepl(":", x, fixed = TRUE), 3L,   # interactions last
         ifelse(grepl("Z", x, fixed = TRUE), 1L,   # Z main effects first
                ifelse(!grepl("Z|~|(Intercept)", x, fixed = F), 2L,   # X main effects second
                       0L)))  # (Intercept) at front
}

make_lfac = function(lvar, lter){
  lfac = matrix(NA, length(lvar), length(lter))
  for(i in 1:nrow(lfac)){
    lfac[i,] = as.numeric(grepl(lvar[i], lter))
  }
  dimnames(lfac) = list(lvar,lter)
  return(lfac)
}

b_check = function(b, form){
  for(i in 1:length(form)){
    lter = attr(terms(form[[i]]), "term.labels")
    lter = c("(Intercept)", lter)
    lter = lter[order(rank_terms(lter))]
    b[,setdiff(colnames(b),lter), i] = 0
  }
  return(b)
}

form4X = function(form){
  out <- vector(mode = "list", length = length(form))
  for(d in 1:length(out)){
    labs  <- attr(terms(form[[d]]), "term.labels")
    resp  <- if (attr(terms(form[[d]]), "response")) all.vars(form[[d]])[1] else NULL
    keep <- labs[!vapply(strsplit(labs, ":", fixed = TRUE),
                         function(v) any(grepl("Z", v, fixed = TRUE)),
                         logical(1))]

    rhs <- if(length(keep)) paste(keep, collapse = " + ") else "1"
    out[[d]] = reformulate(rhs, response = resp,
                           intercept = attr(terms(form[[d]]), "intercept") == 1L)
    environment(out[[d]]) <- environment(form[[d]])
  }
  names(out) = names(form)
  return(out)
}

ortho_Y = function(Y, family, form, control, X = NULL){

  if(is.null(X)){
    return(Y)
  }
  formX = form4X(form)
  fstart = fam4bstart(family)
  Ytmp = array(NA, dim(Y))
  for(i in 1:ncol(Y)){
    tmpY <- Y[,i]
    if(family[i] == "beta") tmpY <- fixy(tmpY)
    for(d in 1:length(formX)){ formX[[d]] <- update(formX[[d]], tmpY ~ .)  }
    tmp <- try(gamlss::gamlss(formX$mu, sigma.formula = formX$sg, nu.formula = formX$nu, tau.formula = formX$ta,
                              family = fstart[[i]], data = as.data.frame(cbind(tmpY,X)),
                              control = gamlss::gamlss.control(trace = F)), silent = T)
    Ytmp[,i] = unname(tmpY - fitted(tmp))
  }
  return(Ytmp)
}
