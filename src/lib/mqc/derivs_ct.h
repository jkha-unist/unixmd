#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

// Routine to calculate dot product from two temporary arrays
static double dot(int nst, double complex *u, double complex *v){

    double complex sum;
    double norm;
    int ist;

    sum = 0.0 + 0.0 * I;
    for(ist = 0; ist < nst; ist++){
        sum += conj(u[ist]) * v[ist];
    }

    norm = creal(sum);
    return norm;

}

// Routine to calculate cdot contribution originated from Ehrenfest term
// Workspace: na_term_work[nst], ct_term_work[nst], rho_work[nst] (pre-allocated by caller)
static void ct_cdot(int nst, double *e, double **dv, double **k_lk, double complex *c, double complex *c_dot,
    double complex *na_term_work, double *ct_term_work, double *rho_work){

    int ist, jst;
    double egs;

    // Calculate densities from current coefficients
    for(ist = 0; ist < nst; ist++){
        rho_work[ist] = creal(conj(c[ist]) * c[ist]);
    }

    for(ist = 0; ist < nst; ist++){
        na_term_work[ist] = 0.0 + 0.0 * I;
        ct_term_work[ist] = 0.0;
        for(jst = 0; jst < nst; jst++){
            if(ist != jst){
                na_term_work[ist] -= dv[ist][jst] * c[jst];
                ct_term_work[ist] += 0.25 * (k_lk[jst][ist] - k_lk[ist][jst]) * rho_work[jst]
                    * (rho_work[ist] + rho_work[jst]) / (nst - 1);
            }
        }
    }

    egs = e[0];
    for(ist = 0; ist < nst; ist++){
        c_dot[ist] = - 1.0 * I * c[ist] * (e[ist] - egs) + na_term_work[ist] + ct_term_work[ist] * c[ist];
    }

}

/*
// Routine to calculate rhodot contribution originated from Ehrenfest term
static void ct_rhodot(int nst, double *e, double **dv, double complex **rho, double complex **rho_dot){

    int ist, jst, kst;

    for(ist = 0; ist < nst; ist++){
        for(jst = 0; jst < nst; jst++){
            rho_dot[ist][jst] = 0.0 + 0.0 * I;
        }
    }

    for(ist = 0; ist < nst; ist++){
        for(jst = 0; jst < nst; jst++){
            if(ist != jst){
                rho_dot[ist][ist] -= dv[ist][jst] * 2.0 * creal(rho[ist][jst]);
            }
        }
    }

    for(ist = 0; ist < nst; ist++){
        for(jst = ist + 1; jst < nst; jst++){
            rho_dot[ist][jst] -=  1.0 * I * (e[jst] - e[ist]) * rho[ist][jst];
            for(kst = 0; kst < nst; kst++){
                rho_dot[ist][jst] -= dv[ist][kst] * rho[kst][jst] + dv[jst][kst] * rho[ist][kst];
            }
            rho_dot[jst][ist] = conj(rho_dot[ist][jst]);
        }
    }

}
*/
