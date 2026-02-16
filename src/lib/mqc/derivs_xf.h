#ifndef DERIVS_XF_H
#define DERIVS_XF_H

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

static void xf_dec(int nat, int ndim, int nst, int *l_coh, double *mass, double **sigma,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double *rho, double **dec);

// Routine to calculate cdot contribution originated from XF term
// Workspace: dec_work[nst][nst], rho_work[nst] (pre-allocated by caller)
static void xf_cdot(int nat, int ndim, int nst, int *l_coh, double *mass, double **sigma,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *c, double complex *xfcdot,
    double **dec_work, double *rho_work){

    int ist, jst;

    // Calculate densities from current coefficients
    for(ist = 0; ist < nst; ist++){
        rho_work[ist] = creal(conj(c[ist]) * c[ist]);
    }

    xf_dec(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, rho_work, dec_work);

    // Get cdot contribution from decoherence term
    for(ist = 0; ist < nst; ist++){
        xfcdot[ist] = 0.0 + 0.0 * I;
        for(jst = 0; jst < nst; jst++){
            xfcdot[ist] -= rho_work[jst] * dec_work[jst][ist] * c[ist];
        }
    }

}

// Routine to print xf debug info
static void xf_print_coef(int nat, int ndim, int nst, int *l_coh, double *mass, double **sigma,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *c, double *dotpopdec){

    double **dec = malloc(nst * sizeof(double*));
    double *rho = malloc(nst * sizeof(double));
    double complex *xfcdot = malloc(nst * sizeof(double complex));

    int ist, jst;

    for(ist = 0; ist < nst; ist++){
        dec[ist] = malloc(nst * sizeof(double));
    }

    // Calculate densities from current coefficients
    for(ist = 0; ist < nst; ist++){
        rho[ist] = creal(conj(c[ist]) * c[ist]);
    }

    xf_dec(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, rho, dec);

    // Get cdot contribution from decoherence term
    for(ist = 0; ist < nst; ist++){
        xfcdot[ist] = 0.0 + 0.0 * I;
        for(jst = 0; jst < nst; jst++){
            xfcdot[ist] -= rho[jst] * dec[jst][ist] * c[ist];
        }
    }

    for(ist = 0; ist < nst; ist++){
        dotpopdec[ist] = 2.0 * creal(xfcdot[ist] * conj(c[ist]));
    }    

    // Deallocate temporary arrays
    for(ist = 0; ist < nst; ist++){
        free(dec[ist]);
    }

    free(dec);
    free(rho);
    free(xfcdot);

}

// Routine to calculate rhodot contribution originated from XF term
// Workspace: dec_work[nst][nst], pop_work[nst] (pre-allocated by caller)
static void xf_rhodot(int nat, int ndim, int nst, int *l_coh, double *mass, double **sigma,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double complex **rho, double complex **xfrhodot,
    double **dec_work, double *pop_work){

    int ist, jst, kst;

    for(ist = 0; ist < nst; ist++){
        pop_work[ist] = creal(rho[ist][ist]);
    }

    xf_dec(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, pop_work, dec_work);

    // Get rhodot contribution from decoherence term
    for(ist = 0; ist < nst; ist++){
        // Diagonal components
        xfrhodot[ist][ist] = 0.0 + 0.0 * I;
        for(kst = 0; kst < nst; kst++){
            xfrhodot[ist][ist] -= 2.0 * dec_work[kst][ist] * rho[ist][kst] * rho[kst][ist];
        }
        // Off-diagonal components
        for(jst = ist + 1; jst < nst; jst++){
            xfrhodot[ist][jst] = 0.0 + 0.0 * I;
            for(kst = 0; kst < nst; kst++){
                xfrhodot[ist][jst] -= (dec_work[kst][ist] + dec_work[kst][jst]) * rho[ist][kst] * rho[kst][jst];
            }
            xfrhodot[jst][ist] = conj(xfrhodot[ist][jst]);
        }
    }

}

// Routine to print xf debug info 
static void xf_print_rho(int nat, int ndim, int nst, int *l_coh, double *mass, double **sigma,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double complex **rho, double *dotpopdec){

    double **dec = malloc(nst * sizeof(double*));
    double *pop = malloc(nst * sizeof(double));
    double complex **xfrhodot = malloc(nst * sizeof(double complex *));

    int ist, jst, kst;

    for(ist = 0; ist < nst; ist++){
        dec[ist] = malloc(nst * sizeof(double));
        xfrhodot[ist] = malloc(nst * sizeof(double complex));
        pop[ist] = creal(rho[ist][ist]);
    }

    xf_dec(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, pop, dec);

    // Get rhodot contribution from decoherence term
    for(ist = 0; ist < nst; ist++){
        // Diagonal components
        xfrhodot[ist][ist] = 0.0 + 0.0 * I;
        for(kst = 0; kst < nst; kst++){
            xfrhodot[ist][ist] -= 2.0 * dec[kst][ist] * rho[ist][kst] * rho[kst][ist];
        }
        // Off-diagonal components
        for(jst = ist + 1; jst < nst; jst++){
            xfrhodot[ist][jst] = 0.0 + 0.0 * I;
            for(kst = 0; kst < nst; kst++){
                xfrhodot[ist][jst] -= (dec[kst][ist] + dec[kst][jst]) * rho[ist][kst] * rho[kst][jst];
            }
            xfrhodot[jst][ist] = conj(xfrhodot[ist][jst]);
        }
    }

    for(ist = 0; ist < nst; ist++){
        dotpopdec[ist] = creal(xfrhodot[ist][ist]);
    }

    // Deallocate temporary arrays
    for(ist = 0; ist < nst; ist++){
        free(dec[ist]);
        free(xfrhodot[ist]);
    }

    free(dec);
    free(pop);
    free(xfrhodot);

}

static void xf_dec(int nat, int ndim, int nst, int *l_coh, double *mass, double **sigma,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double *rho, double **dec){

    int ist, jst, iat, isp;

    // Initialize variables related to decoherence
    for(iat = 0; iat < nat; iat++){
        for(isp = 0; isp < ndim; isp++){
            qmom[iat][isp] = 0.0;
        }
    }

    for(ist = 0; ist < nst; ist++){
        for(jst = 0; jst < nst; jst++){
            dec[ist][jst] = 0.0;
        }
    }

    // Get quantum momentum from auxiliary positions and sigma values
    for(ist = 0; ist < nst; ist++){

        if(l_coh[ist] == 1){
            for(iat = 0; iat < nat; iat++){
                for(isp = 0; isp < ndim; isp++){
                    qmom[iat][isp] += 0.5 * rho[ist] * (pos[iat][isp] - aux_pos[ist][iat][isp])
                        / (sigma[iat][isp] * sigma[iat][isp] * mass[iat]);
                }
            }
        }

    }

    // Get decoherence term from quantum momentum and phase
    for(ist = 0; ist < nst; ist++){
        for(jst = ist + 1; jst < nst; jst++){

            if(l_coh[ist] == 1 && l_coh[jst] == 1){
                for(iat = 0; iat < nat; iat++){
                    for(isp = 0; isp < ndim; isp++){
                        dec[ist][jst] += qmom[iat][isp] * (phase[ist][iat][isp] - phase[jst][iat][isp]);
                    }
                }
            }
            dec[jst][ist] = - 1.0 * dec[ist][jst];

        }
    }
}
#endif
