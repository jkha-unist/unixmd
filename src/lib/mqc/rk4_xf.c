#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "derivs.h"
#include "derivs_xf.h"

// Routine for coefficient propagation scheme in rk4 propagator
static void rk4_coef(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme,
    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *coef,
    int verbosity, double *dotpopdec);

// Routine for density propagation scheme in rk4 propagator
static void rk4_rho(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme,
    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex **rho,
    int verbosity, double *dotpopdec);

// Interface routine for propagation scheme in rk4 propagator
static void rk4(int nat, int ndim, int nst, int nesteps, double dt, char *elec_object, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme, double **nacme_old,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *coef, double complex **rho,
    int verbosity, double *dotpopdec){

    if(strcmp(elec_object, "coefficient") == 0){
        rk4_coef(nat, ndim, nst, nesteps, dt, l_coh, mass, energy, energy_old, sigma,
            nacme, nacme_old, pos, qmom, aux_pos, phase, coef, verbosity, dotpopdec);
    }
    else if(strcmp(elec_object, "density") == 0){
        rk4_rho(nat, ndim, nst, nesteps, dt, l_coh, mass, energy, energy_old, sigma,
            nacme, nacme_old, pos, qmom, aux_pos, phase, rho, verbosity, dotpopdec);
    }

}

// Routine for coefficient propagation scheme in rk4 propagator
static void rk4_coef(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme,
    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex *coef,
    int verbosity, double *dotpopdec){

    double complex *k1 = malloc(nst * sizeof(double complex));
    double complex *k2 = malloc(nst * sizeof(double complex));
    double complex *k3 = malloc(nst * sizeof(double complex));
    double complex *k4 = malloc(nst * sizeof(double complex));
    double complex *kfunction = malloc(nst * sizeof(double complex));
    double complex *variation = malloc(nst * sizeof(double complex));
    double complex *c_dot = malloc(nst * sizeof(double complex));
    double complex *xf_c_dot = malloc(nst * sizeof(double complex));
    double complex *coef_new = malloc(nst * sizeof(double complex));
    double *eenergy = malloc(nst * sizeof(double));
    double **dv = malloc(nst * sizeof(double*));

    int ist, jst, iestep;
    double frac, edt, norm;

    // Precomputed RK4 constants
    const double sqrt2 = 1.41421356237309504880;
    const double rk_c1 = 0.5 * (-1.0 + sqrt2);
    const double rk_c2 = 1.0 - 0.5 * sqrt2;
    const double rk_c3 = -0.5 * sqrt2;
    const double rk_c4 = 1.0 + 0.5 * sqrt2;
    const double rk_c5 = 2.0 - sqrt2;
    const double rk_c6 = 2.0 + sqrt2;

    for(ist = 0; ist < nst; ist++){
        dv[ist] = malloc(nst * sizeof(double));
    }

    // Workspace for xf_cdot - allocated once
    double **xf_dec = malloc(nst * sizeof(double*));
    double *xf_rho = malloc(nst * sizeof(double));
    for(ist = 0; ist < nst; ist++){
        xf_dec[ist] = malloc(nst * sizeof(double));
    }

    // Workspace for cdot - allocated once
    double complex *cdot_na_term = malloc(nst * sizeof(double complex));

    frac = 1.0 / (double)nesteps;
    edt = dt * frac;

    for(iestep = 0; iestep < nesteps; iestep++){

        // Calculate cdot contribution originated from XF term
        xf_cdot(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, coef, xf_c_dot, xf_dec, xf_rho);

        // Interpolate energy and NACME terms between time t and t + dt
        for(ist = 0; ist < nst; ist++){
            eenergy[ist] = energy_old[ist] + (energy[ist] - energy_old[ist]) * (double)iestep * frac;
            for(jst = 0; jst < nst; jst++){
                dv[ist][jst] = nacme_old[ist][jst] + (nacme[ist][jst] - nacme_old[ist][jst])
                    * (double)iestep * frac;
            }
        }

        // Calculate k1
        cdot(nst, eenergy, dv, coef, c_dot, cdot_na_term);

        for(ist = 0; ist < nst; ist++){
            k1[ist] = edt * (c_dot[ist] + xf_c_dot[ist]);
            kfunction[ist] = 0.5 * k1[ist];
            coef_new[ist] = coef[ist] + kfunction[ist];
        }

        // Calculate k2
        cdot(nst, eenergy, dv, coef_new, c_dot, cdot_na_term);

        for(ist = 0; ist < nst; ist++){
            k2[ist] = edt * (c_dot[ist] + xf_c_dot[ist]);
            kfunction[ist] = rk_c1 * k1[ist] + rk_c2 * k2[ist];
            coef_new[ist] = coef[ist] + kfunction[ist];
        }

        // Calculate k3
        cdot(nst, eenergy, dv, coef_new, c_dot, cdot_na_term);

        for(ist = 0; ist < nst; ist++){
            k3[ist] = edt * (c_dot[ist] + xf_c_dot[ist]);
            kfunction[ist] = rk_c3 * k2[ist] + rk_c4 * k3[ist];
            coef_new[ist] = coef[ist] + kfunction[ist];
        }

        // Calculate k4
        cdot(nst, eenergy, dv, coef_new, c_dot, cdot_na_term);

        for(ist = 0; ist < nst; ist++){
            k4[ist] = edt * (c_dot[ist] + xf_c_dot[ist]);
            variation[ist] = (k1[ist] + rk_c5 * k2[ist] + rk_c6 * k3[ist] + k4[ist]) / 6.0;
            coef_new[ist] = coef[ist] + variation[ist];
        }

        // TODO : Is this part necessary?
        // Renormalize the coefficients
        norm = dot(nst, coef_new, coef_new);
        for(ist = 0; ist < nst; ist++){
            coef_new[ist] /= sqrt(norm);
            coef[ist] = coef_new[ist];
        }

    }

    if(verbosity >= 1){
        xf_print_coef(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, coef, dotpopdec);
    }

    for(ist = 0; ist < nst; ist++){
        free(dv[ist]);
        free(xf_dec[ist]);
    }
    free(xf_dec);
    free(xf_rho);
    free(cdot_na_term);

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(kfunction);
    free(variation);
    free(c_dot);
    free(xf_c_dot);
    free(coef_new);
    free(eenergy);
    free(dv);

}

// Routine for density propagation scheme in rk4 propagator
static void rk4_rho(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
    double *mass, double *energy, double *energy_old, double **sigma, double **nacme,
    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex **rho,
    int verbosity, double *dotpopdec){

    double complex **k1 = malloc(nst * sizeof(double complex*));
    double complex **k2 = malloc(nst * sizeof(double complex*));
    double complex **k3 = malloc(nst * sizeof(double complex*));
    double complex **k4 = malloc(nst * sizeof(double complex*));
    double complex **kfunction = malloc(nst * sizeof(double complex*));
    double complex **variation = malloc(nst * sizeof(double complex*));
    double complex **rho_dot = malloc(nst * sizeof(double complex*));
    double complex **xf_rho_dot = malloc(nst * sizeof(double complex*));
    double complex **rho_new = malloc(nst * sizeof(double complex*));
    double *eenergy = malloc(nst * sizeof(double));
    double **dv = malloc(nst * sizeof(double*));

    int ist, jst, iestep;
    double frac, edt;

    // Precomputed RK4 constants
    const double sqrt2 = 1.41421356237309504880;
    const double rk_c1 = 0.5 * (-1.0 + sqrt2);
    const double rk_c2 = 1.0 - 0.5 * sqrt2;
    const double rk_c3 = -0.5 * sqrt2;
    const double rk_c4 = 1.0 + 0.5 * sqrt2;
    const double rk_c5 = 2.0 - sqrt2;
    const double rk_c6 = 2.0 + sqrt2;

    for(ist = 0; ist < nst; ist++){
        k1[ist] = malloc(nst * sizeof(double complex));
        k2[ist] = malloc(nst * sizeof(double complex));
        k3[ist] = malloc(nst * sizeof(double complex));
        k4[ist] = malloc(nst * sizeof(double complex));
        kfunction[ist] = malloc(nst * sizeof(double complex));
        variation[ist] = malloc(nst * sizeof(double complex));
        rho_dot[ist] = malloc(nst * sizeof(double complex));
        xf_rho_dot[ist] = malloc(nst * sizeof(double complex));
        rho_new[ist] = malloc(nst * sizeof(double complex));
        dv[ist] = malloc(nst * sizeof(double));
    }

    // Workspace for xf_rhodot - allocated once
    double **xfrho_dec = malloc(nst * sizeof(double*));
    double *xfrho_pop = malloc(nst * sizeof(double));
    for(ist = 0; ist < nst; ist++){
        xfrho_dec[ist] = malloc(nst * sizeof(double));
    }

    frac = 1.0 / (double)nesteps;
    edt = dt * frac;

    for(iestep = 0; iestep < nesteps; iestep++){

        // Calculate rhodot contribution originated from XF term
        xf_rhodot(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, rho, xf_rho_dot, xfrho_dec, xfrho_pop);

        // Interpolate energy and NACME terms between time t and t + dt
        for(ist = 0; ist < nst; ist++){
            eenergy[ist] = energy_old[ist] + (energy[ist] - energy_old[ist]) * (double)iestep * frac;
            for(jst = 0; jst < nst; jst++){
                dv[ist][jst] = nacme_old[ist][jst] + (nacme[ist][jst] - nacme_old[ist][jst])
                    * (double)iestep * frac;
            }
        }

        // Calculate k1
        rhodot(nst, eenergy, dv, rho, rho_dot);

        for(ist = 0; ist < nst; ist++){
            for(jst = 0; jst < nst; jst++){
                k1[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
                kfunction[ist][jst] = 0.5 * k1[ist][jst];
                rho_new[ist][jst] = rho[ist][jst] + kfunction[ist][jst];
            }
        }

        // Calculate k2
        rhodot(nst, eenergy, dv, rho_new, rho_dot);

        for(ist = 0; ist < nst; ist++){
            for(jst = 0; jst < nst; jst++){
                k2[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
                kfunction[ist][jst] = rk_c1 * k1[ist][jst] + rk_c2 * k2[ist][jst];
                rho_new[ist][jst] = rho[ist][jst] + kfunction[ist][jst];
            }
        }

        // Calculate k3
        rhodot(nst, eenergy, dv, rho_new, rho_dot);

        for(ist = 0; ist < nst; ist++){
            for(jst = 0; jst < nst; jst++){
                k3[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
                kfunction[ist][jst] = rk_c3 * k2[ist][jst] + rk_c4 * k3[ist][jst];
                rho_new[ist][jst] = rho[ist][jst] + kfunction[ist][jst];
            }
        }

        // Calculate k4
        rhodot(nst, eenergy, dv, rho_new, rho_dot);

        for(ist = 0; ist < nst; ist++){
            for(jst = 0; jst < nst; jst++){
                k4[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
                variation[ist][jst] = (k1[ist][jst] + rk_c5 * k2[ist][jst]
                    + rk_c6 * k3[ist][jst] + k4[ist][jst]) / 6.0;
                rho[ist][jst] += variation[ist][jst];
            }
        }

    }

    if(verbosity >= 1){
        xf_print_rho(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, rho, dotpopdec); 
    }

    for(ist = 0; ist < nst; ist++){
        free(k1[ist]);
        free(k2[ist]);
        free(k3[ist]);
        free(k4[ist]);
        free(kfunction[ist]);
        free(variation[ist]);
        free(rho_dot[ist]);
        free(xf_rho_dot[ist]);
        free(rho_new[ist]);
        free(dv[ist]);
        free(xfrho_dec[ist]);
    }
    free(xfrho_dec);
    free(xfrho_pop);

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(kfunction);
    free(variation);
    free(rho_dot);
    free(xf_rho_dot);
    free(rho_new);
    free(eenergy);
    free(dv);

}


