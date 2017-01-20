
// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>        // std::abs
#include <vector>
#include <algorithm>    // std::min_element, std::max_element
#include <ctime>
using namespace std;

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

// Function prototypes

struct Delta_int_params {
    vector<double> coeff_list;
    char prior_set;
    int k;
    int n_c;
    int h;
    double Q;
    double cbar_L;
    double cbar_U;
    double sigma;
    double Delta_k;
};

double Heaviside(double x);

double uniform_cn_likelihood(double cn, double cbar);

double gaussian_cn_likelihood(double cn, double cbar);

double uniform_log_cbar_prior(double cbar, double cbar_lower, double cbar_upper);

double gaussian_log_cbar_prior(double cbar, double sigma);

double likelihood_hash(char prior_set, double cn, double cbar);

double cbar_prior_hash(char prior_set, double cbar, double cbar_lower, double cbar_upper, double sigma);

double den_integrand(double x[], size_t dim, void *parameters);

double num_integrand(double x[], size_t dim, void *parameters);

double Delta_k_posterior(
    vector<double> coeff_list,
    char prior_set,
    int k,
    int n_c,
    int h,
    double Q,
    double cbar_L,
    double cbar_U,
    double sigma,
    double Delta_k);

//*********************************************************************//

int main (void)
{
    std::clock_t start;
    start = clock();

    // Params
    vector<double> coeff_list = {1.0, 1.0, 1.0};
    char prior_set = 'A';
    int k = 2;
    int n_c = 3;
    int h = 4;
    double Q = 0.33;
    double cbar_L = 0.001;
    double cbar_U = 1/cbar_L;
    double sigma = 0.5;
    double Delta_k = 0;


    cout << setprecision(6) << Delta_k_posterior(coeff_list, prior_set, k, n_c, h, Q, cbar_L, cbar_U, sigma, Delta_k) << endl;

    cout << "Time: " << (clock() - start) / (double) CLOCKS_PER_SEC << " seconds" << endl;
    return 0;
}

//*********************************************************************//

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


double Heaviside(double x) {
    return 0.5 * (sgn(x) + 1);
}


double uniform_cn_likelihood(double cn, double cbar) {
    return Heaviside(cbar - abs(cn)) / (2 * cbar);
}


double gaussian_cn_likelihood(double cn, double cbar) {
    return exp(-pow(cn, 2) / (2 * pow(cbar, 2))) / (sqrt(2.0 * M_PI) * cbar);
}


double uniform_log_cbar_prior(double cbar, double cbar_lower, double cbar_upper) {
    double val = 1/(cbar * log(cbar_upper/cbar_lower));
    return val * Heaviside(cbar - cbar_lower) * Heaviside(cbar_upper - cbar);
}


double gaussian_log_cbar_prior(double cbar, double sigma) {
    return exp(-pow(log(cbar) / sigma, 2) / 2) / (sqrt(2*M_PI) * cbar * sigma);
}


double likelihood_hash(char prior_set, double cn, double cbar) {
    if (prior_set == 'A' or prior_set == 'B') {
        return uniform_cn_likelihood(cn, cbar);
    }

    if (prior_set == 'C') {
        return gaussian_cn_likelihood(cn, cbar);
    }
    return 0;
}


double cbar_prior_hash(char prior_set, double cbar, double cbar_lower, double cbar_upper, double sigma) {
    if (prior_set == 'A' or prior_set == 'C') {
        return uniform_log_cbar_prior(cbar, cbar_lower, cbar_upper);
    }

    if (prior_set == 'B') {
        return gaussian_log_cbar_prior(cbar, sigma);
    }
    return 0;
}


double den_integrand(double x[], size_t dim, void *parameters) {
    struct Delta_int_params *params = (struct Delta_int_params *)parameters;

    vector<double> coeff_list = params->coeff_list;
    char prior_set = params->prior_set;
    // double Q = params->Q;
    // double k = params->k;
    double n_c = params->n_c;
    // double h = params->h;
    double cbar_L = params->cbar_L;
    double cbar_U = params->cbar_U;
    double sigma = params->sigma;

    double value = cbar_prior_hash(prior_set, x[0], cbar_L, cbar_U, sigma);

    for( int n = 0; n < n_c; n++ ) {
        value *= likelihood_hash(prior_set, coeff_list[n], x[0]);
    }
    return value;
}


double num_integrand(double x[], size_t dim, void *parameters) {
    struct Delta_int_params *params = (struct Delta_int_params *)parameters;

    // Parameters
    vector<double> coeff_list = params->coeff_list;
    char prior_set = params->prior_set;
    double Q = params->Q;
    double k = params->k;
    // double n_c = params->n_c;
    double h = params->h;
    // double cbar_L = params->cbar_L;
    // double cbar_U = params->cbar_U;
    // double sigma = params->sigma;
    double Delta_k = params->Delta_k;

    double value = den_integrand(x, dim, parameters);

    // Useful variables
    double cbar = x[0];
    double ckplus1 = Delta_k/pow(Q, k+1);
    double cm;

    for (unsigned int m = 1; m < h; m++) {
        // cout << m << endl;
        cm = x[m] * cbar;
        value *= cbar * likelihood_hash(prior_set, cm, cbar);
        ckplus1 -= cm * pow(Q, m);
    }

    value *= likelihood_hash(prior_set, ckplus1, cbar);

    return value;
}


double Delta_k_posterior(
    vector<double> coeff_list,
    char prior_set,
    int k,
    int n_c,
    int h,
    double Q,
    double cbar_L,
    double cbar_U,
    double sigma,
    double Delta_k)
{
    // vector<double> coeff_list(coeff_l, coeff_l + sizeof(coeff_l) / sizeof(coeff_l[0]));
    // Abs of coeffs
    vector<double> abs_coeff_list (coeff_list.size());
    copy(coeff_list.begin(), coeff_list.end(), abs_coeff_list.begin());
    for(unsigned int i = 0; i < abs_coeff_list.size(); i++) {
        if(abs_coeff_list[i] < 0)abs_coeff_list[i] *= -1; //make positive.    _OR_   use numbers[i] = abs(numbers[i]);
        // std::cout<<abs_coeff_list[i]<<std::endl;
    }

    // Set up parameters
    struct Delta_int_params parameters = {coeff_list, prior_set, k, n_c, h, Q, cbar_L, cbar_U, sigma, Delta_k};

    // Limits
    double lower_limit[h];
    double upper_limit[h];

    if (prior_set == 'A') {
        lower_limit[0] = max(*max_element(abs_coeff_list.begin(), abs_coeff_list.end()), cbar_L);
        upper_limit[0] = cbar_U;
        for(int i = 1; i < h; i++) {
            lower_limit[i] = -1;
            upper_limit[i] = 1;
        }
    }
    if (prior_set == 'B') {
        lower_limit[0] = *max_element(abs_coeff_list.begin(), abs_coeff_list.end());
        upper_limit[0] = exp(5 * sigma);
        for(int i = 1; i < h; i++) {
            lower_limit[i] = -1;
            upper_limit[i] = 1;
        }
    }
    if (prior_set == 'C') {
        lower_limit[0] = cbar_L;
        double sum_sq = 0;
        for(int j = 0; j < n_c; j++) {
            sum_sq += pow(coeff_list[j], 2);
        }
        upper_limit[0] = min(cbar_U, 4 * sum_sq);
        for(int i = 1; i < h; i++) {
            lower_limit[i] = -4;
            upper_limit[i] = 4;
        }
    }

    // Set up monte carlo
    double den_result, den_error;     // result and error for denominator
    double num_result, num_error;     // result and error for numerator
    size_t calls = 500000;

    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup ();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    // Evaluate denominator

    gsl_monte_function denominator = { &den_integrand, 1, &parameters };
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (1);
    gsl_monte_vegas_integrate (&denominator, &lower_limit[0], &upper_limit[0], 1,
                               10000, r, s, &den_result, &den_error);

    do
    {
        gsl_monte_vegas_integrate (&denominator, &lower_limit[0], &upper_limit[0], 1, calls / 5, r, s,
                     &den_result, &den_error);
        cout
        << "result = " << setprecision (6) << den_result
        << " sigma = " << setprecision (6) << den_error
        << " chisq/dof = " << setprecision (1) << s->chisq << endl;
    }
    while (fabs (s->chisq - 1.0) > 0.5);



    // Evaluate numerator

    gsl_monte_function numerator = { &num_integrand, h, &parameters };
    gsl_monte_vegas_state *s_num = gsl_monte_vegas_alloc (h);
    gsl_monte_vegas_integrate (&numerator, lower_limit, upper_limit, h,
                               10000, r, s_num, &num_result, &num_error);

    do
    {
        gsl_monte_vegas_integrate (&numerator, lower_limit, upper_limit, h, calls / 5, r, s_num,
                     &num_result, &num_error);
        cout
        << "result = " << setprecision (6) << num_result
        << " sigma = " << setprecision (6) << num_error
        << " chisq/dof = " << setprecision (1) << s_num->chisq << endl;
    }
    while (fabs (s_num->chisq - 1.0) > 0.5);

    // cout << setprecision (6) << num_result / (pow(Q, k+1) * den_result) << endl;
    return num_result / (pow(Q, k+1) * den_result);
}


//*********************************************************************//

