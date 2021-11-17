// Stan code for a basic logistic regression model with a single predictor
// Note that it assumes that the predictors have been standardized to have mean 0 and 
// standard deviation 1

data{
  int<lower = 0> n; // number of data points
  int<lower = 0, upper = 1> y[n]; // array of observed 0-1 outcomes
  vector[n] funny; // vector of funny predictors
  vector[n] quick;
  vector[n] pat;
  vector[n] celeb;
  vector[n] danger;
  vector[n] animals;
  vector[n] sex;
  
  real x_mean; // mean of our single predictor (for re-scaling at the end)
  real<lower = 0> x_sd; // standard deviations of our single predictor (for re-scaling at the end)
  
  int n_grid; // number of points in the grid of x values at which we want to evaluate P(y = 1|x)
  vector[n_grid] x_grid;

  real mu_std_alpha; // prior mean for intercept on standardized scale
  real<lower = 0> sigma_std_alpha; // prior sd for intercept on standardized scale
  real mu_std_beta; // prior mean for slopes
  real<lower = 0> sigma_std_beta;// prior sd for slope 
 
}
parameters{
  real alpha; // intercept 
  real b_funny; // funny coefficient
  real b_quick;
  real b_pat;
  real b_celeb;
  real b_danger;
  real b_animals;
  real b_sex;
}

model{
  alpha ~ normal(mu_std_alpha, sigma_std_alpha);
  b_funny ~ normal(mu_std_beta, sigma_std_beta);
  b_quick ~ normal(mu_std_beta, sigma_std_beta);
  b_pat ~ normal(mu_std_beta, sigma_std_beta);
  b_celeb ~ normal(mu_std_beta, sigma_std_beta);
  b_danger~ normal(mu_std_beta, sigma_std_beta);
  b_animals ~ normal(mu_std_beta, sigma_std_beta);
  b_sex ~ normal(mu_std_beta, sigma_std_beta);
  for(i in 1:n){
    y[i] ~ bernoulli_logit(alpha + funny[i]*b_funny + quick[i]*b_quick + pat[i]*b_pat + celeb[i]*b_celeb + 
    danger[i]*b_danger + animals[i]*b_animals + sex[i]*b_sex);
  }
}
generated quantities{
  real offset; // we have to adjust the intercept back to the original scale
  real alpha;
  real beta;
  vector<lower = 0, upper = 1>[n_grid] prob_grid; // prob. that y = 1 at each grid point
  
  offset = std_beta/x_sd * x_mean;
  alpha = std_alpha - offset;
  beta = std_beta/x_sd;
  
  for(i in 1:n_grid){
    prob_grid[i] = inv_logit(alpha + x_grid[i] * beta);
  }
  
}
