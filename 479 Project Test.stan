// Stan code for a basic logistic regression model with a single predictor
// Note that it assumes that the predictors have been standardized to have mean 0 and 
// standard deviation 1

data{
  int<lower = 0> n; // number of data points
  int<lower = 0> popularity[n]; // total number of likes+dislikes
  int<lower = 0> likes[n]; // total number of likes
  vector[n] funny; // vector of funny predictors
  //vector[n] quick;
  //vector[n] pat;
  //vector[n] celeb;
  //vector[n] danger;
  //vector[n] animals;
  //vector[n] sex;
  
 
  int n_grid; // number of points in the grid of x values at which we want to evaluate P(y = 1|x)
  vector[n_grid] x_grid;

  real mu_alpha; // prior mean for intercept on standardized scale
  real<lower = 0> sigma_alpha; // prior sd for intercept on standardized scale
  real mu_beta; // prior mean for slopes
  real<lower = 0> sigma_beta;// prior sd for slope 
 
}
parameters{
  real alpha; // intercept (on standardized scale)
  real b_funny; // slopes (on standardized scale)
  //real b_quick;
  //real b_pat;
  //real b_celeb;
  //real b_danger;
  //real b_animals;
  //real b_sex;
}

model{
  alpha ~ normal(0, sigma_alpha);
  b_funny ~ normal(0, sigma_beta);
  //b_quick ~ normal(0, sigma_std_beta);
  //b_pat ~ normal(0, sigma_std_beta);
  //b_celeb ~ normal(0, sigma_std_beta);
  //b_danger~ normal(0, sigma_std_beta);
  //b_animals ~ normal(0, sigma_std_beta);
 // b_sex ~ normal(0, sigma_std_beta);
  for(i in 1:n){
    likes[i] ~ binomial_logit(popularity[i], alpha + funny[i]*b_funny);
   // y[i] ~ bernoulli_logit(alpha + funny[i]*b_funny + quick[i]*b_quick + pat[i]*b_pat + celeb[i]*b_celeb + 
    //danger[i]*b_danger + animals[i]*b_animals + sex[i]*b_sex);
    //binomial_logit
  }
}

//generated quantities{
  //vector<lower = 0>[n_grid] prob_grid; // prob. that y = 1 at each grid point
  
  //for(i in 1:n_grid){
    //prob_grid[i] = inv_logit(alpha + x_grid[i] * b_funny);
  //}
  
//}
