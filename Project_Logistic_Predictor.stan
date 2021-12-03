// Stan code for a basic logistic regression model with a single predictor
// Note that it assumes that the predictors have been standardized to have mean 0 and 
// standard deviation 1

data{
  int<lower = 0> n; // number of data points
  
  int<lower = 0> popularity[n]; // total number of likes+dislikes
  int<lower = 0> likes[n]; // total number of likes
  
  vector[n] funny; // vector of funny predictors
  vector[n] quick;
  vector[n] pat;
  vector[n] celeb;
  vector[n] danger;
  vector[n] animals;
  vector[n] sex;
  
 
  int n_grid; // number of points in the grid of x values at which we want to evaluate P(y = 1|x)
  vector[n_grid] funny_grid;
  vector[n_grid] quick_grid;
  vector[n_grid] pat_grid;
  vector[n_grid] celeb_grid;
  vector[n_grid] danger_grid;
  vector[n_grid] animals_grid;
  vector[n_grid] sex_grid;

 // real mu_alpha; // prior mean for intercept on standardized scale
  //real<lower = 0> sigma_alpha; // prior sd for intercept on standardized scale
  //real mu_b_funny; // prior mean for slope of funny variable
  //real<lower = 0> sigma_b_funny;// prior sd for slope of funny variable
 
}
parameters{
  real alpha; // intercept (on standardized scale)
  real b_funny; // slopes (on standardized scale)
  real b_quick;
  real b_pat;
  real b_celeb;
  real b_danger;
  real b_animals;
  real b_sex;
}

model{

  alpha ~ normal(3.27, .1);
  b_funny ~ normal(0, .5);
  b_quick ~ normal(0, 2.75);
  b_pat ~ normal(0, .5);
  b_celeb ~ normal(0, .25);
  b_danger~ normal(0, .5);
  b_animals ~ normal(0, .5);
  b_sex ~ normal(0, .5);
  for(i in 1:n){
    likes[i] ~ binomial_logit(popularity[i], alpha + funny[i]*b_funny + quick[i]*b_quick + pat[i]*b_pat + celeb[i]*b_celeb + 
    danger[i]*b_danger + animals[i]*b_animals + sex[i]*b_sex);
  }
}

generated quantities{
  vector<lower = 0, upper = 1>[n_grid] prob_grid; // likes/popularity for each combination of variables
  
  for(i in 1:n_grid){
    prob_grid[i] = inv_logit(alpha + funny_grid[i] * b_funny + pat_grid[i]*b_quick + celeb_grid[i]*b_celeb +
    danger_grid[i]*b_danger + animals_grid[i]*b_animals + sex_grid[i]*b_sex);
  }
  
}
