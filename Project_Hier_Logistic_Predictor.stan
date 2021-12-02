// Stan code for a basic logistic regression model with a single predictor
// Note that it assumes that the predictors have been standardized to have mean 0 and 
// standard deviation 1

data{
  int<lower = 0> n; // number of data points
  int<lower = 0> J; // number of different intercepts/groups
  
  int<lower = 0> popularity[n]; // total number of likes+dislikes
  int<lower = 0> likes[n]; // total number of likes
  int<lower = 1, upper = J> group_id[n];// indicates to which group observation i belongs
  
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

 
}
parameters{
  real<lower = 0> tau; // sd of the group-specific intercepts across *ALL* groups (on standardized scale)
  real alpha_bar; // mean of group-specific intercepts across *ALL* groups (on standardized scale)
  vector[J] alpha_aux; // auxiliary parameter for group-specific intercepts (on standardized scale)
  real b_funny; // slopes (on standardized scale)
  real b_quick;
  real b_pat;
  real b_celeb;
  real b_danger;
  real b_animals;
  real b_sex;
}
transformed parameters{
  vector[J] alpha; // vector of group-specific intercepts 
  for(j in 1:J){
    alpha[j] = alpha_bar + tau * alpha_aux[j];
  }
}

model{
  alpha_bar ~ normal(1, 1);
  tau ~ student_t(7, 0.0, 1);
  
  for(j in 1:J){
    alpha_aux[j] ~ normal(0,1);
  }
  
  b_funny ~ normal(0, .25);
  b_quick ~ normal(0, .25);
  b_pat ~ normal(0, .25);
  b_celeb ~ normal(0, .25);
  b_danger~ normal(0, .25);
  b_animals ~ normal(0, .25);
  b_sex ~ normal(0, .25);
  for(i in 1:n){
      likes[i] ~ binomial_logit(popularity[i], alpha[group_id[i]] + funny[i]*b_funny + quick[i]*b_quick + pat[i]*b_pat + celeb[i]*b_celeb + 
     danger[i]*b_danger + animals[i]*b_animals + sex[i]*b_sex);

  }
}

generated quantities{
  matrix<lower = 0, upper = 1>[J+1,n_grid] prob_grid; // prob. of popularity at each grid point
  real alpha_new;  // alpha for the new group
  
  alpha_new = normal_rng(alpha_bar,tau);
  
    // make predictions for original J groups first
  for(i in 1:n_grid){
    for(j in 1:J){
      prob_grid[j,i] = inv_logit(alpha[j] + funny_grid[i]*b_funny + quick_grid[i]*b_quick + pat_grid[i]*b_pat + 
      celeb_grid[i]*b_celeb + danger_grid[i]*b_danger + animals_grid[i]*b_animals + sex_grid[i]*b_sex); // prediction for j-th year at i-th grid point
    }
    prob_grid[J+1,i] = inv_logit(alpha_new + funny_grid[i]*b_funny + quick_grid[i]*b_quick + pat_grid[i]*b_pat + 
      celeb_grid[i]*b_celeb + danger_grid[i]*b_danger + animals_grid[i]*b_animals + sex_grid[i]*b_sex); // prediction for new group at i-th grid point

  }
  
}
