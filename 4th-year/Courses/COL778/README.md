# COL778 - Principles of Autonomous System
This repository ccontains assignments, lecture notes, books and other resources for the course [COL778](https://lily-molybdenum-65d.notion.site/COL778-Principles-of-Autonomous-Systems-eb895fb5ac0d4edc860533439cce8fa7)/[864](https://lily-molybdenum-65d.notion.site/COL864-Special-Topics-in-AI-Embodied-AI-28e0e65bfef34ee8a9905375f5e419b3) offered by [Prof. Rohan Paul](https://www.cse.iitd.ac.in/~rohanpaul/index.html) in the semester 2302 at IITD. The [course](https://lily-molybdenum-65d.notion.site/COL778-Principles-of-Autonomous-Systems-eb895fb5ac0d4edc860533439cce8fa7) deals with the algorithmic aspects of intelligent robotics and more generally autonomous systems. 
## Topic wise resources
1) State Representation :
   - Read about homogenous transforms [here](https://mecharithm.com/learning/lesson/homogenous-transformation-matrices-configurations-in-robotics-12#).
   - Read about Camera Calibration [here](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node3.html)

2) Bayes Filtering Algorithm :
   - Read [Bishop](https://github.com/iamsecretlyflash/COL774/blob/main/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) Chapter 8 to learn about Graphical Models
   - Can be broken down into two steps
     - <details>
        <summary>Action update step $p(x_t | z_1,....z_{t-1}, u_1,....u_t)$</summary>
        <br>
        $p(x_t | z_1,....z_{t-1}, u_1,....u_t) = \int_{x_{t-1}}p(x_t | z_1,....z_{t-1}, u_1,....u_t, x_{t-1})p(x_{t-1} | z_1,....z_{t-1},u_1,....u_t)dx_{t-1}$ <br>
       Now, $p(x_t | z_1,....z_{t-1}, u_1,....u_t, x_{t-1}) = p(x_t | x_{t-1}, u_t)$ and $p(x_{t-1} | z_1,....z_{t-1},u_1,....u_t) = Bel(x_{t-1})$
       $\therefore p(x_t | z_1,....z_{t-1}, u_1,....u_t) = \int_{x_{t-1}}p(x_t | x_{t-1}, u_t)Bel(x_{t-1})dx_{t-1}$
       <br>
       or, $\overline{Bel}(X_t) = \int_{x_{t-1}}p(x_t | x_{t-1}, u_t)Bel(x_{t-1})dx_{t-1}$
     
     </details>
     
      - <details>
        <summary>Measurement update step $p(x_t | z_1,....z_{t}, u_1,....u_t)$</summary>
        <br>
        $p(x_t | z_1,....z_{t}, u_1,....u_t) = \eta * p(z_t | x_t,z_1,....z_{t-1}, u_1,....u_t) * p(x_t |z_1,....z_{t-1}, u_1,....u_t)$

        Now, $p(z_t | x_t,z_1,....z_{t-1}, u_1,....u_t) = p(z_t | x_t)$ <br>
        $\therefore Bel(x_t) = \eta * p(z_t | x_t) * \overline{Bel}(x_t)$
     
     </details>
3) State Estimation using Kalman Filters :
   - Read about Conditioned Joint Gaussian PDFs [here](https://bmeyers.github.io/conditional_distribution_for_jointly_gaussian_random_vectors/).
   -  <details>
      <summary>Calculation of co-variance matrix $\sum_{X_tX_{t+1}}$ used in derivation of the action update equations</summary>
      <br>
         
      $\sum_{X_{t+1}X_t}$ = $\mathbb{E}[(X_{t+1} - \mu_{X_{t+1}})(X_t - \mu_{X_t})^T]$
      
      $\sum_{X_{t+1}X_t}$ = $\mathbb{E}[(X_{t+1} - A_t\mu_t - B_t\mu_t)(X_t - \mu_t)^T]$
      
      $\sum_{X_{t+1}X_t}$ = $\mathbb{E}[(A_tX_t - A_t\mu_t + \epsilon_t)(X_t - \mu_t)^T]$
      
      $\sum_{X_{t+1}X_t}$ = $\mathbb{E}[(A_tX_t - A_t\mu_t + \epsilon_t)(X_t^T - \mu_t^T)]$
      
      $\sum_{X_{t+1}X_t}$ = $\mathbb{E}[A_tX_tX_t^T - A_t\mu_tX_t^T + \epsilon_tX_t - A_tX_t\mu_t^T + A_t\mu_t\mu_t^T - \epsilon_t\mu_t^T]$
      
      Since $\epsilon_t$ is an independent zero-mean random variable, all terms with $\epsilon_t$ go to 0
      
      $\sum_{X_{t+1}X_t}$ = $\mathbb{E}[A_tX_tX_t^T - 2 * A_t\mu_tX_t^T + A_t\mu_t\mu_t^T]$
      
      $\sum_{X_{t+1}X_t}$ = $A_t\mathbb{E}[X_tX_t^T] - A_t\mu_t\mu_t^T$
      
      $\sum_{X_{t+1}X_t}$ = $A_t(\mathbb{E}[X_tX_t^T] - \mathbb{E}[X_t]\mathbb{E}[X_t]^T)$
      
      $\sum_{X_{t+1}X_t}$ = $A_t\sum_{t|0:t}$
      
      </details>
4) MDPs :
   - Read about Prioritized Value Iteration [here](https://web2.qatar.cmu.edu/~gdicaro/15281/additional/variants-of-value-iteration.pdf).
   - Read about Forward Search for MPDs [here](https://www.khoury.northeastern.edu/home/camato/5100/mdps.pdf).
