import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pints

#Wrapping model in class for pints.ForwardModel

class SIR_model(pints.ForwardModel):

    def simulate(self, parameters, time):

        initial_con = [0.99, 0.01, 0] #Initial conditions of S, I, R

        def SIRc(y, t, p):

            S, I, R = y
            dsdt = -p[0]*S*I
            didt = (p[0]*S*I) - p[1]*I
            drdt = p[1]*I

            return [dsdt, didt, drdt]

        
        return sp.integrate.odeint(SIRc, initial_con, time, (parameters,))
    
    def n_parameters(self):
        return 2

    def n_outputs(self):
        return 3
    
model = SIR_model()

time = np.linspace(0, 730, 730)
true_p = [0.04, 0.01] #True values of beta and gamma
values = model.simulate(true_p, pints.vector(time)) #Generate data

problem = pints.MultiOutputProblem(model, time, values) #Inference problem with multiple outputs
score = pints.SumOfSquaresError(problem) #Calculates f = sum from i to n of (y_i - x_i)^2, y is data, x is model output, n total data points (not used again?)

sigma = 1e-6 #sigma really small to simulate no noise 
logliklihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma) #noise

x0 = [[0.5, 0.2], [0.4, 0.1]] #Starting points of each markov chain

mcmc= pints.MCMCController(logliklihood, 2, x0) #Samples from Markov chains?
mcmc.set_max_iterations(2000)
mcmc.set_log_to_screen(False)
chains = mcmc.run() #Running the MCMC routine

results = pints.MCMCSummary(chains=chains, time=mcmc.time(), parameter_names=['beta', 'gamma'])
#Calculates and prints key summaries of posterior samples and diagnostic quantities from MCMC chains. 
print(results)

plt.figure()
pints.plot.trace(chains) #Doesnt work
plt.show()
