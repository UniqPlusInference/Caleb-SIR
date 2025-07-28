import numpy as np
import scipy as sp
import pints
import matplotlib.pyplot as plt

#SIR model with demography
class SIR_modelWD(pints.ForwardModel):

    def simulate(self, parameters, time):

        initial_con = [0.99, 0.01, 0] #Initial conditions of S, I, R, mu
        death_rate = float(1/90)
        def SIRWD(y, t, p):
            
            S, I, R = y
            dsdt = death_rate - p[0]*S*I - death_rate*S
            didt = (p[0]*S*I) - p[1]*I - death_rate*I
            drdt = p[1]*I - death_rate*R
            
            return [dsdt, didt, drdt]

        
        return sp.integrate.odeint(SIRWD, initial_con, time, (parameters,))
    
    def n_parameters(self):
        return 2

    def n_outputs(self):
        return 3
    
model = SIR_modelWD()

time = np.linspace(0, 15, 15)
true_p = [1.66, float(1/2.2)] #Values from 1978 influenza incident in an English boarding school
values = model.simulate(true_p, pints.vector(time))

S,I,R = values.T

problem = pints.MultiOutputProblem(model, time, values)

sigma = 1e-6
loglikelihood = pints.GaussianKnownSigmaLogLikelihood(problem, sigma)

x0 = [[0.6, 0.4], [1, 0.6]]
mcmc = pints.MCMCController(loglikelihood, 2, x0)
mcmc.set_max_iterations(2000)
mcmc.set_log_to_screen(False)
chains = mcmc.run()

results = pints.MCMCSummary(chains=chains, time=mcmc.time(), parameter_names=['beta', 'gamma'])
print(results)

plt.figure(figsize=(10, 6))
plt.plot(time, S, label='Susceptible')
plt.plot(time, I, label='Infected')
plt.plot(time, R, label='Recovered')
plt.xlabel('Time(days)')
plt.ylabel('Population proportion')
plt.legend()
plt.show()
