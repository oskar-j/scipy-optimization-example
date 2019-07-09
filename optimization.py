import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class Evaluator:
    '''
    The evaluator class.
    Responsible for calculating score for the candidate solution.
    '''

    @staticmethod
    def generate_range(mu, sigma, acc):
        rv1, rv2 = norm(loc=mu[0], scale=sigma[0]), norm(loc=mu[1], scale=sigma[1])
        start = min(rv1.ppf(acc), rv2.ppf(1 - acc))
        end = max(rv1.ppf(acc), rv2.ppf(1 - acc))
        return start, end

    @staticmethod
    def generate_x(mu_true, sigma_true, mu_est, sigma_est, num, acc):
        start_true, end_true = Evaluator.generate_range(mu_true, sigma_true, acc)
        start_est, end_est = Evaluator.generate_range(mu_est, sigma_est, acc)
        return np.linspace(min(start_true, start_est), max(end_true, end_est), num=num)

    @staticmethod
    def generate_y(x, mu, sigma, weights):
        rv1, rv2 = norm(loc=mu[0], scale=sigma[0]), norm(loc=mu[1], scale=sigma[1])
        cdf = weights[0] * rv1.cdf(x) + weights[1] * rv2.cdf(x)
        cdf = np.append(np.insert(cdf, 0, 0), 1)
        return np.diff(cdf, 1)

    @staticmethod
    def calculate_distance(mu_true, sigma_true, weights_true, mu_est, sigma_est, weights_est, num=1000, acc=0.001):
        x = Evaluator.generate_x(mu_true, sigma_true, mu_est, sigma_est, num, acc)
        y_true = Evaluator.generate_y(x, mu_true, sigma_true, weights_true)
        y_est = Evaluator.generate_y(x, mu_est, sigma_est, weights_est)
        return np.abs(y_true - y_est).sum() / 2


class Sampler:
    '''
    Generates observations from the 2 normal distributions
    '''

    @staticmethod
    def generate_observations(n, true_mu, true_sigma, true_weights):
        '''
        Generates list of n observations from two normal distributions: N(true_mu[0], true_sigma[0]) and
        N(true_mu[1], true_sigma[1]). Each observation has probability true_weights[i] to
        belong to N(true_mu[i], true_sigma[i]
        Args:
            n: integer, number of observations to generate
            true_mu: list of floats, length 2
            true_sigma: list of floats, length 2
            true_weights: list of floats, each element is between 0 and 1, sum(true_weights)==1
        Output:
            result: list of floats, length n, representing random draws from the 2 normal distributions as described.
        '''
        np.random.seed(42)
        assert sum(true_weights) == 1
        n1 = np.random.binomial(n, true_weights[0])
        obs0 = np.random.normal(loc=true_mu[0], scale=true_sigma[0], size=n1).tolist()
        obs1 = np.random.normal(loc=true_mu[1], scale=true_sigma[1], size=n - n1).tolist()
        result = np.array(obs0 + obs1).reshape(-1, 1)
        np.random.shuffle(result)
        return result


class Problem:

    @staticmethod
    def estimate_parameters(obs):
        # Put the solution here
        obs_array = np.array(obs)
        obs_len = len(obs)

        mu, std = norm.fit(obs_array)
        weight_1 = 0.5

        init_guess = [mu, mu, std, std, weight_1]

        def f(parameters):
            mu1, mu2, std1, std2, weight_1 = parameters
            weight_2 = 1.0 - weight_1
            test_obs = Sampler.generate_observations(obs_len, [mu1, mu2], [std1, std2], [weight_1, weight_2])
            dist = np.abs(obs - test_obs).sum() / 2
            return dist

        bounds = [[-5, 10], [-5, 10], [0.1, 3], [0.1, 3], [0.01, 0.99]]

        result = minimize(f, init_guess, bounds=bounds, method='L-BFGS-B', tol=10**-8)

        mu1, mu2, std1, std2, weight_1 = result.x

        estimate_mu = [mu1, mu2]
        estimate_std = [std1, std2]
        estimate_weight = [weight_1, 1 - weight_1]

        return estimate_mu, estimate_std, estimate_weight


# Sample call and print, feel free to adjust
n, true_mu, true_sigma, true_weights = 100, [-3, 5], [2, 1], [0.4, 0.6]

obs = Sampler.generate_observations(n, true_mu, true_sigma, true_weights)
est_mu, est_sigma, est_weights = Problem.estimate_parameters(obs)
print("Estimated parameters:")
print((est_mu, est_sigma, est_weights))
print("Calculated distance to true distribution:")
print(Evaluator.calculate_distance(true_mu, true_sigma, true_weights,
                                   est_mu, est_sigma, est_weights, 1000))

n, true_mu, true_sigma, true_weights = 1000, [10, 1], [2, 2], [0.2, 0.8]

obs = Sampler.generate_observations(n, true_mu, true_sigma, true_weights)
est_mu, est_sigma, est_weights = Problem.estimate_parameters(obs)
print("Estimated parameters:")
print((est_mu, est_sigma, est_weights))
print("Calculated distance to true distribution:")
print(Evaluator.calculate_distance(true_mu, true_sigma, true_weights,
                                   est_mu, est_sigma, est_weights, 1000))

# Hard case - below optimizer fails to find a proper solution
n, true_mu, true_sigma, true_weights = 150, [1, 4], [1.5, 2.5], [0.2, 0.8]

obs = Sampler.generate_observations(n, true_mu, true_sigma, true_weights)
est_mu, est_sigma, est_weights = Problem.estimate_parameters(obs)
print("Estimated parameters:")
print((est_mu, est_sigma, est_weights))
print("Calculated distance to true distribution:")
print(Evaluator.calculate_distance(true_mu, true_sigma, true_weights,
                                   est_mu, est_sigma, est_weights, 1000))
