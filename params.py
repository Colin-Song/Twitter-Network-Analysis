import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mean_threshold = 0.5
std_dev_threshold = 0.1
depths = range(1, 11)
lambdas = [0.2, 0.4, 0.6, 0.8]

def calculate_probabilities(lambda_decay, mean_threshold, std_dev_threshold, depths):
    probabilities = []
    for depth in depths:
        influence = math.exp(-lambda_decay * depth)
        probability = norm.cdf(influence, loc=mean_threshold, scale=std_dev_threshold)
        probabilities.append(probability)
    return probabilities

plt.figure(figsize=(15, 8))

for lambda_decay in lambdas:
    probabilities = calculate_probabilities(lambda_decay, mean_threshold, std_dev_threshold, depths)
    plt.plot(depths, probabilities, marker='o', linestyle='-', label=f'Lambda = {lambda_decay}')

plt.title('Probability that Influence Exceeds Threshold vs. Depth')
plt.xlabel('Depth')
plt.ylabel('Probability that Influence > Threshold')
plt.grid(True)
plt.xticks(list(depths))
#plt.gca().invert_yaxis()
plt.legend()
plt.show()