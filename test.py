import r0481422
from matplotlib import pyplot
import statistics
import pandas as pd

if __name__ == "__main__":


    mean_fitnesses = []
    best_fitnesses = []

    num_runs = 1

    for i in range(num_runs):
        print("The current num_runs = ", i)

        a = r0481422.r0481422()
        current_mean_fitness, current_best_fitness = a.optimize("./tour929.csv")
        mean_fitnesses.append(current_mean_fitness)
        best_fitnesses.append(current_best_fitness)

    avg_mean_fitness = sum(mean_fitnesses) / len(mean_fitnesses)
    avg_best_fitness = sum(best_fitnesses) / len(best_fitnesses)

    # Printing average of the list
    print("Average of the mean_fitnesses =", round(avg_mean_fitness, 2))
    print("Average of the best_fitnesses =", round(avg_best_fitness, 2))

    # Printing best of the list
    print("The best solution in this run =", min(best_fitnesses))
    # Printing best of the list
    print("The STD of best solution in this run =", statistics.pstdev(best_fitnesses))
    print("The STD of average solution in this run =", statistics.pstdev(mean_fitnesses))

    # Plot the histogram of the num_runs result for best and mean_fitnesses

    pyplot.figure(figsize=(8, 6))
    pyplot.hist(best_fitnesses, bins=100, alpha=0.5, label='best', color='blue')
    pyplot.hist(mean_fitnesses, bins=100, alpha=0.5, label='mean', color='yellow')
    pyplot.xlabel("Objective value")  # Add ", fontsize = #" to control fontsize
    pyplot.ylabel("Frequency")
    pyplot.legend(loc='upper right')
    # pyplot.show()

    pyplot.savefig("best_mean_histogram.png")

    df = pd.DataFrame(best_fitnesses)
    df.to_csv('best_fitnesses.csv', index=False)

    df = pd.DataFrame(mean_fitnesses)
    df.to_csv('mean_fitnesses.csv', index=False)



