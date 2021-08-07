# GeneticAlgorithm_TSP


Build an advanced genetic algorithm with Python to solve different sizes of traveling sales problems. The algorithm includes the following components of genetic algorithm:


- Initialization: hybrid random and simple greedy initalization for the first population
- Selection: k-tournament selection to choose the parents for recombination
- Mutation: hybrid mutation scheme including swap, inversion, and scramble mutation
- Recombination: hybrid recombination scheme including PMX and HGrex crossover 
- Elimination: (\lambda+\mu) elimination
- Local search operator: hybrid local search scheme including inversion local search and adjacent swap local search
- Diversity promotion scheme: fitness sharing using normalzied hamming distance
