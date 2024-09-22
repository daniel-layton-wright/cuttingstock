# Cutting Stock Solution

This implements a solution to the cutting stock problem using the Dantzig-Wolfe decomposition and column generation.

## Usage and Input

Details about usage can be found by running:

```
python cutting_stock.py --help
```

To run the script and obtain a solution, use the following command:

```
python cutting_stock.py <input_file> <output_file>
```

The format of the input file is as follows:

* The first line should have the length $L$ of each roll
* The subsequent lines should have the length $l_i$ and demand $d_i$ for each order:

```
L
l_1 d_1
l_2 d_2
...
l_n d_n
```

For example:

```
100
20 10
30 5
40 15
```

### Time Limit
After column generation is complete, the program solves the integer master problem but this can be slow depending on the problem. The solver will stop and use the best solution found after a specified time limit. To set the limit use the `--time_limit` flag:

```
python cutting_stock.py <input_file> <output_file> --time_limit <time_limit>
```

The default time limit is 60 seconds.

## Output

The output file is written in the following format:
* The first line is the objective function value
* The second line is the value of $\theta_p$ for each pattern $p$ (column of $A$)
* The next line is left blank
* The subsequent lines are the matrix $A$ in the master problem

For example:
```
10
4 3 2 1 

1 0 0 0 
0 1 0 0 
0 0 1 0 
0 0 0 1 
```