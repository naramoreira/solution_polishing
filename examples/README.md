The execution of the toy example provided generates a file `polish.log` like the extraction below. The log summarizes the progress of the algorithm during the solving process.

Notice that the percentage (`perc`) of variables being fixed (within the sub-set allowed to be fixed) starts high (80%) and decreases as the algorithm evolves, up to the point that no variables are being fixed (`perc=0`, which means the original problem). Of course, because this is a toy example, this happens very fast. For real examples (difficult models), the algorithm will spend more time in multiple iterations of the same step (same value of `perc`), and will maybe reach another stopping condition other than `perc=0`.

```
2025/05/19 12:28:31:DEBUG:: Polish called with total time limit = 600
2025/05/19 12:28:31:DEBUG:: Total time left to polish = 600
2025/05/19 12:28:31:DEBUG:: ----- Starting polish iteration 1 with perc = 0.800
2025/05/19 12:28:31:DEBUG:: Time since last improve = 0
2025/05/19 12:28:31:DEBUG:: Time limit = 30.0
2025/05/19 12:28:31:DEBUG:: Solution improved to -510.0 at iteration 1 with perc=0.800. Iteration's optimality gap = 0.0%, Solution improve = 51000.0%, Improve rate = 1.0
2025/05/19 12:28:31:DEBUG:: First solution's value = 0.0
2025/05/19 12:28:31:DEBUG:: Time per iteration increased using condition max_perc_time_find_1_sol. Iteration time limit changed from 60 to 100
2025/05/19 12:28:31:DEBUG:: Total time left to polish = 600
2025/05/19 12:28:31:DEBUG:: ----- Starting polish iteration 2 with perc = 0.800
2025/05/19 12:28:31:DEBUG:: Time since last improve = 0
2025/05/19 12:28:31:DEBUG:: Time limit = 30.0
2025/05/19 12:28:31:DEBUG:: Solution improved to -517.0 at iteration 2 with perc=0.800. Iteration's optimality gap = 0.0%, Solution improve = 1.3725490196078431%, Improve rate = 2.691272587466359e-05
2025/05/19 12:28:31:DEBUG:: Polish step updated by condition 1
2025/05/19 12:28:31:DEBUG:: Total time left to polish = 600
2025/05/19 12:28:31:DEBUG:: ----- Starting polish iteration 3 with perc = 0.650
2025/05/19 12:28:31:DEBUG:: Time since last improve = 0
2025/05/19 12:28:31:DEBUG:: Time limit = 30.0
2025/05/19 12:28:31:DEBUG:: Solution improved to -529.0 at iteration 3 with perc=0.650. Iteration's optimality gap = 0.0%, Solution improve = 2.321083172147002%, Improve rate = 1.6910748825642443
2025/05/19 12:28:31:DEBUG:: Total time left to polish = 600
2025/05/19 12:28:31:DEBUG:: ----- Starting polish iteration 4 with perc = 0.650
2025/05/19 12:28:31:DEBUG:: Time since last improve = 0
2025/05/19 12:28:31:DEBUG:: Time limit = 30.0
2025/05/19 12:28:31:DEBUG:: Solution improved to -535.0 at iteration 4 with perc=0.650. Iteration's optimality gap = 0.0%, Solution improve = 1.1342155009451798%, Improve rate = 0.4886578449905482
2025/05/19 12:28:31:DEBUG:: Polish step updated by condition 3
2025/05/19 12:28:31:DEBUG:: Total time left to polish = 600
...
2025/05/19 12:28:32:DEBUG:: ----- Starting polish iteration 11 with perc = 0.150
2025/05/19 12:28:32:DEBUG:: Time since last improve = 0
2025/05/19 12:28:32:DEBUG:: Time limit = 30.0
2025/05/19 12:28:32:DEBUG:: No improvements at iteration 11 with perc=0.150. Best known solution -550.0 found at iteration 9 with perc=0.300.
2025/05/19 12:28:32:DEBUG:: Polish step updated by condition 1
2025/05/19 12:28:32:DEBUG:: Total time left to polish = 599
2025/05/19 12:28:32:DEBUG:: ----- Starting polish iteration 12 with perc = 0.000
2025/05/19 12:28:32:DEBUG:: Time since last improve = 0
2025/05/19 12:28:32:DEBUG:: Time limit = 30.0
2025/05/19 12:28:33:DEBUG:: No improvements at iteration 12 with perc=0.000. Best known solution -550.0 found at iteration 9 with perc=0.300.
2025/05/19 12:28:33:INFO:-- Finished polishing the solution.
2025/05/19 12:28:33:INFO:-- Runtime 1 seconds.
2025/05/19 12:28:33:INFO:-- Final fixed percentage: 0.000.
2025/05/19 12:28:33:INFO:-- Best solution value -550.0 found at iteration 9 with perc  0.300.
2025/05/19 12:28:32:DEBUG:: Time since last improve = 0
2025/05/19 12:28:32:DEBUG:: Time limit = 30.0
2025/05/19 12:28:32:DEBUG:: Solution improved to -548.0 at iteration 7 with perc=0.400. Iteration's optimality gap = 0.0%, Solution improve = 0.3663003663003663%, Improve rate = 0.9963369963369964
2025/05/19 12:28:32:DEBUG:: Total time left to polish = 599
```
