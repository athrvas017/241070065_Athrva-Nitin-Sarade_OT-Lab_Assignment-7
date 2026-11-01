# Track E Prompt

Implement main case study: College Exam Timetable and Invigilation Scheduling:
1. Synthetic dataset: 30 exams, 200 students, 6 rooms (capacities 25 to 55), seed 241070065.
2. Hard constraints: no student clash, room capacity not exceeded, single room per exam per slot.
3. Soft objective: minimize slots used + consecutive exam spread penalty.
4. Solvers: Naive sequential first-fit, DSATUR graph coloring, and Simulated Annealing (>= 5 runs with mean +/- std).
5. Construct strategy selection table and analyze why continuous optimizers (Adam) and quantum annealers (QAOA) are ill-suited for this problem.
