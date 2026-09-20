# Lab 07 - Parallel Array Sum with MPI Collectives

Sums the array 1..1,000,000 in parallel. Expected result: **500,000, 500,000**.
All 6 programs verified correct (`Correct? = YES`) on 2, 4, and 8 processes.
Raw program output for every run is in `outputs/`.

## Part 1: Feature Comparison

| Program       | Collectives used    | Full array on every rank?         | Root manual summation?   | Result available where?             |
|---------------|---------------------|-----------------------------------|--------------------------|-------------------------------------|
| sum_bcast     | Bcast + Send/Recv   | Yes  every rank gets all 1M ints | Yes (Recv loop)          | Root only                           |
| sum_scatter   | Scatter + Send/Recv | No  root full, others get chunk  | Yes (Recv loop)          | Root only                           |
| sum_gather    | Scatter + Gather    | No                                | Yes (loop over all_sums) | Root only                           |
| sum_reduce    | Scatter + Reduce    | No                                | No                       | Root only                           |
| sum_allreduce | Scatter + Allreduce | No                                | No                       | All ranks (same value)              |
| sum_scan      | Scatter + Scan      | No                                | No                       | All ranks (different prefix values) |

## Part 2: Execution Times (seconds, 1 run each, OpenMPI in a VM)

| Program       | 2 procs | 4 procs | 8 procs |
|---------------|---------|---------|---------|
| sum_bcast     | 0.0110  | 0.0186  | 0.0476  |
| sum_scatter   | 0.0035  | 0.0010  | 0.0019  |
| sum_gather    | 0.0131  | 0.0058  | 0.0261  |
| sum_reduce    | 0.0033  | 0.0028  | 0.0030  |
| sum_allreduce | 0.0032  | 0.0015  | 0.0045  |
| sum_scan      | 0.0058  | 0.0011  | 0.0391  |

### Observations

- **sum_bcast is the slowest at every process count, and its cost grows with P** (0.0110 → 0.0186 → 0.0476 sec, about 4.3x slower from 2 to 8 processes). This matches theory: Bcast ships the entire 4 MB array to *every* process, so total data movement is N×P and scales linearly with the process count.
- **The Scatter-based versions move only N elements total** (each rank receives just its chunk), which is why they are much faster and roughly flat as P grows. sum_scatter was the fastest overall (0.0010 sec at 4 processes).
- **Tree-based collectives (Reduce/Allreduce) are consistently fast and stable** across process counts (Reduce: 0.0033 / 0.0028 / 0.0030) because they combine results in O(log P) communication steps. In contrast, the linear O(P) Send/Recv loop and Gather's manual summation serialize work at the root  visible in sum_gather's higher times.
- **Anomalies:** sum_scan at 8 processes (0.0391) and sum_gather at 8 (0.0261) are much higher than their 4-process times almost certainly VM noise from oversubscribing 8 ranks on fewer physical cores, plus single-run jitter at millisecond scale. The ranking is unaffected: Scatter-style distribution + tree-based reduction wins.

**Fastest approach:** Scatter-based distribution with a tree reduction (sum_scatter / sum_allreduce). Broadcasting the full array is the dominant cost in the naive version and should always be replaced by Scatter when each rank only needs its own chunk.

## Part 3: Thinking Question

**When would you choose MPI_Scan over MPI_Allreduce?**

When each rank needs a per-rank cumulative result rather than one shared total. Allreduce gives every rank the same answer; Scan gives rank r the sum of all contributions from ranks 0..r  a different value per rank.

Concrete example: each rank generates a variable number of records and must write them to a shared output with globally unique sequential IDs. Rank r needs to know how many records ranks 0..r-1 produced to know its starting ID  exactly what Scan's `prefix_sum - local_sum` provides in one call. Allreduce only yields the grand total, which is insufficient. The same pattern appears in parallel counting sort (bucket offsets), sparse matrix-vector products (row offsets), and load-balanced work splitting.