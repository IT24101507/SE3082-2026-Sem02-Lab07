CC     = mpicc
CFLAGS = -Wall -O2
NP     = 4
MPIRUN = mpirun --oversubscribe

PROGS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

all: $(PROGS)

sum_bcast: sum_bcast.c
    $(CC) $(CFLAGS) -o $@ $<

sum_scatter: sum_scatter.c
    $(CC) $(CFLAGS) -o $@ $<

sum_gather: sum_gather.c
    $(CC) $(CFLAGS) -o $@ $<

sum_reduce: sum_reduce.c
    $(CC) $(CFLAGS) -o $@ $<

sum_allreduce: sum_allreduce.c
    $(CC) $(CFLAGS) -o $@ $<

sum_scan: sum_scan.c
    $(CC) $(CFLAGS) -o $@ $<

run: all
    $(MPIRUN) -np $(NP) ./sum_bcast
    $(MPIRUN) -np $(NP) ./sum_scatter
    $(MPIRUN) -np $(NP) ./sum_gather
    $(MPIRUN) -np $(NP) ./sum_reduce
    $(MPIRUN) -np $(NP) ./sum_allreduce
    $(MPIRUN) -np $(NP) ./sum_scan

clean:
    rm -f $(PROGS)

.PHONY: all run clean