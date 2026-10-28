using MPI
include("../src/PersistentC.jl")
using .PersistentC
using Printf

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

const NITER = 50000
const WARMUP = 100
const SIZES = (64, 256, 1024, 4096, 16384, 65536, 262144, 1048576) # размеры в байтах

function bench_regular_bcast!(buf, comm, niter)
    for _ in 1:WARMUP; MPI.Bcast!(buf, 0, comm); end
    MPI.Barrier(comm)
    t = @elapsed for _ in 1:niter; MPI.Bcast!(buf, 0, comm); end
    return t
end

function bench_persistent_bcast!(buf, comm, niter)
    req, keep = PersistentC.bcast_init!(buf, 0, comm)
    for _ in 1:WARMUP; PersistentC.start!(req); PersistentC.wait!(req); end
    MPI.Barrier(comm)
    t = @elapsed for _ in 1:niter
        PersistentC.start!(req); PersistentC.wait!(req)
    end
    PersistentC.free!(req)
    return t
end

function bench_regular_allreduce!(sendbuf, recvbuf, comm, niter)
    for _ in 1:WARMUP; MPI.Allreduce!(sendbuf, recvbuf, +, comm); end
    MPI.Barrier(comm)
    t = @elapsed for _ in 1:niter; MPI.Allreduce!(sendbuf, recvbuf, +, comm); end
    return t
end

function bench_persistent_allreduce!(sendbuf, recvbuf, comm, niter)
    req, keep = PersistentC.allreduce_init!(sendbuf, recvbuf, MPI.SUM, comm)
    for _ in 1:WARMUP; PersistentC.start!(req); PersistentC.wait!(req); end
    MPI.Barrier(comm)
    t = @elapsed for _ in 1:niter
        PersistentC.start!(req); PersistentC.wait!(req)
    end
    PersistentC.free!(req)
    return t
end

if rank == 0
    @printf("\n--- Bcast (%d итераций, %d процессов) ---\n", NITER, nprocs)
    @printf("%10s | %12s | %12s | %8s\n", "size(B)", "regular µs", "persist µs", "speedup")
    @printf("%s\n", "-"^50)
end

for sz in SIZES
    n_elements = sz ÷ 8
    buf = rand(Float64, n_elements)
    t_reg = bench_regular_bcast!(buf, comm, NITER)
    t_pers = bench_persistent_bcast!(buf, comm, NITER)
    if rank == 0
        @printf("%10d | %12.2f | %12.2f | %7.2fx\n", sz, t_reg/NITER*1e6, t_pers/NITER*1e6, t_reg/t_pers)
    end
end

if rank == 0
    @printf("\n--- Allreduce (%d итераций, %d процессов) ---\n", NITER, nprocs)
    @printf("%10s | %12s | %12s | %8s\n", "size(B)", "regular µs", "persist µs", "speedup")
    @printf("%s\n", "-"^50)
end

for sz in SIZES
    n_elements = sz ÷ 8
    sendbuf = rand(Float64, n_elements)
    recvbuf = similar(sendbuf)
    t_reg = bench_regular_allreduce!(sendbuf, recvbuf, comm, NITER)
    t_pers = bench_persistent_allreduce!(sendbuf, recvbuf, comm, NITER)
    if rank == 0
        @printf("%10d | %12.2f | %12.2f | %7.2fx\n", sz, t_reg/NITER*1e6, t_pers/NITER*1e6, t_reg/t_pers)
    end
end

MPI.Finalize()