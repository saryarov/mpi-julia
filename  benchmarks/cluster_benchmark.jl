using MPI
using Printf
include("../src/PersistentCollectives.jl")
using .PersistentCollectives

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

sizes = [10^2, 10^4, 10^6] 
iterations = 500          
root = 0

if rank == root
    println("==========================================================================")
    println("                  БЕНЧМАРК: ОБЫЧНЫЕ VS ПЕРСИСТЕНТНЫЕ КОЛЛЕКТИВЫ          ")
    println("==========================================================================")
    @printf("%-12s | %-12s | %-18s | %-18s | %-10s\n", "Операция", "Размер (N)", "Время обычн. (с)", "Время персист. (с)", "Ускорение")
    println("-"^80)
end

for N in sizes
    sendbuf = rand(Float64, N)
    recvbuf = zeros(Float64, N)
    
    # --- ALLREDUCE ---
    MPI.Barrier(comm)
    t0 = MPI.Wtime()
    for i in 1:iterations
        MPI.Allreduce!(sendbuf, recvbuf, MPI.SUM, comm)
    end
    t_plain = MPI.Wtime() - t0

    req_allred = Allreduce_init!(sendbuf, recvbuf, MPI.SUM, comm)
    MPI.Barrier(comm)
    t0 = MPI.Wtime()
    for i in 1:iterations
        MPI.Start(req_allred)
        MPI.Wait!(req_allred)
    end
    t_persist = MPI.Wtime() - t0
    MPI.free(req_allred)

    if rank == root
        @printf("%-12s | %12d | %18.5f | %18.5f | %9.2fx\n", "Allreduce", N, t_plain, t_persist, t_plain / t_persist)
    end

    # --- BCAST ---
    MPI.Barrier(comm)
    t0 = MPI.Wtime()
    for i in 1:iterations
        MPI.Bcast!(sendbuf, root, comm)
    end
    t_plain = MPI.Wtime() - t0

    req_bcast = Bcast_init!(sendbuf, root, comm)
    MPI.Barrier(comm)
    t0 = MPI.Wtime()
    for i in 1:iterations
        MPI.Start(req_bcast)
        MPI.Wait!(req_bcast)
    end
    t_persist = MPI.Wtime() - t0
    MPI.free(req_bcast)

    if rank == root
        @printf("%-12s | %12d | %18.5f | %18.5f | %9.2fx\n", "Bcast", N, t_plain, t_persist, t_plain / t_persist)
        println("-"^80)
    end
end

MPI.Finalize()
