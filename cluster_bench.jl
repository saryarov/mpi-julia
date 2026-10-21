using MPI
using Printf
using Statistics

include("persistent_mpi.jl")
using .PersistentMPI

function run_bench()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if size < 2
        rank == 0 && error("Нужно ≥ 2 процесса. Запускайте через mpiexec/srun.")
        MPI.Finalize()
        return
    end

    dst = mod(rank + 1, size)
    src = mod(rank - 1, size)

    # ------------------- Параметры -------------------
    sizes_bytes = (8, 1024, 8192, 65536, 262144)
    warmup      = 500
    iters       = 5000
    # -------------------------------------------------

    rows = NamedTuple[]

    for nbytes in sizes_bytes
        N = max(1, nbytes ÷ 8)
        sendbuf = fill(Float64(rank), N)
        recvbuf = zeros(Float64, N)

        # -------- 1. Isend / Irecv --------
        for _ in 1:warmup
            r = MPI.Irecv!(recvbuf, comm; source=src, tag=1)
            s = MPI.Isend(sendbuf,  comm; dest=dst,  tag=1)
            MPI.Waitall([s, r])
        end
        MPI.Barrier(comm)
        GC.gc()
        t_normal = @timed begin
            for _ in 1:iters
                r = MPI.Irecv!(recvbuf, comm; source=src, tag=1)
                s = MPI.Isend(sendbuf,  comm; dest=dst,  tag=1)
                MPI.Waitall([s, r])
            end
        end
        MPI.Barrier(comm)

        # -------- 2. Persistent (ваш модуль) --------
        sreq = persistent_send_init(sendbuf, dst, 2, comm)
        rreq = persistent_recv_init!(recvbuf, src, 2, comm)
        rset = PersistentRequestSet([rreq, sreq])

        for _ in 1:warmup
            start_all!(rset)
            wait_all!(rset)
        end
        MPI.Barrier(comm)
        GC.gc()
        t_persist = @timed begin
            for _ in 1:iters
                start_all!(rset)
                wait_all!(rset)
            end
        end
        MPI.Barrier(comm)

        # -------- 3. Sendrecv (blocking, reference) --------
        for _ in 1:warmup
            MPI.Sendrecv!(sendbuf, recvbuf, comm;
                          dest=dst, source=src, sendtag=3, recvtag=3)
        end
        MPI.Barrier(comm)
        GC.gc()
        t_sendrecv = @timed begin
            for _ in 1:iters
                MPI.Sendrecv!(sendbuf, recvbuf, comm;
                              dest=dst, source=src, sendtag=3, recvtag=3)
            end
        end
        MPI.Barrier(comm)

        free_set!(rset)

        push!(rows, (
            bytes          = nbytes,
            normal_ms      = t_normal.time  * 1000,
            normal_alloc   = t_normal.bytes,
            persist_ms     = t_persist.time * 1000,
            persist_alloc  = t_persist.bytes,
            sendrecv_ms    = t_sendrecv.time * 1000,
            sendrecv_alloc = t_sendrecv.bytes,
        ))

        rank == 0 && @printf("  [размер %6d Б] готово\n", nbytes)
    end

    # ------------------- Отчёт -------------------
    if rank == 0
        println("\n" * "="^112)
        @printf("%-10s | %-13s | %-13s | %-13s | %-13s | %-13s | %-9s\n",
                "Size(B)", "Isend ms", "Isend alloc",
                "Persist ms", "Persist alloc", "Sendrecv ms", "Speedup")
        println("-"^112)
        for r in rows
            speedup = (r.normal_ms - r.persist_ms) / r.normal_ms * 100
            @printf("%-10d | %-13.3f | %-13d | %-13.3f | %-13d | %-13.3f | %+.1f%%\n",
                    r.bytes,
                    r.normal_ms,  r.normal_alloc,
                    r.persist_ms, r.persist_alloc,
                    r.sendrecv_ms, speedup)
        end
        println("="^112)
        println("Speedup = (Isend_time - Persistent_time) / Isend_time × 100%")
        println("Persist alloc должен быть 0 (или равно числу итераций — нет, должно быть 0).")

        # CSV для статьи
        open("bench_results_rank0.csv", "w") do io
            println(io, "bytes,isend_ms,isend_alloc,persist_ms,persist_alloc,sendrecv_ms,sendrecv_alloc")
            for r in rows
                println(io, "$(r.bytes),$(r.normal_ms),$(r.normal_alloc)," *
                            "$(r.persist_ms),$(r.persist_alloc)," *
                            "$(r.sendrecv_ms),$(r.sendrecv_alloc)")
            end
        end
        println("→ Результаты сохранены в bench_results_rank0.csv")
    end

    MPI.Finalize()
end

run_bench()