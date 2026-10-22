using MPI
using Printf
using Statistics

include("persistent_mpi.jl")
using .PersistentMPI

function measure(comm, f, warmup::Int, iters::Int; reps::Int = 5)
    for _ in 1:warmup
        f()
    end
    times = Vector{Float64}(undef, reps)
    allocs = 0
    for k in 1:reps
        MPI.Barrier(comm)
        GC.gc()
        MPI.Barrier(comm)
        t = @timed for _ in 1:iters
            f()
        end
        times[k] = t.time * 1000
        allocs = t.bytes
    end
    return (med = median(times), min = minimum(times),
            max = maximum(times), alloc = allocs)
end

function grid_dims(n::Int)
    cols = isqrt(n)
    while n % cols != 0
        cols -= 1
    end
    rows = n ÷ cols
    return (rows, cols)
end

function grid_neighbors(rank::Int, rows::Int, cols::Int)
    r = rank ÷ cols
    c = rank % cols
    up    = ((r - 1 + rows) % rows) * cols + c
    down  = ((r + 1) % rows) * cols + c
    left  = r * cols + ((c - 1 + cols) % cols)
    right = r * cols + ((c + 1) % cols)
    return (up=up, down=down, left=left, right=right)
end

# 1=up, 2=down, 3=left, 4=right. OPP[d] — противоположное направление.
const OPP = (2, 1, 4, 3)

function run_bench()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if size < 2
        rank == 0 && error("Нужно >= 2 процесса. Запускайте через mpiexec/srun.")
        MPI.Finalize()
        return
    end

    sizes_bytes = (8, 1024, 8192, 65536, 262144)
    warmup      = 500
    iters       = 5000
    reps        = 5

    # ----- ring -----
    dst = mod(rank + 1, size)
    src = mod(rank - 1, size)

    # ----- grid -----
    grows, gcols = grid_dims(size)
    gn = grid_neighbors(rank, grows, gcols)
    gnbrs = [gn.up, gn.down, gn.left, gn.right]
    grid_ok = (grows > 1 && gcols > 1 && all(n != rank for n in gnbrs))

    ring_rows = NamedTuple[]
    grid_rows = NamedTuple[]

    for nbytes in sizes_bytes
        N = max(1, nbytes ÷ 8)
        sendbuf = fill(Float64(rank), N)
        recvbuf = zeros(Float64, N)

        # ======= RING =======
        f_isend_waitall = () -> begin
            r = MPI.Irecv!(recvbuf, comm; source=src, tag=1)
            s = MPI.Isend(sendbuf,  comm; dest=dst,  tag=1)
            MPI.Waitall([s, r])
        end
        m_isend_waitall = measure(comm, f_isend_waitall, warmup, iters; reps=reps)

        f_isend_wait = () -> begin
            r = MPI.Irecv!(recvbuf, comm; source=src, tag=1)
            s = MPI.Isend(sendbuf,  comm; dest=dst,  tag=1)
            MPI.Wait(s)
            MPI.Wait(r)
        end
        m_isend_wait = measure(comm, f_isend_wait, warmup, iters; reps=reps)

        sreq = persistent_send_init(sendbuf, dst, 2, comm)
        rreq = persistent_recv_init!(recvbuf, src, 2, comm)
        rset = PersistentRequestSet([rreq, sreq])
        f_persist_ring = () -> begin
            start_all!(rset)
            wait_all!(rset)
        end
        m_persist_ring = measure(comm, f_persist_ring, warmup, iters; reps=reps)
        free_set!(rset)

        f_sendrecv = () -> begin
            MPI.Sendrecv!(sendbuf, recvbuf, comm;
                          dest=dst, source=src, sendtag=3, recvtag=3)
        end
        m_sendrecv = measure(comm, f_sendrecv, warmup, iters; reps=reps)

        push!(ring_rows, (
            bytes = nbytes,
            isend_waitall_ms = m_isend_waitall.med,
            isend_wait_ms    = m_isend_wait.med,
            persist_ms       = m_persist_ring.med,
            persist_alloc    = m_persist_ring.alloc,
            sendrecv_ms      = m_sendrecv.med,
        ))

        # ======= GRID =======
        if grid_ok
            sendbufs = [fill(Float64(rank), N) for _ in 1:4]
            recvbufs = [zeros(Float64, N) for _ in 1:4]

            f_grid_normal = () -> begin
                reqs = Vector{MPI.Request}(undef, 8)
                for d in 1:4
                    reqs[d] = MPI.Irecv!(recvbufs[d], comm; source=gnbrs[d], tag=100+d)
                end
                for d in 1:4
                    reqs[4+d] = MPI.Isend(sendbufs[d], comm; dest=gnbrs[d], tag=100+OPP[d])
                end
                MPI.Waitall(reqs)
            end
            m_grid_normal = measure(comm, f_grid_normal, warmup, iters; reps=reps)

            greqs  = [persistent_recv_init!(recvbufs[d], gnbrs[d], 100+d,    comm) for d in 1:4]
            gsreqs = [persistent_send_init(sendbufs[d], gnbrs[d], 100+OPP[d], comm) for d in 1:4]
            gset = PersistentRequestSet([greqs; gsreqs])
            f_grid_persist = () -> begin
                start_all!(gset)
                wait_all!(gset)
            end
            m_grid_persist = measure(comm, f_grid_persist, warmup, iters; reps=reps)
            free_set!(gset)

            push!(grid_rows, (
                bytes = nbytes,
                normal_ms    = m_grid_normal.med,
                normal_alloc = m_grid_normal.alloc,
                persist_ms   = m_grid_persist.med,
                persist_alloc = m_grid_persist.alloc,
            ))
        end

        rank == 0 && @printf("  [размер %6d Б] готово\n", nbytes)
    end

    if rank == 0
        ncpu = try Sys.CPU_THREADS catch; -1 end
        println()
        println("nproc() = $ncpu, MPI size = $size, reps = $reps, iters = $iters")
        println("grid: $(grows)x$(gcols), grid_ok = $grid_ok")

        # ----- RING table -----
        println("\n" * "="^100)
        println("RING (2 neighbors: prev, next)")
        println("="^100)
        @printf("%-10s | %-13s | %-13s | %-13s | %-13s | %-13s | %-9s\n",
                "Size(B)", "Isend+WAll", "Isend+W s/r", "Persist", "Sendrecv",
                "Persist alloc", "Persist vs WAll")
        println("-"^100)
        for r in ring_rows
            sp = (r.isend_waitall_ms - r.persist_ms) / r.isend_waitall_ms * 100
            @printf("%-10d | %-13.3f | %-13.3f | %-13.3f | %-13.3f | %-13d | %+.1f%%\n",
                    r.bytes, r.isend_waitall_ms, r.isend_wait_ms,
                    r.persist_ms, r.sendrecv_ms, r.persist_alloc, sp)
        end

        # ----- GRID table -----
        if grid_ok && !isempty(grid_rows)
            println("\n" * "="^100)
            println("GRID $(grows)x$(gcols) (4 neighbors: up, down, left, right)")
            println("="^100)
            @printf("%-10s | %-15s | %-15s | %-13s | %-13s | %-9s\n",
                    "Size(B)", "Irecv*4+Isend*4", "Persist x8",
                    "Normal alloc", "Persist alloc", "Speedup")
            println("-"^100)
            for r in grid_rows
                sp = (r.normal_ms - r.persist_ms) / r.normal_ms * 100
                @printf("%-10d | %-15.3f | %-15.3f | %-13d | %-13d | %+.1f%%\n",
                        r.bytes, r.normal_ms, r.persist_ms,
                        r.normal_alloc, r.persist_alloc, sp)
            end
        else
            println("\n[grid mode SKIPPED: size=$size is not suitable for 2D grid]")
        end

        # ----- CSV -----
        open("bench_results_rank0.csv", "w") do io
            println(io, "pattern,bytes,isend_waitall_ms,isend_wait_ms,persist_ms,sendrecv_ms," *
                        "persist_alloc,grid_normal_ms,grid_persist_ms,grid_normal_alloc,grid_persist_alloc")
            for (i, r) in enumerate(ring_rows)
                g = (grid_ok && i <= length(grid_rows)) ? grid_rows[i] : nothing
                tail = g === nothing ? ",,,," :
                    ",$(g.normal_ms),$(g.persist_ms),$(g.normal_alloc),$(g.persist_alloc)"
                println(io,
                    "ring,$(r.bytes),$(r.isend_waitall_ms),$(r.isend_wait_ms)," *
                    "$(r.persist_ms),$(r.sendrecv_ms),$(r.persist_alloc)" * tail)
            end
        end
        println("\n-> Результаты сохранены в bench_results_rank0.csv")
    end

    MPI.Finalize()
end

run_bench()
