using MPI
using Test

include("persistent_mpi.jl")
using .PersistentMPI
import .PersistentMPI: persistent_send_init, persistent_recv_init!, mpi_persistent_free!

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if size < 2
        println("Ошибка: Запустите тест минимум на 2 процессах через mpiexec -n 2")
        MPI.Finalize()
        return
    end

    @testset "Проверка корректности работы каркаса (Ранг $rank)" begin
        send_buf = [Float64(rank) + 100.0]
        recv_buf = [0.0]

        dst = mod(rank + 1, size)
        src = mod(rank - 1, size)

        let
            sreq = persistent_send_init(send_buf, dst, 123, comm)
            rreq = persistent_recv_init!(recv_buf, src, 123, comm)

            MPI.Start(rreq)
            MPI.Start(sreq)

            MPI.Wait!(sreq)
            MPI.Wait!(rreq)

            @test recv_buf[1] == Float64(src) + 100.0


            mpi_persistent_free!(sreq)
            mpi_persistent_free!(rreq)


            finalizer(x -> nothing, sreq)
            finalizer(x -> nothing, rreq)
        end
    end

    MPI.Barrier(comm)

    if rank == 0
        println("\n=== ПРОВЕРКА ЭФФЕКТИВНОСТИ НА ЭМУЛЯЦИИ НАГРУЗКИ ===")
    end

    N = 100
    iterations = 5000
    bench_send = fill(Float64(rank), N)
    bench_recv = zeros(Float64, N)
    dst = mod(rank + 1, size)
    src = mod(rank - 1, size)

    let
        sreq = persistent_send_init(bench_send, dst, 456, comm)
        rreq = persistent_recv_init!(bench_recv, src, 456, comm)

        stats = @timed begin
            for step in 1:iterations
                MPI.Start(rreq)
                MPI.Start(sreq)
                MPI.Wait!(sreq)
                MPI.Wait!(rreq)
            end
        end

        if rank == 0
            println("Результаты каркаса MVP за $iterations шагов:")
            println("  -> Выделено памяти в цикле: ", stats.bytes, " БАЙТ")
            if stats.bytes == 0
                println("[УСПЕХ] Достигнута пиковая эффективность: 0 аллокаций памяти!")
            end
        end

        mpi_persistent_free!(sreq)
        mpi_persistent_free!(rreq)

        finalizer(x -> nothing, sreq)
        finalizer(x -> nothing, rreq)
    end

    GC.gc()


    MPI.Finalize()
end

main()
