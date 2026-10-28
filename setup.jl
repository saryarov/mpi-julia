using MPI
include("../src/PersistentC.jl")
using .PersistentC

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if rank == 0; println("Проверка поддержки MPI-4.0..."); end
@assert PersistentC.supports_persistent_collectives() "MPI-4.0 persistent collectives не поддерживаются"

buf = [Float64(rank + 1), Float64(rank + 2)]
preq, _keep = PersistentC.bcast_init!(buf, 0, comm)
PersistentC.start!(preq); PersistentC.wait!(preq)
@assert buf == [1.0, 2.0] "Bcast: неверный результат на ранге $rank"
PersistentC.free!(preq)

sendbuf = [Float64(rank + 1)]
recvbuf = [0.0]
preq, _keep2 = PersistentC.allreduce_init!(sendbuf, recvbuf, MPI.SUM, comm)
PersistentC.start!(preq); PersistentC.wait!(preq)
expected = sum(1.0:nprocs)
@assert recvbuf[1] == expected "Allreduce: $recvbuf[1] != $expected на ранге $rank"
PersistentC.free!(preq)

if rank == 0
    println("OK: все проверки на $nprocs процессах прошли успешно!")
end
MPI.Finalize()