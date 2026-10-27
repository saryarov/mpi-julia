using MPI
include("../src/PersistentCollectives.jl")
using .PersistentCollectives

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

N = 3
bcast_buf  = (rank == 0) ? fill(55.0, N) : zeros(Float64, N)
send_buf   = fill(Float64(rank + 1), N)
allred_buf = zeros(Float64, N)

req_bcast  = Bcast_init!(bcast_buf, 0, comm)
req_allred = Allreduce_init!(send_buf, allred_buf, MPI.SUM, comm)

MPI.Startall([req_bcast, req_allred])
MPI.Waitall!([req_bcast, req_allred])

expected_sum = sum(1:size)
if rank == 0
    @assert all(bcast_buf .== 55.0) "Ошибка валидации Bcast"
    @assert all(allred_buf .== expected_sum) "Ошибка валидации Allreduce"
    println("СТАТУС MVP: Все математические проверки пройдены успешно.")
end

MPI.free(req_bcast)
MPI.free(req_allred)
MPI.Finalize()
