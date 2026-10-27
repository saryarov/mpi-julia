module PersistentCollectives

using MPI

export Bcast_init!, Reduce_init!, Allreduce_init!, Gather_init!

const libmpi = MPI.libmpi

function Bcast_init!(buffer, root::Integer, comm::MPI.Comm)
    req = MPI.Request()
    count = length(buffer)
    datatype = MPI.Datatype(eltype(buffer))
    
    err = ccall(
        (:MPI_Bcast_init, libmpi), Cint,
        (Ptr{Cvoid}, Cint, MPI.Datatype, Cint, MPI.Comm, MPI.Info, Ptr{MPI.Request}),
        pointer(buffer), count, datatype, root, comm, MPI.INFO_NULL, req
    )
    if err != 0; error("Ошибка в MPI_Bcast_init: $err"); end
    return req
end

function Reduce_init!(sendbuf, recvbuf, op::MPI.Op, root::Integer, comm::MPI.Comm)
    req = MPI.Request()
    count = length(sendbuf)
    datatype = MPI.Datatype(eltype(sendbuf))
    
    sendptr = sendbuf === MPI.IN_PLACE ? MPI.MPI_IN_PLACE : pointer(sendbuf)
    recvptr = pointer(recvbuf)

    err = ccall(
        (:MPI_Reduce_init, libmpi), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, MPI.Datatype, MPI.Op, Cint, MPI.Comm, MPI.Info, Ptr{MPI.Request}),
        sendptr, recvptr, count, datatype, op, root, comm, MPI.INFO_NULL, req
    )
    if err != 0; error("Ошибка в MPI_Reduce_init: $err"); end
    return req
end

function Allreduce_init!(sendbuf, recvbuf, op::MPI.Op, comm::MPI.Comm)
    req = MPI.Request()
    count = length(sendbuf)
    datatype = MPI.Datatype(eltype(sendbuf))
    
    sendptr = sendbuf === MPI.IN_PLACE ? MPI.MPI_IN_PLACE : pointer(sendbuf)
    recvptr = pointer(recvbuf)

    err = ccall(
        (:MPI_Allreduce_init, libmpi), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, MPI.Datatype, MPI.Op, MPI.Comm, MPI.Info, Ptr{MPI.Request}),
        sendptr, recvptr, count, datatype, op, comm, MPI.INFO_NULL, req
    )
    if err != 0; error("Ошибка в MPI_Allreduce_init: $err"); end
    return req
end

function Gather_init!(sendbuf, recvbuf, root::Integer, comm::MPI.Comm)
    req = MPI.Request()
    sendcount = length(sendbuf)
    datatype = MPI.Datatype(eltype(sendbuf))
    
    sendptr = sendbuf === MPI.IN_PLACE ? MPI.MPI_IN_PLACE : pointer(sendbuf)
    recvptr = pointer(recvbuf)

    err = ccall(
        (:MPI_Gather_init, libmpi), Cint,
        (Ptr{Cvoid}, Cint, MPI.Datatype, Ptr{Cvoid}, Cint, MPI.Datatype, Cint, MPI.Comm, MPI.Info, Ptr{MPI.Request}),
        sendptr, sendcount, datatype, recvptr, sendcount, datatype, root, comm, MPI.INFO_NULL, req
    )
    if err != 0; error("Ошибка в MPI_Gather_init: $err"); end
    return req
end

end
