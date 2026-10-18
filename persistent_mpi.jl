module PersistentMPI

using MPI

mutable struct SafePersistentRequest{B}
    request::MPI.Request
    buffer::B

    function SafePersistentRequest(raw_req::MPI.Request, buffer::B) where B
        req = new{B}(raw_req, buffer)
        finalizer(mpi_persistent_free!, req)
        return req
    end
end

function mpi_persistent_free!(req::SafePersistentRequest)
    # Защита: вызываем free в Си только если среда MPI активна
    if MPI.Initialized() && !MPI.Finalized()
        if req.request.val != MPI.API.MPI_REQUEST_NULL
            MPI.API.MPI_Request_free(Ref(req.request.val))
            req.request = MPI.Request()
        end
    end
    return nothing
end

MPI.Start(req::SafePersistentRequest) = MPI.Start(req.request)
MPI.Wait!(req::SafePersistentRequest) = MPI.Wait!(req.request)

function persistent_send_init(buf::AbstractArray, dest::Integer, tag::Integer, comm::MPI.Comm)
    mpi_buf = MPI.Buffer(buf)
    raw_req = MPI.Send_init(mpi_buf, comm; dest=dest, tag=tag)
    return SafePersistentRequest(raw_req, buf)
end

function persistent_recv_init!(buf::AbstractArray, source::Integer, tag::Integer, comm::MPI.Comm)
    mpi_buf = MPI.Buffer(buf)
    raw_req = MPI.Recv_init(mpi_buf, comm; source=source, tag=tag)
    return SafePersistentRequest(raw_req, buf)
end

export SafePersistentRequest, persistent_send_init, persistent_recv_init!, mpi_persistent_free!

end # module
