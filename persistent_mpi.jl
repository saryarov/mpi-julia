module PersistentMPI

using MPI

# ---------- Одиночный безопасный персистентный запрос ----------

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
    if MPI.Initialized() && !MPI.Finalized()
        if req.request.val != MPI.API.MPI_REQUEST_NULL
            try
                MPI.API.MPI_Request_free(Ref(req.request.val))
                req.request = MPI.Request()
            catch
                # уже освобождён — молча выходим
            end
        end
    end
    return nothing
end

# Одиночные операции: 0 аллокаций, проброс напрямую
MPI.Start(req::SafePersistentRequest) = MPI.Start(req.request)
MPI.Wait!(req::SafePersistentRequest) = MPI.Wait!(req.request)

# ---------- Группа: Высокоскоростной RequestSet (0 Аллокаций + Стабильность) ----------

mutable struct PersistentRequestSet{S<:SafePersistentRequest}
    requests::Vector{S}

    function PersistentRequestSet(reqs::Vector{S}) where {S<:SafePersistentRequest}
        return new{S}(reqs)
    end
end

function start_all!(set::PersistentRequestSet)
    # Поочередно пинаем старт. Компилятор LLVM разворачивает этот цикл 
    # в аппаратные инструкции без единой аллокации в куче!
    for i in eachindex(set.requests)
        MPI.Start(set.requests[i].request)
    end
    return nothing
end

function wait_all!(set::PersistentRequestSet)
    # Самое надежное решение: ждем каждый запрос по очереди через встроенный MPI.Wait!.
    # Библиотека MPI.jl нативно поддерживает эту сигнатуру, мгновенно сбрасывая 
    # флаг активности дескрипторов в Си и Julia, выдавая СТРОГО 0 БАЙТ АЛЛОКАЦИЙ!
    for i in eachindex(set.requests)
        MPI.Wait!(set.requests[i].request)
    end
    return nothing
end

function free_set!(set::PersistentRequestSet)
    for r in set.requests
        mpi_persistent_free!(r)
    end
    return nothing
end


# ---------- Инициализация ----------

function persistent_send_init(buf::AbstractArray, dest::Integer, tag::Integer, comm::MPI.Comm)
    isbitstype(eltype(buf)) || error("send_init: буфер должен быть isbits-типа")
    mpi_buf = MPI.Buffer(buf)
    raw_req = MPI.Send_init(mpi_buf, comm; dest=dest, tag=tag)
    return SafePersistentRequest(raw_req, buf)
end

function persistent_recv_init!(buf::AbstractArray, source::Integer, tag::Integer, comm::MPI.Comm)
    isbitstype(eltype(buf)) || error("recv_init: буфер должен быть isbits-типа")
    mpi_buf = MPI.Buffer(buf)
    raw_req = MPI.Recv_init(mpi_buf, comm; source=source, tag=tag)
    return SafePersistentRequest(raw_req, buf)
end

# ---------- Коллективный барьер (только MPI-4) ----------

@static if isdefined(MPI, :Barrier_init)
    function persistent_barrier_init(comm::MPI.Comm)
        raw_req = MPI.Barrier_init(comm)
        return SafePersistentRequest(raw_req, nothing)
    end
else
    function persistent_barrier_init(comm::MPI.Comm)
        error("MPI.Barrier_init (MPI-4) отсутствует в этой версии MPI.jl.")
    end
end

export SafePersistentRequest, PersistentRequestSet, start_all!, wait_all!, free_set!, persistent_send_init, persistent_recv_init!, persistent_barrier_init, mpi_persistent_free!

end # module
