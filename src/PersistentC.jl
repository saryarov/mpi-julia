module PersistentC

using MPI
import Libdl

# Загружаем библиотеку с флагами GLOBAL и LAZY
const libmpi_path = "/home/sar/.local/openmpi-5.0.6/lib/libmpi.so"
const libmpi_handle = Libdl.dlopen(libmpi_path, Libdl.RTLD_GLOBAL | Libdl.RTLD_LAZY)

function supports_persistent_collectives()
    ptr = Libdl.dlsym(libmpi_handle, "MPI_Bcast_init"; throw_error=false)
    return ptr !== nothing
end

function bcast_init!(buf::Array, root::Integer, comm::MPI.Comm;
                     info::MPI.Info = MPI.INFO_NULL)
    req_ref = Ref{MPI.API.MPI_Request}()
    fptr = Libdl.dlsym(libmpi_handle, :MPI_Bcast_init)
    
    ccall(fptr, Cint,
          (Ptr{Cvoid}, Cint, MPI.API.MPI_Datatype, Cint, MPI.API.MPI_Comm, MPI.API.MPI_Info, Ptr{MPI.API.MPI_Request}),
          pointer(buf), Cint(length(buf)), MPI.Datatype(eltype(buf)),
          Cint(root), comm, info, req_ref)
    
    req = MPI.Request(req_ref[], buf)
    return req, buf
end

function allreduce_init!(sendbuf::Array, recvbuf::Array,
                         op::MPI.Op, comm::MPI.Comm;
                         info::MPI.Info = MPI.INFO_NULL)
    req_ref = Ref{MPI.API.MPI_Request}()
    fptr = Libdl.dlsym(libmpi_handle, :MPI_Allreduce_init)
    
    ccall(fptr, Cint,
          (Ptr{Cvoid}, Ptr{Cvoid}, Cint, MPI.API.MPI_Datatype, MPI.API.MPI_Op, MPI.API.MPI_Comm, MPI.API.MPI_Info, Ptr{MPI.API.MPI_Request}),
          pointer(sendbuf), pointer(recvbuf), Cint(length(sendbuf)),
          MPI.Datatype(eltype(sendbuf)), op, comm, info, req_ref)
          
    req = MPI.Request(req_ref[], (sendbuf, recvbuf))
    return req, (sendbuf, recvbuf)
end

# Функции жизненного цикла
start!(req::MPI.Request) = (MPI.Start(req); req)
wait!(req::MPI.Request)  = (MPI.Wait(req);  req)
free!(req::MPI.Request)  = (MPI.free(req);   nothing)

end # module