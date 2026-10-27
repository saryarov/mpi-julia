#!/bin/bash
#SBATCH --job-name=mpi_persistent_job
#SBATCH --output=result_output.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00

# 1. ПОДМЕНА ПУТЕЙ (Использование локального MPI 4.0+ из вашей домашней папки)
export PATH=$HOME/local_mpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local_mpi/lib:$LD_LIBRARY_PATH

# Выводим техническую информацию в лог для проверки подмены путей
echo "=== ПРОВЕРКА ОКРУЖЕНИЯ НА КЛАСТЕРЕ ==="
echo "Абсолютный путь к mpirun:" $(which mpirun)
echo "Используемая версия MPI:" $(mpirun --version | head -n 1)

# Загружаем модуль Julia (если он необходим в конфигурации вашего кластера)
module load julia 2>/dev/null || echo "Системный модуль julia не найден, используется локальный из PATH"

# 2. ЖЕСТКАЯ ПРИВЯЗКА JULIA К ВАШЕМУ ЛОКАЛЬНОМУ OPEN MPI
echo -e "\n=== СВЯЗЫВАНИЕ ПАКЕТОВ JULIA С ЛОКАЛЬНЫМ MPI ==="
julia -e '
using Pkg
Pkg.activate(".")
using Pkg; Pkg.add("MPIPreferences")
using MPIPreferences
MPIPreferences.use_system_binary(
    library_names = [joinpath(homedir(), "local_mpi", "lib", "libmpi.so")],
    mpiexec = joinpath(homedir(), "local_mpi", "bin", "mpiexec")
)
Pkg.build("MPI")
'

# 3. ЗАПУСК MVP (Проверка корректности математических расчетов)
echo -e "\n=== ЗАПУСК 1: ПРОВЕРКА КОРРЕКТНОСТИ ДАННЫХ (MVP) ==="
mpirun -np 8 julia --project=. test/test_validation.jl

# 4. ЗАПУСК БЕНЧМАРКА (Сбор статистики для научной работы)
echo -e "\n=== ЗАПУСК 2: СБОР СТАТИСТИКИ ПРОИЗВОДИТЕЛЬНОСТИ ==="
mpirun -np 8 julia --project=. benchmarks/cluster_benchmark.jl
