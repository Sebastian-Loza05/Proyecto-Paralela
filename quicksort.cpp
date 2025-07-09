// quicksort_mpi.cpp
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>     // rand()
#include <numeric>     // iota
#include <iterator>

// --- Quicksort secuencial sencillo ---
template<class It>
void quicksort_seq(It first, It last) {
    if (std::distance(first, last) <= 1) return;
    auto pivot = *(first + std::distance(first, last) / 2);
    It middle1 = std::partition(first, last, [pivot](auto x){ return x < pivot; });
    It middle2 = std::partition(middle1, last, [pivot](auto x){ return x == pivot; });
    quicksort_seq(first, middle1);
    quicksort_seq(middle2, last);
}

// --- Elegir pivote global (mediana de medianas local) ---
int choose_global_pivot(const std::vector<int>& local,
                        MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Mediana local
    int local_pivot = local.empty() ? 0 : local[local.size()/2];

    // Reunir todas las medianas locales y tomar su mediana
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    std::vector<int> local_pivots(comm_size);
    MPI_Allgather(&local_pivot, 1, MPI_INT,
                  local_pivots.data(), 1, MPI_INT, comm);

    std::nth_element(local_pivots.begin(),
                     local_pivots.begin() + comm_size/2,
                     local_pivots.end());
    return local_pivots[comm_size/2];
}

// --- Hipercubo quicksort recursivo ---
void parallel_quicksort(std::vector<int>& local,
                        MPI_Comm comm) {

    int comm_size, rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    if (comm_size == 1) {               // Caso base
        quicksort_seq(local.begin(), local.end());
        return;
    }

    // 1. Elegir pivote global
    int pivot = choose_global_pivot(local, comm);

    // 2. Particionar bloque local
    std::vector<int> lower, upper;
    std::partition_copy(local.begin(), local.end(),
                        std::back_inserter(lower),
                        std::back_inserter(upper),
                        [pivot](int x){ return x < pivot; });

    // 3. Pareja con XOR
    int dim = 0;
    while ((1 << dim) < comm_size) ++dim;
    dim--;   // la dimensión más alta en el sub-cubo actual
    int partner = rank ^ (1 << dim);

    // 4. Intercambiar bloques apropiados
    std::vector<int>& send_buf = (rank & (1 << dim)) ? lower : upper;
    std::vector<int>& keep_buf = (rank & (1 << dim)) ? upper : lower;

    // Enviar tamaño primero
    int send_count = static_cast<int>(send_buf.size());
    int recv_count = 0;
    MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0,
                 &recv_count, 1, MPI_INT, partner, 0,
                 comm, MPI_STATUS_IGNORE);

    // Reservar y mandar datos
    std::vector<int> recv_buf(recv_count);
    MPI_Sendrecv(send_buf.data(), send_count, MPI_INT, partner, 1,
                 recv_buf.data(), recv_count, MPI_INT, partner, 1,
                 comm, MPI_STATUS_IGNORE);

    // 5. Combinar con los que me quedo
    keep_buf.insert(keep_buf.end(), recv_buf.begin(), recv_buf.end());
    local.swap(keep_buf);

    // 6. Dividir comunicador y recursión
    int color = (rank & (1 << dim)) ? 1 : 0; // 0 = bajos, 1 = altos
    MPI_Comm subcomm;
    MPI_Comm_split(comm, color, rank, &subcomm);
    parallel_quicksort(local, subcomm);
    MPI_Comm_free(&subcomm);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // ------- 0. Generar / leer datos en el proceso 0 --------
    int N = (argc >= 2) ? std::atoi(argv[1]) : 32;
    std::vector<int> data;
    std::vector<int> counts(world_size), displs(world_size);
    if (world_rank == 0) {
        data.resize(N);
        std::iota(data.begin(), data.end(), 0);
        std::shuffle(data.begin(), data.end(), std::mt19937{42});

        // Reparto balanceado
        int base = N / world_size, rem = N % world_size;
        int offset = 0;
        for (int p = 0; p < world_size; ++p) {
            counts[p] = base + (p < rem);
            displs[p] = offset;
            offset += counts[p];
        }
    }
    // Broadcast de counts
    MPI_Bcast(counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // ------- 1. Scatterv -------
    std::vector<int> local(counts[world_rank]);
    MPI_Scatterv(data.data(), counts.data(), displs.data(),
                 MPI_INT, local.data(), counts[world_rank],
                 MPI_INT, 0, MPI_COMM_WORLD);

    // ------- 2. Orden paralelo -------
    parallel_quicksort(local, MPI_COMM_WORLD);

    // ------- 3. Recolectar resultado -------
    MPI_Gatherv(local.data(), counts[world_rank], MPI_INT,
                data.data(), counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Vector ordenado:\n";
        for (auto v : data) std::cout << v << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
