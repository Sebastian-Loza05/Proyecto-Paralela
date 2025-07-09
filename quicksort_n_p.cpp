#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <mpi.h>

int pickPivot(const std::vector<int>& local_s, MPI_Comm comm, int i_PE, int p_prime) {
    int pivot;
    int pivot_owner_rank;
    if (i_PE == 0) {
        srand(time(NULL) + p_prime + i_PE);
        pivot_owner_rank = rand() % p_prime;
    }
    MPI_Bcast(&pivot_owner_rank, 1, MPI_INT, 0, comm);
    if (i_PE == pivot_owner_rank) {
        pivot = local_s.empty() ? 0 : local_s[local_s.size() / 2];
    }
    MPI_Bcast(&pivot, 1, MPI_INT, pivot_owner_rank, comm);
    return pivot;
}

void parQuickSort(std::vector<int>& local_s, MPI_Comm comm) {
    int i_PE, p_prime;
    MPI_Comm_rank(comm, &i_PE);
    MPI_Comm_size(comm, &p_prime);

    if (p_prime <= 1) {
        if (!local_s.empty()) std::sort(local_s.begin(), local_s.end());
        return;
    }

    int v = pickPivot(local_s, comm, i_PE, p_prime);
    
    auto it = std::partition(local_s.begin(), local_s.end(), [v](int e) { return e <= v; });
    std::vector<int> a(local_s.begin(), it);
    std::vector<int> b(it, local_s.end());
    
    long long n_a = 0, n_b = 0;
    long long local_na = a.size(), local_nb = b.size();
    MPI_Allreduce(&local_na, &n_a, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(&local_nb, &n_b, 1, MPI_LONG_LONG, MPI_SUM, comm);

    int k;
    if (n_a + n_b == 0) k = p_prime / 2;
    else {
        double k_prime_ideal = static_cast<double>(n_a) / (n_a + n_b) * p_prime;
        k = round(k_prime_ideal);
    }
    k = std::max(1, k);
    k = std::min(p_prime - 1, k);

    // --- CORRECCIÓN DEL DEADLOCK ---
    // La recolección de datos debe ocurrir ANTES de la división lógica.
    // TODOS los procesos en 'comm' participan en estas dos recolecciones.

    // 1. Recolectar todos los datos 'a' en el proceso 0.
    std::vector<int> a_sizes(p_prime), a_displs(p_prime, 0);
    int local_a_size = a.size();
    MPI_Gather(&local_a_size, 1, MPI_INT, a_sizes.data(), 1, MPI_INT, 0, comm);
    
    std::vector<int> gathered_a;
    if (i_PE == 0) {
        gathered_a.resize(n_a);
        for(int i = 1; i < p_prime; ++i) a_displs[i] = a_displs[i-1] + a_sizes[i-1];
    }
    MPI_Gatherv(a.data(), a.size(), MPI_INT, gathered_a.data(), a_sizes.data(), a_displs.data(), MPI_INT, 0, comm);

    // 2. Recolectar todos los datos 'b' en el proceso k.
    std::vector<int> b_sizes(p_prime), b_displs(p_prime, 0);
    int local_b_size = b.size();
    MPI_Gather(&local_b_size, 1, MPI_INT, b_sizes.data(), 1, MPI_INT, k, comm);

    std::vector<int> gathered_b;
    if (i_PE == k) {
        gathered_b.resize(n_b);
        for(int i = 1; i < p_prime; ++i) b_displs[i] = b_displs[i-1] + b_sizes[i-1];
    }
    MPI_Gatherv(b.data(), b.size(), MPI_INT, gathered_b.data(), b_sizes.data(), b_displs.data(), MPI_INT, k, comm);

    // 3. Ahora que los datos están recolectados, dividimos el comunicador.
    int color = (i_PE < k) ? 0 : 1;
    MPI_Comm new_comm;
    MPI_Comm_split(comm, color, i_PE, &new_comm);

    // 4. Distribuir los datos recolectados DENTRO de los nuevos comunicadores.
    if (color == 0) { // Grupo 'a'
        std::vector<int> scatter_counts(k), scatter_displs(k, 0);
        if (i_PE == 0) { // El raíz (0) calcula cómo repartir
            for (int i = 0; i < k; ++i) {
                scatter_counts[i] = n_a / k;
                if (i < n_a % k) scatter_counts[i]++;
            }
            for (int i = 1; i < k; ++i) scatter_displs[i] = scatter_displs[i-1] + scatter_counts[i-1];
        }
        int my_new_size;
        MPI_Scatter(scatter_counts.data(), 1, MPI_INT, &my_new_size, 1, MPI_INT, 0, new_comm);
        local_s.assign(my_new_size, 0);
        MPI_Scatterv(gathered_a.data(), scatter_counts.data(), scatter_displs.data(), MPI_INT, local_s.data(), my_new_size, MPI_INT, 0, new_comm);
    } else { // Grupo 'b'
        int new_rank;
        MPI_Comm_rank(new_comm, &new_rank);
        int b_group_size = p_prime - k;
        std::vector<int> scatter_counts(b_group_size), scatter_displs(b_group_size, 0);
        if (new_rank == 0) { // El nuevo raíz (originalmente k) calcula
            for (int i = 0; i < b_group_size; ++i) {
                scatter_counts[i] = (n_b > 0) ? n_b / b_group_size : 0;
                if (i < n_b % b_group_size) scatter_counts[i]++;
            }
            for (int i = 1; i < b_group_size; ++i) scatter_displs[i] = scatter_displs[i-1] + scatter_counts[i-1];
        }
        int my_new_size;
        MPI_Scatter(scatter_counts.data(), 1, MPI_INT, &my_new_size, 1, MPI_INT, 0, new_comm);
        local_s.assign(my_new_size, 0);
        MPI_Scatterv(gathered_b.data(), scatter_counts.data(), scatter_displs.data(), MPI_INT, local_s.data(), my_new_size, MPI_INT, 0, new_comm);
    }
    
    // 5. Llamada recursiva con el grupo y los datos correctos.
    parQuickSort(local_s, new_comm);
    MPI_Comm_free(&new_comm);
}

// --- El main se mantiene igual ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N_ELEMENTS = 200;
    std::vector<int> global_data;
    if (world_rank == 0) {
        printf("Distributed Quicksort con %d elementos y %d procesos.\n", N_ELEMENTS, world_size);
        global_data.resize(N_ELEMENTS);
        srand(time(NULL)); 
        std::cout << "Datos originales: ";
        for (int i = 0; i < N_ELEMENTS; ++i) {
            global_data[i] = rand() % 100;
            std::cout << global_data[i] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<int> send_counts(world_size), displs(world_size, 0);
    if (world_rank == 0) {
      int count_per_proc = N_ELEMENTS / world_size;
      int remainder = N_ELEMENTS % world_size;
      for (int i = 0; i < world_size; ++i) {
          send_counts[i] = count_per_proc + (i < remainder ? 1 : 0);
      }
      for (int i = 1; i < world_size; ++i) {
          displs[i] = displs[i-1] + send_counts[i-1];
      }
    }
    MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> local_data(send_counts[world_rank]);
    MPI_Scatterv(global_data.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
    
    parQuickSort(local_data, MPI_COMM_WORLD);

    int local_size = local_data.size();
    std::vector<int> gather_counts(world_size);
    MPI_Gather(&local_size, 1, MPI_INT, gather_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> final_displs(world_size, 0);
    if(world_rank == 0) {
        if(N_ELEMENTS > 0) {
           for(int i = 1; i < world_size; ++i) final_displs[i] = final_displs[i-1] + gather_counts[i-1];
        }
    }
    MPI_Gatherv(local_data.data(), local_data.size(), MPI_INT, global_data.data(), gather_counts.data(), final_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Datos ordenados:  ";
        for (int i = 0; i < N_ELEMENTS; ++i) {
            std::cout << global_data[i] << " ";
        }
        std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
}
