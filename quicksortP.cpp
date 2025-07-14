#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <random>
#include <tuple>


using namespace std;

void generate_unique_array(std::vector<int>& arr, int n) {
  arr.resize(n);

  std::vector<int> pool(n * 10);
  std::iota(pool.begin(), pool.end(), 0);

  // baraja aleatoriamente
  std::mt19937 g(12345);
  std::shuffle(pool.begin(), pool.end(), g);

  // copia los primeros n valores únicos
  std::copy(pool.begin(), pool.begin() + n, arr.begin());
}

int getPivot(vector<int>& local_s, MPI_Comm comm) {
  int i_PE, p_prime;
  MPI_Comm_rank(comm, &i_PE);
  MPI_Comm_size(comm, &p_prime);

  int pivot;
  int pivot_index;

  if (i_PE == 0) {
    srand(time(NULL));
    //% Eleccion del dueño del pivote
    pivot_index = rand() % p_prime;
  }

  //% Se indica quien es dueño del pivote
  MPI_Bcast(&pivot_index, 1, MPI_INT, 0, comm);

  if (i_PE == pivot_index) {
    if (local_s.empty()) {
      pivot = 0; //% Pivot por defecto
    }
    else {
      srand(time(NULL) + i_PE);
      //% Eleccion del pivote
      pivot = local_s[rand() % local_s.size()];
    }
  }

  //% Se envia el pivote a todos los PE
  MPI_Bcast(&pivot, 1, MPI_INT, pivot_index, comm);

  return pivot;
}

tuple<vector<int>, vector<int>> partition_vector(vector<int>& vec, int pivot) {
  vector<int> local_s = vec;

  auto it = partition(local_s.begin(), local_s.end(), [pivot](int x) { return x <= pivot;});

  vector<int> left(local_s.begin(), it);
  vector<int> right(it, local_s.end());

  return make_tuple(left, right);
}

void quicksortP(vector<int>& local_s, MPI_Comm comm) {
  int i_PE, p_prime;
  MPI_Comm_rank(comm, &i_PE);
  MPI_Comm_size(comm, &p_prime);

  //% Solo hay un proceso, se ordena localmente
  if (p_prime <= 1) {
    if (!local_s.empty()) {
      sort(local_s.begin(), local_s.end());
    }
    return;
  }

  int pivot = getPivot(local_s, comm);

  vector<int> a, b;
  tie(a, b) = partition_vector(local_s, pivot);

  long long n_a = 0, n_b = 0;
  long long n_a_local = a.size(), n_b_local = b.size();

  MPI_Allreduce(&n_a_local, &n_a, 1, MPI_LONG_LONG, MPI_SUM, comm);
  MPI_Allreduce(&n_b_local, &n_b, 1, MPI_LONG_LONG, MPI_SUM, comm);

  int k;
  if (n_a + n_b == 0) {
    k = p_prime / 2;
  }
  else {
    k = round(static_cast<double>(n_a) / (n_a + n_b) * p_prime);
  }

  k = max(1, k);
  k = min(p_prime - 1, k);

  // % Reparto de elementos a sus procesos correspondientes
  //
  //
  // % Planificacion de que datos no le corresponde a este PE
  vector<int> send_counts(p_prime, 0);

  if (i_PE < k) {
    //% enviar datos grandes, si recibire datos pequeños
    int p_large_group = p_prime - k;
    for (long i = 0; i < b.size(); ++i) {
      send_counts[k + (i % p_large_group)]++;
    }
  }
  else {
    //% enviar datos pequeños, si recibire datos grandes
    for (long i = 0; i < a.size(); ++i) {
      send_counts[i % k]++;
    }
  }

  vector<int> recv_counts(p_prime, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

  //% Calculo de la cantidad de los buffers de envio y recepcion
  //
  vector<int> cant_send_buffer(p_prime, 0), cant_recv_buffer(p_prime, 0);

  for (int i = 1; i < p_prime; ++i) {
    cant_send_buffer[i] = cant_send_buffer[i-1] + send_counts[i-1];
    cant_recv_buffer[i] = cant_recv_buffer[i-1] + recv_counts[i-1];
  }

  // % Buffer envio
  vector<int> send_buffer;
  if (i_PE < k) {
    send_buffer = b;
  }
  else {
    send_buffer = a;
  }

  // % Buffer de recepcion
  int total_recv = accumulate(recv_counts.begin(), recv_counts.end(), 0);
  vector<int> recv_buffer(total_recv);

  //% Envio y recepcion de datos, intercambio entre procesos

  MPI_Alltoallv(send_buffer.data(), send_counts.data(), cant_send_buffer.data(), MPI_INT,
                recv_buffer.data(), recv_counts.data(), cant_recv_buffer.data(), MPI_INT, comm);


  //% Actualizacion de los datos locales
  if (i_PE < k) {
    //% Proceso con datos pequeños
    local_s = a;
    local_s.insert(local_s.end(), recv_buffer.begin(), recv_buffer.end());
  }
  else {
    //% Proceso con datos grandes
    local_s = b;
    local_s.insert(local_s.end(), recv_buffer.begin(), recv_buffer.end());
  }

  int color = (i_PE < k) ? 0 : 1;
  MPI_Comm new_comm;
  //% division de comunicaciones entre grupo grande y grupo pequeño
  MPI_Comm_split(comm, color, i_PE, &new_comm);

  quicksortP(local_s, new_comm);

  MPI_Comm_free(&new_comm);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2) {
        if (world_rank == 0) {
            std::cerr << "Uso: mpirun -np <num_procesos> " << argv[0] << " <num_elementos>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    const int N_ELEMENTS = std::stoi(argv[1]);

    std::vector<int> global_data;
    if (world_rank == 0) {
        generate_unique_array(global_data, N_ELEMENTS);
    }

    double t0, t1;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size, 0);
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
    MPI_Scatterv(global_data.data(), send_counts.data(), displs.data(), MPI_INT, 
                 local_data.data(), send_counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

    quicksortP(local_data, MPI_COMM_WORLD);

    int local_size = local_data.size();
    std::vector<int> gather_counts(world_size);
    MPI_Gather(&local_size, 1, MPI_INT, gather_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> final_displs(world_size, 0);
    if (world_rank == 0) {
       for(int i = 1; i < world_size; ++i) {
           final_displs[i] = final_displs[i-1] + gather_counts[i-1];
       }
       global_data.resize(N_ELEMENTS);
    }

    MPI_Gatherv(local_data.data(), local_data.size(), MPI_INT,
                global_data.data(), gather_counts.data(), final_displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    if (world_rank == 0) {
        std::cout << world_size << "," << N_ELEMENTS << "," << t1 - t0 << std::endl;

        bool sorted = true;
        for (int i = 0; i < N_ELEMENTS - 1; ++i) {
            if (global_data[i] > global_data[i+1]) {
                sorted = false;
                break;
            }
        }

        if (sorted) {
            std::cout << "VERIFICACION: El arreglo esta correctamente ordenado. ✔️" << std::endl;
        } else {
            std::cout << "VERIFICACION: ¡ERROR! El arreglo NO esta ordenado. ❌" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
