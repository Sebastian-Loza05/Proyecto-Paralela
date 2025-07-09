#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <mpi.h>
#include <cmath>
#include <ctime>

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

void print_array(vector<int> arr, int n) {
  for (int i = 0; i < n; i++) {
    cout << arr[i] << " ";
  }
  cout << endl;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Constants
  double t0,t1;
  int n = 36;
  int P = sqrt(size);
  int N = n / P;
  int row = rank / P;
  int col = rank % P;
  MPI_Request requests[P];

  // arrays
  vector<int> global_arr(n, 0);
  vector<int> global_ranking(n, 0);
  vector<int> local_row(N, 0);
  vector<int> local_col(N, 0);

  // Generate array and boradcast cols and rows
  if (rank == 0) {
    generate_unique_array(global_arr, n);
    cout << "Generated array: ";
    print_array(global_arr, n);
  }

  // Empezamos la medición del tiempo en paralelo
  t0=MPI_Wtime();


  MPI_Comm row0Comm;
  int inFirstRow = (row == 0) ? 1 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, inFirstRow, col, &row0Comm);

  if (row == 0) {
    MPI_Scatter(global_arr.data(), N, MPI_INT, local_row.data(), N, MPI_INT, 0, row0Comm);
  }

  MPI_Comm colComm;
  MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);

  MPI_Bcast(local_row.data(), N, MPI_INT, 0, colComm);

  MPI_Comm_free(&colComm);
  
  // Broadcast rows
  MPI_Comm rowComm;
  MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);

  if (col == row) {
    local_col = local_row;
  }

  MPI_Bcast(local_col.data(), N, MPI_INT, row, rowComm);
  
  // Sort
  sort(local_row.begin(), local_row.end());


  // Local Ranking (promedio O(N log N)) -> usando binary search
  vector<int> local_ranking(N, 0);

  for (int i = 0; i < N; i++) {
    auto it = std::upper_bound(local_row.begin(), local_row.end(), local_col[i]);
    local_ranking[i] = int(it - local_row.begin());
  }

  // Reduce local ranking
  
  vector<int> row_ranking(N, 0);
  MPI_Reduce(local_ranking.data(),
             row_ranking.data(),
             N,
             MPI_INT, MPI_SUM,
             row,
             rowComm
             );

  MPI_Comm_free(&rowComm);

  // Print reduced Ranking
  for (int r = 0; r < size; r++) {
    MPI_Barrier(MPI_COMM_WORLD);  // sincroniza antes de imprimir
    if (rank == r && col == row) {
      printf("Rank %d: \n", rank);
      for (int i = 0; i < N; i++) {
        cout << local_col[i] << " ";
      }
      cout << endl;
      printf("Reduced ranking: %d", rank);
      for (int i = 0; i < N; i++) {
        cout << row_ranking[i] << " ";
      }
      cout << endl;
    }
  }

  // Gathering rankings in the root process
  
  // El gather es más eficiente O(log(P)) que el send/recv O(P - 1)
  int isDiag = (col == row) ? 1 : MPI_UNDEFINED;
  MPI_Comm diagComm;
  MPI_Comm_split(MPI_COMM_WORLD, isDiag, 0, &diagComm);

  if (isDiag == 1) {
    MPI_Gather(row_ranking.data(),
               N, MPI_INT,
               global_ranking.data(),
               N, MPI_INT,
               0,
               diagComm);

    MPI_Comm_free(&diagComm);
  }

  // Hasta aquí se calcula el tiempo
  t1=MPI_Wtime();
  
  // Ordenamiento final en O(n) y print
  if (rank == 0) {
    vector<int> ordered_array(n, 0);
    for (int i = 0; i < n; i++) {
      size_t pos = global_ranking[i] - 1;
      ordered_array[pos] = global_arr[i];
    }

    cout << "Ordered array: ";
    print_array(ordered_array, n);
  }

  // Print time
  if (rank == 0) {
    printf("Time taken: %f seconds\n", t1 - t0);
  }

  MPI_Finalize();
  return 0;
}
