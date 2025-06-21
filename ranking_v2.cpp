#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <mpi.h>
#include <cmath>

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
  int n = 18;
  int P = sqrt(size);
  int N = n / P;
  int row = rank / P;
  int col = rank % P;
  MPI_Request requests[3];

  // arrays
  vector<int> global_arr(n, 0);
  vector<int> local_row(N, 0);
  vector<int> local_col(N, 0);

  // Generate array and boradcast cols
  if (rank == 0) {
    generate_unique_array(global_arr, n);
    cout << "Generated array: ";
    print_array(global_arr, n);

    for (int i = 1; i < P; i++)
      MPI_Isend(&global_arr[i*N], N, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i]);

    for (int i = 0; i < N; i++)
      local_row[i] = global_arr[i];
  }
  
  if ( rank > 0 && rank < P) {
    MPI_Irecv(local_row.data(), N, MPI_INT, 0, 0, MPI_COMM_WORLD, &requests[rank]);
    MPI_Wait(&requests[rank], MPI_STATUS_IGNORE);
  }

  if (rank < P) {
    MPI_Send(local_row.data(), N, MPI_INT, rank + P, 0, MPI_COMM_WORLD);
  }
  
  if (rank >= P && rank < size - P) {
    MPI_Recv(local_row.data(), N, MPI_INT, rank - P, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(local_row.data(), N, MPI_INT, rank + P, 0, MPI_COMM_WORLD);
  }

  if (rank >= size - P) {
    MPI_Recv(local_row.data(), N, MPI_INT, rank - P, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

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

  // Print reduced Ranking
  for (int r = 0; r < size; r++) {
    MPI_Barrier(MPI_COMM_WORLD);  // sincroniza antes de imprimir
    if (rank == r && col == row) {
      printf("Rank %d: \n", rank);
      for (int i = 0; i < N; i++) {
        cout << local_col[i] << " ";
      }
      cout << endl;
      printf("Reduced ranking: ", rank);
      for (int i = 0; i < N; i++) {
        cout << row_ranking[i] << " ";
      }
      cout << endl;
    }
  }

  MPI_Comm_free(&rowComm);

  MPI_Finalize();
  return 0;
}
