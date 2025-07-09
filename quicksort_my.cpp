#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <mpi.h>
#include <vector>
#include <algorithm>

using namespace std;

int pickPivot(const vector<int>& local_a, MPI_Comm comm, int iP, int nP) {
  int pivot;
  int pivotP;
  if (iP == 0) {
    srand(time(NULL) + nP + iP);
    pivotP = rand() % nP;
  }

  MPI_Bcast(&pivotP, 1, MPI_INT, 0, comm);

  if (iP == pivotP) {
    pivot = local_a.empty() ? 0 : local_a[local_a.size() / 2];
  }

  MPI_Bcast(&pivot, 1, MPI_INT, pivotP, comm);

  return pivot;
}

pair<vector<int>, vector<int>> partition(const vector<int>& local_a, int pivot) {
  vector<int> less, greater;
  for (int value : local_a) {
    if (value < pivot) {
      less.push_back(value);
    } else {
      greater.push_back(value);
    }
  }
  return make_pair(less, greater);
}

void pQuickSort(vector<int> & local_a, MPI_Comm comm) {
  int iP; // % Id del procesador
  int nP; // % Numero de procesadores: p'
  
  MPI_Comm_rank(comm, &iP);
  MPI_Comm_size(comm, &nP);

  if (nP == 1){
    sort(local_a.begin(), local_a.end());
    return;
  }

  int pivot = pickPivot(local_a, comm, iP, nP); // % Valor pivot
  auto [a, b] = partition(local_a, pivot); // % Particionamiento
  
  long long n_a = 0, n_b = 0;
  long long local_na = a.size();
  long long local_nb = b.size();

  MPI_Allreduce(&local_na, &n_a, 1, MPI_LONG_LONG, MPI_SUM, comm);
  MPI_Allreduce(&local_nb, &n_b, 1, MPI_LONG_LONG, MPI_SUM, comm);

  int k;
  if ()



}
