// #include <bits/stdc++.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <mpi.h>

int globalRandInt(int nP) {
    int randValue;
    int rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        randValue = rand() % nP;
    }

    MPI_Bcast(&randValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return randValue;
}

int getPivot(int item, MPI_Comm comm, int nP) {
  int pivot = item;
  int pivotPE = globalRandInt(nP);

  MPI_Bcast(&pivot, 1, MPI_INT, pivotPE, comm);
  return pivot;
}

void count(int value, int *sum, int *allSum, MPI_Comm comm, int nP) {
  MPI_Scan(&value, sum, 1, MPI_INT, MPI_SUM, comm);
  *allSum = *sum;
  MPI_Bcast(allSum, 1, MPI_INT, nP - 1, comm);
}


int pQuickSort(int item, MPI_Comm comm) {
  int iP; // % Id del procesador
  int nP; // % Número total de procesadores en el comunicador actual  
  int small; // % si el item < pivot, entonces vale 1, si no 0
  int allSmall; // % numero total de items < pivot
  int pivot; // % pivote global
  MPI_Comm newComm; // % Comunicador para la recursión
  MPI_Status status;
  
  // % Obtener indice local y tamaño del comunicador
  MPI_Comm_rank(comm, &iP);
  MPI_Comm_size(comm, &nP);

  if (nP == 1) {
    return item;
  }
  else {
    pivot = getPivot(item, comm, nP);
    count(item < pivot, &small, &allSmall, comm, nP);
    if (item < pivot) {
      MPI_Bsend(&item, 1, MPI_INT, small - 1, 8, comm);
    }
    else {
      MPI_Bsend(&item, 1, MPI_INT, allSmall + iP - small, 8, comm);
    }
    MPI_Recv(&item, 1, MPI_INT, MPI_ANY_SOURCE, 8, comm, &status);
    MPI_Comm_split(comm, iP < allSmall, 0, &newComm);
    int sorted = pQuickSort(item, newComm);
    MPI_Comm_free(&newComm);
    return  sorted;

  }
}



int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Siembra aleatoria
  if (rank == 0) {
    srand(time(NULL) + rank);
  }

  // Preparamos buffer para MPI_Bsend
  int bufsize = size * (MPI_BSEND_OVERHEAD + sizeof(int));
  void* bbuf = malloc(bufsize);
  MPI_Buffer_attach(bbuf, bufsize);

  // Datos de prueba
  // int testData[] = {44, 77, 11, 55, 22, 54, 10};
  int testData[] = {44, 77, 11, 55};
  int item       = testData[rank];
  int sortedItem = pQuickSort(item, MPI_COMM_WORLD);

  // Impresión ordenada
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < size; ++i) {
    if (rank == i) {
      printf("Proceso %d: item ordenado = %d\n", rank, sortedItem);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Desprendemos buffer y finalizamos
  void* detached_buf;
  int detached_size;
  MPI_Buffer_detach(&detached_buf, &detached_size);
  free(detached_buf);

  MPI_Finalize();
  return 0;
}

