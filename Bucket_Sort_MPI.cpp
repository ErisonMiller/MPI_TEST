#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <math.h>

#define USE_OMP 0

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


#define DATA_SIZE 10000000
#define NUM_BUCKETS 12

#define MPI_TYPE MPI_INT
typedef int Type;
typedef std::vector<Type> Bucket;

Type Rand_Data() {
	return rand()%(NUM_BUCKETS * 100);
}

int getBucket(const Type &t) {
	return floor(t * 0.01f);
}

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
	int	numtasks = 0,				/* number of tasks in partition */
		taskid,						/* a task identifier */
		source,						/* task id of message source */
		mtype,						/* message type */
		rows,						/* rows of matrix A sent to each worker */
		averow, extra, offset,		/* used to determine rows sent to each worker */
		j, k, rc = 0;			/* misc */

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	if (numtasks < 2) {
		fprintf(stderr, "Need at least two MPI tasks. Quitting...\n");

		MPI_Abort(MPI_COMM_WORLD, rc);
		exit(1);
	}

	const int numworkers = numtasks - 1;

	Type *data = new Type[DATA_SIZE]();


	// Process 0
	if (taskid == MASTER) {
		
		std::cout << "mpi_mm has started with %d tasks " << numtasks << "\n";
		// printf("Initializing arrays...\n");
		for (int i = 0; i < DATA_SIZE; i++) {
			data[i] = Rand_Data();
		}
	
		Bucket buckets[NUM_BUCKETS];
		for (int i = 0; i < NUM_BUCKETS; i++) {
			buckets[i].reserve(DATA_SIZE / NUM_BUCKETS);
		}
	
	
		/* Measure start time */
		double start = MPI_Wtime();
		
		//#pragma omp declare reduction (merge : std::vector<Type> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

		
		#if !USE_OMP

		for (int i = 0; i < DATA_SIZE; i++) {
			buckets[getBucket(data[i])].push_back(data[i]);
		}
		#else

		#pragma omp parallel
		{
			int ithread = omp_get_thread_num();
			int nthreads = omp_get_num_threads();
		
			int start, size;
			size = (ithread == nthreads - 1) ? DATA_SIZE / nthreads + DATA_SIZE % nthreads : DATA_SIZE / nthreads;
			start = ithread * DATA_SIZE / nthreads;
			
			Bucket private_buckets[NUM_BUCKETS];
			//#pragma omp for nowait
			for (int i = start; i < start + size; i++) {
				private_buckets[getBucket(data[i])].push_back(data[i]);
			}
			for (int i = 0; i < NUM_BUCKETS; i++) {
				#pragma omp critical
				buckets[i].insert(buckets[i].end(), private_buckets[i].begin(), private_buckets[i].end());
			}
		}
		#endif


		averow = NUM_BUCKETS / numworkers;
		extra = NUM_BUCKETS % numworkers;
		offset = 0;

		mtype = FROM_MASTER;
				
		//#pragma omp parallel for
		for (int dest = 1; dest <= numworkers; dest++)
		{
			rows = (dest == numworkers) ? averow + extra : averow;
			MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			for (j = 0; j < rows; j++) {
				unsigned bucket_size = buckets[offset + j].size();
				MPI_Send(&bucket_size, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				//dividing data btw nodes
				MPI_Send(&buckets[offset + j][0], bucket_size, MPI_TYPE, dest, mtype, MPI_COMM_WORLD);
			}
			offset = offset + rows;
		}
		


		/* Receive results from worker tasks */
		mtype = FROM_WORKER;
		int posi = 0;
		for (int i = 0; i < numworkers; i++)
		{
			source = i+1;
			MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(data + posi, rows, MPI_TYPE, source, mtype, MPI_COMM_WORLD, &status);
			posi += rows;
		}
		
		/* Measure finish time */
		double finish = MPI_Wtime();
		printf("Done in %f seconds.\n", finish - start);
		
		for (int i = 1; i < DATA_SIZE; i++) {
			if (data[i - 1] > data[i])std::cout << "deu errado\n";
		}
	}
	else {
		mtype = FROM_MASTER;
		
		int buckets = 0;
		MPI_Recv(&buckets, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		
		int *buckets_position = (int*)alloca(buckets + 1);

		buckets_position[0] = 0;
		int posi = 0;
		for (int i = 0; i < buckets; i++) {

			MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

			MPI_Recv(data + posi, rows, MPI_TYPE, MASTER, mtype, MPI_COMM_WORLD, &status);
			
			//std::sort(data + posi, data + posi + rows);
			posi += rows;
			buckets_position[i + 1] = posi;
		}

		#pragma omp parallel for
		for (int i = 0; i < buckets; i++) {
			std::sort(data + buckets_position[i], data + buckets_position[i + 1]);
		}
		
		
		mtype = FROM_WORKER;
		MPI_Send(&posi, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&data[0], posi, MPI_TYPE, MASTER, mtype, MPI_COMM_WORLD);
	
	}

	free(data);
	
	MPI_Finalize();
	
}
