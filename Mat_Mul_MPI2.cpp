#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("avx2")  //Enable AVX

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <cstring>
#include <vector>

#define MATSIZE 2048

#define NUM_DIVISIONS 4

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


inline void MatMull(const float* _a, const float* _b, float *_c, int a_r, int b_c) {
	
	//what line of a 
	#pragma omp parallel for
	for (int i = 0; i < a_r; i++) {
		//what line of b
		for (unsigned l = 0; l < b_c; l++) {
			_c[i * b_c + l] = 0;
			for (unsigned j = 0; j < MATSIZE; j ++) {
				_c[i * b_c + l] += _a[i * MATSIZE + j] * _b[l * MATSIZE + j];
			}
			//_c[i * b_c + l] = 256;
		}
	}

}


int actual_row = 0, actual_col = 0, x = 0, y = 0,
	numworkers;			/* number of worker tasks */

std::vector<int> free_nodes;

bool running = true;

float* a;
float* b;
float* c;
float* ax;

struct Block {
	int x, y, rows, cols;
};

void Master_Sender() {
	//std::unique_lock<std::mutex> lk(m);
	//cv.wait(lk, [] {return free_nodes.empty(); });

	/* Send matrix data to the worker tasks */
	int averow = MATSIZE / NUM_DIVISIONS;
	int extra = MATSIZE % NUM_DIVISIONS;
	
	int mtype = FROM_MASTER;

	int rows = (actual_row == (NUM_DIVISIONS-1)) ? (averow + extra) : averow;
	int cols = (actual_col == (NUM_DIVISIONS-1)) ? (averow + extra) : averow;

	Block block{ x,y,rows, cols };
	//printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
	int dest = free_nodes.back();
	free_nodes.pop_back();
	
	MPI_Send(&block, 4, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	//MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
	MPI_Send(&a[y * MATSIZE], rows * MATSIZE, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
	MPI_Send(&b[x * MATSIZE], cols * MATSIZE, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);


	x += cols;
	actual_col++;
	if (actual_col == NUM_DIVISIONS) {
		actual_col = 0;
		actual_row++;
		x = 0;
		y += rows;
	}
	if (actual_row == NUM_DIVISIONS) {
		running = false;
	}
}

void Master_Recieve() {
	MPI_Status status;
	Block block;
	int finished_nodes = 0;
	while (finished_nodes < numworkers)
	{
		
		MPI_Recv(&block, 4, MPI_INT, MPI_ANY_SOURCE, FROM_WORKER, MPI_COMM_WORLD, &status);

		MPI_Recv(ax, block.rows * block.cols, MPI_FLOAT, status.MPI_SOURCE, FROM_WORKER, MPI_COMM_WORLD, &status);

		for (int i = 0; i < block.rows; i++) {
			std::memcpy(c + (block.y + i) * MATSIZE + block.x, ax + i * block.cols, block.cols * sizeof(float));
		}

		free_nodes.push_back(status.MPI_SOURCE);
		
		//finished_nodes++;
		//running = false;
		MPI_Send(&running, 1, MPI_CHAR, status.MPI_SOURCE, FROM_MASTER, MPI_COMM_WORLD);
		if (running) {
			Master_Sender();
		}
		else {
			finished_nodes++;
		}
		//finished_nodes++;

	}


}

int main(int argc, char* argv[])
{

	//float* a = new float[MATSIZE * MATSIZE]();	/* matrix A to be multiplied */
	//float* b = new float[MATSIZE * MATSIZE]();	/* matrix B to be multiplied */
	//float* c = new float[MATSIZE * MATSIZE]();	/* result matrix C */
	//for (int i = 0; i < MATSIZE; i++) {
	//	for (int j = 0; j < MATSIZE; j++) {
	//		a[i * MATSIZE + j] = 1;
	//		b[i * MATSIZE + j] = -2;
	//	}
	//}
	//
	//clock_t t;
	//t = clock();
	//
	//MatMull(a, b, c, MATSIZE, MATSIZE);
	//
	//t = clock() - t;
	//std::cout << "levou " << t << " clocks ou " << ((float)t) / CLOCKS_PER_SEC << " segundos ou " << 1.0f / (((float)t) / CLOCKS_PER_SEC) << " fps\n";
	//
	//delete[] a;
	//delete[] b;
	//delete[] c;

	int	numtasks = 0,				/* number of tasks in partition */
		taskid,						/* a task identifier */
		source,						/* task id of message source */
		dest,						/* task id of message destination */
		mtype,						/* message type */
		rows,						/* rows of matrix A sent to each worker */
		averow, extra, offset,		/* used to determine rows sent to each worker */
		rc = 0;			/* misc */
	
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	if (numtasks < 2) {
		printf("Need at least two MPI tasks. Quitting...\n");
	
		MPI_Abort(MPI_COMM_WORLD, rc);
		exit(1);
	}
	numworkers = numtasks - 1;
	
	a = new float[MATSIZE * MATSIZE]();	/* matrix A to be multiplied */
	b = new float[MATSIZE * MATSIZE]();	/* matrix B to be multiplied */
	c = new float[MATSIZE * MATSIZE]();	/* result matrix C */
	const int s = MATSIZE / NUM_DIVISIONS + MATSIZE % NUM_DIVISIONS;
	ax = new float[s * s]();	/* result matrix C */




	/**************************** master task ************************************/
	if (taskid == MASTER)
	{
		std::cout << "mpi_mm has started with %d tasks " << numtasks << "\n";
		for (int i = 0; i < MATSIZE; i++) {
			for (int j = 0; j < MATSIZE; j++) {
				a[i * MATSIZE + j] = 1;
				b[i * MATSIZE + j] = 1;
				c[i * MATSIZE + j] = -1;
			}
		}
		/* Measure start time */
		double start = MPI_Wtime();

		for (int i = 0; i < numworkers; i++) {
			free_nodes.push_back(i + 1);
			Master_Sender();
		}
	
		Master_Recieve();
		
		/* Measure finish time */
		double finish = MPI_Wtime();
		printf("Done in %f seconds.\n", finish - start);
	
		int s = MATSIZE * MATSIZE;
		for (int i = 0; i < s; i++) {
			if (c[i] != MATSIZE * 1) {
				std::cout << "dif " << c[i] << "\n";
			}
		}
		printf("Done in %f seconds.\n", finish - start);

	}
	
	
	/**************************** worker task ************************************/
	if (taskid > MASTER)
	{
		//MPI_Bcast(b, MATSIZE * MATSIZE, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

		while (running) {
			mtype = FROM_MASTER;
			Block block;
			MPI_Recv(&block, 4, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(a, block.rows * MATSIZE, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(b, block.cols * MATSIZE, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);

			MatMull(a, b, c, block.rows, block.cols);
			//for (k = 0; k < NCB; k++){
			//	for (i = 0; i < rows; i++)
			//	{
			//		c[i][k] = 0.0;
			//		for (j = 0; j < NCA; j++)
			//			c[i][k] = c[i][k] + a[i][j] * b[j][k];
			//	}
			//}
			mtype = FROM_WORKER;
			MPI_Send(&block, 4, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Send(c, block.rows * block.cols, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);


			MPI_Recv(&running, 1, MPI_CHAR, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
			//if (block.x + block.cols == MATSIZE-1 && block.y + block.rows == MATSIZE-1) {
			//	running = false;
			//}
		}

		printf("Woerker end.\n");
	}
	
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] ax;
	
	MPI_Finalize();
}
