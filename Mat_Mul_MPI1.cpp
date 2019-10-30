#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("avx2")  //Enable AVX

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>

#define MATSIZE 2048

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


#define GET_FLOAT_M128(v128,I) _mm_cvtss_f32(_mm_shuffle_ps(v128, v128, I))


inline void MatMull(const float* a, const float* b, float *c, int a_r, int a_c) {
	
	//what line of a
	#pragma omp parallel for shared(a, b, c)
	for (int i = 0; i < a_r; i++) {
		//what line of b
		for (unsigned l = 0; l < a_c; l++) {
			c[i * a_c + l] = 0;
			for (unsigned j = 0; j < a_c; j ++) {
				c[i * a_c + l] += a[i * a_c + j] * b[l * a_c + j];
			}
		}
	}
	return;

	//#pragma omp parallel for
	//for (int i = 0; i < a_r; i++) {
	//	//what line of b
	//	for (unsigned l = 0; l < a_c; l++) {
	//		//__m256 VecSum[8];
	//		//waht colun of a and b
	//		const float* aa = a + i * a_c;
	//		const float* bb = b + l * a_c;
	//		float v1 = 0;
	//		float v2 = 0;
	//		float v3 = 0;
	//		float v4 = 0;
	//		float v5 = 0;
	//		float v6 = 0;
	//		float v7 = 0;
	//		float v8 = 0;
	//		for (unsigned j = 0; j < a_c; j += 64) {
	//			v1 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//			v2 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//			v3 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//			v4 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//			v5 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//			v6 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//			v7 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//			v8 += _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_dp_ps(_mm256_load_ps(aa), _mm256_load_ps(bb), 0xff))); aa += 8; bb += 8;
	//
	//		}
	//		c[i * a_r + l] = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
	//	}
	//}
	

	//#pragma omp parallel for
	//for (int i = 0; i < a_r; i++) {
	//	//what line of b
	//	for (unsigned l = 0; l < a_c; l++) {
	//		//__m256 VecSum[8];
	//		//waht colun of a and b
	//		c[i * a_r + l] = 0;
	//		for (unsigned j = 0; j < a_c; j += 8) {
	//			
	//			c[i * a_r + l] += _mm_cvtss_f32(_mm256_castps256_ps128(
	//				_mm256_dp_ps(_mm256_load_ps(a + i * a_c + j + 00), _mm256_load_ps(b + l * a_c + j + 00), 0xff)
	//			));
	//		}
	//	}
	//}



	#pragma omp parallel for shared(a,b,c)
	for (int i = 0; i < a_r; i++) {
		//what line of b
		for (int l = 0; l < a_c; l++) {
			const float* aa = a + i * a_c;
			const float* bb = b + l * a_c;
			__m256 VecSum[8]{
				_mm256_setzero_ps(),
				_mm256_setzero_ps(),
				_mm256_setzero_ps(),
				_mm256_setzero_ps(),
				_mm256_setzero_ps(),
				_mm256_setzero_ps(),
				_mm256_setzero_ps(),
				_mm256_setzero_ps()
			};
			//what colun of a and b
			for (int j = 0; j < a_c; j += 64) {
				//to use 100% of cpu
VecSum[0] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[0]);
aa += 8; bb += 8;

VecSum[1] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[1]);
aa += 8; bb += 8;

VecSum[2] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[2]);
aa += 8; bb += 8;

VecSum[3] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[3]);
aa += 8; bb += 8;

VecSum[4] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[4]);
aa += 8; bb += 8;

VecSum[5] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[5]);
aa += 8; bb += 8;

VecSum[6] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[6]);
aa += 8; bb += 8;

VecSum[7] = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(aa), _mm256_load_ps(bb)),
VecSum[7]);
aa += 8; bb += 8;
			}
			//sum the vectors to discover the value of c[i][l]
			VecSum[0] = _mm256_add_ps(VecSum[0], VecSum[4]);
			VecSum[1] = _mm256_add_ps(VecSum[1], VecSum[5]);
			VecSum[2] = _mm256_add_ps(VecSum[2], VecSum[6]);
			VecSum[3] = _mm256_add_ps(VecSum[3], VecSum[7]);
			
			VecSum[0] = _mm256_add_ps(VecSum[0], VecSum[1]);
			VecSum[2] = _mm256_add_ps(VecSum[2], VecSum[3]);
			
			VecSum[0] = _mm256_add_ps(VecSum[0], VecSum[2]);
			
			__m128 r = _mm_add_ps(_mm256_extractf128_ps(VecSum[0], 0), _mm256_extractf128_ps(VecSum[0], 1));
			c[i * a_c + l] = GET_FLOAT_M128(r, 0) + GET_FLOAT_M128(r, 1) + GET_FLOAT_M128(r, 2) + GET_FLOAT_M128(r, 3);
	
		}
	}
}




int main(int argc, char* argv[])
{
	int	numtasks = 0,				/* number of tasks in partition */
		taskid,						/* a task identifier */
		numworkers,					/* number of worker tasks */
		source,						/* task id of message source */
		dest,						/* task id of message destination */
		mtype,						/* message type */
		rows,						/* rows of matrix A sent to each worker */
		averow, extra, offset,		/* used to determine rows sent to each worker */
		i, j, k, rc = 0;			/* misc */
	
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
	
	float* a = new float[MATSIZE * MATSIZE]();	/* matrix A to be multiplied */
	float* b = new float[MATSIZE * MATSIZE]();	/* matrix B to be multiplied */
	float* c = new float[MATSIZE * MATSIZE]();	/* result matrix C */
	


	/**************************** master task ************************************/
	if (taskid == MASTER)
	{
		std::cout << "mpi_mm has started with %d tasks " << numtasks << "\n";
		for (int i = 0; i < MATSIZE; i++) {
			for (int j = 0; j < MATSIZE; j++) {
				a[i * MATSIZE + j] = 1;
				b[i * MATSIZE + j] = 1;
			}
		}
		/* Measure start time */
		double start = MPI_Wtime();

		MPI_Bcast(b, MATSIZE * MATSIZE, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

		/* Send matrix data to the worker tasks */
		averow = MATSIZE / numworkers;
		extra = MATSIZE % numworkers;
		offset = 0;
		mtype = FROM_MASTER;
		for (dest = 1; dest <= numworkers; dest++)
		{
			rows = (dest == numworkers) ? averow + extra : averow;
			//printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
			MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[offset*MATSIZE], rows * MATSIZE, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
			//MPI_Send(b, MATSIZE * MATSIZE, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
			offset = offset + rows;
		}
	
		/* Receive results from worker tasks */
		mtype = FROM_WORKER;
		for (i = 1; i <= numworkers; i++)
		{
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset*MATSIZE], rows * MATSIZE, MPI_FLOAT, source, mtype, MPI_COMM_WORLD, &status);
		}
	
		/* Measure finish time */
		double finish = MPI_Wtime();
		printf("Done in %f seconds.\n", finish - start);
	
		int s = MATSIZE * MATSIZE;
		for (i = 0; i < s; i++) {
			if (c[i] != MATSIZE * 1) {
				std::cout << "dif " << c[i] << "\n";
			}
		}
	}


	/**************************** worker task ************************************/
	if (taskid > MASTER)
	{
		MPI_Bcast(b, MATSIZE * MATSIZE, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

		mtype = FROM_MASTER;
		MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(a, rows * MATSIZE, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
		//MPI_Recv(b, MATSIZE * MATSIZE, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);

		MatMull(a, b, c, rows, MATSIZE);
		//for (k = 0; k < NCB; k++){
		//	for (i = 0; i < rows; i++)
		//	{
		//		c[i][k] = 0.0;
		//		for (j = 0; j < NCA; j++)
		//			c[i][k] = c[i][k] + a[i][j] * b[j][k];
		//	}
		//}
		mtype = FROM_WORKER;
		MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(c, rows * MATSIZE, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
	}
	
	delete[] a;
	delete[] b;
	delete[] c;
	
	MPI_Finalize();
}
