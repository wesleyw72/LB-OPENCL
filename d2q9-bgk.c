/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <xmmintrin.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#define SPEED(numspeeds, x, y, numX, numY) \
  ((numspeeds * numY * numX) + (y * numX) + x)
#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define OCLFILE "kernels.cl"
#define WORKSIZEX (16)
#define WORKSIZEY (8)
int tot_cells = 0; /* no. of cells used in calculation */
float tot_u;       /* accumulated magnitudes of velocity for each cell */
/* struct to hold the parameter values */
typedef struct {
  int nx; /* no. of cells in x-direction */
  int ny; /* no. of cells in y-direction */
  size_t work_group_size;
  int num_work_groups;
  int maxIters;     /* no. of iterations */
  int reynolds_dim; /* dimension for Reynolds number */
  float density;    /* density per link */
  float accel;      /* density redistribution */
  float omega;      /* relaxation parameter */
} t_param;
/* struct to hold OpenCL objects */
typedef struct {
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;

  cl_program program;
  cl_kernel accelerate_flow;
  cl_kernel collision;

  cl_mem partialSums;
  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
} t_ocl;
/* struct to hold the 'speed' values */
typedef struct {
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/
void printGrid(const t_param params, float* cells) {
  FILE* fp;
  fp = fopen("printedGrid", "w");

  for (int y = 0; y < params.ny; y++) {
    for (int x = 0; x < params.nx; x++) {
      for (int s = 0; s < 9; s++) {
        fprintf(fp, "%d %d %d %f\n", x, y, s,
                cells[SPEED(s, x, y, params.nx, params.ny)]);
      }
    }
  }
  fclose(fp);
}
/* load params, allocate memory, load obstacles & initialise fluid particle
 * densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params,
               float** cells_ptr, float** tmp_cells_ptr, int** obstacles_ptr,
               float** av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, int* obstacles, t_ocl ocl, int iteration);
int accelerate_flow(const t_param params, int* obstacles, t_ocl ocl);
int collision(const t_param params, int* obstacles, t_ocl ocl, int iteration);
int write_values(const t_param params, float* cells, int* obstacles,
                 float* av_vels);
int accelerate_flow2(const t_param params, float* cells, int* obstacles);
/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);
void checkError(cl_int err, const char* op, const int line);
/* compute average velocity */
float av_velocity(const t_param params, float* cells, int* obstacles);
cl_device_id selectOpenCLDevice();
/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
void swap_pointers(float** cells, float** tmp_cells) {
  float* temp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = temp;
  // printf("asd\n");
}

int accelerate_flow2(const t_param params, float* cells, int* obstacles) {
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.0;
  float w2 = params.density * params.accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = params.ny - 2;
  for (int jj = 0; jj < params.nx; jj++) {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii * params.nx + jj] &&
        (cells[SPEED(7, jj, ii, params.nx, params.ny)] - w1) > 0.0 &&
        (cells[SPEED(1, jj, ii, params.nx, params.ny)] - w2) > 0.0 &&
        (cells[SPEED(4, jj, ii, params.nx, params.ny)] - w2) > 0.0) {
      /* increase 'east-side' densities */
      cells[SPEED(8, jj, ii, params.nx, params.ny)] += w1;
      cells[SPEED(7, jj, ii, params.nx, params.ny)] -= w1;
      cells[SPEED(3, jj, ii, params.nx, params.ny)] += w2;
      cells[SPEED(1, jj, ii, params.nx, params.ny)] -= w2;
      cells[SPEED(4, jj, ii, params.nx, params.ny)] -= w2;
      cells[SPEED(6, jj, ii, params.nx, params.ny)] += w2;
      /* decrease 'west-side' densities */
    }
  }
  return EXIT_SUCCESS;
}
void init2(float* cells_ptr, t_param params);
int main(int argc, char* argv[]) {
  char* paramfile = NULL;    /* name of the input parameter file */
  char* obstaclefile = NULL; /* name of a the input obstacle file */
  t_param params;            /* struct to hold parameter values */
  float* cells = NULL;
  float* tmp_cells = NULL;
  int* obstacles = NULL; /* grid indicating which cells are blocked */
  float* av_vels =
      NULL; /* a record of the av. velocity computed for each timestep */
  struct timeval timstr; /* structure to hold elapsed time */
  struct rusage ru;      /* structure to hold CPU time--system and user */
  double tic,
      toc; /* floating point numbers to calculate elapsed wallclock time */
  double usrtim; /* floating point number to record elapsed user CPU time */
  double systim; /* floating point number to record elapsed system CPU time */
  t_ocl ocl;
  cl_int err;

  /* parse the command line */
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles,
             &av_vels, &ocl);
  init2(cells, params);
  for (int i = 0; i < params.maxIters; i++) {
    av_vels[i] = 0;
  }
  printf("Num work groups: %d\n", params.num_work_groups);
  printf("Work group size: %zu", params.work_group_size);
  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(ocl.queue, ocl.obstacles, CL_TRUE, 0,
                             sizeof(cl_int) * params.nx * params.ny, obstacles,
                             0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  printf("tot density: %.12E\n", total_density(params, cells));
  err = clEnqueueWriteBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                             sizeof(float) * params.nx * params.ny * NSPEEDS,
                             cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);
  for (int tt = 0; tt < params.maxIters; tt++) {
    // Write cells to device
    timestep(params, obstacles, ocl, tt);
    cl_mem temp;
    temp = ocl.cells;
    ocl.cells = ocl.tmp_cells;
    ocl.tmp_cells = temp;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  printf("a");
  err = clEnqueueReadBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                            sizeof(float) * params.nx * params.ny * NSPEEDS,
                            cells, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);
  float* temp2 =
      malloc(sizeof(float) * params.num_work_groups * params.maxIters);
  err = clEnqueueReadBuffer(
      ocl.queue, ocl.partialSums, CL_TRUE, 0,
      sizeof(float) * params.num_work_groups * params.maxIters, temp2, 0, NULL,
      NULL);
  checkError(err, "reading cells data", __LINE__);
  for (int iter = 0; iter < params.maxIters; iter++) {
    for (int i = 0; i < params.num_work_groups; i++) {
      av_vels[iter] += temp2[iter * params.num_work_groups + i];
    }
    av_vels[iter] = av_vels[iter] / tot_cells;
  }
  free(temp2);

  // printGrid(params,cells);
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n",
         calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, int* obstacles, t_ocl ocl, int iteration) {
  accelerate_flow(params, obstacles, ocl);
  collision(params, obstacles, ocl, iteration);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, int* obstacles, t_ocl ocl) {
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err =
      clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow, 1, NULL, global,
                               NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;
}

int collision(const t_param params, int* obstacles, t_ocl ocl, int iteration) {
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.collision, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting collision arg 0", __LINE__);
  err = clSetKernelArg(ocl.collision, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting collision arg 1", __LINE__);
  err = clSetKernelArg(ocl.collision, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting collision arg 2", __LINE__);
  err = clSetKernelArg(ocl.collision, 3, sizeof(cl_mem), &ocl.partialSums);
  checkError(err, "setting collision arg 7", __LINE__);
  err = clSetKernelArg(ocl.collision, 4,
                       sizeof(cl_float) * params.work_group_size, NULL);
  checkError(err, "setting collision arg 8", __LINE__);
  err = clSetKernelArg(ocl.collision, 5, sizeof(cl_int), &iteration);
  checkError(err, "setting collision arg 5", __LINE__);
  // Enqueue kernel
  size_t global[1] = {params.nx};
  size_t local[1] = {WORKSIZEY};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.collision, 1, NULL, global, local,
                               0, NULL, NULL);
  checkError(err, "enqueueing collision kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for collision kernel", __LINE__);

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, float* cells, int* obstacles) {
  tot_u = 0;
/* initialise */

/* loop over all non-blocked cells */
#pragma omp for schedule(static) reduction(+ : tot_u)
  for (int ii = 0; ii < params.ny; ii++) {
    for (int jj = 0; jj < params.nx; jj++) {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj]) {
        /* local density total */
        float local_density = 0.0;

        local_density += cells[SPEED(0, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(1, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(2, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(3, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(4, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(5, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(6, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(7, jj, ii, params.nx, params.ny)];
        local_density += cells[SPEED(8, jj, ii, params.nx, params.ny)];

        /* x-component of velocity */
        float u_x = (cells[SPEED(1, jj, ii, params.nx, params.ny)] +
                     cells[SPEED(5, jj, ii, params.nx, params.ny)] +
                     cells[SPEED(8, jj, ii, params.nx, params.ny)] -
                     (cells[SPEED(3, jj, ii, params.nx, params.ny)] +
                      cells[SPEED(6, jj, ii, params.nx, params.ny)] +
                      cells[SPEED(7, jj, ii, params.nx, params.ny)])) /
                    local_density;
        /* compute y velocity component */
        float u_y = (cells[SPEED(2, jj, ii, params.nx, params.ny)] +
                     cells[SPEED(5, jj, ii, params.nx, params.ny)] +
                     cells[SPEED(6, jj, ii, params.nx, params.ny)] -
                     (cells[SPEED(4, jj, ii, params.nx, params.ny)] +
                      cells[SPEED(7, jj, ii, params.nx, params.ny)] +
                      cells[SPEED(8, jj, ii, params.nx, params.ny)])) /
                    local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
      }
    }
  }

  return tot_u / (float)tot_cells;
}
void init2(float* cells_ptr, t_param params) {
  float w0 = params.density * 4.0 / 9.0;
  float w1 = params.density / 9.0;
  float w2 = params.density / 36.0;
  for (int ii = 0; ii < params.ny; ii++) {
    for (int jj = 0; jj < params.nx; jj++) {
      /* centre */
      cells_ptr[SPEED(0, jj, ii, params.nx, params.ny)] = w0;
      /* axis directions */
      cells_ptr[SPEED(8, jj, ii, params.nx, params.ny)] = w1;
      cells_ptr[SPEED(2, jj, ii, params.nx, params.ny)] = w1;
      cells_ptr[SPEED(7, jj, ii, params.nx, params.ny)] = w1;
      cells_ptr[SPEED(5, jj, ii, params.nx, params.ny)] = w1;
      /* diagonals */
      cells_ptr[SPEED(3, jj, ii, params.nx, params.ny)] = w2;
      cells_ptr[SPEED(1, jj, ii, params.nx, params.ny)] = w2;
      cells_ptr[SPEED(4, jj, ii, params.nx, params.ny)] = w2;
      cells_ptr[SPEED(6, jj, ii, params.nx, params.ny)] = w2;
    }
  }
}
int initialise(const char* paramfile, const char* obstaclefile, t_param* params,
               float** cells_ptr, float** tmp_cells_ptr, int** obstacles_ptr,
               float** av_vels_ptr, t_ocl* ocl) {
  char message[1024]; /* message buffer */
  FILE* fp;           /* file pointer */
  int xx, yy;         /* generic array indices */
  int blocked;        /* indicates whether a cell is blocked by an obstacle */
  int retval;         /* to hold return value for checking */
  char* ocl_src;      /* OpenCL kernel source */
  long ocl_size;      /* size of OpenCL kernel source */
  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1)
    die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1)
    die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1)
    die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr =
      (float*)_mm_malloc(sizeof(t_speed) * (params->ny * params->nx), 64);

  if (*cells_ptr == NULL)
    die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr =
      (float*)_mm_malloc(sizeof(t_speed) * (params->ny * params->nx), 64);

  if (*tmp_cells_ptr == NULL)
    die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++) {
    for (int jj = 0; jj < params->nx; jj++) {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if (retval != 3)
      die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1)
      die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1)
      die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1)
      die("obstacle blocked value should be 1", __LINE__, __FILE__);
    ++tot_cells;
    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  tot_cells = params->nx * params->ny - tot_cells;
  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL) {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(ocl->context, 1,
                                           (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);
  params->work_group_size = WORKSIZEX;
  params->num_work_groups = params->ny / params->work_group_size;
  // Build OpenCL program
  // char options[] = "-cl-unsafe-math-optimizations -cl-mad-enable";
  char buff[100];
  sprintf(buff,
          "-D nx=%d -D ny=%d -D omega=%f -D NUMWORKGROUPS=%d "
          "-cl-fast-relaxed-math -DMAC -D NUMI=%d",
          params->nx, params->ny, params->omega, params->num_work_groups,
          params->ny);

  err = clBuildProgram(ocl->program, 1, &ocl->device, buff, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t sz;
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &sz);
    char* buildlog = malloc(sz);
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, sz,
                          buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->collision = clCreateKernel(ocl->program, "collision", &err);
  checkError(err, "creating collision kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                              sizeof(float) * params->nx * params->ny * NSPEEDS,
                              NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(
      ocl->context, CL_MEM_READ_WRITE,
      sizeof(float) * params->nx * params->ny * NSPEEDS, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(
      ocl->context, CL_MEM_READ_WRITE,
      sizeof(float) * params->nx * params->ny * NSPEEDS, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);
  printf("MAXITERS:%d", params->maxIters);

  ocl->partialSums = clCreateBuffer(
      ocl->context, CL_MEM_READ_WRITE,
      sizeof(float) * params->num_work_groups * params->maxIters, NULL, &err);
  checkError(err, "Creating buffer partialSums", __LINE__);
  printf("NUM WORK GROUPS: %d\n", params->num_work_groups);

  int compute_units;
  size_t max_group_size;
  clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                  &compute_units, NULL);
  clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                  &max_group_size, NULL);
  printf("Compute units: %d, max group size: %d", compute_units,
         max_group_size);
  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl) {
  /*
  ** free up allocated memory
  */
  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.collision);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, float* cells, int* obstacles) {
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim /
         viscosity;
}

float total_density(const t_param params, float* cells) {
  float total = 0.0; /* accumulator */

  for (int ii = 0; ii < params.ny; ii++) {
    for (int jj = 0; jj < params.nx; jj++) {
      for (int kk = 0; kk < NSPEEDS; kk++) {
        total += cells[SPEED(kk, jj, ii, params.nx, params.ny)];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, int* obstacles,
                 float* av_vels) {
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  float local_density;          /* per grid cell sum of densities */
  float pressure;               /* fluid pressure in grid cell */
  float u_x;                    /* x-component of velocity in grid cell */
  float u_y;                    /* y-component of velocity in grid cell */
  float u; /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++) {
    for (int jj = 0; jj < params.nx; jj++) {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj]) {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++) {
          local_density += cells[SPEED(kk, jj, ii, params.nx, params.ny)];
        }

        /* compute x velocity component */
        u_x = (cells[SPEED(8, jj, ii, params.nx, params.ny)] +
               cells[SPEED(3, jj, ii, params.nx, params.ny)] +
               cells[SPEED(6, jj, ii, params.nx, params.ny)] -
               (cells[SPEED(7, jj, ii, params.nx, params.ny)] +
                cells[SPEED(1, jj, ii, params.nx, params.ny)] +
                cells[SPEED(4, jj, ii, params.nx, params.ny)])) /
              local_density;
        /* compute y velocity component */
        u_y = (cells[SPEED(2, jj, ii, params.nx, params.ny)] +
               cells[SPEED(3, jj, ii, params.nx, params.ny)] +
               cells[SPEED(1, jj, ii, params.nx, params.ny)] -
               (cells[SPEED(5, jj, ii, params.nx, params.ny)] +
                cells[SPEED(4, jj, ii, params.nx, params.ny)] +
                cells[SPEED(6, jj, ii, params.nx, params.ny)])) /
              local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u,
              pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++) {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice() {
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++) {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES - total_devices, devices + total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++) {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char* dev_env = getenv("OCL_DEVICE");
  if (dev_env) {
    char* end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end)) die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices) {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME, MAX_DEVICE_NAME, name,
                  NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
void checkError(cl_int err, const char* op, const int line) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}
