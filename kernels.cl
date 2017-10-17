#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define SPEED(numspeeds,x,y,numX,numY) ((numspeeds * numY * numX) + (y * numX) + x)
//#define SPEED(numspeeds,x,y,numX,numY) ((9 * numX * y) + (x * numY) + numspeeds)
//#define SPEED(numspeeds,x,y,numX,numY) ((numY * 9 * x) + (y * 9) + numspeeds)
//#define SPEED(numspeeds,x,y,numX,numY) ((numspeeds * numY * numX) + (x * numY) + y)
void reduce(                                          
   __local  float*,                          
   __global float*,int iteration);
kernel void collision(global float* cells,global float* tmp_cells,global int* obstacles,global float* g_velocities,local float* l_velocities,private int iteration)
{
  l_velocities[get_local_id(0)]=0;
  //int ii = get_global_id(1);
  int jj = get_global_id(0);
  //printf("%d",NUMI);

  for(int ii=0;ii<NUMI;ii++)
  {
    int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
      //t_speed localSpeeds;
  float t0,t1,t2,t3,t4,t5,t6,t7,t8;
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */

  t0 = cells[SPEED(0,jj,ii,nx,ny)];
  t1 = cells[SPEED(8,x_w,ii,nx,ny)];
  t2 = cells[SPEED(2,jj,y_s,nx,ny)];
  t3 = cells[SPEED(7,x_e,ii,nx,ny)];
  t4 = cells[SPEED(5,jj,y_n,nx,ny)];
  t5 = cells[SPEED(3,x_w,y_s,nx,ny)];
  t6 = cells[SPEED(1,x_e,y_s,nx,ny)];
  t7 = cells[SPEED(4,x_e,y_n,nx,ny)];
  t8 = cells[SPEED(6,x_w,y_n,nx,ny)];
        /* compute local density total */
  float local_density = t0;
  local_density += t1;
  local_density += t2;
  local_density += t3;
  local_density += t4;
  local_density += t5;
  local_density += t6;
  local_density += t7;
  local_density += t8;   
        /* compute x velocity component */
  float u_x = (t1+ t5+ t8- (t3+ t6+ t7))/ local_density;
        /* compute y velocity component */
  float u_y = (t2+ t5+ t6- (t4+ t7+ t8))/ local_density;
  const float ld1 =local_density*omega/9.0f;
  const float ld2 = local_density*omega/36.0f;
  const float u_s = u_x + u_y;
  const float u_d = -u_x +u_y; 
  const float u_sq = (u_x*u_x+u_y*u_y);
  const float d_eq = (1.0f - 1.5f*(u_sq));
  tmp_cells[SPEED(8,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t3 : ld1*(d_eq + 4.5f*u_x*(2.0f/3.0f + u_x)) - 0.85f*t1;
  tmp_cells[SPEED(7,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t1 :ld1*(d_eq - 4.5f*u_x*(2.0f/3.0f - u_x)) - 0.85f*t3;
  tmp_cells[SPEED(2,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t4 :ld1*(d_eq + 4.5f*u_y*(2.0f/3.0f + u_y)) - 0.85f*t2;
  tmp_cells[SPEED(5,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t2 :ld1*(d_eq - 4.5f*u_y*(2.0f/3.0f - u_y)) - 0.85f*t4;
  tmp_cells[SPEED(3,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t7 :ld2*(d_eq + 4.5f*u_s*(2.0f/3.0f + u_s)) - 0.85f*t5;
  tmp_cells[SPEED(4,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t5 :ld2*(d_eq - 4.5f*u_s*(2.0f/3.0f - u_s)) - 0.85f*t7;
  tmp_cells[SPEED(1,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t8 :ld2*(d_eq + 4.5f*u_d*(2.0f/3.0f + u_d)) - 0.85f*t6;
  tmp_cells[SPEED(6,jj,ii,nx,ny)] = obstacles[ii * nx + jj] ? t6 :ld2*(d_eq - 4.5f*u_d*(2.0f/3.0f - u_d)) - 0.85f*t8;
  tmp_cells[SPEED(0,jj,ii,nx,ny)] = d_eq*local_density*omega*(4.0f/9.0f) - 0.85f*t0;      
  l_velocities[get_local_id(0)]+= obstacles[ii * nx + jj] ? 0 :  sqrt((u_x * u_x) + (u_y * u_y));
  }
  
  
  
  //int group_id       = get_group_id(0);
  
  //printf("Group id:%d\n",group_id);
   barrier(CLK_LOCAL_MEM_FENCE);
  reduce(l_velocities, g_velocities,iteration);      
}
kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            float density, float accel)
{
  /* compute weighting factors */
  double w1 = density * accel / 9.0;
  double w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[SPEED(7,jj,ii,nx,ny)] - w1) > 0.0
      && (cells[SPEED(1,jj,ii,nx,ny)] - w2) > 0.0
      && (cells[SPEED(4,jj,ii,nx,ny)] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[SPEED(8,jj,ii,nx,ny)] += w1;
    cells[SPEED(7,jj,ii,nx,ny)] -= w1;
    cells[SPEED(3,jj,ii,nx,ny)] += w2;
    /* decrease 'west-side' densities */
    cells[SPEED(1,jj,ii,nx,ny)] -= w2;
    cells[SPEED(4,jj,ii,nx,ny)] -= w2;
    cells[SPEED(6,jj,ii,nx,ny)] += w2;
  }
}
void reduce(                                          
   __local  float*    local_sums,                          
   __global float*    partial_sums,int iteration)                        
{                                                          
   int num_wrk_items  = get_local_size(0);                 
   int local_id       = get_local_id(0); 
   //int local_id       = get_group_id(0)+get_group_size(0)*get_group_id(1);               
   int group_id       = get_group_id(0);               
   
   float sum;                              
   int i;           
   for(int offset = get_local_size(0)/2;offset>0;offset=offset/2)
   {
    if(local_id<offset)
    {
      float other = local_sums[local_id+offset];
      float mine = local_sums[local_id];
      local_sums[local_id] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
   }
   if(local_id==0)
   {
    partial_sums[iteration*NUMWORKGROUPS+group_id]=local_sums[0];
   }



   
   // if (local_id == 0) {    
   // //printf("ITERATION %d\n",NUMITERS);                  
   //    sum = 0.0f;   
   //    //printf("GLOBAL SIZE: %d",get_global_size(0));                         
   
   //    for (i=0; i<num_wrk_items; i++) {        
   //        sum += local_sums[i];             
   //    }                                     
   //   // printf("GLOBAL ID: %d %d %d %d\n",get_num_groups(0),get_num_groups(1),get_group_id(0),get_group_id(1));
   //   // printf("index:%d, NUMITERS:%d group_id :%d iteration:%d\n ",iteration*NUMWORKGROUPS+group_id,NUMWORKGROUPS,group_id,iteration);
   //    partial_sums[iteration*NUMWORKGROUPS+group_id] = sum;         
   // }
}
kernel void Mreduce(__global float* buffer,
            __local float* scratch,
            __const int length,
            __global float* result) {

  int global_index = get_global_id(0);
  float accumulator = 0;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float element = buffer[global_index];
    accumulator = accumulator + element;
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}

