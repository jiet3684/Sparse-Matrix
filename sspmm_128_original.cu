#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
//#include <cub/cub.cuh>

#define ERR fprintf(stderr, "ERR\n");

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FTYPE float
#define STYPE int

#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-2147483648)
#define THRESHOLD (8*2)
#define BH (128/1)
#define BW (128/1)
#define MIN_OCC (BW)	// BW*3/4
//#define MIN_OCC (BW/4)
//#define BW (
#define SBSIZE (1024/8)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)	//1024
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define SC_SIZE (2048)

//#define SIM_VALUE

#define GPRINT(x,y) int *tt0=(int *)malloc(sizeof(int)*(y));\
	fprintf(stderr, "\n");\
	cudaMemcpy(tt0, x, sizeof(int)*(y), cudaMemcpyDeviceToHost);\
	for(int i=0;i<(y);i++) if(tt0[i] == 0) fprintf(stderr,"(%d %d) ", i, tt0[i]); fprintf(stderr,"\n");\
	free(tt0);

#define GPRINT2(x,y) int *tt1=(int *)malloc(sizeof(int)*(y));\
	fprintf(stderr, "\n");\
	cudaMemcpy(tt1, x, sizeof(int)*(y), cudaMemcpyDeviceToHost);\
	for(int i=0;i<(y);i++) fprintf(stderr,"%d ", tt1[i]); fprintf(stderr,"\n");\
	free(tt1);


int gran=1;

struct v_struct {
	int row, col;
	FTYPE val;
	int grp;
};

double avg, vari;
struct v_struct *temp_v, *gold_temp_v;
int sc, nr, nc, ne, gold_ne, npanel, mne, mne_nr;
int nr0;

int *csr_v; 
int *csr_e0;
FTYPE *csr_ev0;


long datavol;

int compare0(const void *a, const void *b)
{
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}

void ready(int argc, char **argv)
{

        FILE *fp;
        int *loc;
        char buf[300];
        int nflag, sflag;
        int pre_count=0;
        int i;

		srand(time(NULL));
		sc = 128;
        int mode = atoi(argv[2]); // 1 : matrix-market, 2 : R-MAT
        fp = fopen(argv[1], "r");

    if (mode == 1) {
        fgets(buf, 300, fp);
        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;

#ifdef SYM
        sflag = 1;
#endif

        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

        fp = fopen(argv[1], "r");
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &nr, &nc, &ne);
        nr0 = nr;
        ne *= (sflag+1);

        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
        gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));

        for(i=0;i<ne;i++) {
                fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
				temp_v[i].grp = INIT_GRP;
                temp_v[i].row--; 
                temp_v[i].col--;

                if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
                        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
                        exit(0);
                }
                if(nflag == 0) temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
                else if(nflag == 1) {
                        FTYPE ftemp;
                        fscanf(fp, " %f ", &ftemp);
                        temp_v[i].val = ftemp;
                } else { // complex
                        FTYPE ftemp1, ftemp2;
                        fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
                        temp_v[i].val = ftemp1;
                }
                //temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;

                if(sflag == 1) {
                        i++;
                        temp_v[i].row = temp_v[i-1].col;
                        temp_v[i].col = temp_v[i-1].row;
                        temp_v[i].val = temp_v[i-1].val;
        		temp_v[i].grp = INIT_GRP;
	        }
        }
    }
    else if (mode == 2) {

        fscanf(fp, "%d %d %d", &nc, &nr, &ne);
        nc = nr;

        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
        gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));

        for(i=0;i<ne;i++) {
			fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
			if(temp_v[i].row == nr) temp_v[i].row--;
			if(temp_v[i].col == nr) temp_v[i].col--;
            temp_v[i].grp = INIT_GRP;
            if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
                fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
                exit(0);
            }
            temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
		}
	}
	
        qsort(temp_v, ne, sizeof(struct v_struct), compare0);

        loc = (int *)malloc(sizeof(int)*(ne+1));

        memset(loc, 0, sizeof(int)*(ne+1));
        loc[0]=1;
        for(i=1;i<ne;i++) {
                if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
                        loc[i] = 0;
                else loc[i] = 1;
        }
        for(i=1;i<=ne;i++)
                loc[i] += loc[i-1];
        for(i=ne; i>=1; i--)
                loc[i] = loc[i-1];
        loc[0] = 0;

        for(i=0;i<ne;i++) {
                temp_v[loc[i]].row = temp_v[i].row;
                temp_v[loc[i]].col = temp_v[i].col;
                temp_v[loc[i]].val = temp_v[i].val;
                temp_v[loc[i]].grp = temp_v[i].grp;
        }
        ne = loc[ne];
        temp_v[ne].row = nr;
        gold_ne = ne;
        for(i=0;i<=ne;i++) {
                gold_temp_v[i].row = temp_v[i].row;
                gold_temp_v[i].col = temp_v[i].col;
                gold_temp_v[i].val = temp_v[i].val;
                gold_temp_v[i].grp = temp_v[i].grp;
        }
        free(loc);

        csr_v = (int *)malloc(sizeof(int)*(nr+1));
        csr_e0 = (int *)malloc(sizeof(int)*ne+256);
        csr_ev0 = (FTYPE *)malloc(sizeof(FTYPE)*ne+256);
        memset(csr_v, 0, sizeof(int)*(nr+1));

        for(i=0;i<ne;i++) {
                csr_e0[i] = temp_v[i].col;
                csr_ev0[i] = temp_v[i].val;
                csr_v[1+temp_v[i].row] = i+1;
        }

        for(i=1;i<nr;i++) {
                if(csr_v[i] == 0) csr_v[i] = csr_v[i-1];
        }
        csr_v[nr] = ne;

        //fprintf(stdout,"TTAAGG,%s,%d,%d,%d,",argv[1],nr0,nc,ne);
        //fprintf(fpo2,"%s,",argv[1]);

}

__global__ 
void spmv_1(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, FTYPE *vin, FTYPE *vout)
{
    int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
        int lane = (threadIdx.x&(MFACTOR-1));
        int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
        int offset2 = offset + MFACTOR;
        int i, j;

	FTYPE r=0.0f;
	FTYPE r2 = 0.0f;
	int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

	int buf; FTYPE buf2;

	int jj = 0, l;
	for(l=loc1; l<loc2; l++) {
		if(jj == 0) {
			buf = csr_e[l+lane];
			buf2 = csr_ev[l+lane];
		}
		FTYPE v1 = __shfl(buf2, jj,MFACTOR);
		int i1 = __shfl(buf, jj,MFACTOR)*sc;
			r += v1 * vin[i1+offset];
			r2 += v1 * vin[i1+offset2];

		jj = (jj+1) % (MFACTOR-1);
	}


    
	vout[idx*sc + offset] = r;
	vout[idx*sc + offset2] = r2;
}

__global__
void spmv_2(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, FTYPE *vin, FTYPE *vout)
{
        int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
        int lane = (threadIdx.x&(MFACTOR-1));
        int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
        int offset2 = offset + MFACTOR;
        int i, j;

	FTYPE r=0.0f;
	FTYPE r2=0.0f;
	int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

        int buf; FTYPE buf2;
        int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

    int jj=0, l;

    for(l=loc1; l<interm3; l+=2) {
		if(jj == 0) {
		        buf = csr_e[l+lane];
		        buf2 = csr_ev[l+lane];
		}
		FTYPE v1 = __shfl(buf2, jj,MFACTOR);
		FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
		int i1 = __shfl(buf, jj,MFACTOR)*sc;
		int i2 = __shfl(buf, jj+1,MFACTOR)*sc;
                r += v1 * vin[i1+offset];
                r2 += v1 * vin[i1+offset2];
                r += v2 * vin[i2+offset];
                r2 += v2 * vin[i2+offset2];

		jj = ((jj+2)&(MFACTOR-1));
        }
        if(interm3 < loc2 && jj == 0) {
                buf = csr_e[l+lane];
                buf2 = csr_ev[l+lane];
        }
        if(interm3 < loc2) {
                r += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR)*sc + offset];
                r2 += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR)*sc + offset2];
        }

	vout[idx*sc + offset] = r;
	vout[idx*sc + offset2] = r2;
}

__global__
void spmv_4(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, FTYPE *vin, FTYPE *vout)
{
	int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
	int lane = (threadIdx.x&(MFACTOR-1));
	int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
	int offset2 = offset + MFACTOR;
	int i, j;

FTYPE r=0.0f;
FTYPE r2=0.0f;
int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

	int buf; FTYPE buf2;
	int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
	int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

int jj=0, l;	
	for(l=loc1; l<interm2; l+=4) {
			if(jj == 0) {
					buf = csr_e[l+lane]*sc;
					buf2 = csr_ev[l+lane];
			}
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
			r += v1 * vin[i1+offset];
			r2 += v1 * vin[i1+offset2];
			r += v2 * vin[i2+offset];
			r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
			r += v3 * vin[i3+offset];
			r2 += v3 * vin[i3+offset2];
			r += v4 * vin[i4+offset];
			r2 += v4 * vin[i4+offset2];

			jj = ((jj+4)&(MFACTOR-1));
	}
	if(interm2 < loc2 && jj == 0) {
			buf = csr_e[l+lane]*sc;
			buf2 = csr_ev[l+lane];
	}
	if(interm2 < interm3) {
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
			r += v1 * vin[i1+offset];
			r2 += v1 * vin[i1+offset2];
			r += v2 * vin[i2+offset];
			r2 += v2 * vin[i2+offset2];

			jj = (jj+2);
	}
	if(interm3 < loc2) {
			r += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset];
			r2 += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset2];
	}
vout[idx*sc + offset] = r;
vout[idx*sc + offset2] = r2;
}

__global__
void spmv_8(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, FTYPE *vin, FTYPE *vout)
{
	int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
	int lane = (threadIdx.x&(MFACTOR-1));
	int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
	int offset2 = offset + MFACTOR;
	int i, j;

	FTYPE r=0.0f;
	FTYPE r2=0.0f;
	int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

		int buf; FTYPE buf2;
		int interm = loc1 + (((loc2 - loc1)>>3)<<3);
		int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
		int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

	int jj=0, l;	
	for(l=loc1; l<interm; l+=8) {
		if(jj == 0) {
				buf = csr_e[l+lane]*sc;
				buf2 = csr_ev[l+lane];
		}
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];

	FTYPE v5 = __shfl(buf2, jj+4,MFACTOR);
	FTYPE v6 = __shfl(buf2, jj+5,MFACTOR);
	int i5 = __shfl(buf, jj+4,MFACTOR);
	int i6 = __shfl(buf, jj+5,MFACTOR);
		r += v5 * vin[i5+offset];
		r2 += v5 * vin[i5+offset2];
		r += v6 * vin[i6+offset];
		r2 += v6 * vin[i6+offset2];

	FTYPE v7 = __shfl(buf2, jj+6,MFACTOR);
	FTYPE v8 = __shfl(buf2, jj+7,MFACTOR);
	int i7 = __shfl(buf, jj+6,MFACTOR);
	int i8 = __shfl(buf, jj+7,MFACTOR);
		r += v7 * vin[i7+offset];
		r2 += v7 * vin[i7+offset2];
		r += v8 * vin[i8+offset];
		r2 += v8 * vin[i8+offset2];

		jj = ((jj+8)&(MFACTOR-1));
	}
	if(interm < loc2 && jj == 0) {
		buf = csr_e[l+lane]*sc;
		buf2 = csr_ev[l+lane];
	}
	if(interm < interm2) {
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];


		jj = (jj+4);
	}
	if(interm2 < interm3) {
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

		jj = (jj+2);
	}
	if(interm3 < loc2) {
		r += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset];
		r2 += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset2];
	}
	vout[idx*sc + offset] = r;
	vout[idx*sc + offset2] = r2;
}

__global__
void spmv_16(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, FTYPE *vin, FTYPE *vout)
{
	int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
	int lane = (threadIdx.x&(MFACTOR-1));
	int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
	int offset2 = offset + MFACTOR;
	int i, j;

	FTYPE r=0.0f;
	FTYPE r2=0.0f;
	int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

        int buf; FTYPE buf2;
        
		int interm0 = loc1 + (((loc2 - loc1)>>4)<<4);
		int interm = loc1 + (((loc2 - loc1)>>3)<<3);
		int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
		int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

	int jj=0, l;	
	for(l=loc1; l<interm0; l+=16) {
		if(jj == 0) {
				buf = csr_e[l+lane]*sc;
				buf2 = csr_ev[l+lane];
		}
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];

	FTYPE v5 = __shfl(buf2, jj+4,MFACTOR);
	FTYPE v6 = __shfl(buf2, jj+5,MFACTOR);
	int i5 = __shfl(buf, jj+4,MFACTOR);
	int i6 = __shfl(buf, jj+5,MFACTOR);
		r += v5 * vin[i5+offset];
		r2 += v5 * vin[i5+offset2];
		r += v6 * vin[i6+offset];
		r2 += v6 * vin[i6+offset2];

	FTYPE v7 = __shfl(buf2, jj+6,MFACTOR);
	FTYPE v8 = __shfl(buf2, jj+7,MFACTOR);
	int i7 = __shfl(buf, jj+6,MFACTOR);
	int i8 = __shfl(buf, jj+7,MFACTOR);
		r += v7 * vin[i7+offset];
		r2 += v7 * vin[i7+offset2];
		r += v8 * vin[i8+offset];
        r2 += v8 * vin[i8+offset2];
        
    FTYPE v9 = __shfl(buf2, jj+8,MFACTOR);
	FTYPE v10 = __shfl(buf2, jj+9,MFACTOR);
	int i9 = __shfl(buf, jj+8,MFACTOR);
	int i10 = __shfl(buf, jj+9,MFACTOR);
		r += v9 * vin[i1+offset];
		r2 += v9 * vin[i1+offset2];
		r += v10 * vin[i2+offset];
		r2 += v10 * vin[i2+offset2];

	FTYPE v11 = __shfl(buf2, jj+10,MFACTOR);
	FTYPE v12 = __shfl(buf2, jj+11,MFACTOR);
	int i11 = __shfl(buf, jj+10,MFACTOR);
	int i12 = __shfl(buf, jj+11,MFACTOR);
		r += v11 * vin[i3+offset];
		r2 += v11 * vin[i3+offset2];
		r += v12 * vin[i4+offset];
		r2 += v12 * vin[i4+offset2];

	FTYPE v13 = __shfl(buf2, jj+12,MFACTOR);
	FTYPE v14 = __shfl(buf2, jj+13,MFACTOR);
	int i13 = __shfl(buf, jj+12,MFACTOR);
	int i14 = __shfl(buf, jj+13,MFACTOR);
		r += v13 * vin[i5+offset];
		r2 += v13 * vin[i5+offset2];
		r += v14 * vin[i6+offset];
		r2 += v14 * vin[i6+offset2];

	FTYPE v15 = __shfl(buf2, jj+14,MFACTOR);
	FTYPE v16 = __shfl(buf2, jj+15,MFACTOR);
	int i15 = __shfl(buf, jj+14,MFACTOR);
	int i16 = __shfl(buf, jj+15,MFACTOR);
		r += v15 * vin[i7+offset];
		r2 += v15 * vin[i7+offset2];
		r += v16 * vin[i8+offset];
		r2 += v16 * vin[i8+offset2];

		jj = ((jj+16)&(MFACTOR-1));
	}
	if(interm0 < loc2 && jj == 0) {
		buf = csr_e[l+lane]*sc;
		buf2 = csr_ev[l+lane];
    }
    if(interm0 < interm){
    FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];

	FTYPE v5 = __shfl(buf2, jj+4,MFACTOR);
	FTYPE v6 = __shfl(buf2, jj+5,MFACTOR);
	int i5 = __shfl(buf, jj+4,MFACTOR);
	int i6 = __shfl(buf, jj+5,MFACTOR);
		r += v5 * vin[i5+offset];
		r2 += v5 * vin[i5+offset2];
		r += v6 * vin[i6+offset];
		r2 += v6 * vin[i6+offset2];

	FTYPE v7 = __shfl(buf2, jj+6,MFACTOR);
	FTYPE v8 = __shfl(buf2, jj+7,MFACTOR);
	int i7 = __shfl(buf, jj+6,MFACTOR);
	int i8 = __shfl(buf, jj+7,MFACTOR);
		r += v7 * vin[i7+offset];
		r2 += v7 * vin[i7+offset2];
		r += v8 * vin[i8+offset];
        r2 += v8 * vin[i8+offset2];

        jj = (jj+8);
    }
	if(interm < interm2) {
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];


		jj = (jj+4);
	}
	if(interm2 < interm3) {
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

		jj = (jj+2);
	}
	if(interm3 < loc2) {
		r += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset];
		r2 += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset2];
	}
	vout[idx*sc + offset] = r;
	vout[idx*sc + offset2] = r2;
}

__global__
void spmv_32(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, FTYPE *vin, FTYPE *vout)
{
	int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
	int lane = (threadIdx.x&(MFACTOR-1));
	int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
	int offset2 = offset + MFACTOR;
	int i, j;

	FTYPE r=0.0f;
	FTYPE r2=0.0f;
	int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

        int buf; FTYPE buf2;
        
		int interm00 = loc1 + (((loc2 - loc1)>>5)<<5);
		int interm0 = loc1 + (((loc2 - loc1)>>4)<<4);
		int interm = loc1 + (((loc2 - loc1)>>3)<<3);
		int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
		int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

	int jj=0, l;	
	for(l=loc1; l<interm00; l+=32) {
		if(jj == 0) {
				buf = csr_e[l+lane]*sc;
				buf2 = csr_ev[l+lane];
		}
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];

	FTYPE v5 = __shfl(buf2, jj+4,MFACTOR);
	FTYPE v6 = __shfl(buf2, jj+5,MFACTOR);
	int i5 = __shfl(buf, jj+4,MFACTOR);
	int i6 = __shfl(buf, jj+5,MFACTOR);
		r += v5 * vin[i5+offset];
		r2 += v5 * vin[i5+offset2];
		r += v6 * vin[i6+offset];
		r2 += v6 * vin[i6+offset2];

	FTYPE v7 = __shfl(buf2, jj+6,MFACTOR);
	FTYPE v8 = __shfl(buf2, jj+7,MFACTOR);
	int i7 = __shfl(buf, jj+6,MFACTOR);
	int i8 = __shfl(buf, jj+7,MFACTOR);
		r += v7 * vin[i7+offset];
		r2 += v7 * vin[i7+offset2];
		r += v8 * vin[i8+offset];
        r2 += v8 * vin[i8+offset2];
        
    FTYPE v9 = __shfl(buf2, jj+8,MFACTOR);
    FTYPE v10 = __shfl(buf2, jj+9,MFACTOR);
    int i9 = __shfl(buf, jj+8,MFACTOR);
    int i10 = __shfl(buf, jj+9,MFACTOR);
        r += v9 * vin[i1+offset];
        r2 += v9 * vin[i1+offset2];
        r += v10 * vin[i2+offset];
        r2 += v10 * vin[i2+offset2];

    FTYPE v11 = __shfl(buf2, jj+10,MFACTOR);
    FTYPE v12 = __shfl(buf2, jj+11,MFACTOR);
    int i11 = __shfl(buf, jj+10,MFACTOR);
    int i12 = __shfl(buf, jj+11,MFACTOR);
        r += v11 * vin[i3+offset];
        r2 += v11 * vin[i3+offset2];
        r += v12 * vin[i4+offset];
        r2 += v12 * vin[i4+offset2];

    FTYPE v13 = __shfl(buf2, jj+12,MFACTOR);
    FTYPE v14 = __shfl(buf2, jj+13,MFACTOR);
    int i13 = __shfl(buf, jj+12,MFACTOR);
    int i14 = __shfl(buf, jj+13,MFACTOR);
        r += v13 * vin[i5+offset];
        r2 += v13 * vin[i5+offset2];
        r += v14 * vin[i6+offset];
        r2 += v14 * vin[i6+offset2];

    FTYPE v15 = __shfl(buf2, jj+14,MFACTOR);
    FTYPE v16 = __shfl(buf2, jj+15,MFACTOR);
    int i15 = __shfl(buf, jj+14,MFACTOR);
    int i16 = __shfl(buf, jj+15,MFACTOR);
        r += v15 * vin[i7+offset];
        r2 += v15 * vin[i7+offset2];
        r += v16 * vin[i8+offset];
        r2 += v16 * vin[i8+offset2];
        
    FTYPE v17 = __shfl(buf2, jj+16,MFACTOR);
	FTYPE v18 = __shfl(buf2, jj+17,MFACTOR);
	int i17 = __shfl(buf, jj+16,MFACTOR);
	int i18 = __shfl(buf, jj+17,MFACTOR);
		r += v17 * vin[i1+offset];
		r2 += v17 * vin[i1+offset2];
		r += v18 * vin[i2+offset];
		r2 += v18 * vin[i2+offset2];

	FTYPE v19 = __shfl(buf2, jj+18,MFACTOR);
	FTYPE v20 = __shfl(buf2, jj+19,MFACTOR);
	int i19 = __shfl(buf, jj+18,MFACTOR);
	int i20 = __shfl(buf, jj+19,MFACTOR);
		r += v19 * vin[i3+offset];
		r2 += v19 * vin[i3+offset2];
		r += v20 * vin[i4+offset];
		r2 += v20 * vin[i4+offset2];

	FTYPE v21 = __shfl(buf2, jj+20,MFACTOR);
	FTYPE v22 = __shfl(buf2, jj+21,MFACTOR);
	int i21 = __shfl(buf, jj+20,MFACTOR);
	int i22 = __shfl(buf, jj+21,MFACTOR);
		r += v21 * vin[i5+offset];
		r2 += v21 * vin[i5+offset2];
		r += v22 * vin[i6+offset];
		r2 += v22 * vin[i6+offset2];

	FTYPE v23 = __shfl(buf2, jj+22,MFACTOR);
	FTYPE v24 = __shfl(buf2, jj+23,MFACTOR);
	int i23 = __shfl(buf, jj+22,MFACTOR);
	int i24 = __shfl(buf, jj+23,MFACTOR);
		r += v23 * vin[i7+offset];
		r2 += v23 * vin[i7+offset2];
		r += v24 * vin[i8+offset];
        r2 += v24 * vin[i8+offset2];
        
    FTYPE v25 = __shfl(buf2, jj+24,MFACTOR);
	FTYPE v26 = __shfl(buf2, jj+25,MFACTOR);
	int i25 = __shfl(buf, jj+24,MFACTOR);
	int i26 = __shfl(buf, jj+25,MFACTOR);
		r += v25 * vin[i1+offset];
		r2 += v25 * vin[i1+offset2];
		r += v26 * vin[i2+offset];
		r2 += v26 * vin[i2+offset2];

	FTYPE v27 = __shfl(buf2, jj+26,MFACTOR);
	FTYPE v28 = __shfl(buf2, jj+27,MFACTOR);
	int i27 = __shfl(buf, jj+26,MFACTOR);
	int i28 = __shfl(buf, jj+27,MFACTOR);
		r += v27 * vin[i3+offset];
		r2 += v27 * vin[i3+offset2];
		r += v28 * vin[i4+offset];
		r2 += v28 * vin[i4+offset2];

	FTYPE v29 = __shfl(buf2, jj+28,MFACTOR);
	FTYPE v30 = __shfl(buf2, jj+29,MFACTOR);
	int i29 = __shfl(buf, jj+28,MFACTOR);
	int i30 = __shfl(buf, jj+29,MFACTOR);
		r += v29 * vin[i5+offset];
		r2 += v29 * vin[i5+offset2];
		r += v30 * vin[i6+offset];
		r2 += v30 * vin[i6+offset2];

	FTYPE v31 = __shfl(buf2, jj+30,MFACTOR);
	FTYPE v32 = __shfl(buf2, jj+31,MFACTOR);
	int i31 = __shfl(buf, jj+30,MFACTOR);
	int i32 = __shfl(buf, jj+31,MFACTOR);
		r += v31 * vin[i7+offset];
		r2 += v31 * vin[i7+offset2];
		r += v32 * vin[i8+offset];
		r2 += v32 * vin[i8+offset2];

		jj = ((jj+32)&(MFACTOR-1));
	}
	if(interm00 < loc2 && jj == 0) {
		buf = csr_e[l+lane]*sc;
		buf2 = csr_ev[l+lane];
    }
    if(interm00 < interm0){
        FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];

	FTYPE v5 = __shfl(buf2, jj+4,MFACTOR);
	FTYPE v6 = __shfl(buf2, jj+5,MFACTOR);
	int i5 = __shfl(buf, jj+4,MFACTOR);
	int i6 = __shfl(buf, jj+5,MFACTOR);
		r += v5 * vin[i5+offset];
		r2 += v5 * vin[i5+offset2];
		r += v6 * vin[i6+offset];
		r2 += v6 * vin[i6+offset2];

	FTYPE v7 = __shfl(buf2, jj+6,MFACTOR);
	FTYPE v8 = __shfl(buf2, jj+7,MFACTOR);
	int i7 = __shfl(buf, jj+6,MFACTOR);
	int i8 = __shfl(buf, jj+7,MFACTOR);
		r += v7 * vin[i7+offset];
		r2 += v7 * vin[i7+offset2];
		r += v8 * vin[i8+offset];
        r2 += v8 * vin[i8+offset2];
        
    FTYPE v9 = __shfl(buf2, jj+8,MFACTOR);
    FTYPE v10 = __shfl(buf2, jj+9,MFACTOR);
    int i9 = __shfl(buf, jj+8,MFACTOR);
    int i10 = __shfl(buf, jj+9,MFACTOR);
        r += v9 * vin[i1+offset];
        r2 += v9 * vin[i1+offset2];
        r += v10 * vin[i2+offset];
        r2 += v10 * vin[i2+offset2];

    FTYPE v11 = __shfl(buf2, jj+10,MFACTOR);
    FTYPE v12 = __shfl(buf2, jj+11,MFACTOR);
    int i11 = __shfl(buf, jj+10,MFACTOR);
    int i12 = __shfl(buf, jj+11,MFACTOR);
        r += v11 * vin[i3+offset];
        r2 += v11 * vin[i3+offset2];
        r += v12 * vin[i4+offset];
        r2 += v12 * vin[i4+offset2];

    FTYPE v13 = __shfl(buf2, jj+12,MFACTOR);
    FTYPE v14 = __shfl(buf2, jj+13,MFACTOR);
    int i13 = __shfl(buf, jj+12,MFACTOR);
    int i14 = __shfl(buf, jj+13,MFACTOR);
        r += v13 * vin[i5+offset];
        r2 += v13 * vin[i5+offset2];
        r += v14 * vin[i6+offset];
        r2 += v14 * vin[i6+offset2];

    FTYPE v15 = __shfl(buf2, jj+14,MFACTOR);
    FTYPE v16 = __shfl(buf2, jj+15,MFACTOR);
    int i15 = __shfl(buf, jj+14,MFACTOR);
    int i16 = __shfl(buf, jj+15,MFACTOR);
        r += v15 * vin[i7+offset];
        r2 += v15 * vin[i7+offset2];
        r += v16 * vin[i8+offset];
        r2 += v16 * vin[i8+offset2];

        jj = (jj+16);
    }
    if(interm0 < interm){
    FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];

	FTYPE v5 = __shfl(buf2, jj+4,MFACTOR);
	FTYPE v6 = __shfl(buf2, jj+5,MFACTOR);
	int i5 = __shfl(buf, jj+4,MFACTOR);
	int i6 = __shfl(buf, jj+5,MFACTOR);
		r += v5 * vin[i5+offset];
		r2 += v5 * vin[i5+offset2];
		r += v6 * vin[i6+offset];
		r2 += v6 * vin[i6+offset2];

	FTYPE v7 = __shfl(buf2, jj+6,MFACTOR);
	FTYPE v8 = __shfl(buf2, jj+7,MFACTOR);
	int i7 = __shfl(buf, jj+6,MFACTOR);
	int i8 = __shfl(buf, jj+7,MFACTOR);
		r += v7 * vin[i7+offset];
		r2 += v7 * vin[i7+offset2];
		r += v8 * vin[i8+offset];
        r2 += v8 * vin[i8+offset2];

        jj = (jj+8);
    }
	if(interm < interm2) {
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

	FTYPE v3 = __shfl(buf2, jj+2,MFACTOR);
	FTYPE v4 = __shfl(buf2, jj+3,MFACTOR);
	int i3 = __shfl(buf, jj+2,MFACTOR);
	int i4 = __shfl(buf, jj+3,MFACTOR);
		r += v3 * vin[i3+offset];
		r2 += v3 * vin[i3+offset2];
		r += v4 * vin[i4+offset];
		r2 += v4 * vin[i4+offset2];


		jj = (jj+4);
	}
	if(interm2 < interm3) {
	FTYPE v1 = __shfl(buf2, jj,MFACTOR);
	FTYPE v2 = __shfl(buf2, jj+1,MFACTOR);
	int i1 = __shfl(buf, jj,MFACTOR);
	int i2 = __shfl(buf, jj+1,MFACTOR);
		r += v1 * vin[i1+offset];
		r2 += v1 * vin[i1+offset2];
		r += v2 * vin[i2+offset];
		r2 += v2 * vin[i2+offset2];

		jj = (jj+2);
	}
	if(interm3 < loc2) {
		r += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset];
		r2 += __shfl(buf2, jj,MFACTOR) * vin[__shfl(buf, jj,MFACTOR) + offset2];
	}
	vout[idx*sc + offset] = r;
	vout[idx*sc + offset2] = r2;
}

void process()
{

	int i, j;

	int *_csr_v; int *_csr_e0; FTYPE *_csr_ev0;
	int *_csr_e; FTYPE *_csr_ev;


        cudaMalloc((void **) &_csr_v, sizeof(int)*(nr+1));
        cudaMalloc((void **) &_csr_e0, sizeof(int)*ne+256);
        cudaMalloc((void **) &_csr_ev0, sizeof(FTYPE)*ne+256);

        cudaMemset(_csr_v, 0, sizeof(int)*(nr+1));
        cudaMemset(_csr_e0, 0, sizeof(int)*ne+256);
        cudaMemset(_csr_ev0, 0, sizeof(FTYPE)*ne+256);

        cudaMemcpy(_csr_v, csr_v, sizeof(int)*(nr+1), cudaMemcpyHostToDevice);
        cudaMemcpy(_csr_e0, csr_e0, sizeof(int)*(ne+1), cudaMemcpyHostToDevice);
        cudaMemcpy(_csr_ev0, csr_ev0, sizeof(FTYPE)*ne, cudaMemcpyHostToDevice);


        FTYPE *vin, *_vin, *vout, *_vout;
        FTYPE *vout_gold;
        vin = (FTYPE *)malloc(sizeof(FTYPE)*nc*sc);
        vout = (FTYPE *)malloc(sizeof(FTYPE)*nr*sc);
        vout_gold = (FTYPE *)malloc(sizeof(FTYPE)*nr*sc);

        cudaError_t err = cudaSuccess;

        err = cudaMalloc((void **) &_vin, sizeof(FTYPE)*nc*sc);
        if(err != 0) exit(0);
        err = cudaMalloc((void **) &_vout, sizeof(FTYPE)*nr*sc);
        if(err != 0) exit(0);

        cudaMemset(_vout, 0, sizeof(FTYPE)*nr*sc);
        for(i=0;i<nc*sc;i++) {
                vin[i] = (FTYPE)(rand()%1048576)/1048576;
#ifdef SIM_VALUE
		vin[i] = 1;
#endif
        }
		cudaMemcpy(_vin, vin, sizeof(FTYPE)*nc*sc, cudaMemcpyHostToDevice);
		
        cudaStream_t stream1, stream2, stream3;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
		cudaStreamCreate(&stream3);
		
	// process
	dim3 s_gridsize(nr/SBF, 1, CEIL(sc, MFACTOR*2));
	dim3 s_blocksize(SBSIZE, 1, 1);
	dim3 ss_gridsize(nr, 1, 1);
	dim3 ss_blocksize(SBSIZE, 1, 1);


	float tot_ms;
	cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

    cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
    spmv_1<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e0, _csr_ev0, _vin, _vout);
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&tot_ms, event1, event2);

	cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
    spmv_2<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e0, _csr_ev0, _vin, _vout);
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&tot_ms, event1, event2);

	cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
    spmv_4<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e0, _csr_ev0, _vin, _vout);
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&tot_ms, event1, event2);

	cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
    spmv_8<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e0, _csr_ev0, _vin, _vout);
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    
    cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
    spmv_16<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e0, _csr_ev0, _vin, _vout);
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    
    cudaDeviceSynchronize();
	cudaEventRecord(event1,0);
    spmv_32<<<s_gridsize, s_blocksize, 0, stream1>>>(sc, _csr_v, _csr_e0, _csr_ev0, _vin, _vout);
	cudaEventRecord(event2,0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&tot_ms, event1, event2);

//	fprintf(stdout, "\n");
        free(vin); free(vout); cudaFree(_vin); cudaFree(_vout);
        free(vout_gold);
	printf("\n");
//printf("st ; %d\n", SSTRIDE);
}

int main(int argc, char **argv)
{
	ready(argc, argv);
	//gen_structure();
	process();
}

