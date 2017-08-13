#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX(A, B)  ((A>B)? A:B)
#define MIN(A, B)  ((A<B)? A:B)

#define TRUE 1
#define FALSE 0

#define CHANGE_WINDOW 100
#define CHANGE_DELTA 0.5
#define SPSA_DELTA .5

#define C_FROM_QP 0.50 
#define SIGMA_SQUARED 3.89789 
double scilab_centre[]={0.650, 0.375, 0.477, 0.454, 0.431, 0.513, 0.235, 0.530, 0.476, 0.196, 0.491, 0.617, 0.127, 0.270};
double scilab_radius=0.671;

struct 	test_output	{ 
			int patterns;
			int typical;
			float accuracy;
			int false_positives;
			int missed_positives;			
			};

typedef struct test_output tes0t_output;

test_output testing(double *a, double R, char *testfile_name);
double fcompute(double *a, double R, char *testfile_name);
main(int argc, char *argv[])
    {
       
    double R, R_previous, R_plus, R_minus, *a, *a_previous, *a_plus, *a_minus, *A_plus, *A_minus, *x_i, delta=SPSA_DELTA;
    double norm_2, loss_sum_plus, loss_sum_minus, max_dist, lowest_norm_2;
    double testing_plus, testing_minus;
    int n, N, N_positive, N_negative, d, i, j, sign, mod_i, mod_j, k, m, m_hadamard, M, y_i, accuracy, false_positives, missed_positives, **hadamard_array;
    int SPSA_check_index_right, SPSA_check_index_left;
    bool minimalchange_flag=FALSE, change_array[CHANGE_WINDOW];
    test_output A, B;

    FILE *training_file, *testing_file;
 
    training_file=fopen(argv[1], "r");
    fscanf(training_file,"%d %d", &N, &d);

    a=(double *)malloc(d*sizeof(double));
    a_previous=(double *)malloc(d*sizeof(double));
    a_plus=(double *)malloc(d*sizeof(double));
    a_minus=(double *)malloc(d*sizeof(double));
    A_plus=(double *)malloc(d*sizeof(double));
    A_minus=(double *)malloc(d*sizeof(double));
    x_i=(double *)malloc(d*sizeof(double));

    SPSA_check_index_left=  floor(d/2);
    SPSA_check_index_right= floor(d/2);
    for (m=0; m<d; m++) { if (m>SPSA_check_index_left && m<SPSA_check_index_right) a[m]=scilab_centre[m]; else a[m]=0.0; a_plus[m]=0.0; a_minus[m]=0.0; }
    R=scilab_radius;

    for (m=1; ; m++) { if(pow(2.0, m)>(d+2)) break; }
    /*m would be the order of the Hadamard matrix of {+1, -1} peturbations*/
    m_hadamard=m;
    M=((int)(pow(2.0, m)));
    hadamard_array=(int **)malloc(M*sizeof(int *));

    for (m=0; m<M; m++)
	{
	    hadamard_array[m]=(int *)malloc(M*sizeof(int));
	}

    for (i=0; i<M; i++)
	{
	printf("\n");
     	for (j=0; j<M; j++)
		{
		sign=1;
		mod_i=i;
		mod_j=j;

		for(m=m_hadamard-1; m>0; m--)
			{
			if ( mod_i >= pow(2, m) && mod_j >= pow(2, m))
				sign=(-1.0)*sign;
			else
				sign=sign;

			mod_i=mod_i%((int)(pow(2, m)));
			mod_j=mod_j%((int)(pow(2, m)));
			if(m==1)
				if(mod_i==1 && mod_j==1) sign=(-1.0)*sign;
			}
		hadamard_array[i][j]=sign;
		printf("%d\t", hadamard_array[i][j]);
		}
	}

    fclose(training_file);
    

    for (m=0; m<atoi(argv[3]) && (!minimalchange_flag || m<=CHANGE_WINDOW); m++) 
    {

    lowest_norm_2=2.0;
    loss_sum_plus=0.0;
    loss_sum_minus=0.0;
    N_positive=0;
    N_negative=0;
    max_dist=0.0;

    for (k=0; k<d; k++)
        { 
        //a_plus[k]=( ((float)rand()/(float)RAND_MAX) > 0.5)? 1.0: -1.0;
        a_plus[k]=(hadamard_array[m%M][k+1]*1.0);
        a_minus[k]=0.0-a_plus[k];
	A_plus[k]=a[k]+delta*a_plus[k];
	A_minus[k]=a[k]+delta*a_minus[k];
        }

    //R_plus=( ((float)rand()/(float)RAND_MAX) > 0.5)? 1.0: -1.0;
    R_plus=(hadamard_array[m%M][d+1]*1.0);
    R_minus=0.0-R_plus;

    training_file=fopen(argv[1], "r");
    
     for (i=0; i<N; i++)
        {
        for (j=0; j<d; j++)
            fscanf(training_file, "%lf", &x_i[j]);

	if (i==0 && m==0) { for (j=0; j<d; j++) a[j]=x_i[j];  }

        fscanf(training_file,"%d", &y_i);
         t1=t2=0;
        if(y_i==1)
	        {

	        norm_2=0.0;
	        for (k=0; k<d; k++)
	            norm_2+=pow((a[k]+delta*a_plus[k]-x_i[k]), 2.0);

		norm_2=2.0-2.0*exp((0.0-norm_2)/(2.0*SIGMA_SQUARED));
		max_dist=MAX( pow(norm_2, 0.5), max_dist);
	       
                loss_sum_plus+=MAX( (norm_2 - pow(R+delta*R_plus, 2.0)), 0);
		lowest_norm_2=MIN(norm_2, lowest_norm_2);
	        
	        norm_2=0.0;
	        for (k=0; k<d; k++)
	            norm_2+=pow((a[k]+delta*a_minus[k]-x_i[k]), 2.0);

		norm_2=2.0-2.0*exp((0.0-norm_2)/(2.0*SIGMA_SQUARED));
		max_dist=MAX( pow(norm_2, 0.5), max_dist);
	        
                loss_sum_minus+=MAX( (norm_2 - pow(R+delta*R_minus, 2.0)), 0);

		lowest_norm_2=MIN(norm_2, lowest_norm_2);

		N_positive++;
                  
       
	        }


	else
		{


	        norm_2=0.0;

	        for (k=0; k<d; k++)
	            norm_2+=pow((a[k]+delta*a_plus[k]-x_i[k]), 2.0);

		norm_2=2.0-2.0*exp((0.0-norm_2)/(2.0*SIGMA_SQUARED));
		max_dist=MAX( pow(norm_2, 0.5), max_dist);
	        loss_sum_plus+=MAX( (pow(R+delta*R_plus, 2.0) - norm_2), 0);
	        
		lowest_norm_2=MIN(norm_2, lowest_norm_2);

	        norm_2=0.0;
	        for (k=0; k<d; k++)
	            norm_2+=pow((a[k]+delta*a_minus[k]-x_i[k]), 2.0);

		norm_2=2.0-2.0*exp((0.0-norm_2)/(2.0*SIGMA_SQUARED));
		max_dist=MAX( pow(norm_2, 0.5), max_dist);
	        loss_sum_minus+=MAX( (pow(R+delta*R_minus, 2.0) - norm_2), 0);

		lowest_norm_2=MIN(norm_2, lowest_norm_2);

		N_negative++;


		}


        }


    fclose(training_file);


    n=N_positive+N_negative;

    loss_sum_plus=pow((R+delta*R_plus), 2.0)+C_FROM_QP*loss_sum_plus;
    loss_sum_minus=pow((R+delta*R_minus), 2.0)+C_FROM_QP*loss_sum_minus;

    A=testing(A_plus, R+delta*R_plus, argv[1]);
    B=testing(A_minus, R+delta*R_minus, argv[1]);

  printf("\niteration: %d f+: %0.2f (%0.2f+%0.2f) A+: %0.2f f-: %0.2f (%0.2f+%0.2f) A-: %0.2f |d|: %0.2f ", m+1, loss_sum_plus, pow((R+delta*R_plus), 2.0), loss_sum_plus-pow((R+delta*R_plus), 2.0), A.accuracy, loss_sum_minus, pow((R+delta*R_minus), 2.0), loss_sum_minus-pow((R+delta*R_minus), 2.0), B.accuracy, lowest_norm_2);

    for (k=0; k<d; k++)
	{
	a_previous[k]=a[k];

	if(k>SPSA_check_index_left && k<SPSA_check_index_right) ; else a[k] = a[k]-(1.0/(m+1.0))*(loss_sum_plus-loss_sum_minus)/(2.0*delta*a_plus[k]);

	a[k] = MIN( MAX(a[k], 0.0), 1.0);
	}

    R_previous=R;
    R =  R-(1/(m+1.0))*(loss_sum_plus-loss_sum_minus)/(2.0*delta*R_plus);
    R =  MIN( MAX(R, 0.0), max_dist);

    minimalchange_flag=TRUE;
    for (k=0; k<d; k++)
	if( fabs(a_previous[k]-a[k]) > CHANGE_DELTA) { minimalchange_flag=FALSE; break; }

    if( fabs(R_previous-R)>CHANGE_DELTA ) minimalchange_flag=FALSE;

    change_array[m%CHANGE_WINDOW]=minimalchange_flag;

    minimalchange_flag=TRUE;
    for (k=0; k<CHANGE_WINDOW; k++)
	{
	minimalchange_flag&=change_array[k];
	if (!minimalchange_flag) break;
	}

   printf(" R: %0.3f%c a:", R, (R_plus==1.0)?'+':'-');
    for (k=0; k<d; k++) //printf(" %0.2f%c", a[k], (a_plus[k]==1.0)?'+':'-');
  fflush(stdout);

    A=testing(a, R, argv[1]);
    printf("  Accuracy: %0.2f F-value: %0.2f", A.accuracy, fcompute(a, R, argv[1]));
    fflush(stdout);
    }

    A=testing(a, R, argv[2]);
    printf("\nAccuracy on %s calculated by subroutine: %0.2f. False +ves: %d Missed +ves: %d.", argv[2], A.accuracy, A.false_positives, A.missed_positives);
    fflush(stdout);
    
    A=testing(scilab_centre, scilab_radius, argv[2]);
    printf("\nAccuracy on %s using SciLab's Fast-SVDD: %0.2f. False +ves: %d Missed +ves: %d.", argv[2], A.accuracy, A.false_positives, A.missed_positives);
    fflush(stdout);

    A=testing(a, R, argv[1]);
    printf("\nAccuracy on %s calculated by subroutine: %0.2f. False +ves: %d Missed +ves: %d.", argv[1], A.accuracy, A.false_positives, A.missed_positives);
    fflush(stdout);

    A=testing(scilab_centre, scilab_radius, argv[1]);
   printf("\nAccuracy on %s using SciLab's Fast-SVDD: %0.2f. False +ves: %d Missed +ves: %d.", argv[1], A.accuracy, A.false_positives, A.missed_positives);
    fflush(stdout);

    printf("\nObjective Values in SciLab's Fast-SVDD: f-test %0.2f, f-train %0.2f", fcompute(scilab_centre, scilab_radius, argv[2]), fcompute(scilab_centre, scilab_radius, argv[1]));
    fflush(stdout);

  printf("\nObjective Function seen by Primal SVDD: f-test %0.2f, f-train %0.2f", fcompute(a, R, argv[2]), fcompute(a, R, argv[1]));
    fflush(stdout);
    
    testing_file=fopen(argv[2], "r");
    fscanf(testing_file,"%d %d", &N, &d);

    accuracy=0;
    false_positives=0;
    missed_positives=0;
    int typical=0;
    int atypical=0;

    for (i=0; i<N; i++)
        {

        for (j=0; j<d; j++)

            fscanf(testing_file, "%lf", &x_i[j]);

        fscanf(testing_file,"%d", &y_i);
        norm_2=0.0;
        for (k=0; k<d; k++)
            norm_2+=pow((a[k]-x_i[k]), 2.0);

	norm_2=2.0-2.0*exp((0.0-norm_2)/(2.0*SIGMA_SQUARED));
        accuracy+=( (norm_2 - pow(R, 2.0)) > 0) ? ( (y_i==-1)? 1:0) : ( (y_i==1)? 1:0);
        false_positives+=( (norm_2 - pow(R, 2.0)) > 0) ? ( (y_i==1)? 1:0 ) : 0;
        missed_positives+=( (norm_2 - pow(R, 2.0)) < 0) ? ( (y_i==-1)? 1:0) : 0;
        
	if (y_i==1) typical++; else atypical++;

        }    


        fclose(testing_file);



        printf("\nTesting File statistics: typical %d (%0.2f) atypical %d (%0.2f)", typical, 100.0*(float)typical/(float)(typical+atypical), atypical, 100.0*(float)atypical/(float)(typical+atypical));

        printf("\nTesting File accuracy: %0.2f False Positives: %0.2f Missed Positives: %0.2f\n", ((float)accuracy/(float)N)*100.0, (float)(false_positives*100.0)/(float)N, (float)(missed_positives*100.0)/(float)N);
        fflush(stdout);
        

    }

test_output testing(double *a, double R, char *testfile_name)
    {
    int N, d, y_i, accuracy, false_positives, missed_positives;
    int i, j, k;
    double norm_2;
    test_output A;

    FILE *testing_file=fopen(testfile_name, "r");
    fscanf(testing_file,"%d %d", &N, &d);
    double *x_i=(double *)malloc(d*sizeof(double));

    A.patterns=N;
    A.typical=0;
    accuracy=0;
    A.false_positives=0;
    A.missed_positives=0;

    for (i=0; i<N; i++)
        {

        for (j=0; j<d; j++)
            fscanf(testing_file, "%lf", &x_i[j]);

        fscanf(testing_file,"%d", &y_i);

	if (y_i==1) A.typical+=1;
        
	norm_2=0.0;
        for (k=0; k<d; k++)
            norm_2+=pow((a[k]-x_i[k]), 2.0);

	norm_2=2.0-2.0*exp((0.0-norm_2)/(2.0*SIGMA_SQUARED));
        accuracy+=( (norm_2 - pow(R, 2.0)) > 0) ? ( (y_i==-1)? 1:0) : ( (y_i==1)? 1:0);
        A.false_positives+=( (norm_2 - pow(R, 2.0)) > 0) ? ( (y_i==1)? 1:0 ) : 0;
        A.missed_positives+=( (norm_2 - pow(R, 2.0)) < 0) ? ( (y_i==-1)? 1:0) : 0;
        }    

    fclose(testing_file);

    A.accuracy=((float)accuracy/(float)N)*100.0;
    return A;
    }

double fcompute(double *a, double R, char *testfile_name)
    {
    int N, d, y_i;
    int i, j, k;
    double norm_2, objective_function;
    
    FILE *testing_file=fopen(testfile_name, "r");
    fscanf(testing_file,"%d %d", &N, &d);
    double *x_i=(double *)malloc(d*sizeof(double));

    objective_function=pow(R, 2.0);

    for (i=0; i<N; i++)
        {

        for (j=0; j<d; j++)
            fscanf(testing_file, "%lf", &x_i[j]);

        fscanf(testing_file,"%d", &y_i);
        norm_2=0.0;
        for (k=0; k<d; k++)
            norm_2+=pow((a[k]-x_i[k]), 2.0);

	norm_2=2.0-2.0*exp((0.0-norm_2)/(2.0*SIGMA_SQUARED));
        if (y_i==1) objective_function+= ( (norm_2 - pow(R, 2.0) > 0) ? C_FROM_QP*(norm_2 - pow(R, 2.0)):0.0);
        }    

 
    fclose(testing_file);

    return (objective_function);

    }

