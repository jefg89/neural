#ifndef _ANN_H
#define _ANN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int N_hidden=0; /*number of hidden neurons in layer*/
int N_in=0;  /*number of input variables*/
int N_iter=0; /*number of training iterations*/
int N_dat=0;  /*number of training data*/

/*activation function mode*/
int actFunc=0; 
/*0 = logsig , 1= linear  /*
/*------------------------*/



float eta=0.0; /*learning speed*/
float *b;

float * x_pi;
float *y_pj; //hidden out
float *y_pk; // output out
float delta_pk; //output layer delta
float * delta_pj; //hidden layer delta
float **w_ji;
float *w_kj;

/*initialize neural network architecture*/
void init_network_1_D();

/*one dimensional ann structural function set*/
void input_layer_1_D(float *in, float *out); 
void hidden_layer_1_D(float * x_pi, float ** w_ji, float * y_pj);
void output_layer_1_D(float *x , float *w_kj,float *b, float *Y ); 

/*N input - single output ANN*/
void ANN (float * In, float ** w_ji, float  * w_kj, float *b, float * y_pj,float * y_pk);


/*one dimensional input - backpropagation algorithm*/
void BPA (float **In,float *d_pk, float *b); //esta



#endif

