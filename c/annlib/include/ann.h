#ifndef _ANN_H
#define _ANN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int N_hidden=0; /*number of hidden neurons in layer*/


/*one dimensional ann structural function set*/
void input_layer_1_D(float in, float *out); 
void hidden_layer_1_D(int N_hidden,float x_pi, float * w_ji, float * y_pj,int actFunc);
void output_layer_1_D(int N_hidden, float *x , float *w_kj,float *b, float *Y ); //esta

/*single input - single output ANN*/
void ANN (int N_hidden,float In, float * w_ji, float  * w_kj, float *b, float * y_pj,float * y_pk); //esta


/*one dimensional input - backpropagation algorithm*/
void BPA (int N, int N_dat,float eta, float *In,float *d_pk, float *w_ji, float *w_kj, float *b); //esta



#endif

