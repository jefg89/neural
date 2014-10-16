
#include <ann.h>

void init_network_1_D() {
x_pi = (float*)malloc(N_in* sizeof(float)); /*dynamic mem allocation for input*/

//memory allocation for hidden output and delta
y_pj = (float*)malloc(N_hidden* sizeof(float));   
y_pk = (float*)malloc(1* sizeof(float));
w_kj=  (float*)malloc(N_hidden* sizeof(float));  
delta_pj= (float*)malloc(N_hidden * sizeof(float));
w_ji=(float**)malloc(N_hidden* sizeof(float*)); 
b=(float*)malloc(1* sizeof(float)); 
int i=0;
for (i=0; i<N_hidden; i++){
	w_ji[i]= (float*)malloc(N_in * sizeof(float));
	}
	
}

// ------------------Input Layer----------------------------------------------
void input_layer_1_D(float *in, float *out) {
     int idx;
     for (idx=0;idx<N_in;idx++) {
		out[idx]=in[idx];
	 }
     
           
     }
//--------------------------------------------------------------------------------

// ----------------Hidden Layer-------------------------------------------------
void hidden_layer_1_D(float *x_pi, float ** w_ji, float * y_pj){	
	int i=0;
	int j=0;
 
	for(i;i<N_hidden;i++) {
		y_pj[i]=0;
		for (j=0;j<N_in;j++) {
			switch (actFunc){
			case 0: /*logsig activation function*/
				{
					y_pj[i]= y_pj[i] + tanh(x_pi[j]/12.0 * w_ji[i][j]); //mejorar esto//
				}	
		
			case 1: /*linear activation function*/
				{
					y_pj[i]=y_pj[i]+x_pi[j] * w_ji[i][j];
				}	
			default:
				{
					y_pj[i]= tanh(x_pi[j]/12.0 * w_ji[i][j]); // y esto//
				}
			}	
		}
	}


}
//-------------------------------------------------------------------------------------

//--------------------Output Layer-----------------------------------------------------
void output_layer_1_D(float *x , float *w_kj,float *b, float *Y ) 
{
     int n=0;
     float add=0.0;
     for (n; n<N_hidden; n++)
	{ 	
		add= add + x[n] * w_kj[n]; 
     }
     *Y=add + *b;
}
//-----------------------------------------------------------------------------

//------------------ANN construction------------------------------------------
void ANN (float *In, float ** w_ji, float  * w_kj, float *b, float * y_pj,float * y_pk)  {

input_layer_1_D(In,x_pi);

hidden_layer_1_D(x_pi, w_ji, y_pj);

output_layer_1_D(y_pj,w_kj,b,y_pk);

}

//-------------------------------------------------------------

//------------Backpropagation algorithm-------------------------
void BPA (float **In,float *d_pk, float * b) {
int i = 0;
int j=0;
int iter=0;

//------------ Initial random weights generation//

srand ((unsigned) (time(0)));
for (i=0;i<N_hidden;i++){
	w_kj [i] =(float) (rand()) / (float) (RAND_MAX);
	for (j=0;j<N_in;j++) {
        w_ji [i][j]=(float) (rand()) / (float) (RAND_MAX);
		}
	}



//bias random generation
//b= static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
*b=-1.72;

//------------ End of random weigths generation

//----------Back_propagation-----------------------//
for (iter;iter<N_iter;iter++) {
	int pos =0;
	for (pos;pos<N_dat;pos++){
		ANN (In[pos], w_ji, w_kj,b,y_pj,y_pk);//ANN run
		//printf("salida %d = %f \n",pos,*y_pk);
 
		//output delta computing
		delta_pk=(d_pk[pos]-*y_pk); 
		//hidden delta computing
		 for (i=0;i<N_hidden;i++) {
			delta_pj[i]= y_pj[i] *( 1.0 - y_pj[i])* delta_pk * w_kj[i];
			w_kj[i] =  w_kj[i] + eta * delta_pk * y_pj[i];  //output weights adjustment
			for (j=0;j<N_in;j++) {
				w_ji[i][j] =  w_ji[i][j] + eta * delta_pj[i] * In[pos][j]; // hidden weight adjustment
			} 
			
			}
			*b=*b + eta * delta_pk ;
		}
	}

}
