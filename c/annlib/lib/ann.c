
#include <ann.h>

// ------------------Input Layer----------------------------------------------
void input_layer_1_D(float in, float *out) {
     *out=in;      
     }
//--------------------------------------------------------------------------------

// ----------------Hidden Layer-------------------------------------------------
void hidden_layer_1_D(int N_hidden,float x_pi, float * w_ji, float * y_pj, int actFunc){	
	int n=0;
 
	for (n;n<N_hidden;n++) {
		
		switch (actFunc){
		
		case 0: /*logsig activation function*/
			{
				y_pj[n]= tanh(x_pi/12.0 * w_ji[n]); //mejorar esto//
			}	
	
		case 1: /*linear activation function*/
			{
				y_pj[n]=x_pi * w_ji[n];
			}	
		default:
			{
				y_pj[n]= tanh(x_pi/12.0 * w_ji[n]); // y esto//
			}
		}
	
	}


}
//-------------------------------------------------------------------------------------

//--------------------Output Layer-----------------------------------------------------
void output_layer_1_D(int N_hidden, float *x , float *w_kj,float *b, float *Y ) 
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
void ANN (int N_hidden,float In, float * w_ji, float  * w_kj, float *b, float * y_pj,float * y_pk)  {
float x_pi;

input_layer_1_D(In,&x_pi);

hidden_layer_1_D(N_hidden,x_pi, w_ji, y_pj,0);

output_layer_1_D(N_hidden,y_pj,w_kj,b,y_pk);

}
//-------------------------------------------------------------

//------------Backpropagation algorithm-------------------------


void BPA (int N_hidden, int N_dat,float eta, float *In,float *d_pk, float *w_ji, float *w_kj, float * b) {

float *y_pj; //hidden out
float *y_pk; // output out
float delta_pk; //output layer delta
float * delta_pj; //hidden layer delta
int x = 0;

//memory allocation for hidden output and delta
y_pj = (float*)malloc(N_hidden* sizeof(float));
delta_pj = (float*)malloc(N_hidden* sizeof(float));
y_pk = (float*)malloc(1* sizeof(float));




//------------ Initial random weights generation//

srand ((unsigned) (time(0)));
for (x;x<N_hidden;x++)
	{
        w_ji [x] =(float) (rand()) / (float) (RAND_MAX);
       
	    w_kj [x] =(float) (rand()) / (float) (RAND_MAX);
	    
	}
	
//bias random generation

//b= static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
*b=-1.72;

//------------ End of random weigths generation





int iter=0;
for (iter;iter<40;iter++) {
	int pos =0;
	
for (pos;pos<N_dat;pos++){
	int y=0;
	
	ANN (N_hidden,In[pos], w_ji, w_kj,b,y_pj,y_pk);//ANN run
	
	//output delta computing
	delta_pk=(d_pk[pos]-*y_pk); 
	

	//hidden delta computing
	 for (y;y<N_hidden;y++) {
		 
         delta_pj[y]= y_pj[y] *( 1.0 - y_pj[y])* delta_pk * w_kj[y];
         w_kj[y] =  w_kj[y] + eta * delta_pk * y_pj[y];  //output weights adjustment
         w_ji[y] =  w_ji[y] + eta * delta_pj[y] * In[pos]; // hidden weight adjustment
	
		}
	*b=*b + eta * delta_pk ;
	
	}
}

}

