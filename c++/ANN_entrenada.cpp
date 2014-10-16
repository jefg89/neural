#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// ------------------Input Layer----------------------------------------------
void input_layer (float a, float &out) {
     out=a;      
     }
//--------------------------------------------------------------------------------

// ----------------Hidden Layer-------------------------------------------------
void hidden_layer(int N,float x_pi, float * w_ji, float * y_pj){	
	int n=0;
	for (n;n<N;n++) {
		//y_pj[n]= 1/ (1 + exp(-x_pi * w_ji[n]));	 //sigmoidal output	
		y_pj[n]=x_pi * w_ji[n];	
	}


}
//-------------------------------------------------------------------------------------

//--------------------Output Layer-----------------------------------------------------
void output_layer(int N, float *x , float *w_kj, float &Y ) 
{
     int n=0;
     float add=0.0;
     for (n; n<N; n++)
	{ 	
		add= add + x[n] * w_kj[n]; 
     }
     Y=add;
}

//-----------------------------------------------------------------------------

//------------------ANN construction------------------------------------------
void ANN (int N,float In, float * w_ji, float  * w_kj, float * y_pj,float & y_pk)  {
float x_pi;

input_layer(In,x_pi);

hidden_layer(N,x_pi, w_ji, y_pj);

output_layer(N,y_pj,w_kj,y_pk);

}
//-------------------------------------------------------------

//------------Backpropagation algorithm-------------------------


void BPA (int N, int ix,float eta, float *In,float *d_pk, float *w_ji, float *w_kj) {

float *y_pj; //hidden out
float y_pk; // output out
y_pj = (float*)malloc(N * sizeof(float)); //memory allocation for hidden output

float delta_pk; //output layer delta
float delta_pj; //hidden layer delta


int x = 0;
// Initial random weights generation//
srand (static_cast <unsigned> (time(0)));
for (x;x<N;x++)
	{
        w_ji [x] =static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        //printf ("%f \n",w_ji [x]);
	    w_kj [x] =static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	    //printf ("%f \n",w_kj [x]);
	}




// BP algorithm


int z=0;
for (z;z<40;z++) {
	int iter =0;
for (iter;iter<ix;iter++){
	int y=0;
	ANN (N,In[iter], w_ji, w_kj, y_pj,y_pk);//ANN run
	//printf("salida %f " , y_pk);
	//printf("deseado %f \n", d_pk[iter]);
	delta_pk=(d_pk[iter]-y_pk); //linear case delta
	//printf("delta %f \n", delta_pk);
	int d_add =0;    //sum term for delta_pj
	 for (y;y<N;y++) {
		d_add= d_add + delta_pk * w_kj[y];
		
		}
         // delta_pj= y_pj[y] *( 1 - y_pj[y])* d_add; //sigmoidal case delta
         delta_pj=  d_add;

	 for (y;y<N;y++) {
		w_kj[y] =  w_kj[y] + eta * delta_pk * y_pj[y]; //output weights adjustment
		w_ji[y] =  w_ji[y] + eta * delta_pj * In[iter]; // hidden weight adjustment
		
		}

	
	}
}

}


int main () {

int N=9; // 5 hidden layers
int ix = 500; // 20 iterations


float *w_ji;
float *w_kj;
float *y_pj;
float *In;
float *d_pk;
float y_pk;
y_pj= (float*)malloc(N * sizeof(float));
w_ji = (float*)malloc(N* sizeof(float));;//memory allocation  for hidden w
w_kj = (float*)malloc(N* sizeof(float));;//memory allocation  for output w
//In = (float*) malloc (ix);//memory allocation  for inputs
In = (float*)malloc(ix * sizeof(float));
d_pk = (float*)malloc(ix * sizeof(float));//memory allocation  for desired values


float eta = -0.00000000000000002;  // 0.01

//input and desired output generation
int c=0;
for (c;c<ix;c++) {
	In[c]=(2.0*c);
	d_pk[c]=1.8 + 2.0*c;
	
	
}


//Training

//BPA (N, ix,eta, In,d_pk, float *w_ji, float *w_kj) 

BPA (N,ix,eta, In,d_pk, w_ji, w_kj);
//testing
int c2=0;
for (c2;c2<ix;c2++) {
	In[c2]= log(c2);
	ANN (N,In[c2], w_ji, w_kj, y_pj,y_pk);
	printf ("%f \t", y_pk);
	printf ("%f \n", 1.8 +2.0*In[c2]);
}

c=0;


return 0;
}
