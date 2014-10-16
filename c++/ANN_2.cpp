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
		y_pj[n]= tanh(x_pi/12.0 * w_ji[n]);	
		//y_pj[n]=x_pi * w_ji[n];	
	}


}
//-------------------------------------------------------------------------------------

//--------------------Output Layer-----------------------------------------------------
void output_layer(int N, float *x , float *w_kj,float b, float &Y ) 
{
     int n=0;
     float add=0.0;
     for (n; n<N; n++)
	{ 	
		add= add + x[n] * w_kj[n]; 
     }
     Y=add + b;
}

//-----------------------------------------------------------------------------

//------------------ANN construction------------------------------------------
void ANN (int N,float In, float * w_ji, float  * w_kj, float b, float * y_pj,float & y_pk)  {
float x_pi;

input_layer(In,x_pi);

hidden_layer(N,x_pi, w_ji, y_pj);

output_layer(N,y_pj,w_kj,b,y_pk);

}
//-------------------------------------------------------------

//------------Backpropagation algorithm-------------------------


void BPA (int N, int N_dat,float eta, float *In,float *d_pk, float *w_ji, float *w_kj, float & b) {

float *y_pj; //hidden out
float y_pk; // output out
float delta_pk; //output layer delta
float * delta_pj; //hidden layer delta
int x = 0;

//memory allocation for hidden output and delta
y_pj = (float*)malloc(N * sizeof(float));
delta_pj = (float*)malloc(N* sizeof(float));




//------------ Initial random weights generation//

srand (static_cast <unsigned> (time(0)));
for (x;x<N;x++)
	{
        w_ji [x] =static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
       
	    w_kj [x] =static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	    
	}
	
//bias random generation

//b= static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
b=-1.72;

//------------ End of random weigths generation


// BP algorithm trained for 50 interations


int iter=0;
for (iter;iter<40;iter++) {
	int pos =0;
	
for (pos;pos<N_dat;pos++){
	int y=0;
	
	ANN (N,In[pos], w_ji, w_kj,b,y_pj,y_pk);//ANN run
	
	//output delta computing
	delta_pk=(d_pk[pos]-y_pk); 
	

	//hidden delta computing
	 for (y;y<N;y++) {
		 
         delta_pj[y]= y_pj[y] *( 1.0 - y_pj[y])* delta_pk * w_kj[y];
         w_kj[y] =  w_kj[y] + eta * delta_pk * y_pj[y];  //output weights adjustment
         w_ji[y] =  w_ji[y] + eta * delta_pj[y] * In[pos]; // hidden weight adjustment
	
		}
	b=b + eta * delta_pk ;
	
	}
}

}


int main () {

int N=10; // number of hidden layers
int N_dat = 102; // number of training data
int c=0;


float *w_ji;
float *w_kj;
float *y_pj;
//float In [34] ;
//float d_pk [34] ;
float y_pk;
float b=0.0;

y_pj= (float*)malloc(N * sizeof(float));
w_ji = (float*)malloc(N* sizeof(float));;//memory allocation  for hidden w
w_kj = (float*)malloc(N* sizeof(float));;//memory allocation  for output w

float In [192] = {11.800009,16.400026,17.400030,16.400026,12.600012,16.200026,1.800000,13.600016,14.200018,15.000021,30.400080,17.200029,13.400015,17.800032,15.200022,13.600016,
13.200014,13.600016,14.000017,13.600016,14.200018,17.000029,8.999998,11.400007,13.600016,13.600016,29.000074,13.600016,32.800072,17.800032,13.600016,11.400007,11.600008,8.999998,
11.800009,16.400026,13.600016,8.999998,13.600016,11.200006,21.200045,14.000017,20.000040,14.000017,2.000000,32.000084,32.000084,32.200081,27.600069,26.000063,1.800000,1.800000,17.000029,
1.800000,25.400061,12.200010,1.800000,14.200018,16.400026,11.800009,13.600016,13.600016,13.400015,1.800000,11.800009,44.799889,13.600016,16.600027,11.200006,1.800000,14.200018,24.200056,
32.200081,1.800000,32.200081,16.200026,13.800016,33.200066,13.600016,16.400026,35.000038,14.200018,13.600016,16.400026,1.800000,11.800009,12.200010,12.200010,15.000021,1.800000,17.000029,
16.800028,13.600016,1.800000,1.800000,17.200029,32.200081,13.800016,14.200018,13.600016,14.200018,16.400026,17.200029,32.400078,12.200010,1.800000,1.800000,11.600008,13.600016,13.600016,11.000006,
13.600016,32.600075,32.400078,1.800000,14.200018,2.000000,11.600008,1.800000,13.600016,32.000084,17.200029,13.600016,39.399971,14.000017,2.000000,33.200066,15.000021,14.000017,11.600008,32.200081,
13.400015,1.800000,33.600060,19.600039,1.800000,14.200018,17.400030,17.200029,13.600016,17.200029,17.200029,14.200018,13.600016,14.400019,16.400026,17.200029,16.600027,16.400026,13.800016,13.600016,
6.999996,17.200029,16.400026,32.200081,17.200029,16.400026,14.000017,2.000000,16.400026,17.600031,32.600075,13.800016,17.600031,20.600042,21.000044,2.000000,14.000017,15.000021,13.600016,17.400030,
14.000017,20.600042,15.000021,11.000006,9.199999,13.600016,12.200010,14.200018,14.200018,13.600016,14.200018,32.200081,23.800055,13.400015,32.200081,17.200029,14.200018,13.600016,17.600031,17.200029,20.800043};
float d_pk [192] ;
for (c;c<192;c++) {
	if (In [c] < 11.6) d_pk[c]=-1;
	else d_pk[c]=1;
}

//In = (float*)malloc(N_dat * sizeof(float));
//d_pk = (float*)malloc(N_dat * sizeof(float));//memory allocation  for desired values


float eta = 0.000000002;  // 0.0000272

//input and desired output generation
/*int c=1;
for (c;c<N_dat;c++) {
	In[c]=(c);
	d_pk[c]=1.8 + 2.0* sqrt(c)/(10.0) + log(c);
	
	
	
}
* */


//Training

//BPA (N, N_dat,eta, In,d_pk, float *w_ji, float *w_kj,b) 

BPA (N,N_dat,eta, In,d_pk, w_ji, w_kj,b);
//testing
int c2=0;
for (c2;c2<192;c2++) {
	printf("%f \t", In [c2]);
	//In[c2]= 2.0+c2;
	ANN (N,In[c2], w_ji, w_kj,b, y_pj,y_pk);
	printf ("%f \t", y_pk);
	//printf ("%f \n", 1.8 +2.0*sqrt(In[c2])/10.0 + log(In[c2]));
	printf ("%f \n", d_pk[c2]);
}

c=0;

for (c;c<N;c++) {
	printf ("%f \t", w_ji[c]);
	printf ("%f \n", w_kj[c]);
}
printf("b= %f \n",b);
return 0;
}
