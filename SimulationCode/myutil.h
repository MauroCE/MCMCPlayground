#define PI  3.141592653589793
void opengfsr(void),
	closegfsr(void);
int disrand(int l, int t),intrand(void),poidev(float xm);
float gfsr4(void),norm4(void);
double expdev(void);
double gfsr8(void),norm8(void);

void printerr(char *s);
void printerr2(char *s,int indic);

int bnldev(float pp, int n);

double rgamma(double a, double scale);
double lgamma(double arg);
double lfactl(int n);
void isort(char dir,int n,int * x);
void isorti(char dir, int n, int * x, int *indx);
void isorti2(char dir, int n, int *x, int *priv, int *indx);
void fsort(char dir,int n,float * x);
void dsort(char dir,int n,double * x);
void dsorti(char dir,int n,double * x,int *indx);
void mom(double x[],int n,double *x1,double *x2,double *x3,double *x4,
		double *min,double *max);












