#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "myutil_new.h"

/* old_edit 28 May 1996 */
/* last_edit 27 June 1996 (corrected isort and isorti bugs) */
/* edited 20 March 1997 to remove NR gammln, replace with lgamma from R
   Also wrote function lfactl */
/* edited 26 March 1997 to correct error in lfactl */
/* edited 1 April 1997 to remove sorting routines - replaced with
new ones based on R - isort and isorti*/
/* edited 10 April to remove possibility of 0 and 1 in expdev() and gfsr4()*/
/* edited 8 July to have gfsr8() */
/* edited 4 March 1998 to correct error in 10 April corrections - the
functions still gave 0 */
/* edited 13 March 1998  to make printerr print to a file as well*/
/* edited 20 May 1998 to make fsort */
/* edited 16 July 1998 to make dsort */
/* edited 27 August 1998 to make expdev double */
/* edited 10 February 2000 to make isorti2 (doesn't have malloc) */
/*edited 7 December 2000 to make norm8() and to make all references to
random numbers in rgamma() to be doubles - i.e. gfsr8() and norm8() */
/* edited 8 February 2000 to make a version of printerr that will
only stop if non-zero integer is supplied - call it printerr2*/

int 	rand_table[98],jindic;


void printerr2(char *s,int indic)
{
	FILE *erf;
	erf = fopen("ERRFILE","a");
	printf("error:: %s\n",s);
	fprintf(erf,"error:: %s\n",s);
	fclose(erf);
	if(indic)exit(1);
}

void printerr(char *s)
{
	FILE *erf;
	erf = fopen("ERRFILE","w");
	printf("error:: %s\n",s);
	fprintf(erf,"error:: %s\n",s);
	fclose(erf);
	exit(1);
}


int intrand(void)

{
      int k;
      ++jindic;
      if(jindic>97)jindic=0;
      k = jindic+27;
      if(k>97)k=k-98;
      rand_table[jindic] = rand_table[jindic]^rand_table[k];
      return(rand_table[jindic]);
}

int disrand(int l,int t)
{
      int k;
      if(t<l){
      	printf("error in disrand\n");
      	exit(1);
      }
      ++jindic;
      if(jindic>97)jindic=0;
      k = jindic+27;
      if(k>97)k=k-98;
      rand_table[jindic] = rand_table[jindic]^rand_table[k];
	return((unsigned)rand_table[jindic]%(t-l+1)+l);
}

float gfsr4(void)
{
      int k;
      ++jindic;
      if(jindic>97)jindic=0;
      k = jindic+27;
      if(k>97)k=k-98;
      rand_table[jindic] = rand_table[jindic]^rand_table[k];
	return(((unsigned)rand_table[jindic] + 1.0)/4294967298.0);
}


double gfsr8(void)
{
      int k;
      ++jindic;
      if(jindic>97)jindic=0;
      k = jindic+27;
      if(k>97)k=k-98;
      rand_table[jindic] = rand_table[jindic]^rand_table[k];
	return(((unsigned)rand_table[jindic] + 1.0)/4294967298.0);
}

#define repeat for(;;)

double rgamma(double a, double scale)
{

static double a1 = 0.3333333;
static double a2 = -0.250003;
static double a3 = 0.2000062;
static double a4 = -0.1662921;
static double a5 = 0.1423657;
static double a6 = -0.1367177;
static double a7 = 0.1233795;
static double e1 = 1.0;
static double e2 = 0.4999897;
static double e3 = 0.166829;
static double e4 = 0.0407753;
static double e5 = 0.010293;
static double q1 = 0.04166669;
static double q2 = 0.02083148;
static double q3 = 0.00801191;
static double q4 = 0.00144121;
static double q5 = -7.388e-5;
static double q6 = 2.4511e-4;
static double q7 = 2.424e-4;
static double sqrt32 = 5.656854;

static double aa = 0.;
static double aaa = 0.;

	static double b, c, d, e, p, q, r, s, t, u, v, w, x;
	static double q0, s2, si;
	double ret_val;

	if (a < 1.0) {
		/* alternate method for parameters a below 1 */
		/* 0.36787944117144232159 = exp(-1) */
		aa = 0.0;
		b = 1.0 + 0.36787944117144232159 * a;
		repeat {
			p = b * gfsr8();
			if (p >= 1.0) {
				ret_val = -log((b - p) / a);
				if (expdev() >= (1.0 - a) * log(ret_val))
					break;
			} else {
				ret_val = exp(log(p) / a);
				if (expdev() >= ret_val)
					break;
			}
		}
		return scale * ret_val;
	}
	/* Step 1: Recalculations of s2, s, d if a has changed */
	if (a != aa) {
		aa = a;
		s2 = a - 0.5;
		s = sqrt(s2);
		d = sqrt32 - s * 12.0;
	}
	/* Step 2: t = standard normal deviate, */
	/* x = (s,1/2)-normal deviate. */
	/* immediate acceptance (i) */

	t = norm8();
	x = s + 0.5 * t;
	ret_val = x * x;
	if (t >= 0.0)
		return scale * ret_val;

	/* Step 3: u = 0,1 - uniform sample. squeeze acceptance (s) */
	u = gfsr8();
	if (d * u <= t * t * t) {
		return scale * ret_val;
	}
	/* Step 4: recalculations of q0, b, si, c if necessary */

	if (a != aaa) {
		aaa = a;
		r = 1.0 / a;
		q0 = ((((((q7 * r + q6) * r + q5) * r + q4)
			* r + q3) * r + q2) * r + q1) * r;

		/* Approximation depending on size of parameter a */
		/* The constants in the expressions for b, si and */
		/* c were established by numerical experiments */

		if (a <= 3.686) {
			b = 0.463 + s + 0.178 * s2;
			si = 1.235;
			c = 0.195 / s - 0.079 + 0.16 * s;
		} else if (a <= 13.022) {
			b = 1.654 + 0.0076 * s2;
			si = 1.68 / s + 0.275;
			c = 0.062 / s + 0.024;
		} else {
			b = 1.77;
			si = 0.75;
			c = 0.1515 / s;
		}
	}
	/* Step 5: no quotient test if x not positive */

	if (x > 0.0) {
		/* Step 6: calculation of v and quotient q */
		v = t / (s + s);
		if (fabs(v) <= 0.25)
			q = q0 + 0.5 * t * t * ((((((a7 * v + a6)
					    * v + a5) * v + a4) * v + a3)
						 * v + a2) * v + a1) * v;
		else
			q = q0 - s * t + 0.25 * t * t + (s2 + s2)
			    * log(1.0 + v);


		/* Step 7: quotient acceptance (q) */

		if (log(1.0 - u) <= q)
			return scale * ret_val;
	}
	/* Step 8: e = standard exponential deviate */
	/* u= 0,1 -uniform deviate */
	/* t=(b,si)-double exponential (laplace) sample */

	repeat {
		e = expdev();
		u = gfsr8();
		u = u + u - 1.0;
		if (u < 0.0)
			t = b - si * e;
		else
			t = b + si * e;
		/* Step  9:  rejection if t < tau(1) = -0.71874483771719 */
		if (t >= -0.71874483771719) {
			/* Step 10:  calculation of v and quotient q */
			v = t / (s + s);
			if (fabs(v) <= 0.25)
				q = q0 + 0.5 * t * t * ((((((a7 * v + a6)
					    * v + a5) * v + a4) * v + a3)
						 * v + a2) * v + a1) * v;
			else
				q = q0 - s * t + 0.25 * t * t + (s2 + s2)
				    * log(1.0 + v);
			/* Step 11:  hat acceptance (h) */
			/* (if q not positive go to step 8) */
			if (q > 0.0) {
				if (q <= 0.5)
					w = ((((e5 * q + e4) * q + e3)
					      * q + e2) * q + e1) * q;
				else
					w = exp(q) - 1.0;
				/* if t is rejected */
				/* sample again at step 8 */
				if (c * fabs(u) <= w * exp(e - 0.5 * t * t))
					break;
			}
		}
	}
	x = s + 0.5 * t;
	return scale * x * x;
}

#undef repeat

int bnldev(float pp, int n)
{
	int j;
	static int nold=(-1);
	float am,em,g,angle,p,bnl,sq,t,y;
	static float pold=(-1.0),pc,plog,pclog,en,oldg;

	p=(pp <= 0.5 ? pp : 1.0-pp);
	am=n*p;
	if (n < 25) {
		bnl=0.0;
		for (j=1;j<=n;j++)
			if (gfsr4() < p) ++bnl;
	} else if (am < 1.0) {
		g=exp(-am);
		t=1.0;
		for (j=0;j<=n;j++) {
			t *= gfsr4();
			if (t < g) break;
		}
		bnl=(j <= n ? j : n);
	} else {
		if (n != nold) {
			en=n;
			oldg=lgamma(en+1.0);
			nold=n;
		} if (p != pold) {
			pc=1.0-p;
			plog=log(p);
			pclog=log(pc);
			pold=p;
		}
		sq=sqrt(2.0*am*pc);
		do {
			do {
				angle=PI*gfsr4();
				y=tan(angle);
				em=sq*y+am;
			} while (em < 0.0 || em >= (en+1.0));
			em=floor(em);
			t=1.2*sq*(1.0+y*y)*exp(oldg-lgamma(em+1.0)
				-lgamma(en-em+1.0)+em*plog+(en-em)*pclog);
		} while (gfsr4() > t);
		bnl=em;
	}
	if (p != pp) bnl=n-bnl;
	return (int) bnl + 0.5;
}



int poidev(float xm)
{
	static float sq,alxm,g,oldm=(-1.0);
	float em,t,y;

	if (xm < 12.0) {
		if (xm != oldm) {
			oldm=xm;
			g=exp(-xm);
		}
		em = -1;
		t=1.0;
		do {
			em += 1.0;
			t *= gfsr4();
		} while (t > g);
	} else {
		if (xm != oldm) {
			oldm=xm;
			sq=sqrt(2.0*xm);
			alxm=log(xm);
			g=xm*alxm-lgamma(xm+1.0);
		}
		do {
			do {
				y=tan(PI*gfsr4());
				em=sq*y+xm;
			} while (em < 0.0);
			em=floor(em);
			t=0.9*(1.0+y*y)*exp(em*alxm-lgamma(em+1.0)-g);
		} while (gfsr4() > t);
	}
	return (int) em+0.5;
}
/* Inserted from R0.16::lgamma.c */

/*
 *  R : A Computer Langage for Statistical Data Analysis
 *  Copyright (C) 1995, 1996  Robert Gentleman and Ross Ihaka
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/* Modification by MAB 20.3.97  removed #include "Mathlib.h"
inserted definition of M_PI, copied from Mathlib.h */
#undef M_PI
#define M_PI		3.141592653589793238462643383276

int signgamR1 = 0;

/* log(2*pi)/2 and pi */

static double hl2pi = 0.9189385332046727417803297;
static double xpi = M_PI;

 /* Coefficients from Cheney and Hart */

#define M 6
#define N 8
static double p1[] =
{
	0.83333333333333101837e-1,
	-.277777777735865004e-2,
	0.793650576493454e-3,
	-.5951896861197e-3,
	0.83645878922e-3,
	-.1633436431e-2,
};
static double p2[] =
{
	-.42353689509744089647e5,
	-.20886861789269887364e5,
	-.87627102978521489560e4,
	-.20085274013072791214e4,
	-.43933044406002567613e3,
	-.50108693752970953015e2,
	-.67449507245925289918e1,
	0.0,
};
static double q2[] =
{
	-.42353689509744090010e5,
	-.29803853309256649932e4,
	0.99403074150827709015e4,
	-.15286072737795220248e4,
	-.49902852662143904834e3,
	0.18949823415702801641e3,
	-.23081551524580124562e2,
	0.10000000000000000000e1,
};

static double posarg(double);
static double negarg(double);
static double asform(double);

double lgamma(double arg)
{
	signgamR1 = 1.0;
	if (arg <= 0.0)
		return (negarg(arg));
	if (arg > 8.0)
		return (asform(arg));
	return (log(posarg(arg)));
}

/* Equation 6.1.41 Abramowitz and Stegun */
/* See also ACM algorithm 291 */

static double asform(arg)
double arg;
{
	double log();
	double n, argsq;
	int i;

	argsq = 1. / (arg * arg);
	for (n = 0, i = M - 1; i >= 0; i--) {
		n = n * argsq + p1[i];
	}
	return ((arg - .5) * log(arg) - arg + hl2pi + n / arg);
}

static double negarg(arg)
double arg;
{
	double temp;
	double log(), sin(), posarg();

	arg = -arg;
	temp = sin(xpi * arg);
	if (temp == 0.0)
		/* removed DOMAIN_ERROR */ printerr("negarg: temp == 0.0");
	if (temp < 0.0)
		temp = -temp;
	else
		signgamR1 = -1;
	return (-log(arg * posarg(arg) * temp / xpi));
}

static double posarg(arg)
double arg;
{
	double n, d, s;
	int register i;

	if (arg < 2.0)
		return (posarg(arg + 1.0) / arg);
	if (arg > 3.0)
		return ((arg - 1.0) * posarg(arg - 1.0));

	s = arg - 2.;
	for (n = 0, d = 0, i = N - 1; i >= 0; i--) {
		n = n * s + p2[i];
		d = d * s + q2[i];
	}
	return (n / d);
}

/* end of lgamma insertion */


double lfactl(int n)
{
	static double lookup[1000];
	if(n < 0)printerr("lfactl: n < 0");
	if(n <= 1)return 0.0;
/*	return lgamma(n+1.0); */
	if(n >= 1000)return lgamma(n+1.0);
 	if(lookup[n] == 0.0)lookup[n] = lgamma(n+1.0);
	return lookup[n];
}


double expdev(void)
{

      int k;
      ++jindic;
      if(jindic>97)jindic=0;
      k = jindic+27;
      if(k>97)k=k-98;
      rand_table[jindic] = rand_table[jindic]^rand_table[k];
	return(-log(((unsigned)rand_table[jindic] + 1.0)/4294967298.0));
}

void opengfsr(void)
{

	FILE 	*rt;
	int 	j;

	rt = fopen("INTFILE","r");
	if(rt==NULL) {
		printf("I need INTFILE! Where is INTFILE?\n");
		exit(1);
	}
	for(j=0;j<98;++j) {fscanf(rt,"%d",&rand_table[j]);
	//printf("ASDF%d\n", rand_table[j]);/*TEST*/
	}
	fscanf(rt,"%d",&jindic);
	fclose(rt);
}

void closegfsr(void)
{
	FILE 	*rt;
	int 	j;

	rt = fopen("INTFILE","w");
	for(j=0;j<98;++j)fprintf(rt,"%d\n",rand_table[j]);
	fprintf(rt,"%d\n",jindic);
	fclose(rt);
}

float norm4(void)
{
	static int iset=0;
	static float gset;
	float fac,rsq,v1,v2;

	if  (iset == 0) {
		do {
			v1=2.0*gfsr4()-1.0;
			v2=2.0*gfsr4()-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}


double norm8(void)
{
	static int iset=0;
	static double gset;
	double fac,rsq,v1,v2;

	if  (iset == 0) {
		do {
			v1=2.0*gfsr8()-1.0;
			v2=2.0*gfsr8()-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}

void mom(double x[],int n,double *x1,double *x2,double *x3,double *x4,
		double *min,double *max)
{
	int i;
	double s1,s2,s3,s4,an,an1,dx,dx2,xi,var;

	s1 = x[0];
	s2 = 0.0;
	s3 = 0.0;
	s4 = 0.0;
	*min = s1;
	*max = s1;
	for(i=1;i<n;++i){
		xi = x[i];
		an = i+1;
		an1 = i;
		dx = (xi-s1)/an;
		dx2 = dx*dx;
		s4 -= dx*(4.0*s3-dx*(6.0*s2+an1*(1.0+pow(an1,3.0))*dx2));
		s3 -= dx*(3.0*s2-an*an1*(an-2.0)*dx2);
		s2 += an*an1*dx2;
		s1 += dx;
		if(xi<*min)*min=xi;
		if(xi>*max)*max=xi;
	}
	*x1 = s1;
	var = n>1 ? s2/(n-1) : 0.0;
	*x2 = sqrt(var);
	*x3 = var>0.0 ? 1.0/(n-1)*s3/pow(var,1.5) : 0.0;
	*x4 = var>0.0 ? 1.0/(n-1)*s4/pow(var,2.0)-3.0 : 0.0;
	return;
}

void isort(char dir,int n,int * x)  /* This is adapted from R 0.16 */
{
	int i, j, h, asc;
	int xtmp;

	if(dir == 'a' || dir == 'A')asc = 1;
	else asc = 0;
	h = 1;
	do {
		h = 3 * h + 1;
	}
	while (h <= n);

	do {
		h = h / 3;
		for (i = h; i < n; i++) {
			xtmp = x[i];
			j = i;
			if(asc){
				while (x[j - h] > xtmp) {
					x[j] = x[j - h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}
			else{
				while (x[j - h] < xtmp) {
					x[j] = x[j - h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}

		end:	x[j] = xtmp;
		}
	} while (h != 1);
}

void fsort(char dir,int n,float * x)  /* This is adapted from R 0.16 */
{
	int i, j, h, asc;
	float xtmp;

	if(dir == 'a' || dir == 'A')asc = 1;
	else asc = 0;
	h = 1;
	do {
		h = 3 * h + 1;
	}
	while (h <= n);

	do {
		h = h / 3;
		for (i = h; i < n; i++) {
			xtmp = x[i];
			j = i;
			if(asc){
				while (x[j - h] > xtmp) {
					x[j] = x[j - h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}
			else{
				while (x[j - h] < xtmp) {
					x[j] = x[j - h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}

		end:	x[j] = xtmp;
		}
	} while (h != 1);
}

void dsort(char dir,int n,double * x)  /* This is adapted from R 0.16 */
{
	int i, j, h, asc;
	double xtmp;

	if(dir == 'a' || dir == 'A')asc = 1;
	else asc = 0;
	h = 1;
	do {
		h = 3 * h + 1;
	}
	while (h <= n);

	do {
		h = h / 3;
		for (i = h; i < n; i++) {
			xtmp = x[i];
			j = i;
			if(asc){
				while (x[j - h] > xtmp) {
					x[j] = x[j - h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}
			else{
				while (x[j - h] < xtmp) {
					x[j] = x[j - h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}

		end:	x[j] = xtmp;
		}
	} while (h != 1);
}

void dsorti(char dir,int n,double * x,int *indx)  /* This is adapted from R 0.16 */
{
	int i, j, h, asc,indtmp;
	double xtmp,*priv;

	priv = (double *)malloc(n*sizeof(double));
	for(j=0;j<n;++j)priv[j] = x[j];

	if(dir == 'a' || dir == 'A')asc = 1;
	else asc = 0;

	h = 1;
	for(j=0;j<n;++j)indx[j] = j;
	do {
		h = 3 * h + 1;
	}
	while (h <= n);

	do {
		h = h / 3;
		for (i = h; i < n; i++) {
			xtmp = priv[i];
			indtmp = indx[i];
			j = i;
			if(asc){
				while (priv[j - h] > xtmp) {
					priv[j] = priv[j - h];
					indx[j] = indx[j-h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}
			else{
				while (priv[j - h] < xtmp) {
					priv[j] = priv[j - h];
					indx[j] = indx[j-h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}

		end:	priv[j] = xtmp;indx[j] = indtmp;;
		}
	} while (h != 1);
	free(priv);
}


void isorti(char dir, int n, int * x, int *indx)
{
	int i, j, h, asc,indtmp;
	int xtmp,*priv;

	priv = (int *)malloc(n*sizeof(int));
	for(j=0;j<n;++j)priv[j] = x[j];
	if(dir == 'a' || dir == 'A')asc = 1;
	else asc = 0;
	for(j=0;j<n;++j)indx[j] = j;
	h = 1;
	do {
		h = 3 * h + 1;
	}
	while (h <= n);

	do {
		h = h / 3;
		for (i = h; i < n; i++) {
			xtmp = priv[i];
			indtmp = indx[i];
			j = i;
			if(asc){
				while (priv[j - h] > xtmp) {
					priv[j] = priv[j - h];
					indx[j] = indx[j-h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}
			else{
				while (priv[j - h] < xtmp) {
					priv[j] = priv[j - h];
					indx[j] = indx[j-h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}

		end:	priv[j] = xtmp;indx[j] = indtmp;
		}
	} while (h != 1);
	free(priv);
}

void isorti2(char dir, int n, int *x, int *priv, int *indx)
{
	int i, j, h, asc,indtmp;
	int xtmp;

	for(j=0;j<n;++j)priv[j] = x[j];
	if(dir == 'a' || dir == 'A')asc = 1;
	else asc = 0;
	for(j=0;j<n;++j)indx[j] = j;
	h = 1;
	do {
		h = 3 * h + 1;
	}
	while (h <= n);

	do {
		h = h / 3;
		for (i = h; i < n; i++) {
			xtmp = priv[i];
			indtmp = indx[i];
			j = i;
			if(asc){
				while (priv[j - h] > xtmp) {
					priv[j] = priv[j - h];
					indx[j] = indx[j-h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}
			else{
				while (priv[j - h] < xtmp) {
					priv[j] = priv[j - h];
					indx[j] = indx[j-h];
					j = j - h;
					if (j < h)
						goto end;
				}
			}

		end:	priv[j] = xtmp;indx[j] = indtmp;
		}
	} while (h != 1);
}

