/* Revised version 13/8/97 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <memory.h>
#include "myutil_new.h"

int **cmp_seq,n__seq,n0__seq=0,n__sites, max_str=500,*mult,no_nodes,sample_size,j,i,r,jj,d,*re_nds;
double theta,*sample,*like,c_time,beta;
char *ancestor_seq[2],**str,r_opt;
long no_runs;
FILE *out;

double update_rval(double x);
double update_theta(double x);
double update_tf(double x);
int priorcalc(double theta,double tf,double rval,double thetabounds[2],double tfbounds[2],double rvalbounds[2]);
int mysample(double *vec,int len,double stot);
void checklik(double lik,double *vec,int len);


struct node {
		int nd;
		char type;
		struct node *next;
				} ;

struct node2 {
		char *val;
		int m;
		int no_nb;
		int m_nb;
		struct node2 **nb;
};

struct elnd {
	int nd;
	char type;
} ;

typedef struct node *F;

struct node **l,*kk;
struct node2 **l2;

/* Order columns of sequences, thinking of them as
binary numbers.
*/

int b_cmp(const void *a, const void *b) {
	int i,flag=1,*pj,*pk;
	pj=(int*)a;pk=(int*)b;
	for(i=0;i<n0__seq;i++) {
		flag &= cmp_seq[i][*pj]==cmp_seq[i][*pk];
		if(flag==0) break;
	}
	if(flag==0&&cmp_seq[i][*pj]==1) return 1;
	else if(flag==0&&cmp_seq[i][*pj]==0) return -1;
	else return 1 - 2*(*pk < *pj);
}


int *site_sort(int *s) {
	int j;
	for(j=0;j<n__sites;j++) s[j]=j;
	qsort((void*)s,n__sites,sizeof(int),b_cmp);
return s;
}

struct node *add(int x, char tp,struct node* t1) {			//adds a node at the beggining of a list
		struct node *t=(struct node*)malloc(sizeof(struct node));
		t->nd=x;
		t->type=tp;
		t->next=t1;
		return t;
	}

void insert(struct node* p1,struct node* p2, struct node* q) {
		p1->next=q;
		q->next=p2;}

int count_list(struct node* head) {						//returns the no. of elements in a list
		if(head==NULL) return 0;
		else return (1+count_list(head->next));
	}

void make_tree(int **path, int *s) {

	int i,j,*count,*an,intern,ss,m,*seqidentity,added,root;
	struct node *a,*c,*cc,*k,*se,*temp;


	an=(int*)malloc((n0__seq+n__sites+1)*sizeof(int));
	count=(int*)malloc(n0__seq*sizeof(int));
	seqidentity=(int*)malloc(n0__seq*sizeof(int));

	for(j=0;j<n0__seq;j++) count[j]=0;
	for(j=0;j<n__sites;j++) for(i=0;i<n0__seq;i++)
		if(cmp_seq[i][s[j]]==1) path[i][count[i]++]=s[j]+1;				//constructs rooted tree, containing the paths of mutations from
																		//sequences to root
	for(i=0;i<n0__seq;i++) path[i][count[i]]=0;

	for (i=0;i<n0__seq;i++)
		seqidentity[i]=0;
	for (i=0;i<n0__seq;i++) {
		j=0;
		while (path[i][j]!=0)
		{ seqidentity[i]++;
		j++;}
	}
	an[0]=0;root=0;
	for(i=0;i<n0__seq;i++)
		for (j=seqidentity[i];j>=0;j--)
			if (path[i][j]!=0) {
				an[path[i][j]]=path[i][j+1];
				if (j==0) an[n__sites+i+1]=path[i][j];
			}
			else {
					if (j==0) { an[n__sites+i+1]=0;
								root=i+1;
								}
						}

	l=(struct node**)malloc((3*n__sites+1)*sizeof(struct node *));
	for (i=0;i<=3*n__sites;i++) {										// starts conversion to unrooted tree
		l[i]=(struct node*)malloc(sizeof(struct node));					// l is an array of lists; on positions 1..n__sites there are mutations,
		l[i]=NULL;														//	on n__sites+1 .. n__sites+n0__seq there are sequences,
	}																	//on  n__sites+n0__seq+1 .. n__sites+n0__seq+intern there are internal nodes
	for(i=0;i<n0__seq;i++)												// each element of l is the head of a list containing its neighbours on the unrooted tree
	for (j=seqidentity[i];j>=0;j--)
			if (j!=0) {
				c=l[path[i][j]];
				added=0;
				while(c!=NULL) {
					if(c->nd==path[i][j-1]&&c->type=='m') added=1;
					c=c->next;
				}
				if (added==0) l[path[i][j]]=add(path[i][j-1],'m',l[path[i][j]]);
			}
			else l[path[i][j]]=add(i+1,'s',l[path[i][j]]);

	l[0]=add(0,'r',l[0]);


	for (i=1;i<=n__sites;i++) {
		l[i]=add(an[i],'m',l[i]);
		l[i]=add(i,'m',l[i]);
	}

	for (i=1;i<=n0__seq;i++) {
		l[n__sites+i]=add(an[n__sites+i],'m',l[n__sites+i]);
		l[n__sites+i]=add(i,'s',l[n__sites+i]);
	}

	for (i=1;i<=n__sites;i++) {
		c=l[i]->next->next;
		m=0; ss=0; se=NULL;

		while (c!=NULL) {
			if (c->type=='m') m=1;
			if (c->type=='s') {ss=1; se=c;}
			if (m==1&&ss==1) {
				cc=l[i]->next->next;

				while (cc!=NULL) {
					if (cc->type=='m') {
					kk=(struct node*)malloc(sizeof(struct node));
					kk->nd=cc->nd;
					kk->type=cc->type;
					insert(l[n__sites+se->nd]->next,l[n__sites+se->nd]->next->next,kk);
					l[cc->nd]->next->nd=se->nd;
					l[cc->nd]->next->type='s';
				/*	k=l[cc->nd]->next->next;
					l[cc->nd]->next->next=k->next;
					kk=NULL;*/
					}
					cc=cc->next;
					}
				cc=l[i]->next;

				while (cc->next!=NULL) {
					if (cc->next->type=='m') {
					/*	k=cc->next;
						cc->next=k->next;*/
						temp=cc->next;
						cc->next=cc->next->next;
						free(temp);
					}
					else cc=cc->next;
				}
				break;
			}
			else c=c->next;
		}
	}

	intern=0;
	for (i=1;i<=n__sites;i++)
		if (count_list(l[i])>3) {
			intern++;
			l[n__sites+n0__seq+intern]=add(i,'m',l[n__sites+n0__seq+intern]);
			l[n__sites+n0__seq+intern]=add(intern,'i',l[n__sites+n0__seq+intern]);
		/*	insert(l[i]->next,l[i]->next->next,l[n__sites+n0__seq+intern]); */

			c=l[i]->next->next;
			while (c!=NULL)
			{/*insert (l[n__sites+n0__seq+intern]->next,l[n__sites+n0__seq+intern]->next->next,c);*/
			a=(struct node*)malloc(sizeof(struct node));
			l[c->nd]->next->nd=l[n__sites+n0__seq+intern]->nd;
			l[c->nd]->next->type='i';
			a->nd=c->nd;
			a->type=c->type;
			insert(l[n__sites+n0__seq+intern]->next,l[n__sites+n0__seq+intern]->next->next,a);
		/*	kk=add(c->nd,c->type,kk); */
		/*	k->nd=l[n__sites+n0__seq+intern]->nd;
			k->type=l[n__sites+n0__seq+intern]->type;
			k->next=l[c->nd]->next;
			insert(l[c->nd],l[c->nd]->next,k);
			k=l[c->nd]->next->next;
			l[c->nd]->next->next=k->next;
			a=NULL;*/
			c=c->next;
			}
		/*	l[n__sites+n0__seq+intern]->next->next=kk;*/
			l[i]->next->next->nd=l[n__sites+n0__seq+intern]->nd;
			l[i]->next->next->type='i';
			l[i]->next->next->next=NULL;
		/*	c=l[i]->next->next;
			while (c->next!=NULL)
			{k=c->next;
			c->next=k->next;
			c=c->next;*/

		}


	if (root!=0) {
		free(l[n__sites+root]->next);
		l[n__sites+root]->next=l[0]->next;
		c=l[n__sites+root];
		while(c->next!=NULL) {
			if (c->next->nd==l[n__sites+root]->nd&&c->next->type=='s') {
				k=c->next;
				c->next=k->next;
				free(k);
				break;
			}
			else c=c->next;
		}
	}
	else {
		intern++;
		l[n__sites+n0__seq+intern]=add(intern,'i',l[n__sites+n0__seq+intern]);
		l[n__sites+n0__seq+intern]->next=l[0]->next;
	}
	for (i=1;i<=3*n__sites;i++)
		if (l[i]!=NULL) {
			if (l[i]->next->nd==0) {
				if (root==0) {
					l[i]->next->nd=intern;
					l[i]->next->type='i';
				}
				else {
					l[i]->next->nd=root;
					l[i]->next->type='s';
				}
			}
		}
	no_nodes=n0__seq+intern;
	free(an);
	free(count);
	free(seqidentity);
}
	/*
void make_tree(int **path, int *s) {
	int i,j,*count;
	count=(int*)malloc(n0__seq*sizeof(int));
	for(j=0;j<n0__seq;j++) count[j]=0;
	for(j=0;j<n__sites;j++) for(i=0;i<n0__seq;i++)
		if(cmp_seq[i][s[j]]==1) path[i][count[i]++]=s[j]+1;
	for(i=0;i<n0__seq;i++) path[i][count[i]]=0;
	free(count);
} */



/* return -1 not compatable in broard sense, 0 if not in narrow, 1 if ok */
int check_seq(int **q) {
	int u[2][2],i,j,k,l,x,ok=1;
	for(i=0;i<n__sites;i++) for(j=0;j<i;j++) {
		for(k=0;k<2;k++) for(l=0;l<2;l++) u[k][l]=0;
		for(x=0;x<n__seq;x++) {
			for(k=0;k<2;k++) for(l=0;l<2;l++)
				u[k][l] += (q[x][i]==k && q[x][j]==l);
		}
		if(u[0][1] != 0&&u[1][0] != 0&&u[1][1] != 0) ok=0;
		if(u[0][1] != 0&&u[1][0] != 0&&u[1][1] != 0&&u[0][0] != 0) return (-1);
	}
	if(ok==0) return 0;
	return 1;
}

void compat_matrix(int **seq, int ** compat_matrix) {
	int u[2][2],i,j,k,l,x;
	for(i=0;i<n__sites;i++) for(j=0;j<i;j++) {
		compat_matrix[i][j]=0;
		for(k=0;k<2;k++) for(l=0;l<2;l++) u[k][l]=0;
		for(x=0;x<n__seq;x++) for(k=0;k<2;k++) for(l=0;l<2;l++)
			u[k][l] += seq[x][i]==k && seq[x][j]==l;
		if(u[0][1] != 0&&u[1][0] != 0&&u[1][1] != 0) compat_matrix[i][j]=1;
		if(u[0][1] != 0&&u[1][0] != 0&&u[1][1] != 0&&u[0][0] != 0)
			compat_matrix[j][i]=1;
	}
}

void tree_out(FILE *out, int **path, int row,int col, char *name,int *seq_identity) {
	FILE *in;
	char buffer[100],*p,c,s;
	int i,j,k,count=0,check[3],asc;
	struct node *q;
	if((in=fopen(name,"r"))==NULL) {
		printf("\n\tCannot open sequence file\n");
		exit(1);
	}
	for(i=0;i<row;i++) {
		c=' ';
		while(c!=':') {
			fscanf(in,"%c",&c);
			if(!(c == '\n' || c == '\r')) fprintf(out,"%c",c);
		}
		for(j=0;path[seq_identity[i]][j];j++)
fprintf(out,"%3d",path[seq_identity[i]][j]);
		fprintf(out,"%3d\n",0);
					for(j=0;j<col;j++) {
			c=' ';
			while(isspace(c)!=0) fscanf(in,"%c",&c);
		}
	}

	fprintf(out,"\n");
	i=1;
	for (i=1;i<=no_nodes+n__sites;i++) {
		if (i<=n__sites) fprintf(out,"%d ",l[i]->nd);
		if((i>n__sites)&&(i<=n__sites+n0__seq)) {
			asc=96+l[i]->nd;
			s=asc;
			fprintf(out,"%c ",s);
		}
		if (i>n0__seq+n__sites) fprintf(out,"i%d ",l[i]->nd);
		q=l[i]->next;
		while (q!=NULL) {
			if (q->type=='m') fprintf(out,"%d ",q->nd);
			if (q->type=='s') {
				asc=96+q->nd;
				s=asc;
				fprintf(out,"%c ",s);
			}
			if (q->type=='i') fprintf(out,"i%d ",q->nd);
			q=q->next;
		}
		fprintf(out,"\n");
	}

	if(out!=stdout) fclose(out);
}

void get_size(int *prow,int *pcol,char *name) {
	FILE *in;
	char buffer[100],c='a';
	int r,x,count=0,bug=0;
	if((in=fopen(name,"r"))==NULL) {
		printf("\n\tCannot open sequence file\n");
		exit(1);
	}
	*prow=0;*pcol=0;
	while(fgets(buffer,100,in) != NULL)
		if(strchr(buffer,':')!=NULL) (*prow)++;
	rewind(in);
	buffer[0]='\000';
	while(buffer[0]!=':' && count <max_str)  {
		fscanf(in,"%s",buffer);
		count++;
	}
	while(!(c=='\n' || c == '\r') && bug <10000) {
		r=fscanf(in,"%c",&c);
		if(r==1&&isspace(c)==0) (*pcol)++;
		if(feof(in) != 0) break;
		bug++;
	}
	fclose(in);
	if(*prow <=0||*pcol<=0) {
		printf("\n\tProblem with data input file\n");
		exit(1);
	}
}

/*
void get_seq(int **q,int row,int col,char *name) {
	FILE *in;
	char buffer[100],c;
	int i,j,k,count=0,check[3],quit=0;
	if((in=fopen(name,"r"))==NULL) {
		printf("\n\tCannot open sequence file\n");
		exit(1);
	}
	for(i=0;i<row;i++) {
		buffer[0]='\000';
		count=0;
		while(buffer[0]!=':'&&count < max_str) {
			fscanf(in,"%s",buffer);
			count++;
		}
		for(j=0;j<col;j++) {
			fscanf(in,"%d",&q[i][j]);
		}
	}
	for(j=0;j<col;j++) {
		check[0]=0;check[1]=0;
		for(i=0;i<row;i++) {
			check[0] += q[i][j]==0;
			check[1] += q[i][j]==1;
		}
		if(row>1 &&(check[0]==row||check[1]==row)) {
			printf("\n\tSite %d must be a segregating site",j+1);
			quit=1;
		}
		if((check[0]+check[1]) != row) {
			printf("\n\tSite %d must contain 0 or 1 data entries",j+1);
			quit=1;
		}
	}
	if(quit==1) {
		printf("\n\n");exit(1);
	}
	fclose(in);
}

*/

void get_seq(int **q,int row,int col,char *name, char *aname) {
	FILE *in;
	FILE* ain;
	char **d,buffer[100],c,anc0,anc1;
	int i,j,k,count=0,check[3],quit=0,maj,type1,type2,ic;
	ancestor_seq[0]=(char*)malloc(col*sizeof(char));
	ancestor_seq[1]=(char*)malloc(col*sizeof(char));
	d=(char**)malloc(row*sizeof(char*));
	for(i=0;i<row;i++)
		d[i]=(char*)malloc(col*sizeof(char)); // 10.12.12 MAB edit here
	if(aname==NULL) {														//reads data from file(s)
				for(j=0;j<col;j++) {
				ancestor_seq[0][j]='0';
				ancestor_seq[1][j]='$';
			}
				if((in=fopen(name,"r"))==NULL) {
		printf("\n\tCannot open sequence file\n");
		exit(1);
	}

		buffer[0]='\000';
		count=0;
		while(buffer[0]!=':'&&count < max_str) {
			fscanf(in,"%s",buffer);
			count++;
		}

			c=' ';
			while(isspace(c)!=0) fscanf(in,"%c",&c);
			if (isalpha(c)||ispunct(c)) maj=1;
			if (isdigit(c)) maj=0;}
	else {
		if((ain=fopen(aname,"r"))==NULL) {
			printf("\n\tCannot open ancestor file\n");
			exit(1);
		}
		for(j=0;j<col;j++) {
			ancestor_seq[1][j]='$';
			c=' ';
			while(isspace(c)!=0) fscanf(ain,"%c",&c);
			ancestor_seq[0][j]=c;
		}
	}
	if((in=fopen(name,"r"))==NULL) {
		printf("\n\tCannot open sequence file\n");
		exit(1);
	}
	rewind(in);
	for(i=0;i<row;i++) {
		buffer[0]='\000';
		count=0;
		while(buffer[0]!=':'&&count < max_str) {
			ic = fscanf(in,"%s",buffer);
			if(ic == EOF)printerr(" error 1");
			count++;
			if (count==2) mult[i]=atoi(buffer);
		}
		for(j=0;j<col;j++) {
			c=' ';
			while(isspace(c)!=0){ic = fscanf(in,"%c",&c);if(ic == EOF)printerr("error 2");}
			d[i][j]=c;														// q is a matrix with 0-1 coded strings
			if(c==ancestor_seq[0][j]) q[i][j]=0;
			else if(ancestor_seq[1][j]=='$') {
				ancestor_seq[1][j]=c; q[i][j]=1;
			}
			else if(c==ancestor_seq[1][j]) q[i][j]=1;

		}

	}
	if (maj==1) { for(j=0;j<col;j++){										//uses the majority rule to make q, when the ancestral state is unknown
				anc0=d[0][j];type1=0;type2=0;
					for (i=0;i<row;i++){
						if (d[i][j]==anc0) type1++;
						else {type2++;
								anc1=d[i][j];}
					}
				if (type1>=type2) ancestor_seq[0][j]=anc0;
				if (type1<type2) ancestor_seq[0][j]=anc1;
			}
				for (j=0;j<col;j++)
					for (i=0;i<row;i++)
						if (d[i][j]==ancestor_seq[0][j]) q[i][j]=0;
						else q[i][j]=1;
	}
/*	while (!feof(in))
	fscanf(in,"%lf",&theta);*/
	/* End read of file */
	for(j=0;j<col;j++) {
		check[0]=0;check[1]=0;
		for(i=0;i<row;i++) {
			check[0] += q[i][j]==0;
			check[1] += q[i][j]==1;
		}
		if(row>1 &&(check[0]==row||check[1]==row)) {
			printf("\n\tSite %d must be a segregating site",j+1);
			quit=1;
	}
		}
	if(quit==1) {
		printf("\n\n");exit(1);
	}
	fclose(in);

}

void recode(int **seq) {
	int i,j,u[2];
	for(j=0;j<n__sites;j++) {
		u[0]=0;u[1]=0;
		for(i=0;i<n__seq;i++) u[seq[i][j]]++;
		if(u[1]>u[0])
			for(i=0;i<n__seq;i++) seq[i][j]=1-seq[i][j];
	}
}

void do_tree(char *filename, char *outfile, char *aname) {
	int **q,*s,i,j,**path,**compat,ok,*seq_identity;
	char c;
	FILE *out;
	get_size(&n__seq,&n__sites,filename);
	q=(int**)malloc(n__seq*sizeof(int*));
	for(i=0;i<n__seq;i++)
		q[i]=(int*)malloc(n__sites*sizeof(int));
	get_seq(q,n__seq,n__sites,filename,aname);
	ok=check_seq(q);
	if(ok==(-1)) {
		printf("\n\tSequences are incompatable in broard sense");
		compat=(int**)malloc(n__sites*sizeof(int*));
		for(i=0;i<n__sites;i++)
			compat[i]=(int*)malloc(n__sites*sizeof(int));
		compat_matrix(q,compat);
		printf("\n\tCompatability matrix (& broard, * narrow, . ok)\n");
		printf("\t  ");
		for(j=0;j<n__sites;j++) printf("%d",(j+1)%10);
			for(i=0;i<n__sites;i++) {
				printf("\n\t%d ",(i+1)%10);
				for(j=0;j<n__sites;j++) {
					if(compat[i][j]==1&&i<j) printf("&");
					else if(compat[i][j]==1&&i>j) printf("*");
					else printf(".");
				}
			}
			printf("\n");
			exit(1);
			}
			if(ok==0) {
				printf("\n\tSequences are incompatable in narrow sense,");
				printf("\n\tusing ancestral site (0) most common matrix.\n");
				recode(q);
			}
			s=(int*)malloc(n__sites*sizeof(int));
			path=(int**)malloc(n__seq*sizeof(int*));
			seq_identity=(int*)malloc(n__seq*sizeof(int));
			cmp_seq=(int**)malloc(n__seq*sizeof(int*));
			for(i=0;i<n__seq;i++) {
				path[i]=(int*)malloc((n__sites+1)*sizeof(int));
				seq_identity[i]=(-1);
				for(j=0;j<=n__sites;j++) path[i][j]=0;
			}
			/* Find duplicate sequences */
			for(i=0;i<n__seq;i++) {
				if(seq_identity[i]==(-1)) {
					cmp_seq[n0__seq]=q[i];
					seq_identity[i]=n0__seq;
					for(j=i+1;j<n__seq;j++)
						if(seq_identity[j]==(-1)&&
							memcmp(q[i],q[j],n__sites*sizeof(int))==0)
								seq_identity[j]=n0__seq;
				n0__seq++;
				}
			}
			site_sort(s);
			/* Binary sort order of columns of seqs, debug only
			printf("\t");
			for(i=0;i<n__sites;i++) printf("%d ",s[i]+1);
			printf("\n");
			*/
			make_tree(path,s);
			if(strcmp(outfile,"stdout")==0)
				tree_out(stdout,path,n__seq,n__sites,filename,seq_identity);
			else {
				out=fopen(outfile,"w");
				if(out==NULL) {
					printf("\n\tCan't open tree output file %s\n",outfile);
					exit(1);
				}
			tree_out(out,path,n__seq,n__sites,filename,seq_identity);
			}
		str=(char**)calloc(no_nodes,sizeof(char*));
		for(i=0;i<no_nodes;i++)
		str[i]=(char*)calloc(n__sites+1,sizeof(char));
		for(i=0;i<n__seq;i++) {
	/*	str[i][0]=q[i][0];*/
	/*	strcpy(str[i],q[i]);*/
		for (j=0;j<n__sites;j++)
	/*	strcat(str[i],q[i][j]);*/
		if (q[i][j]==0) str[i][j]='0';
		else str[i][j]='1';
		}
			free(cmp_seq);
}

void freelist() {
	int i;
	struct node *h,*temp;
	free(l[0]);
	for (i=1;i<=3*n__sites;i++) {
	h=l[i];
	while(h!=NULL) {
		temp=h;
		h=h->next;
		free(temp);

	}
}

}

double newtime(double cur_time,int nlin,int use_beta, double beta, double *rval,double *tf, double theta)
/* beta = log(r)/tf, and the models are the same as long as tf is long enough back into the past to make little difference*/
{
	double crate,cutoff,deltat,tt,mu_deltat,mut_newtime,coal_newtime,logr;

	if(use_beta){
		if(fabs(beta) < 1.0e-5){*rval = 1;*tf=10;} // no growth case
		else{// I hope these conditions are OK?? choose a big value for rval and then modify tf accordingly
		     // may need to modify this.
			if(beta < 0)logr = -200.0;
			else logr = 200.0;
			*tf = logr/beta;
			*rval = exp(logr);
		}
	}

	crate = (nlin*(nlin-1)*0.5);// coalescence rate
	deltat = expdev()/crate;// the waiting time for the standard coalescent
	mu_deltat = expdev()/(nlin*theta*0.5); // time for next mutation.
	mut_newtime = mu_deltat +  cur_time;

	/* this is the value of deltat that would correspond to tf when transformed according to formula below.
	   I.e. if you substitute cutoff for deltat in first expression for new_time, you end up with tf.*/
	cutoff = (*rval - pow(*rval,cur_time/(*tf)))*(*tf)/log(*rval);

	/* this is the limiting value for the 3 sets of expressions below
	when r->1 */
	if(fabs(1.0-*rval) < 1.0e-5){ /* minimal change in pop size, so just use standard coalescent */
		coal_newtime = cur_time+deltat;
	}
	else if(cur_time < (*tf)){
		if(deltat < cutoff){ // everything within period of growth
			coal_newtime = log(deltat*log(*rval)/(*tf)+pow(*rval,cur_time/(*tf)))*(*tf)/log(*rval);

		}
		else {
			coal_newtime = (deltat - cutoff)/(*rval) + *tf;// take into account discontinuity at cutoff
		}
	}
	else {
		coal_newtime = deltat/(*rval) + cur_time; // everything after period of growth (looking backwards in time)
	}
	if(coal_newtime < mut_newtime)return coal_newtime;
	else return mut_newtime;

}


void init_list() {
	struct node *q,*c,*cc,*prev;
	int i,j,f,jj,*mut,k,fi;
/*	l2=(struct node2**)calloc(no_nodes,sizeof(struct node2*));*/
	for (i=1;i<=no_nodes;i++) {
	/*	l2[i]=(struct node2*)calloc(1,sizeof(struct node2));
		l2[i]->val=(char*)calloc(n__sites+1,sizeof(char));
		l2[i]->nb=(struct node2**)calloc(3*n__sites,sizeof(struct node2*));
	*/	for(j=1;j<=n__sites;j++)
		l2[i]->val[j]=str[i-1][j-1];
		if (i<=n0__seq) l2[i]->val[n__sites+1]='\0';							//constructs a second list that has only sequences and internal nodes as elements
		l2[i]->val[0]='.';														//each node has multiple links to the nodes it communicates with
		l2[i]->m=mult[i-1];														//ignores mutations, just stores the no. of mutations between 2 neighbouring nodes
		l2[i]->m_nb=count_list(l[n__sites+i])-1;
		if(i>n0__seq) {															//in l2, the first n0 elements are observed sequences and the next are
			l2[i]->val[1]=' ';													//internal nodes, so knowing the position of a node in l2(i) means
			l2[i]->m=0;															//knowing whether it's type 's' or 'i'	in l
		}
	}
	for (i=1;i<=no_nodes;i++) {
		c=l[n__sites+i]->next;
		j=0;
		while(c!=NULL) {
			f=0;
			prev=NULL;
			cc=l[c->nd];
			while(f==0) {
			q=cc;
			cc=cc->next;
			do {
				if (cc!=NULL) {
				if (cc->type=='m'||(cc->nd==l[n__sites+i]->nd&&cc->type==l[n__sites+i]->type)) cc=cc->next;
				else {
					j++;
					if (cc->type=='i') l2[i]->nb[j]=l2[n0__seq+cc->nd];
					if (cc->type=='s') l2[i]->nb[j]=l2[cc->nd];
					f=1;
					break;
				}
				}
			}
			while (cc!=NULL);
			if (f==0) {
			/*	cc=l[c->nd]->next;
				prev=l[c->nd];*/
				cc=q->next;
				if (cc->type!='m') cc=cc->next;
				if (prev!=NULL&&cc->nd==prev->nd&&cc->type==prev->type) cc=cc->next;
				cc=l[cc->nd];
				prev=q;
			}
			else {
				c=c->next;
				break;
			}
			}
		}
		l2[i]->no_nb=j;
	/*	l2[i]->nb=(struct node2**)realloc(l2[i]->nb,l2[i]->no_nb*sizeof(struct node2*));*/
	}
	for (i=n0__seq+1;i<=no_nodes;i++) {
		mut=(int*)malloc(n__sites*sizeof(int));
		cc=l[l[n__sites+i]->next->nd];
		j=0;
		f=0;
		fi=0;
		prev=l[n__sites+i];
		while(f==0) {
			q=cc;
			j++;
			mut[j]=q->nd;
			cc=cc->next;
			do {
				if (cc!=NULL) {
				if (cc->type=='m'||(cc->nd==prev->nd&&cc->type==prev->type)) cc=cc->next;
				else {
					if (cc->type=='i') k=n0__seq+cc->nd;
					else k=cc->nd;
					if (l2[k]->val[1]!=' ') {
					for (jj=1;jj<=n__sites;jj++) {
						if (cc->type=='s') l2[i]->val[jj]=l2[cc->nd]->val[jj];
						if (cc->type=='i') l2[i]->val[jj]=l2[n0__seq+cc->nd]->val[jj];
					}
					for(jj=1;jj<=j;jj++) {
						if (cc->type=='s') {
							if (l2[cc->nd]->val[mut[jj]]=='0') l2[i]->val[mut[jj]]='1';
							else l2[i]->val[mut[jj]]='0';
						}
						if (cc->type=='i') {
						/*	l2[i]->val[mut[jj]]=1-atoi(l2[n0__seq+cc->nd]->val[jj]);*/
							if (l2[n0__seq+cc->nd]->val[mut[jj]]=='0') l2[i]->val[mut[jj]]='1';
							else l2[i]->val[mut[jj]]='0';
						}
					}
					f=1;
					free(mut);
					break;
					}
					else {
						cc=l[cc->nd+n__sites+n0__seq];
						prev=q;
						fi=1;
						break;
					}

				}
				}
			}
			while (cc!=NULL);
			if (f==0&&fi==0) {
				cc=q->next;
				if (cc->type!='m') cc=cc->next;
				if (prev!=NULL&&cc->nd==prev->nd&&cc->type==prev->type) cc=cc->next;
				cc=l[cc->nd];
				prev=q;
			}
			else {
				if (f==0&&fi==1) {
					if (prev!=NULL&&cc->next->nd==prev->nd&&cc->next->type==prev->type) cc=cc->next->next;
					else cc=cc->next;
					prev=l[k+n__sites];
					cc=l[cc->nd];
					fi=0;
				}
				if (f==1&&fi==0) break;
			}
			}
	}
	//MAB 5/01/2016 this used to be commented out (maybe I did this at some stage after Sorina wrote it...)
	// But it looks as though we then reallocate new space to str later, so maybe need to do this
	for(i=0;i<no_nodes;i++)
		free(str[i]);
	free(str);
}

struct elnd *eligible_nodes(struct node2 **l3,struct elnd *en) {

	int i,j,jj;
	j=0;
	for(i=1;i<=no_nodes;i++) {									//finds the nodes that might have been involved in the most recent event back in time
		if (l3[i]->m>1) {
			j++;												//as l, en stores the label and the type of the node
			if (i<=n0__seq) {
				en[j].nd=i;
				en[j].type='s';
			}
			else {
				en[j].nd=i-n0__seq;
				en[j].type='i';
			}
			for(jj=1;jj<=l3[i]->m-1;jj++) {
				en[j+jj].nd=en[j].nd;
				en[j+jj].type=en[j].type;
			}
			j=j+l3[i]->m-1;
		}
		if (l3[i]->m_nb==1&&l3[i]->m==1&&i!=r) {			//the condition (i!=r) is for the rooted version, when a mutation must not occur after the root
			j++;											//is reached. when the unrooted version is run, r=0, which will be always different from i
			if (i<=n0__seq) {								// (since i starts from 1), so this condition will have no influence
				en[j].nd=i;
				en[j].type='s';
			}
			else {
				en[j].nd=i-n0__seq;
				en[j].type='i';
			}
		}
	}
	en[0].nd=j;									//en[0].nd is the number of nodes eligible for change
	return en;
}

int mut_btw(struct node2 *p,struct node2 *q) {
	int n,j;
	n=0;												//finds the number of mutations between 2 nodes
	for (j=1;j<=n__sites;j++)
		if (p->val[j]!=q->val[j]) n++;
	return n;
}

double imp_weights(double theta,double tf,double rval,int c,struct elnd *en,int use_beta,double beta) {
	int n,n0,i,j,k,lbl;
	double v,lambda;
	n0=en[0].nd;										//computes the importance weight for one step back in time
	n=0;
	for (j=1;j<=no_nodes;j++)
		n=n+l2[j]->m;
	c_time=newtime(c_time,n,use_beta,beta,&rval,&tf,theta);
	if (c_time<tf) v=pow(rval,-c_time/tf);
	else v=1/rval;
	lambda=1/v;
	if (en[c].type=='s') i=en[c].nd;
	else i=en[c].nd+n0__seq;
	if (l2[i]->m>1) {
		(l2[i]->m)--;																		//for coalescence
		return n0*(l2[i]->m)*lambda/((l2[i]->m+1)*((n-1)*lambda+theta));
	}
	else {
		if (mut_btw(l2[i],l2[i]->nb[1])==1) {
			(l2[i]->m)--;
			(l2[i]->nb[1]->m)++;
			(l2[i]->nb[1]->m_nb)--;
			if (l2[i]->nb[1]->m==1) return n0*theta/(n*((n-1)*lambda+theta));		//for 1 mutation, evolution into a new state
			else return n0*(l2[i]->nb[1]->m)*theta/(n*((n-1)*lambda+theta));		//for 1 mutation, evolution into an already existing state
		}
		else {
			lbl=disrand(1,mut_btw(l2[i],l2[i]->nb[1]));				//when there are more mutations and you don'tknow their order, pick one randomly
			k=0;
			for (j=1;j<=n__sites;j++)
			if (l2[i]->val[j]!=l2[i]->nb[1]->val[j]) {
				k++;
				if (k==lbl) {
					if (l2[i]->val[j]=='1') l2[i]->val[j]='0';
					else l2[i]->val[j]='1';
					return n0*theta/(n*((n-1)*lambda+theta));
				}
			}
		}
	}
}

double prod(double theta,double tf,double rval,int use_beta, double beta) {
	struct elnd *en;
	int c_node;
	double imp;													//computes the importance weight for the whole history
	imp=1;
	en=(struct elnd*)calloc(sample_size+1,sizeof(struct elnd));
	en=eligible_nodes(l2,en);
	c_time=0.0;
	while(en[0].nd!=0) {
		c_node=disrand(1,en[0].nd);
		imp=imp*imp_weights(theta,tf,rval,c_node,en,use_beta,beta);
		free(en);
		en=(struct elnd*)calloc(sample_size+1,sizeof(struct elnd));
		en=eligible_nodes(l2,en);
	}
	free(en);
	return imp;
}

void reinit_list() {
	int i,j;											//reinitializes the second list for a new run of the IS algorithm
	for (i=1;i<=no_nodes;i++) {
		l2[i]->m=mult[i-1];								//resets the original multiplicities and states
		l2[i]->m_nb=count_list(l[n__sites+i])-1;
		if(i>n0__seq) {
			l2[i]->m=0;
		}
	}
	for (i=1;i<=d;i++)
		strcpy(l2[re_nds[i]]->val,str[i]);
}


double lkl_est(double theta,double tf, double rval,int use_beta,double beta) {
	double p;
	p=0;								//computes a likelihood estimate by averaging the importance weights obtained
	for (j=1;j<=no_runs;j++) {
		p += prod(theta,tf,rval,use_beta,beta);
		reinit_list();
	}
	p=p/no_runs;
	return p;
}

void freelist2() {
	int i;
	struct node2 *h,*temp;
	for (i=1;i<=no_nodes;i++) {
		h=l2[i];
		if (h!=NULL) {
		free(h->val);
	/*	for (j=1;j<=h->no_nb;j++)
				free(h->nb[j]);*/
		free(h->nb);
		free(h);
		}
	}
}

void mcmc (long n, double up_t, double sd_t, double init_t) {
	int n1,n2,n_acc,i;
	double c_t,n_t,n_like,c_like,eps_t;
	sample=(double*)calloc(n,sizeof(double));
	like=(double*)calloc(n,sizeof(double));
	n_acc=0;
	n1=0;
	n2=0;
	c_t=init_t;
	c_like=lkl_est(init_t,3.0,4.0,0,0.0);
	for (i=1;i<=n;i++) {
		eps_t=norm8()*sd_t;
		n_t=c_t+eps_t;
		if (n_t<0) n1++;
		else if (n_t>up_t) n2++;
			 else {
				 n_like=lkl_est(n_t,3.0,4.0,0,0.0);
				 if (gfsr8()<n_like/c_like) {
					 n_acc++;
					 c_t=n_t;
					 c_like=n_like;
				 }
			 }
		sample[i]=c_t;
		like[i]=c_like;
	}
}

double post_mean(int n,double *v) {
	int i;
	double s;
	FILE *out1,*out2;
	s=0.0;
	out1=fopen("thetas.dat","w");
	out2=fopen("likelihoods.dat","w");
	for (i=101;i<=n;i++) {
		s=s+v[i];
		fprintf(out1,"%e%c",v[i],' ');
		fprintf(out2,"%e%c",like[i],' ');
	}
	fclose(out1);
	fclose(out2);
	return s/n;
}

int find_root() {
	int r;												//finds the root of the tree for the rooted version of the algorithm
for (i=1;i<=no_nodes;i++) {
	r=1;
	for (jj=1;jj<=n__sites;jj++)
		if(l2[i]->val[jj]!='0') {
			r=0;
			break;
		}
	if (r==1) {
		return i;

	}
}
printf("%s","Can't find ancestral state,running unrooted version");
return 0;
}

void changed_nodes() {
	d=0;
	//MAB 5/01/2016 hunting down malloc bug; changed from no_nodes to 10*no_nodes to 
	// try to avoid use of realloc
	// this does seem to have fixed things.... 
	// other comments below...
	re_nds=(int*)malloc(10*no_nodes*sizeof(int));			//saves the state of nodes that are followed by more than 1 mutation on a single branch,
	for (i=1;i<=no_nodes;i++)							//because it will be modified when running the IS
		for (j=1;j<=l2[i]->no_nb;j++)					//then re_nds can be used to reinitialize l2, without having to walk again through the unrooted
			if (mut_btw(l2[i],l2[i]->nb[j])>1) {		//tree in order to determine the state of the internal nodes
				d++;
				if(d >= 10*no_nodes){
					printf("d is %d\n",d);
					printf("no_nodes is %d\n",no_nodes);
					printerr("changed_nodes: d >= 10*no_nodes");
				}
				re_nds[d]=i;
				break;
			}
// something odd about this line: a) it occurs outside the loop
// b) it assumes that d is never greater than number of nodes ? (is this right?)
// c) why does it matter to reallocate to make smaller? 
//	if (d<no_nodes) re_nds=realloc(re_nds,(d+1)*sizeof(int));

}


int main(int argc, char**argv) {

	double theta;
	double rval;
	double tf;
	double theta_new,rval_new,tf_new;
	double lambda;
	char *data,*paramfile;
	int iter,no_iter;
	double *thetavec,*tfvec,*rvalvec,*wt,dtemp;
	double *dpoint;
	int pval;
	double lik_cur,lik_new,lik_temp,wsum;
	int ii,inner;
	double thetabounds[2],rvalbounds[2],tfbounds[2];
	FILE *pm;
	int ms_scaling;

	data = argv[1];

	no_iter = atoi(argv[2]);
	no_runs = atoi(argv[3]);
	paramfile = argv[4];
	ms_scaling = atoi(argv[5]);
	
	pm = fopen(paramfile,"r");
	thetavec = (double *)malloc(no_iter*sizeof(double));
	tfvec = (double *)malloc(no_iter*sizeof(double));
	rvalvec = (double *)malloc(no_iter*sizeof(double));
	wt = (double *)malloc(no_iter*sizeof(double));
	for(j=0;j<no_iter;++j){
		fscanf(pm,"%lf %lf %lf",&thetavec[j],&rvalvec[j],&tfvec[j]);
		if(ms_scaling)tfvec[j] = log(2.0) + tfvec[j]; // input param values are in log scale
	}

opengfsr();
mult=(int*)malloc(1000*sizeof(int)); //10.12.12 MAB changed from n0_seq [which is zero originally] to 1000, which should cover most things
do_tree(data,"tree.txt",NULL);

l2=(struct node2**)calloc(no_nodes+1,sizeof(struct node2*));				// in l2[i], m represents the multiplicity of type i, val is a string with the
for (i=1;i<=no_nodes;i++) {												// 0-1 configuration of type i, nb is an array of pointers to the neighbouring
		l2[i]=(struct node2*)calloc(1,sizeof(struct node2));			// nodes. no_nb is the number of sequence neighbours(length of nb) and m_nb is
		l2[i]->val=(char*)calloc(n__sites+2,sizeof(char));				//the number of mutation neighbours (realized later that these 2 are the same
		l2[i]->nb=(struct node2**)calloc(3*n__sites,sizeof(struct node2*));		//actually, so one of them is redundant).
	}
init_list();

changed_nodes();
str=(char**)malloc((d+1)*sizeof(char*));
for(i=1;i<=d;i++) {
	str[i]=(char*)malloc((n__sites+2)*sizeof(char));
	strcpy(str[i],l2[re_nds[i]]->val);
}

for (j=1;j<=no_nodes;j++)
sample_size=sample_size+l2[j]->m;


r_opt = 'r';
//r_opt = 'u';
if (r_opt=='r') r=find_root();
else r=0;
beta = 1.0;


//editing from here

/* 0,beta refers to use_beta (yes, no) and then value of beta. 
So, here, we don't use beta */
for(iter =0;iter<no_iter;++iter){
	wsum = lkl_est(exp(thetavec[iter]),exp(tfvec[iter]),exp(rvalvec[iter]),0,beta);
	wt[iter] = wsum;
	printf("%.16e\n",wt[iter]);
}

closegfsr();
return 0;
}

void checklik(double lik,double *vec,int len)
{
	int j;
	double cum;
	cum = 0;
	for(j=0;j<len;++j){
		cum = cum +vec[j];
	}
	if(fabs((cum -lik*len)/(lik*len)) > 0.00001)printerr("checklik: problem");
}


int mysample(double *vec,int len,double stot)
{
	double val,cum;
	val = stot*gfsr8();
	cum = 0;
	for(j=0;j<len;++j){
		cum = cum + vec[j];
		if(cum >= val)return j;
	}
	printerr("mysample: run through loop");
	
}



double update_rval(double x)
{
	return x + norm8()*0.5;

}

double update_theta(double x)
{
	return x + norm8()*0.5;

}

double update_tf(double x)
{
	return x + norm8()*0.5;

}

int priorcalc(double theta,double tf,double rval,double thetabounds[2],double tfbounds[2],double rvalbounds[2])
{
	if(theta < thetabounds[0] || theta > thetabounds[1])return 0;
	if(tf < tfbounds[0] || tf > tfbounds[1])return 0;
	if(rval < rvalbounds[0] || rval > rvalbounds[1]) return 0;
	return 1;

}


#ifdef SEQTOTR
char *example1=
"  0 6 : 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
  0 1 : 0 0 0 0 1 0 0 0 0 0 0 0 0 0\n\
  0 1 : 0 1 0 0 0 0 0 0 0 0 0 0 0 0\n\
  0 3 : 0 0 1 0 0 0 0 0 0 0 0 0 0 0\n\
  0 1 : 0 0 0 1 0 0 0 0 0 0 0 0 0 0\n\
  0 1 : 0 0 0 0 0 0 0 0 0 0 0 1 0 0\n\
  0 1 : 1 0 0 0 0 0 1 0 0 0 0 1 0 0\n\
  0 1 : 0 0 0 0 0 0 0 0 0 1 0 0 0 0\n\
  0 1 : 1 0 0 0 0 0 0 1 0 0 0 1 0 0\n\
  0 1 : 0 0 0 0 0 0 0 0 1 0 0 0 0 0\n\
  0 1 : 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n\
  0 1 : 1 0 0 0 0 0 0 0 0 0 1 1 0 0\n\
  0 1 : 0 0 0 0 0 0 0 0 0 0 0 0 1 0\n\
  1 7 : 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
  1 3 : 0 0 0 0 1 0 0 0 0 0 0 0 0 0\n\
  1 4 : 1 0 0 0 0 0 0 0 0 0 0 1 0 0\n\
  1 2 : 0 0 0 1 0 0 0 0 0 0 0 0 0 0\n\
  1 1 : 0 0 0 1 0 1 0 0 0 0 0 0 0 0\n\
\n";

char *example2=
"  0 6 :  A . G Z\n\
  1 1 :  A . G X\n\
  1 1 :  T . G Z\n\
  2 3 :  A C G Z\n\
  2 1 :  A . C Z\n\
  2 1 :  A . G Z\n\
\n";

char *ancestor="        A . G Z\n";

void help() {
	printf("  Example 1\n");
	printf("  sequence file named example1.seq\n");
	printf(example1);
	printf("  subpopulations 0 and 1 on left are optional\n");
	printf("  0 ancestor base, 1 is mutant base\n");
	printf("  seq2tr example1.seq example1.tre\n\n");
	printf("  Example 2\n");
	printf("  sequence file named example2.seq\n");
	printf(example2);
	printf("  Spaces between bases are optional\n");
	printf("  ancestor base file named ancestor.bas\n");
	printf(ancestor);
	printf("  seq2tr example2.seq example2.tre ancestor.bas\n\n");
	exit(1);
}


char *usage="\n\tseq2tr seq_file tree_out_file [ancestor_base_file]\n";
char *version="\tVer 1.1 13/8/97, Bob Griffiths\n";
char *helpcmd="\tseq2tr -h for help\n\n";
main(int argc, char**argv) {
	if(argc>1&&(argv[1][1]=='h'||argv[1][1]=='h')) help();
	if(argc<3) {
		printf(usage);
		printf(version);
		printf(helpcmd);
		exit(1);
		}
	if(argc==3) do_tree(argv[1],argv[2],NULL);
	else if(argc==4) do_tree(argv[1],argv[2],argv[3]);
}
#endif
