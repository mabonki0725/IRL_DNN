/******************
  ニューラルネットワーク
  PRML 上巻 5章
  5.3.1 
  m.nakai 20130803
*******************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "neuro.h"

//#define ONLY

#define LIMIT 100000
#define PITCH 0.001

//#define SIG

static char *comMalloc(int size);
static double  tanh(double a);
static double dtanh(double a);



struct node {
  double *w;
  double *dw;
  int    *next;
  int ex;

  //double w;
  double d;
  double a;
  double z;
  
  //double *(func());
  //double *(dfunc());
};

/*****NET WORK data****
           node 1 node 2 node 3  node 4(mj)
level 1       1     1      1         2       <- input
level 2       1     1      1         2       <- hidden1
level 3       1     1      1         2       <- hidden2
level 4(step) 1     1      0         0       <- output
************************/

/****** input out data ********
       inp1 inp2 inp3  out1 out2(nj)
rec 1   x1   x2   x3    t1   t2 
rec 2   x1   x2   x3    t1   t2 
rec 3   x1   x2   x3    t1   t2

rec ni  x1   x2   x3    t1   t2 
********************************/
/************
Quation 
num of input  are same as num of node(except 2) at level 1 
num of output are same as num of node(except 2) at level mi 
if node is 2, node is bias node.
*************/
static double ReLU(double x)
{
   if(x >= 0) return(x);
   else       return(0);
}
/***
 Make neuro
****/
struct node **IRLneuroMake(flag,ni,nj,nexp,ntag,nstep,nhidden,pMj) 
int flag;    /* TAG_VALUE TAG_RAD */
int ni;      /* num of data records */
int nj;      /* num of data dimension */
int nexp;    /* num of data explanation dimension */
int ntag;    /* num of data target dimension */
int nstep;   /* num of layer */
int nhidden; /* num of hidden nodes */
int *pMj;    /* out num of node by step */
{

  int i,j,k,L;
  int mj;
  int nmid;

  int step;

  //double *x;
  //double *t;

  struct node **node;

  //double errOld;

  //int conv;
  //int limit;
  //double pitch;  /* 20171217 add */
  int numInp,numOut;


  /* start Exection */
  if(nstep <= 0) step=3;
  else           step=nstep;

  if(nhidden <= 0) nmid=(int)(ni * 0.5);
  else             nmid=nhidden;


  /* NET領域の獲得 */
  if(nexp > ntag) mj = nexp;
  else            mj = ntag;
  if(nmid > mj  ) mj = nmid + 1;
  else            mj = mj   + 1;

  node=(struct node **)comMalloc(sizeof(struct node)*step);
  for(L=0;L<step;L++) {
    node[L]=(struct node *)comMalloc(sizeof(struct node)*mj);
    for(j=0;j<mj;j++) {
      node[L][j].w=(double *)comMalloc(sizeof(double)*mj);
      node[L][j].dw=(double *)comMalloc(sizeof(double)*mj);
      node[L][j].next=(int *)comMalloc(sizeof(int)*mj);
    }
  }
  
  /* 入力と出力 */
  for(j=0;j<nexp;j++) {
    node[0][j].ex = 1;
  }
  node[0][nexp].ex=2;
  for(j=0;j<ntag;j++) {
    node[step-1][j].ex = 1;
  }
  /* 中間設定 */
  for(i=1;i<step-1;i++) {
    for(j=0;j<nmid;j++) {
      node[i][j].ex=1;
    }
    node[i][j].ex=2;
  }


  /* 入力数と出力数のチェック */
  numInp=0;
  for(j=0;j<mj;j++) {
    if(node[0][j].ex == 0 || node[0][j].ex == 2);
    else numInp++;
  }
  numOut=0;
  for(j=0;j<mj;j++) {
    if(node[step-1][j].ex == 0 || node[step-1][j].ex == 2);
    else numOut++;
  }
  /** 入出ノード数がデータと一致しているかチェック **/
  if(numInp != nexp || numOut != ntag) {
    if(numInp != nexp) fprintf(stderr,"inp node=%d is not same explanation=%d\n",numInp,nexp);
    if(numOut != ntag) fprintf(stderr,"out node=%d is not same target=%d\n",numOut,ntag);

    for(L=0;L<step;L++) {
      for(j=0;j<mj;j++) {
        free(node[L][j].w);
        free(node[L][j].next);
      }
      free(node[L]);
    }
    free(node);

    return(NULL);
  }

  /* ノード間の結線 */
  for(L=0;L<step-1;L++) {
    for(j=0;j<mj;j++) {
      if(!node[L][j].ex) continue;
      for(k=0;k<mj;k++) {
        if(node[L+1][k].ex == 1) {
          node[L][j].next[k]=1;
          node[L][j].w[k]=(double)rand()/(double)RAND_MAX;
        }
      }
      //node[L][j].func=&(tanh());
      //node[L}[j].dfunc=&(dtanh());
    }
  }
  *pMj = mj;
  return(node);
}
/*******
  free Neuro
*******/
IRLfreeNeuro(pNode,step,mj)
struct node **pNode;
int step;
int mj;
{
    int L,j;

    for(L=0;L<step;L++) {
      for(j=0;j<mj;j++) {
        free(pNode[L][j].w);
        free(pNode[L][j].next);
      }
      free(pNode[L]);
    }
    free(pNode);
    
    return(0);
}
/******
  backward
*******/
int IRLbackward(flag,pNode,ni,nj,data,outData,nexp,ntag,step,mj,loop,eta,method)
int flag;    /* TAG_VALUE TAG_RAD */
struct node **pNode;
int ni;
int nj;
double **data;  /* input data nexp+ntag order */
double **outData;  /* out */
int nexp;
int ntag;
int step;   /* num of layer */
int mj;
int loop;    /* num of calc loop */
double eta;
int method;  /* TANH RELU SIGMOID */
{
  int i,j,k,n,L;
  double sum,errOld,err,pitch;
  int limit,conv;
  double *x;
  double *t;
  int numOut;

  conv=0;
  if(loop <= 0) limit=100000;
  else          limit=loop;

  if(eta <= 0) pitch = PITCH;   /* 20171217 add */
  else         pitch = eta;

  numOut=0;
  for(j=0;j<mj;j++) {
    if(pNode[step-1][j].ex == 0 || pNode[step-1][j].ex == 2);
    else numOut++;
  }  

  err=0;
  for(n=0;n < ni;n++) {




    err=0;
    for(loop=0;loop < limit;loop++) { 

      t=&data[n][0];         /* 出力 */
      x=&data[n][numOut];    /* 入力 */

      /* middle data foward **/
      k=0;
      for(j=0;j<mj;j++) {
        if(!pNode[0][j].ex) continue;
        if(pNode[0][j].ex == 2) pNode[0][j].z=1.0;  /* 入力バイアス */
        else                    pNode[0][j].z=x[k++];
      }

      /* foward propagtion */
      for(L=1;L<step;L++) {
        for(k=0;k<mj;k++) {
          if(!pNode[L][k].ex) continue;
          pNode[L][k].a = 0;
          for(j=0;j<mj;j++) {
            if(pNode[L-1][j].next[k]) { /* j -> k */
              pNode[L][k].a += pNode[L-1][j].z * pNode[L-1][j].w[k];
            }
          }
          if(pNode[L][k].ex != 2) {
#ifdef SIG
            if(L < step-0) {
#else
            if(L < step-1) {
#endif
              if(FUNC_TANH) pNode[L][k].z = /*node[L][k].func*/tanh(pNode[L][k].a);
              if(FUNC_RELU) pNode[L][k].z = /*node[L][k].func*/ReLU(pNode[L][k].a);

            }
            else           pNode[L][k].z = pNode[L][k].a;
          }
          else             pNode[L][k].z = 1.0;  /* 結線なしバイアス */
        }
      }
    
      /* backward */

      i=0;
      for(k=0;k<mj;k++) {
        if(!pNode[step-1][k].ex) continue;
        if(flag == TAG_VALUE) pNode[step-1][k].d = pNode[step-1][k].z - t[i];
        if(flag == TAG_GRAD)  pNode[step-1][k].d = - t[i];
        err += fabs(pNode[step-1][k].d);
#ifdef SIG
        pNode[step-1][k].d *= dtanh(pNode[step-1][k].a); /*(1 - pow(node[step-1][k].z,2.0)) */;
#endif
        i++;
      }
      //if(fabs(err) < 0.01) break;
      //if(conv == 1 && fabs(err) > errOld) break;
      //errOld=fabs(err);
      
      /* dw のクリア */
      for(L=0;L<step-1;L++) {
        for(j=0;j<mj;j++) {
          if(!pNode[L][j].ex) continue;
          for(k=0;k<mj;k++) {
            if(pNode[L+1][k].ex == 1) {
              pNode[L][j].dw[k]=0;
            }
          }
        }
      }

      for(L=step-1;L > 0;L--) {
        for(j=0;j<mj;j++) {
          if(!pNode[L-1][j].ex) continue;
          for(k=0;k<mj;k++) {
            if(!pNode[L-1][j].next[k]) continue;
            pNode[L-1][j].dw[k] +=   pNode[L][k].d * pNode[L-1][j].z;
          }
        }
        for(j=0;j<mj;j++) {
          if(!pNode[L-1][j].ex) continue;
          sum=0.0;
          for(k=0;k<mj;k++) {
            if(pNode[L-1][j].next[k]) {
              sum += pNode[L-1][j].w[k] * pNode[L][k].d;
            }
          }
#if 1
          if(FUNC_TANH) pNode[L-1][j].d = /*node[L-1][j].dfunc*/dtanh(pNode[L-1][j].a) * sum;
          if(FUNC_RELU) pNode[L-1][j].d = /*node[L-1][j].dfunc*/ReLU(pNode[L-1][j].a) * sum;
#else
          pNode[L-1][j].d = (1 - pow(pNode[L-1][j].z ,2.0)) * sum;
#endif
        }
      }
      //if(fabs(err) < 0.01) break;
      //if(err < 0) break;
      //if(errOld < err)  break;

      errOld=err;
    
      for(L=0;L<step-1;L++) {
        for(j=0;j<mj;j++) {
          if(!pNode[L][j].ex) continue;
          for(k=0;k<mj;k++) {
            if(!pNode[L][j].next[k]) continue;
            pNode[L][j].w[k] -= pitch * pNode[L][j].dw[k]; /* 20171217 add */
          }
        }
      }
    //if(loop % 1000 == 0) fprintf(stderr,"loop=%d err=%lf\n",loop,err);
    }

    t=&data[n][0];        /* 出力 */
    x=&data[n][numOut];   /* 入力 */

    k=0;
    for(j=0;j<mj;j++) {
      if(!pNode[0][j].ex) continue;
      if(pNode[0][j].ex == 2) pNode[0][j].z=1.0;  /* 入力バイアス */
      else                    pNode[0][j].z=x[k++];
    }

    /* foward propagtion */
    for(L=1;L<step;L++) {
      for(k=0;k<mj;k++) {
        if(!pNode[L][k].ex) continue;
        pNode[L][k].a = 0;
        for(j=0;j<mj;j++) {
          if(pNode[L-1][j].next[k]) { /* j -> k */
            pNode[L][k].a += pNode[L-1][j].z * pNode[L-1][j].w[k];
          }
        }
        if(pNode[L][k].ex != 2) {
#ifdef SIG
          if(L < step-0) {
#else
          if(L < step-1) {
#endif 
            if(FUNC_TANH) pNode[L][k].z = /*node[L][k].func*/tanh(pNode[L][k].a);
            if(FUNC_RELU) pNode[L][k].z = /*node[L][k].func*/ReLU(pNode[L][k].a);
          }
          else             pNode[L][k].z = pNode[L][k].a;
        }
        else               pNode[L][k].z = 1.0;  /* 結線なしバイアス */
      }
    }

    /* 結果の書出し */
    for(j=0;j<ntag;j++) {
      outData[n][j] = pNode[step-1][j].z;
    }
  }
  return(0);

}
/***********
  foward
***********/
IRLfoward(flag,pNode,ni,data,step,mj,loop)
struct node **pNode;
int flag;
double **data;
int step;
int mj;
int loop;
{

  int j,k,n,L;
  double errOld,err;
  int limit,numOut;
  double *x;

  if(loop <= 0) limit=100000;
  else          limit=loop;

  numOut=0;
  for(j=0;j<mj;j++) {
    if(pNode[step-1][j].ex == 0 || pNode[step-1][j].ex == 2);
    else numOut++;
  }  

  errOld = DBL_MAX;
  for(loop=0;loop < limit;loop++) {  /* ニューロの重みの訓練 */

    for(L=0;L<step-1;L++) {
      for(j=0;j<mj;j++) {
        if(!pNode[L][j].ex) continue;
        for(k=0;k<mj;k++) {
          if(pNode[L+1][k].ex == 1) {
            pNode[L][j].dw[k]=0;
          }
        }
      }
    }


    err=0;

    for(n=0;n<ni;n++) {    /* 入出力行繰返し */

      x=&data[n][0];       /* 入力 */


      k=0;
      for(j=0;j<mj;j++) {
        if(!pNode[0][j].ex) continue;
        if(pNode[0][j].ex == 2) pNode[0][j].z=1.0;  /* 入力バイアス */
        else                    pNode[0][j].z=x[k++];
      }

      /* foward propagtion */
      for(L=1;L<step;L++) {
        for(k=0;k<mj;k++) {
          if(!pNode[L][k].ex) continue;
          pNode[L][k].a = 0;
          for(j=0;j<mj;j++) {
            if(pNode[L-1][j].next[k]) { /* j -> k */
              pNode[L][k].a += pNode[L-1][j].z * pNode[L-1][j].w[k];
            }
          }
          if(pNode[L][k].ex != 2) {
#ifdef SIG
            if(L < step-0) {
#else
            if(L < step-1) {
#endif
              pNode[L][k].z = /*node[L][k].func*/tanh(pNode[L][k].a);
            }
            else           pNode[L][k].z = pNode[L][k].a;
          }
          else             pNode[L][k].z = 1.0;  /* 結線なしバイアス */
        }
      }
    }
  }
  return(0);
}
/******
  All neuro
*******/
int IRLneuro(flag,data,ni,nj,nexp,ntag,outData,nstep,loop,eta,nhidden,method)
int flag;    /* TAG_VALUE TAG_RAD */
int ni;      /* num of data records */
int nj;      /* num of data dimension */
int nexp;    /* num of data explanation dimension */
int ntag;    /* num of data target dimension */
double **data;  /* input data nexp+ntag order */
double **outData;  /* out */
int nstep;   /* num of layer */
int loop;    /* num of calc loop */
double eta;  /* learning rate */
int nhidden; /* num of hidden nodes */
int method;  /* TANH RELU SIGMOID */
{

  double sum;

  int i,j,k,L,n;
  int mj;
  int nmid;

  int step;

  double *x;
  double *t;

  struct node **node;

  double err,errOld;

  int conv;
  int limit;
  double pitch;  /* 20171217 add */
  int numInp,numOut;


  /* start Exection */
  if(nstep <= 0) step=3;
  else           step=nstep;

  if(nhidden <= 0) nmid=(int)(ni * 0.5);
  else             nmid=nhidden;

  conv=0;
  if(loop <= 0) limit=100000;
  else          limit=loop;

  if(eta <= 0) pitch = PITCH;   /* 20171217 add */
  else         pitch = eta;

  /* NET領域の獲得 */
  if(nexp > ntag) mj = nexp;
  else            mj = ntag;
  if(nmid > mj  ) mj = nmid + 1;
  else            mj = mj   + 1;

  node=(struct node **)comMalloc(sizeof(struct node)*step);
  for(L=0;L<step;L++) {
    node[L]=(struct node *)comMalloc(sizeof(struct node)*mj);
    for(j=0;j<mj;j++) {
      node[L][j].w=(double *)comMalloc(sizeof(double)*mj);
      node[L][j].dw=(double *)comMalloc(sizeof(double)*mj);
      node[L][j].next=(int *)comMalloc(sizeof(int)*mj);
    }
  }
  
  /* 入力と出力 */
  for(j=0;j<nexp;j++) {
    node[0][j].ex = 1;
  }
  node[0][nexp].ex=2;
  for(j=0;j<ntag;j++) {
    node[step-1][j].ex = 1;
  }
  /* 中間設定 */
  for(i=1;i<step-1;i++) {
    for(j=0;j<nmid;j++) {
      node[i][j].ex=1;
    }
    node[i][j].ex=2;
  }


  /* 入力数と出力数のチェック */
  numInp=0;
  for(j=0;j<mj;j++) {
    if(node[0][j].ex == 0 || node[0][j].ex == 2);
    else numInp++;
  }
  numOut=0;
  for(j=0;j<mj;j++) {
    if(node[step-1][j].ex == 0 || node[step-1][j].ex == 2);
    else numOut++;
  }
  /** 入出ノード数がデータと一致しているかチェック **/
  if(numInp != nexp || numOut != ntag) {
    if(numInp != nexp) fprintf(stderr,"inp node=%d is not same explanation=%d\n",numInp,nexp);
    if(numOut != ntag) fprintf(stderr,"out node=%d is not same target=%d\n",numOut,ntag);

    for(L=0;L<step;L++) {
      for(j=0;j<mj;j++) {
        free(node[L][j].w);
        free(node[L][j].next);
      }
      free(node[L]);
    }
    free(node);

    return(-9);
  }



  /* ノード間の結線 */
  for(L=0;L<step-1;L++) {
    for(j=0;j<mj;j++) {
      if(!node[L][j].ex) continue;
      for(k=0;k<mj;k++) {
        if(node[L+1][k].ex == 1) {
          node[L][j].next[k]=1;
          node[L][j].w[k]=(double)rand()/(double)RAND_MAX;
        }
      }
      //node[L][j].func=&(tanh());
      //node[L}[j].dfunc=&(dtanh());
    }
  }


  errOld = DBL_MAX;
  for(loop=0;loop < limit;loop++) {  /* ニューロの重みの訓練 */

    for(L=0;L<step-1;L++) {
      for(j=0;j<mj;j++) {
        if(!node[L][j].ex) continue;
        for(k=0;k<mj;k++) {
          if(node[L+1][k].ex == 1) {
            node[L][j].dw[k]=0;
          }
        }
      }
    }


    err=0;

    for(n=0;n<ni;n++) {    /* 入出力行繰返し */

      t=&data[n][numOut];  /* 出力 */
      x=&data[n][0];       /* 入力 */


      k=0;
      for(j=0;j<mj;j++) {
        if(!node[0][j].ex) continue;
        if(node[0][j].ex == 2) node[0][j].z=1.0;  /* 入力バイアス */
        else                   node[0][j].z=x[k++];
      }

      /* foward propagtion */
      for(L=1;L<step;L++) {
        for(k=0;k<mj;k++) {
          if(!node[L][k].ex) continue;
          node[L][k].a = 0;
          for(j=0;j<mj;j++) {
            if(node[L-1][j].next[k]) { /* j -> k */
              node[L][k].a += node[L-1][j].z * node[L-1][j].w[k];
            }
          }
          if(node[L][k].ex != 2) {
#ifdef SIG
            if(L < step-0) {
#else
            if(L < step-1) {
#endif
              node[L][k].z = /*node[L][k].func*/tanh(node[L][k].a);
            }
            else           node[L][k].z = node[L][k].a;
          }
          else             node[L][k].z = 1.0;  /* 結線なしバイアス */
        }
      }

      /* back propagation */
      i=0;
      for(k=0;k<mj;k++) {
        if(!node[step-1][k].ex) continue;
        if(flag == TAG_VALUE) node[step-1][k].d = node[step-1][k].z - t[i];
        if(flag == TAG_GRAD)  node[step-1][k].d = - t[i];
        err += fabs(node[step-1][k].d);
#ifdef SIG
        node[step-1][k].d *= dtanh(node[step-1][k].a); /*(1 - pow(node[step-1][k].z,2.0)) */;
#endif
        i++;
      }
      //if(fabs(err) < 0.01) break;
      if(conv == 1 && fabs(err) > errOld) break;
      //errOld=fabs(err);

      for(L=step-1;L > 0;L--) {
        for(j=0;j<mj;j++) {
          if(!node[L-1][j].ex) continue;
          for(k=0;k<mj;k++) {
            if(!node[L-1][j].next[k]) continue;
            node[L-1][j].dw[k] +=   node[L][k].d * node[L-1][j].z;
          }
        }
        for(j=0;j<mj;j++) {
          if(!node[L-1][j].ex) continue;
          sum=0.0;
          for(k=0;k<mj;k++) {
            if(node[L-1][j].next[k]) {
              sum += node[L-1][j].w[k] * node[L][k].d;
            }
          }
#if 1
          node[L-1][j].d = /*node[L-1][j].dfunc*/dtanh(node[L-1][j].a) * sum;
#else
          node[L-1][j].d = (1 - pow(node[L-1][j].z ,2.0)) * sum;
#endif
        }
      }
    }
    if(fabs(err) < 0.01) break;
    //if(err < 0) break;
    //if(errOld < err)  break;

    errOld=err;
    
    /* 重みの更新 */
    for(L=0;L<step-1;L++) {
      for(j=0;j<mj;j++) {
        if(!node[L][j].ex) continue;
        for(k=0;k<mj;k++) {
          if(!node[L][j].next[k]) continue;
          node[L][j].w[k] -= pitch * node[L][j].dw[k]; /* 20171217 add */
        }
      }
    }
    if(loop % 1000 == 0) fprintf(stderr,"loop=%d err=%lf\n",loop,err);

  }
  

  /* write output */
  for(n=0;n<ni;n++) {
    t=&data[n][numOut];  /* 出力 */
    x=&data[n][0];       /* 入力 */
    


    k=0;
    for(j=0;j<mj;j++) {
      if(!node[0][j].ex) continue;
      if(node[0][j].ex == 2) node[0][j].z=1.0;  /* 入力バイアス */
      else                   node[0][j].z=x[k++];
    }

    /* foward propagtion */
    for(L=1;L<step;L++) {
      for(k=0;k<mj;k++) {
        if(!node[L][k].ex) continue;
        node[L][k].a = 0;
        for(j=0;j<mj;j++) {
          if(node[L-1][j].next[k]) { /* j -> k */
            node[L][k].a += node[L-1][j].z * node[L-1][j].w[k];
          }
        }
        if(node[L][k].ex != 2) {
#ifdef SIG
          if(L < step-0) {
#else
          if(L < step-1) {
#endif 
            node[L][k].z = /*node[L][k].func*/tanh(node[L][k].a);
          }
          else             node[L][k].z = node[L][k].a;
        }
        else               node[L][k].z = 1.0;  /* 結線なしバイアス */
      }
    }

    /* 結果の書出し */
    for(j=0;j<ntag;j++) {
      outData[n][j] = node[step-1][j].z;
    }

  }


  for(i=0;i<step;i++) {
    for(j=0;j<mj;j++) {
      free(node[i][j].w);
      free(node[i][j].next);
    }
    free(node[i]);
  }
  free(node);

  return(0);

}
/*************
  領域の確保
**************/
static char *comMalloc(int size)
{
   char *pc;

   pc=(char *)malloc(size);
   memset(pc,'\0',size);

   return(pc);
}
/**************
   活性化関数
***************/
static double tanh(double a)
{
   double v;

   v = (exp(a) - exp(-a)) / (exp(a) + exp(-a));

   return(v);
}
/**************
   活性化関数(微分)
***************/
static double dtanh(double a)
{
   double v;

   v = 1.0 - pow(tanh(a),2.0);

   return(v);

}
#ifdef ONLY
/******************
  パラメータの読込
*******************/
static int readParam(parmNam,parmVal,nLimit)
char *parmNam[];
char *parmVal[];
int nLimit;
{
    int line;
    char record[512],*pc;
    char buff[512];
    int np;

    np=0;
    line=0;
    while(1) {
       fprintf(stdout,"%2d>>",++line);
       pc=fgets(record,512,pFclc);
       if(!pc) {
         /* パラメータが途中でない */
         return(-1);
       }
       //if(is_blank(pc) || is_comment(pc)) continue;

       if(pc == NULL || record[0] == ';') break;
       if(np > 256 -1) continue;

       pc=strtok(record,":=");
       if(pc) {
          strcpy(buff,pc);
          //nki_blank(buff);
          parmNam[np]=(char *)comMalloc(strlen(buff)+1);
          strcpy(parmNam[np],buff);
          if((pc=strtok(NULL,";\n"))) {
            strcpy(buff,pc);
            //nki_blank(buff);
            parmVal[np]=(char *)comMalloc(strlen(buff)+1);
            strcpy(parmVal[np],buff);
          }
          else {
            parmVal[np]=(char *)comMalloc(2);
            strcpy(parmVal[np],"");
          }
          np++;
       }      
    }
    parmNam[np]=NULL;
    return(np);
}
#endif