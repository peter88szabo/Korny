// Training-time value evaluator for the same local graph message kernel as
// src/message_kernel.py. Derivatives and saved model inference use NumPy.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
struct Edge {int a,b;double w;};
static double sw(double x,double lo,double hi) {
    if (x<=lo) return 1.; if(x>=hi) return 0.;
    double t=(x-lo)/(hi-lo);
    return 1.+t*t*t*t*(-35.+t*(84.+t*(-70.+20.*t)));
}
extern "C" void message_sites(int n,int m,int topology,const int *type,const double *r,
                              const double *seeds,double *out) {
    const double radius[]={.31,.76,.71,.66};
    const double centers[]={.84,1.,1.24,1.63,2.03}, widths[]={.095,.115,.16,.26,.35};
    int channels=4*m,dim=24+2*channels+5*topology;
    std::vector<Edge> edges;
    std::vector<double> embed(n*channels), sumsq(n);
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) {
        int a=type[i],b=type[j]; if(a<0||b<0) continue;
        double d2=0.;
        for(int k=0;k<3;k++) {double t=r[3*i+k]-r[3*j+k];d2+=t*t;}
        double x=std::sqrt(d2)/(radius[a]+radius[b]);
        if(x>=2.6) continue;
        double cut=sw(x,1.8,2.6), w=topology?sw(x,1.12,1.65):sw(x,1.13,1.83);
        out[dim*i+b*6]+=w;out[dim*j+a*6]+=w;
        for(int k=0;k<5;k++) {
            double u=(x-centers[k])/widths[k], y=cut*std::exp(-.5*u*u);
            if(topology) {
                const double lo[]={1.06,1.10,1.12,1.16,1.22}, hi[]={1.30,1.43,1.56,1.66,1.78};
                y=sw(x,lo[k],hi[k]);
            }
            out[dim*i+b*6+1+k]+=y;out[dim*j+a*6+1+k]+=y;
        }
        if(w>0.) {edges.push_back({i,j,w});sumsq[i]+=w*w;sumsq[j]+=w*w;}
    }
    for(int i=0;i<n;i++) if(type[i]>=0) {
        for(int j=0;j<m;j++) {
            double d2=0.;
            for(int k=0;k<24;k++) {double u=out[i*dim+k]-seeds[(type[i]*m+j)*24+k];d2+=u*u;}
            embed[i*channels+type[i]*m+j]=std::exp(-.5*d2/(1.25*1.25));
        }
    }
    for(auto e:edges) for(int k=0;k<channels;k++) {
        out[e.a*dim+24+k]+=e.w*embed[e.b*channels+k];
        out[e.b*dim+24+k]+=e.w*embed[e.a*channels+k];
    }
    for(auto e:edges) for(int k=0;k<channels;k++) {
        out[e.a*dim+24+channels+k]+=e.w*out[e.b*dim+24+k];
        out[e.b*dim+24+channels+k]+=e.w*out[e.a*dim+24+k];
    }
    for(int i=0;i<n;i++) for(int k=0;k<channels;k++)
        out[i*dim+24+channels+k]-=sumsq[i]*embed[i*channels+k];
    if(topology) {
        std::vector<double> degree(n,.5), power(n*n), next(n*n);
        for(auto e:edges){degree[e.a]+=e.w; degree[e.b]+=e.w;}
        for(auto &e:edges)e.w/=std::sqrt(degree[e.a]*degree[e.b]);
        for(int i=0;i<n;i++) power[i*n+i]=1.;
        for(int k=1;k<=7;k++){
            std::fill(next.begin(), next.end(), 0.);
            for(auto e:edges) for(int i=0;i<n;i++){
                next[e.a*n+i]+=e.w*power[e.b*n+i];
                next[e.b*n+i]+=e.w*power[e.a*n+i];
            }
            power.swap(next);
            if(k>=3)for(int i=0;i<n;i++)out[i*dim+24+2*channels+k-3]=power[i*n+i];
        }
    }
}
