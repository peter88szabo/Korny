// Energy-only implementation for building the four-body training matrix.
// The deployed potential and its Cartesian Jacobian are in src/four_body.py.
#include <array>
#include <cmath>
#include <vector>
using Vec = std::array<double,3>;
static double dot(const Vec &a,const Vec &b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static double radius(int z) {
    switch(z) {
        case 1:return .31; case 6:return .76; case 7:return .71; case 8:return .66;
        case 9:return .57; case 15:return 1.07; case 16:return 1.05; case 17:return 1.02;
        default:return 0;
    }
}
static double gate(double x,double lo,double hi) {
    if(x<=lo)return 1;
    if(x>=hi)return 0;
    double t=(x-lo)/(hi-lo);
    return 1+t*t*t*t*(-35+t*(84+t*(-70+20*t)));
}
struct Edge { int j; Vec u; double x,w; };
extern "C" void four_values(int n,const int *z,const double *r,double *out) {
    if(n<4)return;
    std::vector<std::vector<Edge>> edges(n);
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) {
        double ri=radius(z[i]), rj=radius(z[j]);
        if(!ri||!rj)continue;
        Vec u;
        for(int k=0;k<3;k++)u[k]=r[3*j+k]-r[3*i+k];
        double d=std::sqrt(dot(u,u)),x=d/(ri+rj);
        if(x>=1.85)continue;
        for(double &k:u)k/=d;
        double w=gate(x,1,1.65);
        edges[i].push_back({j,u,x,w});
        for(double &k:u)k=-k;
        edges[j].push_back({i,u,x,w});
    }
    for(int i=0;i<n;i++) {
        int typ=z[i]==6?0:z[i]==7?1:z[i]==8?2:z[i]==16?3:-1;
        if(typ<0)continue;
        for(int shell=0;shell<2;shell++) {
            double m[3][3]={};
            int neighbors=0;
            for(const auto &a:edges[i]) {
                double w=shell?gate(a.x,1.1,1.85):gate(a.x,.95,1.4);
                if(w<=0)continue;
                neighbors++;
                for(int k=0;k<3;k++)for(int l=0;l<3;l++)m[k][l]+=w*a.u[k]*a.u[l];
            }
            if(neighbors>=3)out[2*typ+shell]+=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
        }
        for(const auto &e:edges[i]) {
            int j=e.j,lo=std::min(z[i],z[j]),hi=std::max(z[i],z[j]);
            if(j<=i||e.w==0)continue;
            int bin=lo==6&&hi==6?8:lo==6&&hi==7?9:lo==6&&hi==8?10:lo==8&&hi==8?11:-1;
            if(bin<0)continue;
            // Project both arms perpendicular to the central bond and require distinct atoms.
            for(const auto &a:edges[i])if(a.j!=j && a.w>0)
                for(const auto &b:edges[j])if(b.j!=i && b.j!=a.j && b.w>0)
                    out[bin]+=e.w*a.w*b.w*(z[a.j]==1?1:1.7)*(z[b.j]==1?1:1.7)
                              *(dot(a.u,b.u)-dot(a.u,e.u)*dot(b.u,e.u));
        }
    }
}
