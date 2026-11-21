// Cache the same bond-centered features as src/bond_environment.py for fitting.
// The deployment implementation supplies analytic derivatives without a compiler.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
struct Bond {int i,j; double length,near,gate; std::array<double,3> u;};
static double gate(double d,double a,double b) {
    if(d<a) return 1;
    if(d>b) return 0;
    double t=(d-a)/(b-a);
    return 1+t*t*t*t*(-35+t*(84+t*(-70+20*t)));
}
extern "C" void bond_values(int n,int width,int dim,const int* z,const double* r,
                            const double* omega,const double* phase,double* out) {
    const double radii[]={.31,.76,.71,.66};
    std::vector<int> type(n);
    std::vector<std::array<double,4>> coord(n);
    std::vector<std::array<std::array<double,3>,4>> vec(n);
    std::vector<Bond> edges;
    for(int i=0;i<n;i++)type[i]=z[i]==1?0:z[i]==6?1:z[i]==7?2:z[i]==8?3:-1;
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) {
        int a=type[i],b=type[j];
        if(a<0||b<0)continue;
        double d=0;
        for(int k=0;k<3;k++) {double v=r[3*i+k]-r[3*j+k];d+=v*v;}
        double x=std::sqrt(d)/(radii[a]+radii[b]);
        if(x>=1.95)continue;
        double near=gate(x,.98,1.65);
        coord[i][b]+=near;coord[j][a]+=near;
        std::array<double,3> u{};
        if(dim==17)for(int k=0;k<3;k++){
            u[k]=(r[3*j+k]-r[3*i+k])/std::sqrt(d);
            vec[i][b][k]+=near*u[k];vec[j][a][k]-=near*u[k];
        }
        edges.push_back({i,j,x,near,gate(x,1.25,1.95),u});
    }
    std::vector<double> angle(width);
    for(const auto& edge:edges) {
        int i=edge.i,j=edge.j;
        if(type[i]>type[j])std::swap(i,j);
        int a=type[i],b=type[j], pair=a*4-a*(a-1)/2+b-a;
        int orientations=a==b?2:1;
        double normalization=std::sqrt(2./width)*edge.gate/orientations;
        for(int o=0;o<orientations;o++) {
            auto l=coord[i],rco=coord[j];
            l[type[j]]-=edge.near;rco[type[i]]-=edge.near;
            std::array<double,4> al{},ar{};
            if(dim==17) {
                double sign=i==edge.i?1:-1;
                for(int h=0;h<4;h++)for(int k=0;k<3;k++) {
                    al[h]+=sign*edge.u[k]*vec[i][h][k];
                    ar[h]-=sign*edge.u[k]*vec[j][h][k];
                }
                al[type[j]]-=edge.near;ar[type[i]]-=edge.near;
            }
            for(int c=0;c<width;c++) {
                int idx=pair*width+c;
                auto w=omega+dim*idx;
                angle[c]=phase[idx]+w[0]*(edge.length-1);
                for(int h=0;h<4;h++)angle[c]+=l[h]*w[1+h]+rco[h]*w[5+h];
                if(dim==17)for(int h=0;h<4;h++)angle[c]+=al[h]*w[9+h]+ar[h]*w[13+h];
            }
            for(int c=0;c<width;c++)out[pair*width+c]+=normalization*std::cos(angle[c]);
            std::swap(i,j);
        }
    }
}
