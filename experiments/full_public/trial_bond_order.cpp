// Batched fitting uses the same continuous two-hop coordination as the Python
// deployment path. Both kernel lengths are evaluated from one squared distance.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
using Shell = std::array<double, 4>;
struct Edge { int i, j; double x, near, short_range, gate; };
static double smooth(double x, double lo, double hi) {
    if (x <= lo) return 1.;
    if (x >= hi) return 0.;
    double t = (x-lo)/(hi-lo);
    return 1.+t*t*t*t*(-35.+t*(84.+t*(-70.+20.*t)));
}
static void evaluate_bonds(int n, const int *z, const double *r,
        const double *centers, const int *offsets, const double *transform, const double *classifier, const double *intercepts, double *wide, double *narrow, const double *learned=nullptr, const double *center_energy=nullptr) {
    const double radii[] = {.31,.76,.71,.66};
    std::vector<int> type(n);
    std::vector<Shell> coord(n), tight(n), second(n);
    std::vector<Edge> edges;
    for(int i=0;i<n;i++) type[i] = z[i]==1?0:z[i]==6?1:z[i]==7?2:z[i]==8?3:-1;
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) {
        int a=type[i], b=type[j];
        if(a<0 || b<0) continue;
        double d2=0.;
        for(int k=0;k<3;k++) {double t=r[3*i+k]-r[3*j+k]; d2+=t*t;}
        double x=std::sqrt(d2)/(radii[a]+radii[b]);
        if(x>=1.95) continue;
        double near=smooth(x,.98,1.65), sh=smooth(x,.83,1.09);
        coord[i][b]+=near; coord[j][a]+=near;
        tight[i][b]+=sh; tight[j][a]+=sh;
        edges.push_back({i,j,x,near,sh,smooth(x,1.25,1.95)});
    }
    for(const auto& e: edges) for(int h=0;h<4;h++) {
        second[e.i][h]+=e.near*(coord[e.j][h]-(h==type[e.i]?e.near:0.));
        second[e.j][h]+=e.near*(coord[e.i][h]-(h==type[e.j]?e.near:0.));
    }
    for(const auto& e: edges) {
        int i=e.i, j=e.j;
        if(type[i]>type[j]) std::swap(i,j);
        int a=type[i], b=type[j], pair=a*4-a*(a-1)/2+b-a;
        int orders=a==b?2:1;
        for(int o=0;o<orders;o++) {
            double desc[25];
            desc[0]=4.5*(e.x-1.);
            for(int h=0;h<4;h++) {
                double ci=coord[i][h]-(h==type[j]?e.near:0.);
                double cj=coord[j][h]-(h==type[i]?e.near:0.);
                desc[1+h]=ci;desc[5+h]=cj;
                desc[9+h]=1.5*(tight[i][h]-(h==type[j]?e.short_range:0.));
                desc[13+h]=1.5*(tight[j][h]-(h==type[i]?e.short_range:0.));
                desc[17+h]=.4*(second[i][h]-e.near*cj);
                desc[21+h]=.4*(second[j][h]-e.near*ci);
            }
            // Project each environment once. Centers arrive already projected,
            // so evaluating all exemplars costs one squared distance apiece.
            if (transform) {
                double projected[25] = {};
                for (int j=0;j<25;j++) for (int k=0;k<25;k++)
                    projected[k] += desc[j]*transform[(pair*25+j)*25+k];
                std::copy(projected,projected+25,desc);
            }
            double logit=classifier?intercepts[pair]:0.;
            double latent=0.;
            std::vector<double> sim(classifier || learned ? offsets[pair+1]-offsets[pair] : 0);
            for(int c=offsets[pair];c<offsets[pair+1];c++) {
                double d=0.;
                for(int h=0;h<25;h++) {double t=desc[h]-centers[25*c+h];d+=t*t;}
                // Squaring the wide kernel gives a length smaller by sqrt(2).
                double val=std::exp(-.5*d/(1.15*1.15));
                if(learned) {
                    sim[c-offsets[pair]]=val;
                    latent+=learned[c]*val;
                } else if(classifier) {
                    sim[c-offsets[pair]]=val;
                    logit+=classifier[c]*val;
                } else {
                    wide[c]+=e.gate/orders*val; narrow[c]+=e.gate/orders*val*val;
                }
            }
            if(learned) {
                for(int c=offsets[pair];c<offsets[pair+1];c++) {
                    double d=(latent-center_energy[c])/1.15;
                    double v=sim[c-offsets[pair]]*std::exp(-.5*d*d);
                    wide[c]+=e.gate/orders*v;
                    narrow[c]+=e.gate/orders*v*v;
                }
            } else if(classifier) {
                double p=1./(1.+std::exp(-logit));
                for(int c=offsets[pair];c<offsets[pair+1];c++) {
                    double v=sim[c-offsets[pair]];
                    wide[c]+=e.gate/orders*p*v; narrow[c]+=e.gate/orders*p*v*v;
                }
            }
            std::swap(i,j);
        }
    }
}

extern "C" void bond_order_gated_values(int n,const int *z,const double *r,const double *centers,
        const int *offsets,const double *transform,const double *classifier,const double *bias,double *wide,double *narrow) {
    evaluate_bonds(n,z,r,centers,offsets,transform,classifier,bias,wide,narrow);
}
extern "C" void bond_order_energy_values(int n,const int *z,const double *r,const double *centers,
        const int *offsets,const double *transform,const double *learned,const double *center_energy,double *wide,double *narrow) {
    evaluate_bonds(n,z,r,centers,offsets,transform,nullptr,nullptr,wide,narrow,learned,center_energy);
}
extern "C" void bond_order_metric_values(int n,const int *z,const double *r,const double *centers,
        const int *offsets,const double *transform,double *wide,double *narrow) {
    bond_order_gated_values(n,z,r,centers,offsets,transform,nullptr,nullptr,wide,narrow);
}
extern "C" void bond_order_values(int n,const int *z,const double *r,const double *centers,
        const int *offsets,double *wide,double *narrow) {
    bond_order_metric_values(n,z,r,centers,offsets,nullptr,wide,narrow);
}
