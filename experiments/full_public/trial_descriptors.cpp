// Training accelerator for the element-resolved inverse-distance Chebyshev and
// Legendre descriptors in src/local.py. Only atomic feature values are needed
// here. MD forces use the analytic Cartesian derivatives in the Python model.
#include <algorithm>
#include <cmath>
#include <vector>
struct Neighbor {
    int type;
    double u[3], radial[8];
};
static void encode_sites(int n, int channels, int nr, int ellmax, int width,
                          double ron, double rc, const int *species, const double *r,
                          const int *layout, double *output, const double *radii,
                          const double *centers, const double *widths, const double *atom_radii=nullptr) {
    const double pi = std::acos(-1.);
    for (int i=0; i<n; ++i) {
        if (species[i]<0) continue;
        double *out=output+i*width;
        std::vector<Neighbor> ns;
        for (int j=0; j<n; ++j) {
            if (j==i || species[j]<0) continue;
            Neighbor a{};
            a.type=species[j];
            double squared=0.;
            for (int k=0;k<3;++k) {a.u[k]=r[3*j+k]-r[3*i+k]; squared+=a.u[k]*a.u[k];}
            double d=std::sqrt(squared);
            double dist=atom_radii?d/(atom_radii[i]+atom_radii[j]):(radii?d/(radii[species[i]]+radii[a.type]):d);
            if (dist>=rc) continue;
            for (double &u:a.u) u/=d;
            double t=std::max(0.,(dist-ron)/(rc-ron));
            double cut=1.+std::pow(t,4)*(-35.+t*(84.+t*(-70.+20.*t)));
            if (centers) {
                for (int k=0;k<nr;k++) {
                    double t=(dist-centers[k])/widths[k];
                    a.radial[k]=cut*std::exp(-.5*t*t);
                }
            } else {
                double x=1./d-1.;
                a.radial[0]=cut;a.radial[1]=cut*x;
                for(int k=2;k<nr;++k) a.radial[k]=2*x*a.radial[k-1]-a.radial[k-2];
            }
            for(int k=0;k<nr;++k) out[a.type*nr+k]+=a.radial[k];
            ns.push_back(a);
        }
        for(const auto &a:ns) for(const auto &b:ns) {
            double cosine=0.;
            for(int k=0;k<3;++k) cosine+=a.u[k]*b.u[k];
            double prev=1., p=cosine;
            for(int l=1;l<=ellmax;++l){
                double factor=p*(2*l+1)/(4*pi);
                for(int u=0;u<nr;++u) for(int v=0;v<nr;++v) {
                    int idx=layout[((l-1)*channels+a.type*nr+u)*channels+b.type*nr+v];
                    if(idx>=0) out[idx]+=factor*a.radial[u]*b.radial[v];
                }
                double next=((2*l+1)*cosine*p-l*prev)/(l+1);
                prev=p;p=next;
            }
        }
    }
}

extern "C" void raw_sites(int n,int channels,int nr,int lmax,int width,double ron,double rc,
                          const int *species,const double *r,const int *layout,double *output) {
    encode_sites(n,channels,nr,lmax,width,ron,rc,species,r,layout,output,nullptr,nullptr,nullptr);
}
extern "C" void density_sites(int n,int channels,int nr,int lmax,int width,double ron,double rc,
                              const int *species,const double *r,const int *layout,double *output,
                              const double *radii,const double *centers,const double *widths) {
    encode_sites(n,channels,nr,lmax,width,ron,rc,species,r,layout,output,radii,centers,widths);
}

extern "C" void periodic_sites(int n,int channels,int nr,int lmax,int width,double ron,double rc,
                               const int *species,const double *r,const int *layout,double *output,
                               const double *atom_radii,const double *centers,const double *widths) {
    encode_sites(n,channels,nr,lmax,width,ron,rc,species,r,layout,output,nullptr,centers,widths,atom_radii);
}
