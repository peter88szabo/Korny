// Local charge-transfer basis for closed-shell Hirshfeld labels. Each unlike
// atom pair adds equal and opposite charge. The cutoff functions and their
// first derivatives are continuous, including where a neighbor leaves a shell.
#include <algorithm>
#include <cmath>
#include <vector>

static double shell(double r, double a, double b) {
    if (r <= a) return 1.;
    if (r >= b) return 0.;
    double t = (r-a)/(b-a);
    return (1.-t)*(1.-t)*(1.+2.*t);
}

// One batch shares an atom count but every sample has its own atom types,
// coordinates and total charge. The Python caller selects only the columns
// that can occur in the batch, keeping even the largest groups inexpensive.
extern "C" void charge_basis(int nframes, int nat, int width,
                             const unsigned char* z, const float* xyz,
                             const signed char* charge, const double* radius,
                             const int* element, double* base, double* x) {
    for (int frame=0;frame<nframes;++frame) {
        const int offset=frame*nat;
        double sum=0.;
        for (int i=0;i<nat;++i) {
            const double r=radius[z[offset+i]];
            base[offset+i]=r*r;
            sum+=r*r;
        }
        for (int i=0;i<nat;++i) {
            base[offset+i]*=charge[frame]/sum;
            const int a=offset+i, za=z[a], ga=4*element[za];
            double* xi=x+static_cast<long>(a)*width;
            for (int j=i+1;j<nat;++j) {
                const int b=offset+j, zb=z[b];
                if (za==zb) continue;
                const double r0=radius[za]+radius[zb];
                double d2=0.;
                for (int k=0;k<3;++k) {
                    const double dx=xyz[3*a+k]-xyz[3*b+k]; d2+=dx*dx;
                }
                if (d2>=7.84*r0*r0) continue;
                const double d=std::sqrt(d2)/r0;
                const double w[4]={shell(d,.78,1.48),shell(d,1.0,1.95),
                                   shell(d,1.40,2.80),charge[frame]*shell(d,1.0,1.95)};
                const int gb=4*element[zb];
                double* xj=x+static_cast<long>(b)*width;
                for (int k=0;k<4;++k) {
                    xi[ga+k]+=w[k]; xi[gb+k]-=w[k];
                    xj[ga+k]-=w[k]; xj[gb+k]+=w[k];
                }
            }
        }
    }
}
