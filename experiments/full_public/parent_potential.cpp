// Evaluate the seven Gaussian pair features already used in ReactionSurface.
// Training projects the reference and model forces onto the same deterministic
// unit vectors. Evaluation contracts each radial force response directly with
// fitted coefficients, recovering all three Cartesian components at every atom.
#include <cmath>
#include <cstdint>
#include <vector>

static uint64_t random_bit(uint64_t& v) {
    v ^= v >> 12; v ^= v << 25; v ^= v >> 27;
    return (v*2685821657736338717ULL)>>63;
}

extern "C" void parent_pair(int frames,int atoms,int width,int ndir,int first,int evaluate,
                           const unsigned char* z,const float* pos,const signed char* charge,
                           const float* reference_force,const double* energy,const int* channel,
                           const int* atom_col,const double* radii,const double* lookup,
                           const double* coeff,double* features,double* reference,
                           double* predicted_force) {
    const int directions=ndir+1;
    std::vector<double> v(ndir*atoms*3);
    for(int n=0;n<frames;++n){
        const int start=n*atoms, q=charge[n];
        double* x=features+static_cast<long>(n)*directions*width;
        double* y=reference+n*directions;
        x[width-2]=q;x[width-1]=q*q;y[0]=energy[n];
        for(int i=0;i<atoms;++i)x[atom_col[z[start+i]]]+=1.;
        uint64_t rng=UINT64_C(0x56be513fa0478971)^(uint64_t(atoms)<<48)^
                     ((static_cast<uint64_t>(first+n)+1)*UINT64_C(0x9e3779b97f4a7c15));
        for(int k=0;k<ndir;++k){
            for(int j=0;j<3*atoms;++j){
                v[k*atoms*3+j]=(random_bit(rng)?1.:-1.)/std::sqrt(3.*atoms);
                y[k+1]+=reference_force[3*start+j]*v[k*atoms*3+j];
            }
        }
        for(int i=0;i<atoms;++i){
            const int ia=start+i, za=z[ia];
            for(int j=i+1;j<atoms;++j){
                const int ib=start+j,zb=z[ib],col=channel[za*119+zb];
                double dr[3],d2=0.;
                for(int a=0;a<3;++a){dr[a]=pos[3*ia+a]-pos[3*ib+a];d2+=dr[a]*dr[a];}
                const double r0=radii[za]+radii[zb];
                if(d2>36.*r0*r0)continue; // discarded Gaussian tails below 1e-28
                const double distance=std::sqrt(d2), s=distance/r0, t=s*1024.;
                const int ix=int(t);const double h=t-ix;
                const double* u=lookup+ix*14;const double* w=u+14;
                double proj[2]={},slope=0.;
                for(int k=0;k<ndir;++k)
                    for(int a=0;a<3;++a)
                        proj[k]-=(v[(k*atoms+i)*3+a]-v[(k*atoms+j)*3+a])*dr[a]/(distance*r0);
                for(int c=0;c<7;++c){
                    // Cubic Hermite values and their derivatives use the same
                    // polynomial, so the returned force is conservative.
                    const double y0=u[c],y1=w[c],a=u[7+c]/1024.,b=w[7+c]/1024.;
                    const double c2=3.*(y1-y0)-2.*a-b,c3=2.*(y0-y1)+a+b;
                    const double val=y0+h*(a+h*(c2+h*c3));
                    const double deriv=1024.*(a+h*(2.*c2+3.*h*c3));
                    x[col+c]+=val;
                    for(int k=0;k<ndir;++k)x[(k+1)*width+col+c]+=proj[k]*deriv;
                    if(evaluate)slope-=coeff[col+c]*deriv/(distance*r0);
                }
                if(evaluate)for(int a=0;a<3;++a){
                    predicted_force[3*ia+a]+=slope*dr[a];
                    predicted_force[3*ib+a]-=slope*dr[a];
                }
            }
        }
    }
}
