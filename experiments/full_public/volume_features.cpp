// Training-only acceleration for the nr=2, lmax=1 descriptor in src/local.py.
// A first-order Legendre contraction is a dot product of vector-valued radial
// sums. This avoids forming all neighbor triples for millions of archive frames.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

static uint64_t mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

// A graph signature protects complete bonding topologies in the anonymous XYZ
// release. Two distance signatures identify repeated geometries independently
// of rotation, translation, atom order and the archive in which they occur.
extern "C" int features(int n, const uint8_t *z, const double *r,
                        const double *radii, const int *types,
                        uint64_t *hashes, double *output) {
    constexpr int channels=16, width=152;
    std::vector<std::vector<int>> graph(n);
    std::vector<std::array<double,channels*3>> vectors(n);
    std::vector<uint64_t> colors(n), next(n);
    uint64_t h1=0, h2=0;
    if (output) std::fill(output, output+n*width, 0.);
    for (int i=0; i<n; ++i) {
        colors[i]=mix(z[i]);
        h1+=colors[i];
        h2+=mix(colors[i]);
        for (int j=0; j<i; ++j) {
            double u[3];
            double d2=0;
            for (int axis=0; axis<3; ++axis) {
                u[axis]=r[3*j+axis]-r[3*i+axis];
                d2+=u[axis]*u[axis];
            }
            double d=std::sqrt(d2);
            if (d<1e-9 || !std::isfinite(d)) return 1;
            uint64_t pair=std::min(z[i],z[j])*128+std::max(z[i],z[j]);
            uint64_t token=(pair<<40)+static_cast<uint64_t>(std::nearbyint(d*1e4));
            h1+=mix(token);
            h2+=mix(token+UINT64_C(0x9e3779b97f4a7c15));
            if (d < 1.2*(radii[z[i]]+radii[z[j]])) {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
            if (!output || d>=3.8) continue;
            if (types[z[i]]<0 || types[z[j]]<0) return 2;
            double t=std::clamp((d-2.)/1.8,0.,1.);
            double cut=1.+t*t*t*t*(-35.+t*(84.+t*(-70.+20.*t)));
            for (int b=0; b<2; ++b) {
                double weight=b ? cut*(2*d/3.8-1) : cut;
                int a=2*types[z[j]]+b, c=2*types[z[i]]+b;
                output[i*width+a]+=weight;
                output[j*width+c]+=weight;
                for (int k=0; k<3; ++k) {
                    vectors[i][3*a+k]+=weight*u[k]/d;
                    vectors[j][3*c+k]-=weight*u[k]/d;
                }
            }
        }
    }
    for (int round=0; round<5; ++round) {
        for (int i=0; i<n; ++i) {
            uint64_t neighbors=0;
            for (int j: graph[i]) neighbors+=mix(colors[j]);
            next[i]=mix(colors[i]+3*neighbors+graph[i].size());
        }
        colors.swap(next);
    }
    uint64_t graph_hash=0;
    for (auto color: colors) graph_hash+=mix(color);
    hashes[0]=graph_hash;
    hashes[1]=h1;
    hashes[2]=h2;
    if (output) {
        for (int i=0; i<n; ++i) {
            int index=channels;
            for (int s=0; s<8; ++s) for (int t=s; t<8; ++t)
                for (int a=2*s; a<2*s+2; ++a)
                    for (int b=(s==t ? a : 2*t); b<2*t+2; ++b) {
                        double dot=0;
                        for (int k=0; k<3; ++k) dot+=vectors[i][3*a+k]*vectors[i][3*b+k];
                        output[i*width+index++]=dot*(3./(4.*std::acos(-1.)));
                    }
        }
    }
    return 0;
}
