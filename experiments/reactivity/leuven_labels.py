"""Extract electronic labels from every distinct private ORCA/Molpro output.

The inventory includes archive-only and nested copies. Store atomic numbers,
Cartesian coordinates (Å), electronic energies (hartree) and the originating
job/stage together. ORCA compound jobs often optimize at DFT and evaluate an
F12 single point at the same coordinates. Those are separate observations.
"""
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

from experiments.reactivity.leuven_files import CACHE
from src.units import BOHR_A

ELEMENT = dict(H=1,C=6,N=7,O=8,F=9,P=15,S=16,Cl=17)
FLOAT = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?'
COORD = re.compile(r'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\s*\n((?:[ \t]*[A-Z][a-z]?[ \t]+[-+.\dEe]+[ \t]+[-+.\dEe]+[ \t]+[-+.\dEe]+[ \t]*\n)+)')
ENERGY = re.compile(r'FINAL SINGLE POINT ENERGY[ \t]+('+FLOAT+r')([^\n]*)')
CMD = re.compile(r'^\s*\|\s*\d+>\s*!\s*(.+)$',re.M)
JOB = re.compile(r'^\s*COMPOUND JOB[^\n]*$',re.M)
LEVELS = ('ccsd(t)-f12/ri','dlpno-ccsd(t)','m062x','m06-2x','pbe0','wb97x-d3','wb97x','wb97m-v','r2scan-3c','r2scan','b97-3c','b3lyp','pbe','bp86','mp2')
BASES = ('aug-cc-pvtz','aug-cc-pvdz','cc-pvdz-f12','cc-pvtz-f12','def2-tzvpp','def2-tzvp','def2-svp','cc-pvtz','cc-pvdz','pc-2','pc-1','pc2','pc1')
BIND = 'INSERT OR IGNORE INTO labels VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)'


def method(command,section,version):
    """Retain basis, dispersion, grid, solvent and program version for comparisons."""
    words=command.lower().split()
    xc=next((w for w in LEVELS if w in words),None)
    basis=next((w for w in BASES if w in words),None)
    if not xc or (not basis and not xc.endswith('-3c')):
        return None
    xc={'m06-2x':'m062x'}.get(xc,xc)
    dispersion=[w for w in words if w in ('d3zero','d3bj','d3','d4','d3zeroatm','d3bjatm')]
    grids=[w for w in words if 'grid' in w]
    solvent=bool(re.search(r'\b(?:cpcm|cosmo|smd|alpb)\b',command,re.I) or
                 re.search(r'^\s*\|\s*\d+>\s*%\s*(?:cpcm|cosmo)\b',section,re.M|re.I))
    return '/'.join(['ORCA'+version,xc,basis or 'composite',*dispersion,*grids,'solution' if solvent else 'gas'])


def atom_table(lines):
    fields=[line.split() for line in lines.splitlines()]
    return ([ELEMENT[row[0]] for row in fields],[[float(x) for x in row[1:4]] for row in fields])


def obs(key,stage,kind,level,q,m,z,r,e,freq,source_name,command):
    xyz=np.asarray(r,dtype='<f8')
    if len(z)==0 or xyz.shape!=(len(z),3) or not np.isfinite(xyz).all() or not np.isfinite(e) or m<1 or (sum(z)-q-m+1)%2:
        raise ValueError('invalid_geometry_or_state')
    return (key,stage,kind,level,q,m,bytes(z),xyz.tobytes(),e,
            json.dumps(freq) if freq is not None else None,source_name,command,len(z))


def orca(key,s,name,issues):
    if 'ORCA TERMINATED NORMALLY' not in s[-10000:]:
        issues['incomplete_ORCA']+=1
        return
    version=re.search(r'Program Version\s+([\d.]+)',s)
    version=version[1] if version else '-unknown'
    for i,section in enumerate(JOB.split(s)):
        commands=CMD.findall(section)
        if not commands:
            continue
        cmd=' '.join(commands)
        level=method(cmd,section,version)
        qs=set(re.findall(r'Total Charge\s+Charge\s+\.+\s*(-?\d+)',section))
        ms=set(re.findall(r'Multiplicity\s+Mult\s+\.+\s*(\d+)',section))
        if not level or len(qs)!=1 or len(ms)!=1 or '$new_job' in section.lower():
            issues['unresolved_method_state_or_multijob']+=1
            continue
        q,m=int(qs.pop()),int(ms.pop())
        energies=list(ENERGY.finditer(section))
        tables=list(COORD.finditer(section))
        if not energies or not tables:
            issues['missing_ORCA_energy_or_coordinates']+=1
            continue
        # A single coordinate block must precede the final energy. Reject
        # ambiguous interleavings, e.g. a final summary following a scan.
        e=energies[-1]
        geom=next((b for b in reversed(tables) if b.end()<=e.start()),None)
        if geom is None or (len(energies)>1 and energies[-2].end()>geom.start()) or 'converg' in e[2].lower():
            issues['ambiguous_last_energy']+=1
            continue
        if 'SCF NOT CONVERGED' in section[geom.end():e.end()].upper():
            issues['unconverged_SCF']+=1
            continue
        z,r=atom_table(geom[1])
        settled='THE OPTIMIZATION HAS CONVERGED' in section[:e.end()]
        freqs=None
        if 'VIBRATIONAL FREQUENCIES' in section[e.end():]:
            spectrum=section[e.end():].rsplit('VIBRATIONAL FREQUENCIES',1)[1]
            values=re.findall(r'^\s*\d+:\s+('+FLOAT+r') cm\*\*-1',spectrum,re.M)
            if len(values)==3*len(z):
                scale=re.search(r'Scaling factor for frequencies\s*=\s*('+FLOAT+r')',spectrum)
                freqs=[float(v)/(float(scale[1]) if scale else 1.) for v in values]
        kind='optimized' if settled else 'single_point'
        if 'opt' in cmd.lower() and not settled:
            kind='other_final_geometry'
        if 'ccsd(t)-f12/ri' in level:
            cc=re.findall(r'F12-E\(CCSD\(T\)\)\s*\.+\s*('+FLOAT+r')',section)
            if not cc or abs(float(cc[-1])-float(e[1]))>1e-6:
                issues['unverified_F12_final_energy']+=1
                continue
        basename=re.search(r'^Basename:\s*(\S+)',section,re.M)
        yield obs(key,i,kind,level,q,m,z,r,float(e[1]),freqs,basename[1] if basename else Path(name).stem,cmd)


def molpro(key,s,name,issues):
    """Use the printed nuclear coordinates (bohr) and the correlated wavefunction."""
    value=re.findall(r'^\s*!(?:RHF-|UHF-)?(?:U|R)?CCSD\(T\)-F12b (?:total )?energy\s+('+FLOAT+r')',s,re.M|re.I)
    table=re.findall(r'(?:U|R)?CCSD\(T\)-F12B/(\S+) energy=\s*('+FLOAT+r')',s,re.I)
    states=set(re.findall(r'\{\s*(?:u|r)?ccsd\(t\)-f12b\s*;\s*wf\s*,\s*(\d+)\s*,\s*\d+\s*,\s*(\d+)',s,re.I))
    if len(value)!=1 or len(table)!=1 or len(states)!=1 or abs(float(value[0])-float(table[0][1]))>1e-7 or 'Molpro calculation terminated' not in s or re.search(r'no convergence|failed to converge',s,re.I):
        issues['unverified_Molpro_method_or_convergence']+=1
        return
    geom=re.search(r'NR\s+ATOM\s+CHARGE\s+X\s+Y\s+Z\s*\n(.*?)(?:\n\s*\n|\Z)',s,re.S)
    if not geom:
        issues['missing_Molpro_coordinates']+=1
        return
    a=[l.split() for l in geom[1].strip().splitlines()]
    z=[ELEMENT[v[1].capitalize()] for v in a]
    r=np.array([v[3:6] for v in a],float)*BOHR_A
    ne,spin=map(int,states.pop())
    level='Molpro/'+re.search(r'^\s*!([^\s]+).*?-F12b',s,re.I|re.M)[0].strip().lstrip('!').lower()+'/'+table[0][0].lower()+'/gas'
    yield obs(key,0,'single_point',level,sum(z)-ne,spin+1,z,r,float(value[0]),None,Path(name).stem,'Molpro printed F12b total')


def run():
    with sqlite3.connect(CACHE/'files.sqlite') as index, sqlite3.connect(CACHE/'labels.sqlite') as dest:
        dest.executescript("""
        CREATE TABLE IF NOT EXISTS labels (hash TEXT,stage INTEGER,kind TEXT,level TEXT,q INTEGER,m INTEGER,z BLOB,r BLOB,energy REAL,freq TEXT,stem TEXT,command TEXT,n INTEGER,PRIMARY KEY(hash,stage));
        CREATE TABLE IF NOT EXISTS visit (hash TEXT PRIMARY KEY, status TEXT);
        """)
        jobs=index.execute("""SELECT f.hash,min(f.name),o.local FROM files f JOIN objects o ON o.hash=f.hash
                              WHERE f.kind='science' AND (lower(f.name) LIKE '%.out' OR lower(f.name) LIKE '%.log') GROUP BY f.hash
                              ORDER BY CASE WHEN min(f.name) LIKE '%Dirac%' THEN 0 WHEN min(f.name) LIKE '%CCSDT%' THEN 1 ELSE 2 END""").fetchall()
        issues=Counter()
        for i,(key,name,path) in enumerate(jobs):
            if dest.execute('SELECT 1 FROM visit WHERE hash=?',(key,)).fetchone():
                continue
            s=Path(path).read_text(errors='replace')
            previous=issues.copy()
            try:
                if 'CARTESIAN COORDINATES (ANGSTROEM)' in s:
                    records=orca(key,s,name,issues)
                elif 'PROGRAM SYSTEM MOLPRO' in s.upper() or 'Molpro calculation terminated' in s:
                    records=molpro(key,s,name,issues)
                else:
                    records=[]; issues['other_output']+=1
                dest.executemany(BIND,records)
            except (ValueError,KeyError,IndexError) as exc:
                issues['invalid_record_'+str(exc)[:60]]+=1
            dest.execute('INSERT INTO visit VALUES(?,?)',(key,json.dumps(issues-previous)))
            if i%250==0:
                dest.commit()
                print(i,len(jobs),dest.execute('SELECT count(*) FROM labels').fetchone()[0],flush=True)
        dest.commit()
        report=list(dest.execute('SELECT kind,level,q,m,count(*),count(freq) FROM labels GROUP BY kind,level,q,m ORDER BY count(*) DESC'))
        (CACHE/'label_summary.json').write_text(json.dumps(report,indent=2))
        print('finished',len(jobs),'outputs;',dest.execute('SELECT count(*) FROM labels').fetchone(),issues,flush=True)


if __name__=='__main__':
    run()
