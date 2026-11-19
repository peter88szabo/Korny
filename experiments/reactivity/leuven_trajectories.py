"""Collect energy-labelled ORCA XYZ frames from the private recursive inventory.

Use the job name in the XYZ comment and the exact input/output directory to
bind a trajectory to its verified reference level and electronic state. Store
per-frame geometry and Hartree energy. Unlabelled geometries and ambiguous
assignments keep their original provenance in the file inventory.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json
import re
import sqlite3
import numpy as np
from experiments.reactivity.leuven_files import CACHE
from experiments.reactivity.leuven_labels import ELEMENT,FLOAT

COMMENT=re.compile(r'^Coordinates from ORCA[- ]job\s+(\S+)\s+E\s+('+FLOAT+r')\s*$',re.I)


def directory(name):
    return name[:-len(name.split('::')[-1].split('/')[-1])]


def run():
    db=sqlite3.connect(CACHE/'trajectories.sqlite')
    db.execute('ATTACH ? AS idx',(str(CACHE/'files.sqlite'),))
    db.execute('ATTACH ? AS lbl',(str(CACHE/'labels.sqlite'),))
    db.executescript("""
    CREATE TABLE IF NOT EXISTS frames (file TEXT,frame INTEGER,level TEXT,q INTEGER,m INTEGER,z BLOB,r BLOB,e_hartree REAL,PRIMARY KEY(file,frame));
    CREATE TABLE IF NOT EXISTS seen (file TEXT PRIMARY KEY,status TEXT);
    """)
    stages=defaultdict(set)
    for name,stem,level,q,m in db.execute("SELECT f.name,l.stem,l.level,l.q,l.m FROM lbl.labels l JOIN idx.files f USING(hash) WHERE l.level LIKE 'ORCA%'"):
        stages[directory(name),stem].add((level,q,m))
    candidates=defaultdict(list)
    for h,name,path in db.execute("SELECT f.hash,f.name,o.local FROM idx.files f JOIN idx.objects o USING(hash) WHERE lower(f.name) LIKE '%.xyz' OR lower(f.name) LIKE '%.allxyz'"):
        candidates[h].append((name,path))
    counts=Counter()
    for h,locations in candidates.items():
        if db.execute('SELECT 1 FROM seen WHERE file=?',(h,)).fetchone():continue
        dirs={directory(name) for name,_ in locations}
        n=0;status='no_ORCA_electronic_energy_comment'
        with Path(locations[0][1]).open(errors='replace') as f:
            try:
                for line in f:
                    if not line.strip():continue
                    nat=int(line);comment=next(f);match=COMMENT.fullmatch(comment.strip())
                    if not match:break
                    assignments=set().union(*(stages.get((d,match[1]),set()) for d in dirs))
                    if len(assignments)!=1:
                        status='ambiguous_or_missing_job_stage';break
                    lev,q,m=assignments.pop()
                    atoms=[next(f).split() for _ in range(nat)]
                    z=bytes(ELEMENT[a[0]] for a in atoms);r=np.array([a[1:4] for a in atoms],dtype='<f8')
                    e=float(match[2])
                    if not np.isfinite(r).all() or not np.isfinite(e) or (sum(z)-q-m+1)%2:raise ValueError
                    db.execute('INSERT OR REPLACE INTO frames VALUES(?,?,?,?,?,?,?,?)',(h,n,lev,q,m,z,r.tobytes(),e))
                    n+=1;status='matched'
            except (IndexError,ValueError,StopIteration,KeyError):
                status='malformed_or_other_XYZ'
        db.execute('INSERT INTO seen VALUES(?,?)',(h,status))
        counts[status]+=1
        if sum(counts.values())%500==0:db.commit()
    db.commit()
    report=dict(frames=db.execute('SELECT count(*) FROM frames').fetchone()[0],matched_files=db.execute('SELECT count(DISTINCT file) FROM frames').fetchone()[0],
                sources=dict(db.execute('SELECT status,count(*) FROM seen GROUP BY status')),levels=dict(db.execute('SELECT level,count(*) FROM frames GROUP BY level')))
    (CACHE/'trajectory_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':run()
