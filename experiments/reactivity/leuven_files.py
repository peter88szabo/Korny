"""Index the private Ready_calc tree, including compressed and nested snapshots.

Original data are opened read-only. Store provenance and archive-only scientific
files in the ignored training directory. A content hash links byte-identical
copies without discarding their filenames, job context, or source archive.
"""
from hashlib import sha256
from pathlib import Path
import argparse
import json
import sqlite3
import tarfile

ROOT = Path("/home/peter/Dropbox/Research_Leuven/LIM1_reevol/Ready_calc")
CACHE = Path(__file__).resolve().parents[2] / "dataset_for_training" / "Leuven_isoprene"
# Keep geometries, programs' text inputs/outputs, derivatives and the author's
# tables. Binary wavefunctions and scheduler wrappers have no independent labels.
EXT = set(".out .log .inp .engrad .hess .carthess .xyz .allxyz .dat .csv .tsv .ods .xlsx .txt .prop .opt .interp .imagfreq .energy .energies .traj .trj .xml .com .gjf .fchk .grad .freq .molden".split())


def digest_file(path):
    with path.open('rb') as stream:
        return sha256(stream.read()).hexdigest() if path.stat().st_size < 2**20 else stream_digest(stream)


def stream_digest(stream):
    hash_ = sha256()
    while block := stream.read(2**20):
        hash_.update(block)
    return hash_.hexdigest()


def category(path):
    name = path.name.lower()
    if name.endswith(('.tar.gz', '.tgz', '.tar', '.tar.bz2', '.tar.xz')):
        return 'archive'
    if name.startswith('slurm-') and name.endswith('.out'):
        return 'scheduler'
    return 'science' if path.suffix.lower() in EXT else 'auxiliary'


def run(root, cache):
    cache.mkdir(exist_ok=True, parents=True)
    blobs = cache/'objects'
    blobs.mkdir(exist_ok=True)
    db = sqlite3.connect(cache/'files.sqlite')
    db.executescript("""
        CREATE TABLE IF NOT EXISTS files (name TEXT PRIMARY KEY, kind TEXT, size INTEGER, hash TEXT);
        CREATE TABLE IF NOT EXISTS objects (hash TEXT PRIMARY KEY, local TEXT);
        CREATE TABLE IF NOT EXISTS archives (name TEXT PRIMARY KEY, status TEXT);
        CREATE INDEX IF NOT EXISTS files_hash ON files(hash);
    """)
    archives = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        key = str(path.relative_to(root))
        kind = category(path)
        if kind == 'archive':
            archives.append((key,path))
        if db.execute('SELECT 1 FROM files WHERE name=?',(key,)).fetchone():
            continue
        h = digest_file(path) if kind=='science' else None
        if h:
            db.execute('INSERT OR IGNORE INTO objects VALUES(?,?)',(h,str(path)))
        db.execute('INSERT INTO files VALUES(?,?,?,?)',(key,kind,path.stat().st_size,h))
    db.commit()
    print('loose inventory',db.execute('SELECT kind,count(*) FROM files WHERE instr(name,"::")=0 GROUP BY kind').fetchall(),flush=True)
    # Tar members are never extracted using the member name; content-addressed
    # paths also prevent a broken snapshot from overwriting another calculation.
    # Resume a nested archive even if its containing archive was already
    # committed before an interrupted run. The list also grows as new nested
    # members are discovered in a fresh scan.
    archives.extend((n,Path(p)) for n,p in db.execute(
        "SELECT f.name,o.local FROM files f JOIN objects o USING(hash) WHERE f.kind='archive' AND instr(f.name,'::')>0"))
    for key,path in archives:
        if db.execute("SELECT 1 FROM archives WHERE name=? AND status='ok'",(key,)).fetchone():
            continue
        try:
            with tarfile.open(path,'r|*') as stream:
                for member in stream:
                    if not member.isfile():
                        continue
                    rel=key+'::'+member.name
                    kind=category(Path(member.name))
                    h=None
                    if kind in ('science','archive'):
                        tmp=cache/'current.member.part'
                        digest=sha256()
                        with tmp.open('wb') as f:
                            source=stream.extractfile(member)
                            while block:=source.read(2**20):
                                f.write(block)
                                digest.update(block)
                        h=digest.hexdigest()
                        row=db.execute('SELECT local FROM objects WHERE hash=?',(h,)).fetchone()
                        if row:
                            tmp.unlink()
                            local=Path(row[0])
                        else:
                            local=blobs/h
                            tmp.replace(local)
                            db.execute('INSERT INTO objects VALUES(?,?)',(h,str(local)))
                        if kind=='archive':
                            archives.append((rel,local))
                    db.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?)',(rel,kind,member.size,h))
            status='ok'
        except (OSError,tarfile.TarError,EOFError) as e:
            status=str(e)
        db.execute('INSERT OR REPLACE INTO archives VALUES(?,?)',(key,status))
        db.commit()
        print(key,status,flush=True)
    counts=dict(db.execute('SELECT kind,count(*) FROM files GROUP BY kind'))
    faults=list(db.execute("SELECT * FROM archives WHERE status!='ok'"))
    (cache/'inventory.json').write_text(json.dumps(dict(counts=counts,archive_errors=faults),indent=2))
    print(counts, 'archive errors',len(faults),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--cache',type=Path,default=CACHE)
    args=parser.parse_args()
    run(args.root,args.cache)
