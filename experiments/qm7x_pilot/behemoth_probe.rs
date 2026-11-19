// One geometry per line; each atom is symbol Z x_bohr y_bohr z_bohr hRAT.
use std::io::{self, BufRead};
use Behemoth::dispersion::dft_d3::{self, D3Config, D3Damping};
use Behemoth::dispersion::ts_pair::disperison_damping_models::{
    TsDampingKind, TsDampingParameters,
};
use Behemoth::dispersion::ts_pair::ts_dispersion_corr::ts_pair_dispersion_gradient_from_atomic;
use Behemoth::utils::free_atoms::load_free_atoms;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let free = load_free_atoms()?;
    let ts = TsDampingParameters::new(TsDampingKind::Fermi, 0.96, 20.);
    let d3 = D3Config::new("pbe0", D3Damping::Rational).with_three_body(true);
    for line in io::stdin().lock().lines() {
        let line = line?;
        let words: Vec<_> = line.split_whitespace().collect();
        if words.is_empty() || words.len() % 6 != 0 {
            return Err("invalid geometry".into());
        }
        let (mut z, mut xyz, mut alpha, mut c6, mut radius) =
            (vec![], vec![], vec![], vec![], vec![]);
        for a in words.chunks_exact(6) {
            let v: f64 = a[5].parse()?;
            let p = free.get(a[0]).ok_or("unknown element")?;
            z.push(a[1].parse()?);
            xyz.push([a[2].parse()?, a[3].parse()?, a[4].parse()?]);
            alpha.push(v * p.alpha0);
            c6.push(v * v * p.c6);
            radius.push(v.cbrt() * p.r_vdw);
        }
        let t =
            ts_pair_dispersion_gradient_from_atomic(&xyz, &alpha, &c6, &radius, None, None, &ts)?;
        let g = dft_d3::evaluate(&z, &xyz, &d3)?;
        print!("{} {}", t.energy, g.energy);
        for x in t.gradient.iter().chain(g.gradient.iter().flatten()) {
            print!(" {x}");
        }
        println!();
    }
    Ok(())
}
