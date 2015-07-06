
pub fn parallel_for<F, T>(col: &[T], f: F) {
    let cores = 8;
    let per_core = col.len() / cores;
    let remainder = col.len() % cores;
    for i in 0..cores-1 {
        let start = i * per_core;
        let end = (i + 1) * per_core;
        std::thread::spawn(move || {
    		for i in start..end {
                f(col[i]);
    		}
    	});
    }
    let start = (cores-1) * per_core;
    let end = (cores) * per_core + remainder;
    for i in start..end {
        f(col[i]);
    }
}
