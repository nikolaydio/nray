
trait Spectrum {
    
}

macro_rules! spectrum_def {
    ($chans:expr, $name:ident) => {
        pub struct $name {
            chans: [f32;$chans]
        }
        impl Spectrum for $name {

        }

    };
}

spectrum_def!(3, RGBSpectrum);
