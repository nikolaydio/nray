
trait Spectrum {

}
use std::ops::{Add,Mul,Div};
use std::clone::Clone;
macro_rules! spectrum_def {
    ($chans:expr, $name:ident) => {
        #[derive(Copy, Clone)]
        pub struct $name {
            pub chans: [f32;$chans]
        }
        impl $name {
            pub fn white() -> $name {
                $name { chans: [1.0f32; $chans]}
            }
            pub fn black() -> $name {
                $name { chans: [0.0f32; $chans]}
            }
        }
        impl Add for $name {
            type Output = $name;

            fn add(self, _rhs: $name) -> $name {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] += _rhs.chans[i];
                }
                v
            }
        }
        impl Mul for $name {
            type Output = $name;

            fn mul(self, _rhs: $name) -> $name {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] *= _rhs.chans[i];
                }
                v
            }

        }
        impl Mul<f32> for $name {
            type Output = $name;

            fn mul(self, _rhs: f32) -> $name {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] *= _rhs;
                }
                v
            }

        }
        impl Div<f32> for $name {
            type Output = $name;

            fn div(self, _rhs: f32) -> $name {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] /= _rhs;
                }
                v
            }

        }
    };
}

spectrum_def!(3, RGBSpectrum);
