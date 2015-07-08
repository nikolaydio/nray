
trait Spectrum {

}
use std::ops::{Add,Mul,Div,Sub};
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
            pub fn new(c: f32) -> $name {
                $name { chans: [c; $chans] }
            }
            pub fn plausable(&self) -> bool {
                self.chans.iter().all(|&c| c <= 1.0f32 && c >= 0.0f32)
            }
            pub fn clamp(&self, min: f32, max: f32) -> RGBSpectrum {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] = v.chans[i].max(min).min(max);
                }
                v
            }
            pub fn sqrt(&self) -> RGBSpectrum {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] = v.chans[i].sqrt();
                }
                v
            }
            pub fn to_sRGB(&self) -> (u8, u8, u8) {
                if !self.plausable() {
                    //println!("Attemp to plot an inplausable spectrum.");
                }
                let cl = self.clamp(0.0f32, 1.0f32) * 255.0f32;
                (cl.chans[0] as u8, cl.chans[1] as u8, cl.chans[2] as u8)
            }
            pub fn from_sRGB(r: u8, g: u8, b: u8) -> $name {
                let mut v = $name::new(0.0f32);
                v.chans[0] = (r as f32) / 255.0f32;
                v.chans[1] = (g as f32) / 255.0f32;
                v.chans[2] = (b as f32) / 255.0f32;
                v
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
        impl Add<f32> for $name {
            type Output = $name;

            fn add(self, _rhs: f32) -> $name {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] += _rhs;
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
        impl Div for $name {
            type Output = $name;

            fn div(self, _rhs: $name) -> $name {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] /= _rhs.chans[i];
                }
                v
            }
        }
        impl Sub for $name {
            type Output = $name;

            fn sub(self, _rhs: $name) -> $name {
                let mut v = $name { chans: self.chans };
                for i in 0..$chans {
                    v.chans[i] -= _rhs.chans[i];
                }
                v
            }
        }
    };
}

spectrum_def!(3, RGBSpectrum);
