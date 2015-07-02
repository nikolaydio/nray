
//define operations
//1. Generate samples
// (samples_per_pixel, area) + sampler_function -> plane_coordinates
//2. Transform coords
// (plane_coordinates, resolution) + generic_trnfm_fn -> plane_coordinates_ndc
//3. Calculate world space rays
// (plane_coordinates_ndc, view_matrix, hfov, vfov) + cam_function -> primary_rays
//4. Intersection(4. rays = primary_rays)
// (scene_geometry, rays) + traversal + intersection -> diff_geoms
//5. Shading sub-process
// (diff_geoms, materials, throughput) + path_integrator -> (secondary_rays, throughput)
//6. loop 2 times + round robin or until emissive object
// Intersection(rays = secondary_rays)
// Shading sub-process
//7. Calculate spectrums
// (throughput, emission) + multiplication -> SPDs
//8. Average over pixels
// (SPDs) + avg -> SPD_pixel
//9. Map to XYZ
// (SPD_pixel, XYZ graphs) -> XYZ_pixel
//10. XYZ to sRGB
// (XYZ_pixel, sRGB graphs) -> sRGB_pixel
