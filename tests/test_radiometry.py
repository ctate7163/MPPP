"""
The 16-bit product is linear and radiometrically traceable to the PDS RAD product.

The chain is  DN --(RADIANCE_SCALING_FACTOR, RADIANCE_OFFSET)--> radiance  --(/ scale_zenith)--> normalised radiance
--(x white balance x brightness)--> colour-processed radiance  --(x scale_rad_to_int16, round)--> uint16, and every
factor is recorded in the image's ``meta`` (the manifest), so that a user can invert the product back to the
PDS radiance.  These tests pin that chain on the two test products and on synthetic data.
"""
import numpy as np
import pytest

from conftest import NLF, ZL0, config_selection_off, needs_data, synthetic_waypoints


def _radiance_from_pds(path):
    """(radiance, valid) straight from the PDS RAD product, as mppp.image reads it."""
    from mppp.labels import label_float, read_pds
    L, dn = read_pds(path)
    scale = label_float(L, "DERIVED_IMAGE_PARMS.RADIANCE_SCALING_FACTOR", "IMAGE.SCALING_FACTOR")
    offset = label_float(L, "DERIVED_IMAGE_PARMS.RADIANCE_OFFSET", "IMAGE.OFFSET", default=0.0)
    invalid = label_float(L, "IMAGE.INVALID_CONSTANT", default=0.0)
    valid = (dn != invalid) if dn.ndim == 2 else np.all(dn != invalid, axis=2)
    return dn.astype(np.float64) * scale + offset, valid, scale, offset


def test_quantise_is_linear_and_invertible():
    """uint16 = round(rad x s16): slope s16, zero intercept, residual <= half a count, zero reserved for invalid."""
    from mppp.radiometry import quantise
    rng = np.random.default_rng(1)
    rad = rng.uniform(0.0, 0.3, size=(40, 50, 3))
    valid = np.ones((40, 50), bool)
    valid[0, :] = False
    cfg = {"scale_rad_to_int16": 2e5, "scale_int8_to_int16": 64, "offset_int8_to_int16": 0.0, "apply_gamma": False}
    im16, im8 = quantise(rad, valid, cfg)
    assert im16.dtype == np.uint16 and im8.dtype == np.uint8
    assert np.all(im16[0] == 0) and np.all(im16[1:] >= 1)
    back = im16[1:].astype(np.float64) / 2e5
    assert np.abs(back - rad[1:]).max() <= 0.5 / 2e5 + 1e-12          # one half-count of quantisation
    x, y = rad[1:].ravel(), im16[1:].ravel().astype(np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    assert abs(slope - 2e5) / 2e5 < 1e-6 and abs(intercept) < 0.6
    # gamma is only ever applied to the 8-bit product
    cfg_g = {**cfg, "apply_gamma": True, "gamma": 2.0}
    im16_g, im8_g = quantise(rad, valid, cfg_g)
    assert np.array_equal(im16_g, im16) and not np.array_equal(im8_g, im8)


@needs_data
@pytest.mark.parametrize("img", [ZL0, NLF], ids=["ZL0", "NLF"])
def test_png16_traces_back_to_pds_radiance(img):
    """With the colour processing at unity (no tau correction, white balance 1, brightness 1) the 16-bit product is
    round(radiance x scale) of the PDS RAD values, pixel for pixel; with the default processing the recorded
    factors invert it back to the PDS radiance within one count."""
    from mppp import MPPPImage
    s16 = 2e5
    unity = config_selection_off({"masking": {"infer_mask": False, "mask_invalid": True},
                                  "radiometry": {"apply_tau_correction": False},
                                  "color": {"enhance": False, "brightness_by_family": {"N": 1.0, "Z": 1.0}, "scale_rad_to_int16": s16},
                                  "resize": {"apply_padding": False}})
    im = MPPPImage(img, unity, synthetic_waypoints())
    rad, valid, scale, offset = _radiance_from_pds(img)
    assert im.meta["radiance_scaling_factor"] == scale and im.meta["radiance_offset"] == offset
    assert im.meta["white_balance"] == [1.0, 1.0, 1.0] and im.meta["brightness"] == 1.0
    assert im.meta["scale_zenith"] is None
    m = (im.mask_valid > 0)
    assert im.image_int16.shape[:2] == rad.shape[:2] and m.shape == valid.shape
    if rad.ndim == 2:
        from mppp.labels import label_get, read_pds
        L, _ = read_pds(img, load_image=False)
        cfa = str(label_get(L, "INSTRUMENT_STATE_PARMS.CFA_TYPE", default="")).upper()
        bayer = str(label_get(L, "INSTRUMENT_STATE_PARMS.BAYER_METHOD", default="")).upper()
        if bayer == "RAW_BAYER" or (not bayer and "BAYER" in cfa):
            # a raw-Bayer product is demosaiced (Malvar 2004, a linear filter): apply the same filter here
            import colour_demosaicing
            rad = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(rad, cfa.replace("BAYER_", "") or "RGGB")
        else:
            rad = np.repeat(rad[..., None], 3, axis=2)
    expected = np.clip(np.rint(rad * s16), 1, 65535)
    v = m & valid
    assert np.array_equal(im.image_int16[v], expected[v].astype(np.uint16))
    assert np.all(im.image_int16[~m] == 0)
    # the default processing: invert with the factors the manifest records
    default = config_selection_off({"masking": {"infer_mask": False}, "resize": {"apply_padding": False}})
    im2 = MPPPImage(img, default, synthetic_waypoints())
    meta = im2.meta
    factor = np.asarray(meta["white_balance"]) * meta["brightness"] * s16 / (meta["scale_zenith"] or 1.0)
    recovered = im2.image_int16.astype(np.float64) / factor.reshape(1, 1, 3)
    lin = rad * factor.reshape(1, 1, 3)
    v2 = (im2.mask_valid > 0) & valid & np.all((lin >= 1) & (lin <= 65535), axis=2)   # clipped pixels cannot invert
    err_counts = np.abs(recovered - rad)[v2] * factor.reshape(1, 3)                 # per channel, in uint16 counts
    assert err_counts.max() <= 0.5 + 1e-6
    assert np.isfinite(recovered[v2]).all()


@needs_data
def test_png16_file_round_trip(tmp_path):
    """The written PNG16 holds the uint16 product unchanged (lossless), with the mask in the alpha channel."""
    from mppp import MPPPImage, process_images
    cfg = config_selection_off({"masking": {"infer_mask": False}, "export": {"formats": ["PNG16"],
                                                                            "store_mask_in_alpha": True},
                                "resize": {"apply_padding": False}})
    man = process_images([ZL0], tmp_path, cfg, synthetic_waypoints())
    rec = man["images"][0]
    out = tmp_path / rec["outputs"]["PNG16"]
    import cv2
    png = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert png.dtype == np.uint16 and png.shape[2] == 4
    im = MPPPImage(ZL0, cfg, synthetic_waypoints())
    assert np.array_equal(png[..., :3][..., ::-1], im.image_int16)      # OpenCV reads BGR
    assert np.array_equal(png[..., 3] > 0, im.mask > 0)
