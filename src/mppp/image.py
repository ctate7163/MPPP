"""
``MPPPImage`` — one PDS image product taken through the MPPP pre-processing:

    PDS IMG -> radiance -> RGB -> opacity/illumination normalisation -> white
    balance -> mask -> full-detector-frame padding -> (optional) undistortion
    -> 16-bit linear / 8-bit products, with intrinsics, pose prior and metadata.

Processing order and arithmetic follow the validated pre-package ``image.py``;
differences are listed in CHANGELOG.md.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from . import radiometry
from .camera import CAHVOR, Intrinsics, Pose, intrinsics_from_metashape_xml, pose_from_label
from .config import load_config
from .filenames import parse_filename
from .labels import first, label_float, label_get, read_pds
from .paths import params_dir, resolve_resource
from .waypoints import lonlat_of, offset_lonlat, waypoint_for_site_drive

PathLike = Union[str, Path]

# Full detector frame (width, height) at full resolution, by camera family.
FULL_FRAME = {"N": (5120, 3840), "F": (5120, 3840), "R": (5120, 3840), "Z": (1648, 1200)}
LANDING_SITE = 3            # M2020 site index of the landing frame used as world origin


def _pad(arr: np.ndarray, left: int, right: int, top: int, bottom: int) -> np.ndarray:
    pads = ((top, bottom), (left, right)) + (((0, 0),) if arr.ndim == 3 else ())
    return np.pad(arr, pads, mode="constant")


class MPPPImage:
    """
    Parameters
    ----------
    img_path : PDS ``.IMG`` file (the file name is parsed and never changed).
    config   : dict, path to JSON, or None for defaults.
    waypoints: GeoJSON dict from :func:`mppp.load_waypoints`, or None.  Without
               waypoints the pose prior is expressed in the image's own site
               frame and no Mars lat/lon is available.

    Main results
    ------------
    image_int16 / image_int8 : H x W x 3, zero = invalid pixel
    mask_valid, mask         : uint8, 255 = valid / include in reconstruction
    intrinsics (Intrinsics), pose (Pose), geo (dict), meta (dict)
    """

    def __init__(self, img_path: PathLike, config: Optional[Union[PathLike, Dict[str, Any]]] = None,
                 waypoints: Optional[Dict[str, Any]] = None):
        self.path = Path(img_path)
        self.config = load_config(config)
        cfg = self.config
        self.fn = parse_filename(self.path)
        self.log: list = []

        self.label, dn = read_pds(self.path)
        self._identify()
        self._camera_from_label()
        self._position(waypoints)
        self._illumination()

        rad, self.mask_valid = self._radiance_rgb(dn)
        if cfg["radiometry"]["apply_tau_correction"]:
            rad /= self.scale_zenith
        rad *= self.white_balance.reshape(1, 1, 3)

        # keep valid pixels strictly positive so that zero uniquely flags "invalid"
        eps = 1.0 / float(cfg["color"]["scale_rad_to_int16"])
        valid = self.mask_valid > 0
        rad = np.where(valid[..., None], np.maximum(rad, eps), 0.0)

        self.mask = self.mask_valid.copy()
        self.mask_card: Optional[Dict[str, Any]] = None
        self.mask_inference_skipped = False
        if cfg["masking"]["infer_mask"]:
            from .config import parse_stations
            if (self.site, self.drive) in parse_stations(cfg["masking"].get("skip_inference_at")):
                self.mask_inference_skipped = True
                self._say(f"mask inference off at site {self.site} drive {self.drive} (masking.skip_inference_at): "
                          f"only invalid pixels are masked")
            else:
                self._infer_mask(rad)
        if not cfg["masking"]["mask_invalid"]:
            warnings.warn("masking.mask_invalid=False: invalid pixels stay flagged in mask_valid only.")

        rad = self._pad_to_full_frame(rad)
        self._apply_xml_intrinsics()
        rad = self._extra_padding(rad)
        self._apply_static_mask()
        self.undistorted = bool(cfg["resize"]["undistort"])
        if self.undistorted:
            rad = self._undistort(rad)

        self.image_rad = rad.astype(np.float32)
        self.image_int16, self.image_int8 = radiometry.quantise(rad, self.mask_valid > 0, cfg["color"])
        self.height, self.width = self.image_int16.shape[:2]
        assert (self.width, self.height) == (self.intrinsics.width, self.intrinsics.height)

    # ------------------------------------------------------------------ steps
    def _say(self, msg: str) -> None:
        self.log.append(msg)
        if self.config.get("verbose"):
            print(f"[mppp] {self.fn.stem}: {msg}")

    def _identify(self) -> None:
        rmc = label_get(self.label, "ROVER_MOTION_COUNTER")
        if rmc is None:
            raise ValueError(f"No ROVER_MOTION_COUNTER in the label of {self.path.name}")
        self.site, self.drive = int(rmc[0]), int(rmc[1])
        self.sol = self.fn.sol
        if self.fn.site is not None and (self.fn.site, self.fn.drive) != (self.site, self.drive):
            self._say(f"file name site/drive {self.fn.site}/{self.fn.drive} != label {self.site}/{self.drive}; label used")

    def _camera_from_label(self) -> None:
        gcm = label_get(self.label, "GEOMETRIC_CAMERA_MODEL")
        if gcm is None:
            raise ValueError("No GEOMETRIC_CAMERA_MODEL in the label")
        ref = str(gcm.get("REFERENCE_COORD_SYSTEM_NAME", "ROVER_NAV_FRAME"))
        if ref != "ROVER_NAV_FRAME":
            raise ValueError(f"Camera model is expressed in {ref}; only ROVER_NAV_FRAME is supported")
        w, h = int(label_get(self.label, "IMAGE.LINE_SAMPLES")), int(label_get(self.label, "IMAGE.LINES"))
        self.cahvor = CAHVOR.from_label(gcm)
        self.intrinsics, self.R_cam_rnav = self.cahvor.decompose(w, h)
        self.intrinsics_label = self.intrinsics
        self.native_size = (w, h)

    def _position(self, waypoints: Optional[Dict[str, Any]]) -> None:
        rcs = label_get(self.label, "ROVER_COORDINATE_SYSTEM")
        offset_ned = np.asarray(rcs["ORIGIN_OFFSET_VECTOR"], dtype=np.float64)   # rover in site frame
        quat = rcs["ORIGIN_ROTATION_QUATERNION"]
        self.geo: Dict[str, Any] = {"lon_east_deg": None, "lat_deg": None, "elev_geoid_m": None}

        use_wp = waypoints is not None and self.config["camera_model"]["extrinsics_from_waypoints"]
        if not use_wp:
            origin, frame, src = offset_ned, f"site{self.site}_enu", "label ORIGIN_OFFSET_VECTOR (site frame)"
        else:
            wp0 = waypoint_for_site_drive(waypoints, LANDING_SITE, 0) or waypoints["features"][0]
            p0 = wp0.get("properties", wp0)
            wp = waypoint_for_site_drive(waypoints, self.site, self.drive, exact=True)
            extra = np.zeros(3)
            if wp is None:                                   # site origin + telemetry offset
                wp = waypoint_for_site_drive(waypoints, self.site, 0, exact=True)
                if wp is None:
                    src_info = waypoints.get("_mppp_source", {})
                    hint = (" The packaged waypoint snapshot may predate this site: try "
                            "mppp.load_waypoints(refresh=True).") if src_info.get("packaged_snapshot") else ""
                    raise ValueError(f"Site {self.site} has no drive-0 waypoint; cannot place "
                                     f"{self.path.name} in the landing frame.{hint}")
                extra = offset_ned
                src = f"waypoint site {self.site} drive 0 + label ORIGIN_OFFSET_VECTOR"
            else:
                src = f"waypoint site {self.site} drive {self.drive} (exact)"
            p = wp.get("properties", wp)
            origin = np.array([float(p["northing"]) - float(p0["northing"]) + extra[0],
                               float(p["easting"]) - float(p0["easting"]) + extra[1],
                               -(float(p["elev_geoid"]) - float(p0["elev_geoid"])) + extra[2]])
            frame = f"site{LANDING_SITE}_enu"
            self.waypoint = p
            ll = lonlat_of(wp)
            if ll is not None:
                lon, lat = offset_lonlat(ll[0], ll[1], d_east=extra[1], d_north=extra[0])
                self.geo.update(lon_east_deg=lon, lat_deg=lat)
            self.geo["elev_geoid_m"] = float(p["elev_geoid"]) - extra[2]

        self.pose: Pose = pose_from_label(self.R_cam_rnav, self.cahvor.C, quat, origin, frame, src)
        if self.geo["elev_geoid_m"] is not None:             # camera, not rover origin
            self.geo["elev_geoid_m"] += float(self.pose.C[2] - (-origin[2]))
        self._say(f"pose prior in {frame} from {src}")

    def _illumination(self) -> None:
        L, r = self.label, self.config["radiometry"]
        self.L_s = label_float(L, "SOLAR_LONGITUDE")
        self.solar_az = label_float(L, "SITE_DERIVED_GEOMETRY_PARMS.SOLAR_AZIMUTH")
        self.solar_el = label_float(L, "SITE_DERIVED_GEOMETRY_PARMS.SOLAR_ELEVATION")
        self.LMST = first(label_get(L, "LOCAL_MEAN_SOLAR_TIME"))
        self.LTST = first(label_get(L, "LOCAL_TRUE_SOLAR_TIME"))
        self.tau_ref = float(r["tau_reference"])
        self.tau_estimated = self.scale_zenith = self.solar_mu = None
        if r["apply_tau_correction"]:
            if self.L_s is None or self.solar_el is None:
                raise ValueError("tau correction needs SOLAR_LONGITUDE and SOLAR_ELEVATION in the label")
            table = resolve_resource(r["tau_table"], params_dir())
            self.tau_estimated = radiometry.interpolate_table(table, self.L_s)
            self.scale_zenith, self.solar_mu = radiometry.zenith_scale(
                self.solar_el, self.tau_estimated, self.tau_ref, r["zenith_min"])

        c = self.config["color"]
        if not c["enhance"]:
            wb = [1.0, 1.0, 1.0]
        elif self.fn.filter == "M":
            wb = c["white_balance_vce"]
        elif self.fn.family in ("N", "F", "R"):
            wb = c["white_balance_ecam"]
        elif self.fn.family == "Z":
            wb = c["white_balance_zcam"]
        else:
            wb = [1.0, 1.0, 1.0]
        self.white_balance = np.asarray(wb, dtype=np.float64)

    def _radiance_rgb(self, dn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = self.label
        scale = label_float(L, "DERIVED_IMAGE_PARMS.RADIANCE_SCALING_FACTOR", "IMAGE.SCALING_FACTOR")
        offset = label_float(L, "DERIVED_IMAGE_PARMS.RADIANCE_OFFSET", "IMAGE.OFFSET", default=0.0)
        if scale is None:
            raise ValueError(f"{self.path.name}: no RADIANCE_SCALING_FACTOR — is this a RAD product?")
        self.scale_dn_to_rad, self.offset_dn_to_rad = scale, offset
        self.dtype_original = str(dn.dtype)

        invalid_dn = label_float(L, "IMAGE.INVALID_CONSTANT", default=0.0)
        valid = (dn != invalid_dn) if dn.ndim == 2 else np.all(dn != invalid_dn, axis=2)
        rad = radiometry.dn_to_radiance(dn, scale, offset)
        if rad.ndim == 2:
            cfa = str(label_get(L, "INSTRUMENT_STATE_PARMS.CFA_TYPE", default="")).upper()
            bayer = str(label_get(L, "INSTRUMENT_STATE_PARMS.BAYER_METHOD", default="")).upper()
            if self.fn.family in ("Z", "S") and (bayer == "RAW_BAYER" or (not bayer and "BAYER" in cfa)):
                import colour_demosaicing
                pattern = cfa.replace("BAYER_", "") or "RGGB"
                rad = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(rad, pattern)
                self._say(f"demosaiced ({pattern}, Malvar 2004)")
            else:
                rad = np.repeat(rad[..., None], 3, axis=2)
        elif rad.shape[2] != 3:
            raise ValueError(f"Expected 1 or 3 bands, got {rad.shape[2]}")
        return np.ascontiguousarray(rad, dtype=np.float64), (valid * 255).astype(np.uint8)

    def _infer_mask(self, rad: np.ndarray) -> None:
        from .mask import infer_mask                           # lazy: torch/timm
        c, m = self.config["color"], self.config["masking"]
        img8 = rad * float(c["scale_rad_to_int16"]) / float(c["scale_int8_to_int16"]) + float(c["offset_int8_to_int16"])
        mask, self.mask_probability, self.mask_card = infer_mask(
            img8, m["checkpoint"], m["device"], m["threshold"], m["dilate_kernel"])
        mask[self.mask_valid == 0] = 0
        self.mask = mask
        self._say(f"mask inferred: {100.0 * (mask > 0).mean():.1f}% included")

    def _pad_to_full_frame(self, rad: np.ndarray) -> np.ndarray:
        self.padding = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        if not self.config["resize"]["apply_padding"] or self.fn.family not in FULL_FRAME:
            return rad
        s = self.fn.downsample_scale
        W, H = (int(round(v * s)) for v in FULL_FRAME[self.fn.family])
        h, w = rad.shape[:2]
        L = self.label
        fs = first(label_get(L, "IMAGE.FIRST_LINE_SAMPLE", "MINI_HEADER.FIRST_LINE_SAMPLE", default=1))
        fl = first(label_get(L, "IMAGE.FIRST_LINE", "MINI_HEADER.FIRST_LINE", default=1))
        # FIRST_LINE(_SAMPLE) are 1-based full-resolution detector coordinates
        left, top = int(round((float(fs) - 1) * s)), int(round((float(fl) - 1) * s))
        if left + w > W or top + h > H:
            raise ValueError(f"{self.path.name}: sub-frame ({w}x{h} at {left},{top}) does not fit the "
                             f"{W}x{H} detector frame — check the downsample code / label.")
        right, bottom = W - left - w, H - top - h
        self.padding = {"left": left, "right": right, "top": top, "bottom": bottom}
        if any(self.padding.values()):
            rad = _pad(rad, left, right, top, bottom)
            self.mask_valid = _pad(self.mask_valid, left, right, top, bottom)
            self.mask = _pad(self.mask, left, right, top, bottom)
            self._say(f"padded {w}x{h} -> {W}x{H} (left {left}, right {right}, top {top}, bottom {bottom})")
        self.intrinsics = self.intrinsics.shifted(left, top, W, H)
        return rad

    def _apply_xml_intrinsics(self) -> None:
        cm = self.config["camera_model"]
        pattern = cm["xml_by_family"].get(self.fn.family) if cm["intrinsics_from_xml"] else None
        if not pattern:
            return
        if not self.config["resize"]["apply_padding"]:
            raise ValueError("intrinsics_from_xml requires resize.apply_padding (XML models are full-frame).")
        xml = params_dir() / "m20_cmods" / pattern.format(eye=self.fn.eye or "")
        scale = self.fn.downsample_scale / float(cm["xml_scale_by_family"].get(self.fn.family, 1.0))
        intr = intrinsics_from_metashape_xml(xml, scale)
        if (intr.width, intr.height) != (self.intrinsics.width, self.intrinsics.height):
            raise ValueError(f"XML calibration is {intr.width}x{intr.height} after scaling by {scale:g}, "
                             f"image is {self.intrinsics.width}x{self.intrinsics.height}")
        self.intrinsics = intr
        self._say(f"intrinsics replaced by {xml.name}")

    def _extra_padding(self, rad: np.ndarray) -> np.ndarray:
        r = self.config["resize"]
        if not (r["extra_padding"] and r["apply_padding"]):
            return rad
        h, w = rad.shape[:2]
        px, py = int(w * r["extra_fraction"] / 2), int(h * r["extra_fraction"] / 2)
        rad = _pad(rad, px, px, py, py)
        self.mask_valid = _pad(self.mask_valid, px, px, py, py)
        self.mask = _pad(self.mask, px, px, py, py)
        self.intrinsics = self.intrinsics.shifted(px, py, w + 2 * px, h + 2 * py)
        self.padding_extra = {"x": px, "y": py}
        return rad

    def _apply_static_mask(self) -> None:
        d = self.config["masking"]["static_masks_dir"]
        if not d:
            return
        f = Path(d) / f"{self.fn.camera_group}.png"
        if not f.is_file():
            return
        sm = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if sm is None or sm.shape != self.mask.shape:
            raise ValueError(f"Static mask {f} must be a {self.mask.shape[1]}x{self.mask.shape[0]} grayscale PNG")
        self.mask[sm == 0] = 0
        self._say(f"static mask {f.name} applied")

    def _undistort(self, rad: np.ndarray) -> np.ndarray:
        intr = self.intrinsics
        h, w = rad.shape[:2]
        f_new = 0.5 * (intr.fx + intr.fy)
        if self.config["resize"]["recenter_principal_point"]:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        else:
            cx, cy = intr.cx, intr.cy
        K_new = np.array([[f_new, 0, cx], [0, f_new, cy], [0, 0, 1.0]])
        m1, m2 = cv2.initUndistortRectifyMap(intr.K, intr.opencv_dist(), np.eye(3), K_new, (w, h), cv2.CV_32FC1)
        remap = lambda a, interp: cv2.remap(a, m1, m2, interpolation=interp, borderMode=cv2.BORDER_CONSTANT)
        rad = remap(rad.astype(np.float32), cv2.INTER_LINEAR).astype(np.float64)
        # a pixel is valid / included only if all its sources were
        self.mask_valid = np.where(remap(self.mask_valid, cv2.INTER_LINEAR) == 255, 255, 0).astype(np.uint8)
        self.mask = np.where(remap(self.mask, cv2.INTER_LINEAR) == 255, 255, 0).astype(np.uint8)
        rad[self.mask_valid == 0] = 0.0
        self.intrinsics = Intrinsics(w, h, K_new, source=intr.source + " -> undistorted")
        return rad

    # ---------------------------------------------------------------- outputs
    def rgba(self, bits: int = 16) -> np.ndarray:
        """RGB + alpha; alpha = reconstruction mask at full scale."""
        im = self.image_int16 if bits == 16 else self.image_int8
        alpha = np.where(self.mask > 0, np.iinfo(im.dtype).max, 0).astype(im.dtype)
        return np.dstack([im, alpha])

    @property
    def reference(self) -> list:
        """[stem, X(E), Y(N), Z(U), yaw, pitch, roll] for Metashape reference import."""
        return [self.fn.stem, *self.pose.C.tolist(), *self.pose.metashape_ypr_deg().tolist()]

    @property
    def meta(self) -> Dict[str, Any]:
        from . import VERSION_TAG
        L = self.label
        return {
            "mppp_version": VERSION_TAG,
            "source_product": self.path.name,
            "filename": self.fn.to_dict(),
            "site": self.site, "drive": self.drive, "sol": self.sol,
            "sclk": self.fn.sclk, "start_time_utc": str(label_get(L, "START_TIME", default="")),
            "LMST": self.LMST, "LTST": self.LTST, "L_s_deg": self.L_s,
            "solar_azimuth_deg": self.solar_az, "solar_elevation_deg": self.solar_el,
            "tau_estimated": self.tau_estimated, "tau_reference": self.tau_ref,
            "scale_zenith": self.scale_zenith, "white_balance": self.white_balance.tolist(),
            "radiance_scaling_factor": self.scale_dn_to_rad, "radiance_offset": self.offset_dn_to_rad,
            "camera_group": self.fn.camera_group,
            "stereo_partner": self.fn.stereo_partner_stem,
            "zoom_mm": self.fn.zoom_mm,
            "focus_position_count": label_get(L, "INSTRUMENT_STATE_PARMS.FOCUS_POSITION_COUNT"),
            "zoom_position_count": label_get(L, "INSTRUMENT_STATE_PARMS.ZOOM_POSITION_COUNT"),
            "exposure_duration_ms": label_float(L, "INSTRUMENT_STATE_PARMS.EXPOSURE_DURATION"),
            "native_size": list(self.native_size), "padding": self.padding,
            "undistorted": self.undistorted,
            "intrinsics": self.intrinsics.to_dict(),
            "intrinsics_label": self.intrinsics_label.to_dict(),
            "cahvor_O_A_angle_deg": self.cahvor.o_a_angle_deg(),
            "pose": self.pose.to_dict(), "geo": self.geo,
            "mask": {"inferred": self.mask_card is not None,
                     "inference_skipped": bool(getattr(self, "mask_inference_skipped", False)),
                     "included_fraction": float((self.mask > 0).mean()),
                     "valid_fraction": float((self.mask_valid > 0).mean()),
                     "model": None if self.mask_card is None else
                     {k: self.mask_card.get(k) for k in ("release_name", "name", "backbone", "stride4", "threshold",
                                                          "dilate_kernel", "val_iou")}},
            "log": list(self.log),
        }
