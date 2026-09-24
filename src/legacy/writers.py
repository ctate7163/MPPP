from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
from PIL import Image, PngImagePlugin
import tifffile as tiff

def _build_xmp_packet(mppp_json: Dict[str, Any]) -> str:
    payload = json.dumps(mppp_json, separators=(",", ":"), ensure_ascii=False)
    # Minimal XMP wrapper with a custom namespace carrying your JSON
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:mppp="http://mppp.example/ns/1.0/">'
        f'<mppp:payload>{payload}</mppp:payload>'
        '</rdf:Description>'
        '</rdf:RDF>'
        '</x:xmpmeta>'
        '<?xpacket end="w"?>'
    )

def save_image_with_camera_metadata(
    image: np.ndarray,
    save_path: Path | str,
    bit_depth: int = 16,                   # 8 or 16
    K: Optional[np.ndarray] = None,        # 3x3 intrinsics
    distortion: Optional[Dict[str, Any]] = None,  # {"k":[...], "p":[...], ...}
    camera_to_world: Optional[np.ndarray] = None, # 4x4 row-major
    model_type: str = "OPENCV",
    camera_model: Optional[str] = None,    # e.g., "M2020_NAVCAM_LEFT"
    camera_serial: Optional[str] = None,
    color_space: str = "linear_rgb",
    gamma: float = 1.0,
    undistorted: bool = False,
    has_alpha_mask: bool = True,
    extra_text: Optional[Dict[str, str]] = None,   # extra tEXt/iTXt pairs
) -> Path:
    """
    Save a color image (RGB or RGBA) as 8-bit PNG or 16-bit TIFF with structured metadata.
    This function DOES NOT modify pixel values. Any dtype/shape mismatch raises ValueError.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize bit_depth synonyms
    if bit_depth in ("uint8", 8):
        bit_depth = 8
    elif bit_depth in ("uint16", 16):
        bit_depth = 16
    else:
        raise ValueError("bit_depth must be 8 or 16 (or 'uint8'/'uint16').")

    # --- Validate image array without altering it ---
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must be HxWx3 or HxWx4 color. Received shape: %r" % (image.shape,))
    if bit_depth == 8 and image.dtype != np.uint8:
        raise ValueError(f"For 8-bit PNG, image dtype must be uint8 (got {image.dtype}).")
    if bit_depth == 16 and image.dtype != np.uint16:
        raise ValueError(f"For 16-bit TIFF, image dtype must be uint16 (got {image.dtype}).")
    if has_alpha_mask and image.shape[2] != 4:
        raise ValueError("has_alpha_mask=True but image has no alpha channel (expected RGBA).")
    if (not has_alpha_mask) and image.shape[2] == 4:
        # We keep pixels intact; just flag the inconsistency for the caller.
        raise ValueError("has_alpha_mask=False but image includes an alpha channel (RGBA).")

    H, W, C = image.shape

    # --- Build simple key/value tags (strings only) ---
    simple_tags: Dict[str, str] = {}
    if camera_model:
        simple_tags["CameraModel"] = str(camera_model)
    if camera_serial:
        simple_tags["CameraSerial"] = str(camera_serial)

    simple_tags["ModelType"] = model_type
    simple_tags["ImageSizePx"] = f"{W},{H}"
    simple_tags["Channels"] = str(C)
    simple_tags["BitDepthPerChannel"] = str(bit_depth)
    simple_tags["ColorSpace"] = color_space
    simple_tags["Gamma"] = f"{gamma:.6g}"
    simple_tags["Undistorted"] = "true" if undistorted else "false"
    simple_tags["HasAlphaMask"] = "true" if has_alpha_mask else "false"

    if K is not None:
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3; got {K.shape}.")
        simple_tags["FocalPixels"] = f"{K[0,0]:.8g},{K[1,1]:.8g}"
        simple_tags["PrincipalPointPx"] = f"{K[0,2]:.8g},{K[1,2]:.8g}"

    if distortion is not None:
        simple_tags["Distortion"] = json.dumps(distortion, separators=(",", ":"))

    if camera_to_world is not None:
        c2w = np.asarray(camera_to_world, dtype=float)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera_to_world must be 4x4; got {c2w.shape}.")
        simple_tags["CameraToWorld"] = json.dumps(c2w.tolist(), separators=(",", ":"))

    if extra_text:
        for k, v in extra_text.items():
            if isinstance(k, str) and isinstance(v, str):
                simple_tags[k] = v

    # --- Rich XMP payload ---
    mppp_json = {
        "model_type": model_type,
        "K": K.tolist() if K is not None else None,
        "distortion": distortion,
        "camera_to_world": camera_to_world.tolist() if camera_to_world is not None else None,
        "image_size": [W, H],
        "channels": C,
        "bit_depth": bit_depth,
        "undistorted": bool(undistorted),
        "color_space": color_space,
        "gamma": float(gamma),
        "mask_semantics": "alpha=valid_mask" if has_alpha_mask else "alpha=none",
        "camera_model": camera_model,
        "camera_serial": camera_serial,
    }
    xmp_packet = _build_xmp_packet(mppp_json)

    # --- Save paths ---
    if bit_depth == 8:
        # PNG path (8-bit per channel only). Pillow requires mode "RGB"/"RGBA".
        mode = "RGBA" if C == 4 else "RGB"
        img = Image.fromarray(image, mode=mode)

        info = PngImagePlugin.PngInfo()
        # Simple iTXt key/values
        for k, v in simple_tags.items():
            info.add_itxt(k, v, lang="", tkey="")
        # XMP under standard key
        info.add_itxt("XML:com.adobe.xmp", xmp_packet, lang="", tkey="")

        save_path = save_path.with_suffix(".png") if save_path.suffix.lower() not in (".png",) else save_path
        img.save(save_path, pnginfo=info, compress_level=4, format="PNG")
    else:
        # TIFF path (true 16-bit per channel RGB/RGBA)
        # Embed XMP via tag 700 (BYTE); include ImageDescription for easy discovery too.
        save_path = save_path.with_suffix(".tif") if save_path.suffix.lower() not in (".tif", ".tiff") else save_path

        # TIFF extra tags:
        # - 700: XMP packet (BYTE)
        # - 270: ImageDescription (ASCII)
        extratags = [
            (700, 'B', len(xmp_packet.encode('utf-8')), xmp_packet.encode('utf-8'), True),
            (270, 's', 0, json.dumps(simple_tags, separators=(",", ":")), True),
        ]

        # Let tifffile infer samples/pixel from shape; ensure correct photometric
        photometric = 'RGB' if C in (3, 4) else None  # (validated above)
        tiff.imwrite(
            save_path,
            image,
            photometric=photometric,
            metadata=None,           # avoid tifffile JSON; we control XMP + 270
            extratags=extratags,
            compression='deflate',   # lossless; change if you prefer 'none'
        )

    return save_path



# def _build_xmp_packet(mppp_json: Dict[str, Any]) -> str:
#     """
#     Wrap a compact JSON blob in a minimal XMP packet with a custom mppp namespace.
#     This stays well-formed XMP that apps preserve even if they don't interpret it.
#     """
#     # JSON string small/compact
#     payload = json.dumps(mppp_json, separators=(",", ":"), ensure_ascii=False)

#     # Minimal XMP packet with a custom namespace and a CDATA JSON payload
#     xmp = f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
# <x:xmpmeta xmlns:x="adobe:ns:meta/">
#  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
#           xmlns:mppp="http://mppp.example/ns/1.0/">
#   <rdf:Description rdf:about=""
#       xmlns:xmp="http://ns.adobe.com/xap/1.0/">
#     <mppp:CameraModel><![CDATA[{payload}]]></mppp:CameraModel>
#   </rdf:Description>
#  </rdf:RDF>
# </x:xmpmeta>
# <?xpacket end="w"?>"""
#     return xmp


def filter_none_lists(data):
    """
    Filters out None values and returns only lists.

    Args:
        data (list): A list containing sublists and/or None values.

    Returns:
        list: A list of lists with None values removed.
    """
    return [item for item in data if isinstance(item, list)]

def save_refs( refs, save_path, norm=False ):
    
    import csv
    
    refs_save = np.array( filter_none_lists(refs) )

    if norm:
        XYZs = np.array(refs)[1:,1:4].astype('float')
        XYZs_mean = np.floor( np.mean( XYZs, axis=0)/10 )*10
        XYZs_norm = XYZs - XYZs_mean
        refs_save[1:,1:4] = XYZs_norm

    refs_save = refs_save.tolist()

    with open(save_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerows(refs_save)

    print( "saved", save_path)

    return refs_save

